//! Structural equality (`==` / `!=`) for aggregate types: records, enums, and
//! lists.
//!
//! Each aggregate type lowers to a cached `func.func @__eq_<n>(lhs, rhs) -> i1`
//! that compares its operands structurally. The helper is registered *before*
//! its body is lowered so recursive types (a list element that is itself a
//! list, or a self-referential enum) resolve to the in-progress symbol instead
//! of infinitely recursing at codegen time. Scalar fields are compared inline.

use crate::types::{Monotype, TypeFunc};
use melior::dialect::{arith, func, scf};
use melior::ir::{
    attribute::{BoolAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
    operation::OperationBuilder,
    r#type::{FunctionType, IntegerType},
    Block, BlockLike, Location, Region, RegionLike, Type, Value,
};

use super::Module;
use super::apply::default_free_vars;
use super::closures::load_field;
use super::enums::{enum_struct_type, enum_variant_fields};
use super::lists::{cell_struct_type, integer_constant, list_elem, list_is_null};
use super::records::{extract_field, record_fields};
use super::stmt::{ensure_extern, ptrtoint_i64};
use super::types::lower_type;

/// Lower `lhs == rhs` for values of the (defaulted) type `typ`, returning an
/// `i1`. Scalars compare inline; aggregate types lower through a cached helper
/// function.
pub(crate) fn lower_equality<'c, 'a>(
    typ: &Monotype,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    if is_aggregate(typ) {
        let symbol = eq_function(typ, module, location)?;
        let call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, &symbol),
            &[lhs, rhs],
            &[IntegerType::new(module.context, 1).into()],
            location,
        );
        block
            .append_operation(call)
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into)
    } else {
        lower_scalar_eq(typ, lhs, rhs, block, module, location)
    }
}

/// Whether `typ` is a list, enum, or record (types that need a helper).
pub(crate) fn is_aggregate(typ: &Monotype) -> bool {
    matches!(
        typ,
        Monotype::TypeFuncApplication(f, _)
            if matches!(**f, TypeFunc::List | TypeFunc::Enum(_) | TypeFunc::Rec | TypeFunc::Tuple)
    )
}

/// Emit (or reuse) the `func.func @__eq_<n>(lhs: T, rhs: T) -> i1` performing
/// structural equality on `T`, returning its symbol.
fn eq_function<'c>(
    typ: &Monotype,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<String, String> {
    let concrete = default_free_vars(typ);
    let key = format!("{concrete:?}");
    if let Some(symbol) = module.eq_functions.get(&key) {
        return Ok(symbol.clone());
    }

    let symbol = format!("__eq_{}", module.eq_counter);
    module.eq_counter += 1;
    // Register before lowering so recursive references resolve to this symbol.
    module.eq_functions.insert(key, symbol.clone());

    let mlir = lower_type(&concrete, module)?;
    let i1: Type = IntegerType::new(module.context, 1).into();

    let block = Block::new(&[(mlir, location), (mlir, location)]);
    let lhs = block.argument(0).map_err(|e| e.to_string())?.into();
    let rhs = block.argument(1).map_err(|e| e.to_string())?.into();

    let result = eq_body(&concrete, lhs, rhs, &block, module, location)?;
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[mlir, mlir], &[i1]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, &symbol),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.functions += 1;

    Ok(symbol)
}

/// Lower the body of an equality helper for values `lhs`/`rhs` of type `typ`.
fn eq_body<'c, 'a>(
    typ: &Monotype,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    match typ {
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::List) => {
            let elem = list_elem(typ).expect("a list type has an element type");
            lower_list_eq(&elem, lhs, rhs, block, module, location)
        }
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Enum(_)) => {
            lower_enum_eq(typ, lhs, rhs, block, module, location)
        }
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Rec) => {
            lower_record_eq(typ, lhs, rhs, block, module, location)
        }
        Monotype::TypeFuncApplication(f, args) if matches!(**f, TypeFunc::Tuple) => {
            lower_tuple_eq(lhs, rhs, block, module, location, args)
        }
        _ => lower_scalar_eq(typ, lhs, rhs, block, module, location),
    }
}

fn lower_tuple_eq<'c, 'a>(
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
    args: &[Monotype]
) -> Result<Value<'c, 'a>, String> {
    let mut acc = bool_constant(module, block, true, location)?;
    for (i, field_ty) in args.iter().enumerate() {
        let concrete = default_free_vars(field_ty);
        let field_mlir = lower_type(&concrete, module)?;
        let l = extract_field(module, block, lhs, i as i32, field_mlir, location)?;
        let r = extract_field(module, block, rhs, i as i32, field_mlir, location)?;
        let eq = lower_equality(&concrete, l, r, block, module, location)?;
        acc = and_values(block, acc, eq, location)?;
    }
    Ok(acc)
}

/// Inline scalar equality: `arith.cmpi` for int/bool, `arith.cmpf` for float,
/// and `strcmp` for strings.
fn lower_scalar_eq<'c, 'a>(
    typ: &Monotype,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    match typ {
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Int | TypeFunc::Bool) => {
            let op = arith::cmpi(module.context, arith::CmpiPredicate::Eq, lhs, rhs, location);
            block
                .append_operation(op)
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Float) => {
            let op = arith::cmpf(module.context, arith::CmpfPredicate::Oeq, lhs, rhs, location);
            block
                .append_operation(op)
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Str) => {
            let i64_type: Type = IntegerType::new(module.context, 64).into();
            let i32_type: Type = IntegerType::new(module.context, 32).into();
            ensure_extern(module, "strcmp", &[i64_type, i64_type], &[i32_type])?;
            let lhs_i64 = ptrtoint_i64(module, block, lhs, location)?;
            let rhs_i64 = ptrtoint_i64(module, block, rhs, location)?;
            let call = func::call(
                module.context,
                FlatSymbolRefAttribute::new(module.context, "strcmp"),
                &[lhs_i64, rhs_i64],
                &[i32_type],
                location,
            );
            let cmp: Value<'c, 'a> = block
                .append_operation(call)
                .result(0)
                .map_err(|e| e.to_string())?
                .into();
            let zero = integer_constant(module, block, 32, 0, location)?;
            let eq = arith::cmpi(module.context, arith::CmpiPredicate::Eq, cmp, zero, location);
            block
                .append_operation(eq)
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
        other => Err(format!(
            "codegen: unsupported operand type for equality: {other:?}"
        )),
    }
}

/// Record equality: each field must be equal (fields are compared with
/// [`lower_equality`], so nested aggregates lower through their own helpers).
fn lower_record_eq<'c, 'a>(
    typ: &Monotype,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let fields = record_fields(typ)?;
    let mut acc = bool_constant(module, block, true, location)?;
    for (i, (_, field_ty)) in fields.iter().enumerate() {
        let concrete = default_free_vars(field_ty);
        let field_mlir = lower_type(&concrete, module)?;
        let l = extract_field(module, block, lhs, i as i32, field_mlir, location)?;
        let r = extract_field(module, block, rhs, i as i32, field_mlir, location)?;
        let eq = lower_equality(&concrete, l, r, block, module, location)?;
        acc = and_values(block, acc, eq, location)?;
    }
    Ok(acc)
}

/// List equality via an `scf.while` loop: walk both lists in lockstep while
/// both are non-null and the heads are equal; equal iff both end (null) with
/// no mismatched element. Element equality recurses through
/// [`lower_equality`], so nested lists/enums/records are handled.
fn lower_list_eq<'c, 'a>(
    elem_ty: &Monotype,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let ptr: Type = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let i1: Type = IntegerType::new(module.context, 1).into();
    let result_types = [ptr, ptr, i1];

    let init_eq = bool_constant(module, block, true, location)?;

    // Loop-carried `(lhs, rhs, equal_so_far)`. The "before" region continues
    // while both lists are non-null and no mismatch has been seen yet.
    let before_block = Block::new(&[(ptr, location), (ptr, location), (i1, location)]);
    let b_l: Value<'c, '_> = before_block.argument(0).map_err(|e| e.to_string())?.into();
    let b_r: Value<'c, '_> = before_block.argument(1).map_err(|e| e.to_string())?.into();
    let b_eq: Value<'c, '_> = before_block.argument(2).map_err(|e| e.to_string())?.into();
    let b_l_null = list_is_null(b_l, &before_block, module, location)?;
    let b_r_null = list_is_null(b_r, &before_block, module, location)?;
    let b_l_nonnull = not_value(module, &before_block, b_l_null, location)?;
    let b_r_nonnull = not_value(module, &before_block, b_r_null, location)?;
    let b_both = and_values(&before_block, b_l_nonnull, b_r_nonnull, location)?;
    let b_cond = and_values(&before_block, b_both, b_eq, location)?;
    before_block.append_operation(scf::condition(b_cond, &[b_l, b_r, b_eq], location));
    let before_region = Region::new();
    before_region.append_block(before_block);

    // The "after" region compares the two heads and advances both tails.
    let after_block = Block::new(&[(ptr, location), (ptr, location), (i1, location)]);
    let a_l: Value<'c, '_> = after_block.argument(0).map_err(|e| e.to_string())?.into();
    let a_r: Value<'c, '_> = after_block.argument(1).map_err(|e| e.to_string())?.into();
    let a_eq: Value<'c, '_> = after_block.argument(2).map_err(|e| e.to_string())?.into();

    let elem_mlir = lower_type(elem_ty, module)?;
    let cell = cell_struct_type(module, elem_mlir)?;
    let a_l_head = load_field(module, &after_block, a_l, cell, 0, elem_mlir, location)?;
    let a_l_tail = load_field(module, &after_block, a_l, cell, 1, ptr, location)?;
    let a_r_head = load_field(module, &after_block, a_r, cell, 0, elem_mlir, location)?;
    let a_r_tail = load_field(module, &after_block, a_r, cell, 1, ptr, location)?;
    let head_eq = lower_equality(elem_ty, a_l_head, a_r_head, &after_block, module, location)?;
    let new_eq = and_values(&after_block, a_eq, head_eq, location)?;
    after_block.append_operation(scf::r#yield(&[a_l_tail, a_r_tail, new_eq], location));
    let after_region = Region::new();
    after_region.append_block(after_block);

    let while_op = scf::r#while(
        &[lhs, rhs, init_eq],
        &result_types,
        before_region,
        after_region,
        location,
    );
    let appended = block.append_operation(while_op);
    let l_final: Value<'c, 'a> = appended.result(0).map_err(|e| e.to_string())?.into();
    let r_final: Value<'c, 'a> = appended.result(1).map_err(|e| e.to_string())?.into();
    let eq_final: Value<'c, 'a> = appended.result(2).map_err(|e| e.to_string())?.into();

    // Equal iff the loop never found a mismatch and both lists are exhausted.
    let l_null = list_is_null(l_final, block, module, location)?;
    let r_null = list_is_null(r_final, block, module, location)?;
    let both_null = and_values(block, l_null, r_null, location)?;
    and_values(block, eq_final, both_null, location)
}

/// Enum equality: discriminants must match, and (for a non-nullary variant)
/// every payload field must match.
fn lower_enum_eq<'c, 'a>(
    typ: &Monotype,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let enum_name = match typ {
        Monotype::TypeFuncApplication(f, _) => match &**f {
            TypeFunc::Enum(n) => n.clone(),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    let variant_count = module
        .enums
        .get(&enum_name)
        .map(|l| l.variants.len())
        .ok_or_else(|| format!("codegen: unknown enum `{enum_name}`"))?;

    let enum_struct = enum_struct_type(module)?;
    let i32: Type = IntegerType::new(module.context, 32).into();
    let disc_l = load_field(module, block, lhs, enum_struct, 0, i32, location)?;
    let disc_r = load_field(module, block, rhs, enum_struct, 0, i32, location)?;

    let disc_eq_op = arith::cmpi(
        module.context,
        arith::CmpiPredicate::Eq,
        disc_l,
        disc_r,
        location,
    );
    let disc_eq: Value<'c, 'a> = block
        .append_operation(disc_eq_op)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    let bool_type: Type = IntegerType::new(module.context, 1).into();

    // Discriminants equal: compare payloads by branching on the discriminant.
    let then_block = Block::new(&[]);
    let payload_eq = lower_enum_payload_eq(
        &enum_name,
        typ,
        disc_l,
        lhs,
        rhs,
        variant_count,
        0,
        &then_block,
        module,
        location,
    )?;
    then_block.append_operation(scf::r#yield(&[payload_eq], location));
    let then_region = Region::new();
    then_region.append_block(then_block);

    // Different variants are never equal.
    let else_block = Block::new(&[]);
    let f = bool_constant(module, &else_block, false, location)?;
    else_block.append_operation(scf::r#yield(&[f], location));
    let else_region = Region::new();
    else_region.append_block(else_block);

    let if_op = scf::r#if(disc_eq, &[bool_type], then_region, else_region, location);
    block
        .append_operation(if_op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Compare the payloads of `lhs`/`rhs`, whose (shared) discriminant is `disc`,
/// by branching on the discriminant across the enum's variants.
fn lower_enum_payload_eq<'c, 'a: 'b, 'b>(
    enum_name: &str,
    typ: &Monotype,
    disc: Value<'c, 'a>,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    variant_count: usize,
    index: usize,
    block: &'b Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'b>, String> {
    // The last variant is guaranteed to be the one that matched.
    if index + 1 == variant_count {
        return lower_variant_payload_eq(enum_name, typ, index, lhs, rhs, block, module, location);
    }

    let bool_type: Type = IntegerType::new(module.context, 1).into();
    let idx = integer_constant(module, block, 32, index as i64, location)?;
    let cond_op = arith::cmpi(
        module.context,
        arith::CmpiPredicate::Eq,
        disc,
        idx,
        location,
    );
    let cond: Value<'c, 'b> = block
        .append_operation(cond_op)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    let then_block = Block::new(&[]);
    let then_val =
        lower_variant_payload_eq(enum_name, typ, index, lhs, rhs, &then_block, module, location)?;
    then_block.append_operation(scf::r#yield(&[then_val], location));
    let then_region = Region::new();
    then_region.append_block(then_block);

    let else_block = Block::new(&[]);
    let else_val = lower_enum_payload_eq(
        enum_name,
        typ,
        disc,
        lhs,
        rhs,
        variant_count,
        index + 1,
        &else_block,
        module,
        location,
    )?;
    else_block.append_operation(scf::r#yield(&[else_val], location));
    let else_region = Region::new();
    else_region.append_block(else_block);

    let if_op = scf::r#if(cond, &[bool_type], then_region, else_region, location);
    block
        .append_operation(if_op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Compare the payload fields of enum variant `variant_index` (or return
/// `true` for a nullary variant, whose data pointer is always null).
fn lower_variant_payload_eq<'c, 'a>(
    enum_name: &str,
    typ: &Monotype,
    variant_index: usize,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let fields = enum_variant_fields(module, typ, enum_name, variant_index)?;
    if fields.is_empty() {
        return bool_constant(module, block, true, location);
    }

    let enum_struct = enum_struct_type(module)?;
    let ptr: Type = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let data_l = load_field(module, block, lhs, enum_struct, 1, ptr, location)?;
    let data_r = load_field(module, block, rhs, enum_struct, 1, ptr, location)?;

    let field_mlirs: Vec<Type> = fields
        .iter()
        .map(|f| lower_type(&default_free_vars(f), module))
        .collect::<Result<_, _>>()?;
    let names: Vec<String> = field_mlirs.iter().map(|t| t.to_string()).collect();
    let payload_struct = Type::parse(
        module.context,
        &format!("!llvm.struct<({})>", names.join(", ")),
    )
    .ok_or_else(|| "codegen: failed to create payload struct type".to_string())?;

    let mut acc = bool_constant(module, block, true, location)?;
    for (j, f) in fields.iter().enumerate() {
        let fmlir = field_mlirs[j];
        let fl = load_field(module, block, data_l, payload_struct, j as i32, fmlir, location)?;
        let fr = load_field(module, block, data_r, payload_struct, j as i32, fmlir, location)?;
        let eq = lower_equality(&default_free_vars(f), fl, fr, block, module, location)?;
        acc = and_values(block, acc, eq, location)?;
    }
    Ok(acc)
}

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------

/// Append a `true`/`false` `i1` constant to `block`.
fn bool_constant<'c, 'a>(
    module: &Module<'c>,
    block: &'a Block<'c>,
    b: bool,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let op = arith::constant(
        module.context,
        BoolAttribute::new(module.context, b).into(),
        location,
    );
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Logical AND of two `i1` values (`arith.andi`).
fn and_values<'c, 'a>(
    block: &'a Block<'c>,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let op = OperationBuilder::new("arith.andi", location)
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
        .map_err(|e| e.to_string())?;
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Logical negation of an `i1` value (`arith.xori` with `true`).
fn not_value<'c, 'a>(
    module: &Module<'c>,
    block: &'a Block<'c>,
    value: Value<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let one = bool_constant(module, block, true, location)?;
    let op = OperationBuilder::new("arith.xori", location)
        .add_operands(&[value, one])
        .enable_result_type_inference()
        .build()
        .map_err(|e| e.to_string())?;
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}
