//! Enum representation: tagged values `{ disc: i32, data: !llvm.ptr }`.

use crate::types::{Monotype, TypeFunc};
use melior::dialect::arith;
use melior::ir::{
    r#type::IntegerType,
    Block, BlockLike, Location, Type, Value,
};
use std::collections::HashMap;

use super::Module;
use super::closures::{load_field, store_field};
use super::lists::{cell_struct_type, empty_list, integer_constant, malloc_call};
use super::records::extract_field;

// ----------------------------------------------------------------------------
// Enums
// ----------------------------------------------------------------------------

/// An enum value is a heap struct `{ disc: i32, data: !llvm.ptr }`; `data`
/// points to the variant payload (or is null for nullary variants).
pub(crate) fn enum_struct_type<'c>(module: &Module<'c>) -> Result<Type<'c>, String> {
    Type::parse(module.context, "!llvm.struct<(i32, !llvm.ptr)>").ok_or_else(|| {
        "codegen: failed to create enum struct type `!llvm.struct<(i32, !llvm.ptr)>`"
            .to_string()
    })
}

/// Allocate an enum value with discriminant `variant_index` and the given
/// payload (use a null pointer for nullary constructors).
pub(crate) fn build_enum_value<'c, 'a>(
    module: &mut Module<'c>,
    block: &'a Block<'c>,
    variant_index: usize,
    payload: Value<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let enum_struct = enum_struct_type(module)?;
    let value = malloc_call(module, block, 16, location)?;
    let disc = integer_constant(module, block, 32, variant_index as i64, location)?;
    store_field(module, block, value, enum_struct, 0, disc, location)?;
    store_field(module, block, value, enum_struct, 1, payload, location)?;
    Ok(value)
}

/// Build a heap struct of the given fields, sized `size` bytes.
///
/// `size` is the total byte size of the payload struct (computed from the
/// field types by [`super::types::monotype_size`]); a record-valued field is a
/// struct value larger than the single pointer slot this used to assume.
pub(crate) fn build_payload<'c, 'a>(
    module: &mut Module<'c>,
    block: &'a Block<'c>,
    fields: &[(Value<'c, 'a>, Type<'c>)],
    size: i64,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    if fields.is_empty() {
        return empty_list(block, module, location);
    }
    let names: Vec<String> = fields.iter().map(|(_, t)| t.to_string()).collect();
    let payload_struct =
        Type::parse(module.context, &format!("!llvm.struct<({})>", names.join(", ")))
            .ok_or_else(|| "codegen: failed to create payload struct type".to_string())?;
    let ptr = malloc_call(module, block, size, location)?;
    for (i, (value, _)) in fields.iter().enumerate() {
        store_field(module, block, ptr, payload_struct, i as i32, *value, location)?;
    }
    Ok(ptr)
}

/// Bindings produced by destructuring a match pattern.
pub(crate) enum PatternBind<'c> {
    /// `x::xs`: load the head/tail fields of a cons cell.
    Cons { head_name: String, head_type: Type<'c>, tail_name: String },
    /// A constructor pattern: load the variable sub-patterns from the enum's
    /// `data` pointer. `field_types` are the full payload field types in
    /// declaration order (for the payload struct layout); `binds` maps each
    /// bound variable to its payload field index (literals bind nothing).
    Enum { field_types: Vec<Type<'c>>, binds: Vec<(String, usize)> },
    /// `Foo { bar: n, .. }`: extract the named fields of a record scrutinee.
    /// Each entry is `(bound var, field type, field index)`.
    Record { fields: Vec<(String, Type<'c>, usize)> },
}

/// Load the bindings of a match pattern inside `block`.
pub(crate) fn destructure_pattern<'c, 'x>(
    binding: Option<PatternBind<'c>>,
    scrut: Value<'c, 'x>,
    block: &'x Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Vec<(String, Value<'c, 'x>)>, String> {
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    match binding {
        None => Ok(vec![]),
        Some(PatternBind::Cons { head_name, head_type, tail_name }) => {
            let cell = cell_struct_type(module, head_type)?;
            let head = load_field(module, block, scrut, cell, 0, head_type, location)?;
            let tail = load_field(module, block, scrut, cell, 1, ptr, location)?;
            Ok(vec![(head_name, head), (tail_name, tail)])
        }
        Some(PatternBind::Enum { field_types, binds }) => {
            let enum_struct = enum_struct_type(module)?;
            let data = load_field(module, block, scrut, enum_struct, 1, ptr, location)?;
            let names: Vec<String> = field_types.iter().map(|t| t.to_string()).collect();
            let payload_struct = Type::parse(
                module.context,
                &format!("!llvm.struct<({})>", names.join(", ")),
            )
            .ok_or_else(|| "codegen: failed to create payload struct type".to_string())?;
            let mut out = Vec::new();
            for (name, index) in binds {
                let v = load_field(
                    module,
                    block,
                    data,
                    payload_struct,
                    index as i32,
                    field_types[index],
                    location,
                )?;
                out.push((name, v));
            }
            Ok(out)
        }
        Some(PatternBind::Record { fields }) => {
            let mut out = Vec::new();
            for (name, ty, index) in fields {
                let v = extract_field(module, block, scrut, index as i32, ty, location)?;
                out.push((name, v));
            }
            Ok(out)
        }
    }
}

/// The field types of enum variant `variant_index`, with the enum's type
/// parameters substituted from the scrutinee type.
pub(crate) fn enum_variant_fields<'a>(
    module: &Module<'a>,
    scrut_typ: &Monotype,
    enum_name: &str,
    variant_index: usize,
) -> Result<Vec<Monotype>, String> {
    let layout = module
        .enums
        .get(enum_name)
        .ok_or_else(|| format!("codegen: unknown enum `{enum_name}`"))?;
    let (_, fields) = &layout.variants[variant_index];
    let args = match scrut_typ {
        Monotype::TypeFuncApplication(f, args) => match **f {
            TypeFunc::Enum(ref n) if n == enum_name => args.clone(),
            _ => vec![],
        },
        _ => vec![],
    };
    let mut map = HashMap::new();
    for (p, a) in layout.params.iter().zip(args.iter()) {
        map.insert(p.clone(), a.clone());
    }
    Ok(fields.iter().map(|f| f.instantiate(&mut map)).collect())
}

/// Load field `index` of an enum variant payload. `fields` are the payload
/// field MLIR types in declaration order; the payload struct type is derived
/// from them exactly as [`destructure_pattern`] does.
pub(crate) fn load_enum_payload_field<'c, 'x>(
    module: &Module<'c>,
    block: &'x Block<'c>,
    scrut: Value<'c, 'x>,
    fields: &[Type<'c>],
    index: i32,
    location: Location<'c>,
) -> Result<Value<'c, 'x>, String> {
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let enum_struct = enum_struct_type(module)?;
    let data = load_field(module, block, scrut, enum_struct, 1, ptr, location)?;
    let names: Vec<String> = fields.iter().map(|t| t.to_string()).collect();
    let payload_struct = Type::parse(
        module.context,
        &format!("!llvm.struct<({})>", names.join(", ")),
    )
    .ok_or_else(|| "codegen: failed to create payload struct type".to_string())?;
    load_field(
        module,
        block,
        data,
        payload_struct,
        index,
        fields[index as usize],
        location,
    )
}

/// Compare `scrut`'s discriminant against `variant_index`, returning an i1.
pub(crate) fn enum_disc_eq<'c, 'a>(
    module: &mut Module<'c>,
    block: &'a Block<'c>,
    scrut: Value<'c, '_>,
    variant_index: usize,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let enum_struct = enum_struct_type(module)?;
    let i32 = IntegerType::new(module.context, 32).into();
    let disc = load_field(module, block, scrut, enum_struct, 0, i32, location)?;
    let idx = integer_constant(module, block, 32, variant_index as i64, location)?;
    let cmp = arith::cmpi(module.context, arith::CmpiPredicate::Eq, disc, idx, location);
    block
        .append_operation(cmp)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}