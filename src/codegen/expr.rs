//! Expression lowering.

use crate::ast::*;
use crate::codegen::lists::ensure_malloc;
use crate::types::{Monotype, TypeFunc};
use melior::dialect::{arith, func, scf};
use melior::ir::ValueLike;
use melior::ir::{
    attribute::{BoolAttribute, FlatSymbolRefAttribute, IntegerAttribute},
    operation::OperationBuilder,
    r#type::IntegerType,
    Block, BlockLike, Location, Operation, Region, RegionLike, Type, Value,
};

use super::{Env, EnvEntry, Module};
use super::apply::{
    bind_in_env, default_free_vars, lower_abstraction, lower_application, lower_let, lower_literal,
    lower_variable,
};
use super::closures::load_field;
use super::enums::{
    PatternBind, destructure_pattern, enum_disc_eq, enum_variant_fields, load_enum_payload_field,
};
use super::equality::lower_equality;
use super::lists::{
    cell_struct_type, list_elem, list_is_null, lower_cons, lower_list, integer_constant,
};
use super::records::{extract_field, field_index, insert_field, record_fields, record_undef};
use super::stmt::{ensure_extern, ptrtoint_i64, inttoptr_ptr};
use super::types::lower_type;

// ----------------------------------------------------------------------------
// Expressions
// ----------------------------------------------------------------------------

/// Lower `expr` to MLIR ops inside the current function body, returning the
/// SSA value it produces.
///
/// Pointer map (ENode variant -> dialect ops):
///   Block(stmts, e)         -> emit statements into a nested region, return `e`
///   Match(scrut, cases)     -> `scf.if` chain on the discriminant
///   List(es)                -> heap-allocate via `llvm` malloc, or a struct
///                              header + element buffer
///   Cons(h, t)              -> prepend to a list header struct
pub(crate) fn lower_expr<'c, 'a>(
    expr: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<Value<'c, 'a>, String> {
    let location = module.location(&expr.pos);
    match &*expr.e {
        ENode::Literal(lit) => lower_literal(lit, block, module, location),
        ENode::Variable(_) => lower_variable(expr, block, module, env, location),
        ENode::Abstraction(binding, body) => lower_abstraction(expr, binding, body, block, module, env, location),
        ENode::Application(f, x) => lower_application(f, x, block, module, env, location),
        ENode::Let(name, e1, e2) => lower_let(name, e1, e2, block, module, env),
        ENode::IfElse(c, t, e) => lower_ifelse(c, t, e, &expr.typ, block, module, env, location),
        ENode::Block(stmts, e) => lower_block(stmts, e, block, module, env),
        ENode::Match(scrut, cases) => lower_match(scrut, cases, &expr.typ, block, module, env),
        ENode::Comparison(op, a, b) => lower_comparison(op, a, b, block, module, env, location),
        ENode::Arithmetic(op, a, b) => lower_arith(op, a, b, block, module, env, location),
        ENode::Logical(op, a, b) => lower_logical(op, a, b, block, module, env, location),
        ENode::Unary(op, e) => lower_unary(op, e, block, module, env, location),
        ENode::List(es) => lower_list(es, block, module, env, location),
        ENode::Cons(h, t) => lower_cons(h, t, block, module, env, location),
        ENode::FieldAccess(scrut, field) => {
            let scrutinee = lower_expr(scrut, block, module, env)?;
            let fields = record_fields(&default_free_vars(&scrut.typ))?;
            let index = field_index(&fields, field)?;
            let result_type = lower_type(&default_free_vars(&expr.typ), module)?;
            extract_field(module, block, scrutinee, index as i32, result_type, location)
        }
        ENode::Record(_, field_assns) => {
            let fields = record_fields(&default_free_vars(&expr.typ))?;
            let struct_type = lower_type(&default_free_vars(&expr.typ), module)?;
            let mut acc = record_undef(block, struct_type, location)?;
            for (i, (label, _)) in fields.iter().enumerate() {
                let fa = field_assns
                    .iter()
                    .find(|fa| &fa.field == label)
                    .ok_or_else(|| {
                        format!("codegen: missing field `{label}` in record construction")
                    })?;
                let value = lower_expr(&fa.exp, block, module, env)?;
                acc = insert_field(module, block, acc, i as i32, value, location)?;
            }
            Ok(acc)
        }
        ENode::With(scrut, field_assns) => {
            let mut acc = lower_expr(scrut, block, module, env)?;
            let fields = record_fields(&default_free_vars(&scrut.typ))?;
            for FieldAssn { field, exp } in field_assns {
                let index = field_index(&fields, field)?;
                let value = lower_expr(exp, block, module, env)?;
                acc = insert_field(module, block, acc, index as i32, value, location)?;
            }
            Ok(acc)
        }
        ENode::Tuple(exprs) => {
            let struct_type = lower_type(&default_free_vars(&expr.typ), module)?;
            let mut acc = record_undef(block, struct_type, location)?;
            for (i, e) in exprs.iter().enumerate() {
                let value = lower_expr(e, block, module, env)?;
                acc = insert_field(module, block, acc, i as i32, value, location)?;
            }
            Ok(acc)
        }
    }
}

/// Primitive kinds that binary/unary operators dispatch on.
enum Prim {
    Int,
    Float,
    Bool,
    Str,
}

/// Classify a (defaulted) primitive type; anything else is not a scalar
/// operand for these operators.
fn primitive_kind(typ: &Monotype) -> Result<Prim, String> {
    match default_free_vars(typ) {
        Monotype::TypeFuncApplication(f, _) if *f == TypeFunc::Int => Ok(Prim::Int),
        Monotype::TypeFuncApplication(f, _) if *f == TypeFunc::Float => Ok(Prim::Float),
        Monotype::TypeFuncApplication(f, _) if *f == TypeFunc::Bool => Ok(Prim::Bool),
        Monotype::TypeFuncApplication(f, _) if *f == TypeFunc::Str => Ok(Prim::Str),
        other => Err(format!(
            "codegen: unsupported operand type for binary/unary operation: {other:?}"
        )),
    }
}

/// Build a generic two-operand `arith` op with result type inference.
fn arith_binop<'c>(
    name: &str,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Result<Operation<'c>, String> {
    OperationBuilder::new(name, location)
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
        .map_err(|e| e.to_string())
}

/// Append a constant of value `n` of type `i32` to `block`.
fn i32_constant<'c, 'a>(
    module: &Module<'c>,
    block: &'a Block<'c>,
    n: i64,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let op = arith::constant(
        module.context,
        IntegerAttribute::new(IntegerType::new(module.context, 32).into(), n).into(),
        location,
    );
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

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

/// Lower `a OP b` to the integer or float `arith` op selected by the operand
/// type.
fn lower_arith<'c, 'a>(
    op: &ArithOp,
    e1: &Expr,
    e2: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let lhs = lower_expr(e1, block, module, env)?;
    let rhs = lower_expr(e2, block, module, env)?;

    if matches!((op, primitive_kind(&e1.typ)?), (ArithOp::Plus, Prim::Str)) {
        let i64_type: Type = IntegerType::new(module.context, 64).into();
        ensure_malloc(module)?;
        ensure_extern(module, "strlen", &[i64_type], &[i64_type])?;
        ensure_extern(module, "strcpy", &[i64_type, i64_type], &[i64_type])?;
        ensure_extern(module, "strcat", &[i64_type, i64_type], &[i64_type])?;

        let lhs_i64 = ptrtoint_i64(module, block, lhs, location)?;
        let rhs_i64 = ptrtoint_i64(module, block, rhs, location)?;

        let lhs_len_call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "strlen"),
            &[lhs_i64],
            &[i64_type],
            lhs.location()
        );
        let rhs_len_call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "strlen"),
            &[rhs_i64],
            &[i64_type],
            rhs.location()
        );

        let lhs_len: Value<'_, '_> = block
            .append_operation(lhs_len_call)
            .result(0)
            .map_err(|e| e.to_string())?
            .into();
        let rhs_len: Value<'_, '_> = block
            .append_operation(rhs_len_call)
            .result(0)
            .map_err(|e| e.to_string())?
            .into();
        let total = block
            .append_operation(arith::addi(lhs_len, rhs_len, location))
            .result(0)
            .map_err(|e| e.to_string())?
            .into();
        let one_i64 = integer_constant(module, block, 64, 1, location)?;
        let add_one = arith::addi(total, one_i64, location);
        let terminated_len: Value<'_, '_> = block
            .append_operation(add_one)
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into)?;
        let malloc_call_op = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "malloc"),
            &[terminated_len],
            &[i64_type],
            location,
        );
        let buf_i64: Value<'_, '_> = block
            .append_operation(malloc_call_op)
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into)?;
        let strcpy_call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "strcpy"),
            &[buf_i64, lhs_i64],
            &[i64_type],
            lhs.location()
        );
        let dest = block
            .append_operation(strcpy_call)
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into)?;
        let strcat_call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "strcat"),
            &[dest, rhs_i64],
            &[i64_type],
            location,
        );
        let raw_i64: Value<'_, '_> = block
            .append_operation(strcat_call)
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into)?;
        return inttoptr_ptr(module, block, raw_i64, location);
    }

    let op_name = match (op, primitive_kind(&e1.typ)?) {
        (ArithOp::Plus, Prim::Int) => "arith.addi",
        (ArithOp::Plus, Prim::Float) => "arith.addf",
        (ArithOp::Minus, Prim::Int) => "arith.subi",
        (ArithOp::Minus, Prim::Float) => "arith.subf",
        (ArithOp::Times, Prim::Int) => "arith.muli",
        (ArithOp::Times, Prim::Float) => "arith.mulf",
        (ArithOp::Div, Prim::Int) => "arith.divsi",
        (ArithOp::Div, Prim::Float) => "arith.divf",
        (ArithOp::Mod, Prim::Int) => "arith.remsi",
        (ArithOp::Mod, Prim::Float) => {
            return Err("codegen: float modulo not implemented".to_string())
        }
        (_, Prim::Bool) => {
            return Err("codegen: arithmetic on booleans is not supported".to_string())
        }
        (_, Prim::Str) => {
            return Err("codegen: arithmetic on strings is not supported".to_string())
        }
    };

    block
        .append_operation(arith_binop(op_name, lhs, rhs, location)?)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Lower `a OP b` to `arith.cmpi` (int/bool) or `arith.cmpf` (float).
fn lower_comparison<'c, 'a>(
    op: &CompOp,
    e1: &Expr,
    e2: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let lhs = lower_expr(e1, block, module, env)?;
    let rhs = lower_expr(e2, block, module, env)?;

    // `==`/`!=` lower through [`super::equality::lower_equality`], which
    // handles scalar types (int/bool/char/unit via `cmpi`, float via `cmpf`,
    // string via `strcmp`) and structural equality on aggregate types.
    if matches!(op, CompOp::Eq | CompOp::NotEq) {
        let typ = default_free_vars(&e1.typ);
        let eq = lower_equality(&typ, lhs, rhs, block, module, location)?;
        if matches!(op, CompOp::NotEq) {
            let one = bool_constant(module, block, true, location)?;
            return block
                .append_operation(arith_binop("arith.xori", eq, one, location)?)
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into);
        }
        return Ok(eq);
    }

    let operation = match primitive_kind(&e1.typ)? {
        Prim::Int | Prim::Bool => {
            let predicate = match op {
                CompOp::Eq => arith::CmpiPredicate::Eq,
                CompOp::NotEq => arith::CmpiPredicate::Ne,
                CompOp::Less => arith::CmpiPredicate::Slt,
                CompOp::Greater => arith::CmpiPredicate::Sgt,
                CompOp::LessEq => arith::CmpiPredicate::Sle,
                CompOp::GreatEq => arith::CmpiPredicate::Sge,
            };
            arith::cmpi(module.context, predicate, lhs, rhs, location)
        }
        Prim::Float => {
            let predicate = match op {
                CompOp::Eq => arith::CmpfPredicate::Oeq,
                CompOp::NotEq => arith::CmpfPredicate::One,
                CompOp::Less => arith::CmpfPredicate::Olt,
                CompOp::Greater => arith::CmpfPredicate::Ogt,
                CompOp::LessEq => arith::CmpfPredicate::Ole,
                CompOp::GreatEq => arith::CmpfPredicate::Oge,
            };
            arith::cmpf(module.context, predicate, lhs, rhs, location)
        }
        Prim::Str => {
            return Err("codegen: string comparison not implemented".to_string())
        }
    };

    block
        .append_operation(operation)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Lower `a && b`/`a || b` with short-circuit evaluation via `scf.if`,
/// and `a ^ b` to `arith.xori`.
fn lower_logical<'c, 'a>(
    op: &LogicalOp,
    e1: &Expr,
    e2: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let bool_type: Type<'c> = IntegerType::new(module.context, 1).into();
    let lhs = lower_expr(e1, block, module, env)?;

    if matches!(op, LogicalOp::Xor) {
        let rhs = lower_expr(e2, block, module, env)?;
        return block
            .append_operation(arith_binop("arith.xori", lhs, rhs, location)?)
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into);
    }

    let mut e2_env = env.clone();
    let e2_block = Block::new(&[]);
    let e2_val = lower_expr(e2, &e2_block, module, &mut e2_env)?;
    e2_block.append_operation(scf::r#yield(&[e2_val], location));
    let e2_region = Region::new();
    e2_region.append_block(e2_block);

    let const_val = matches!(op, LogicalOp::Or);
    let const_block = Block::new(&[]);
    let c = bool_constant(module, &const_block, const_val, location)?;
    const_block.append_operation(scf::r#yield(&[c], location));
    let const_region = Region::new();
    const_region.append_block(const_block);

    let (then_region, else_region) = if matches!(op, LogicalOp::And) {
        (e2_region, const_region)
    } else {
        (const_region, e2_region)
    };

    block
        .append_operation(scf::r#if(lhs, &[bool_type], then_region, else_region, location))
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Lower `-e` (int: `subi 0, e`; float: `arith.negf`) and `!e` (`xori e, true`).
fn lower_unary<'c, 'a>(
    op: &UnaryOp,
    e: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let value = lower_expr(e, block, module, env)?;

    match op {
        UnaryOp::Negate => match primitive_kind(&e.typ)? {
            Prim::Int => {
                let zero = i32_constant(module, block, 0, location)?;
                block
                    .append_operation(arith_binop("arith.subi", zero, value, location)?)
                    .result(0)
                    .map_err(|e| e.to_string())
                    .map(Into::into)
            }
            Prim::Float => {
                let op = OperationBuilder::new("arith.negf", location)
                    .add_operands(&[value])
                    .enable_result_type_inference()
                    .build()
                    .map_err(|e| e.to_string())?;
                block
                    .append_operation(op)
                    .result(0)
                    .map_err(|e| e.to_string())
                    .map(Into::into)
            }
            _ => Err("codegen: unary negation requires an int or float operand".to_string()),
        },
        UnaryOp::Not => {
            let one = bool_constant(module, block, true, location)?;
            block
                .append_operation(arith_binop("arith.xori", value, one, location)?)
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
    }
}

/// Lower `if c then t else e` to `scf.if`, returning its result value.
///
/// Each branch gets its own copy of the environment (so branch-local bindings
/// do not leak out) and yields its lowered value; the result type is the
/// `if` expression's resolved type, with free type variables defaulted.
fn lower_ifelse<'c, 'a>(
    cond: &Expr,
    then_branch: &Expr,
    else_branch: &Expr,
    result_mono: &Monotype,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let condition = lower_expr(cond, block, module, env)?;
    let result_type = lower_type(&default_free_vars(result_mono), module)?;

    let mut then_env = env.clone();
    let then_block = Block::new(&[]);
    let then_value = lower_expr(then_branch, &then_block, module, &mut then_env)?;
    then_block.append_operation(scf::r#yield(&[then_value], location));
    let then_region = Region::new();
    then_region.append_block(then_block);

    let mut else_env = env.clone();
    let else_block = Block::new(&[]);
    let else_value = lower_expr(else_branch, &else_block, module, &mut else_env)?;
    else_block.append_operation(scf::r#yield(&[else_value], location));
    let else_region = Region::new();
    else_region.append_block(else_block);

    let if_op = scf::r#if(condition, &[result_type], then_region, else_region, location);
    block
        .append_operation(if_op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Lower a block expression `{ stmt; ...; e }`: run its statements in a
/// cloned environment (block-local bindings do not leak out) and return the
/// final expression's value.
fn lower_block<'c, 'a>(
    stmts: &[Stmt],
    final_expr: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<Value<'c, 'a>, String> {
    let mut block_env = env.clone();
    for stmt in stmts {
        lower_block_stmt(stmt, block, module, &mut block_env)?;
    }
    lower_expr(final_expr, block, module, &mut block_env)
}

/// Lower a statement that appears inside a block expression: declarations
/// become local SSA bindings (unlike top-level declarations, which become
/// symbols).
pub(crate) fn lower_block_stmt<'c, 'a>(
    stmt: &Stmt,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<(), String> {
    match &*stmt.s {
        SNode::Decl(e1, _, e2) => {
            let name = match &*e1.e {
                ENode::Variable(n) => n.clone(),
                _ => {
                    return Err(format!(
                        "codegen: expected a variable name in declaration, got {:?}",
                        *e1.e
                    ))
                }
            };
            bind_in_env(&name, e2, block, module, env)
        }
        SNode::Expr(e1) => {
            lower_expr(e1, block, module, env)?;
            Ok(())
        }
        SNode::TypeDecl(_, _) => Err(
            "codegen: type declarations are not allowed inside block expressions".to_string(),
        ),
    }
}

/// Lower `match scrut | pat => e | ...` to an `scf.if` chain. Patterns: a
/// literal compares for equality, `[]` tests for an empty list, `x::xs`
/// destructures a cons cell, and a final variable pattern is the catch-all
/// else branch.
fn lower_match<'c, 'a>(
    scrutinee: &Expr,
    cases: &[MatchCase],
    result_mono: &Monotype,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<Value<'c, 'a>, String> {
    let location = module.location(&scrutinee.pos);
    let scrut = lower_expr(scrutinee, block, module, env)?;
    let result_type = lower_type(&default_free_vars(result_mono), module)?;
    let scrut_typ = default_free_vars(&scrutinee.typ);
    lower_match_cases(
        scrut,
        &scrut_typ,
        cases,
        0,
        result_type,
        location,
        block,
        module,
        env,
    )
}

fn lower_match_cases<'c, 'a: 'b, 'b>(
    scrut: Value<'c, 'a>,
    scrut_typ: &Monotype,
    cases: &[MatchCase],
    index: usize,
    result_type: Type<'c>,
    location: Location<'c>,
    block: &'b Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<Value<'c, 'b>, String> {
    if index == cases.len() {
        // Defensive: the type checker rejects non-exhaustive matches.
        return Err("codegen: non-exhaustive match".to_string());
    }
    let case = &cases[index];
    let last = index + 1 == cases.len();

    // A catch-all variable pattern matches anything and must be last. Enum
    // constructor names (e.g. `None`) are not catch-alls.
    if let ENode::Variable(name) = &*case.val.e && !module.constructors.contains_key(name) {
        if !last {
            return Err(format!(
                "codegen: catch-all pattern `{name}` must be the last case"
            ));
        }
        let mut case_env = env.clone();
        case_env.insert(name.clone(), EnvEntry::Value(scrut));
        return lower_expr(&case.exp, block, module, &mut case_env);
    }

    let binding = case_pattern(case, scrut_typ, module)?;

    // The last case is guaranteed to match (exhaustiveness), so lower it
    // directly instead of wrapping it in one more `scf.if`.
    if last {
        let mut case_env = env.clone();
        for (name, value) in destructure_pattern(binding, scrut, block, module, location)? {
            case_env.insert(name, EnvEntry::Value(value));
        }
        return lower_expr(&case.exp, block, module, &mut case_env);
    }

    let cond = case_condition(case, scrut, scrut_typ, block, module, location)?;

    let mut then_env = env.clone();
    let then_block = Block::new(&[]);
    for (name, value) in destructure_pattern(binding, scrut, &then_block, module, location)? {
        then_env.insert(name, EnvEntry::Value(value));
    }
    let then_value = lower_expr(&case.exp, &then_block, module, &mut then_env)?;
    then_block.append_operation(scf::r#yield(&[then_value], location));
    let then_region = Region::new();
    then_region.append_block(then_block);

    let else_block = Block::new(&[]);
    let else_value = lower_match_cases(
        scrut,
        scrut_typ,
        cases,
        index + 1,
        result_type,
        location,
        &else_block,
        module,
        env,
    )?;
    else_block.append_operation(scf::r#yield(&[else_value], location));
    let else_region = Region::new();
    else_region.append_block(else_block);

    let if_op = scf::r#if(cond, &[result_type], then_region, else_region, location);
    block
        .append_operation(if_op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Unwrap a constructor pattern `Ctor p1 ... pn` (parsed as left-nested
/// applications) into the constructor name and its sub-patterns in application
/// order (outermost argument last; `.rev()` recovers declaration order).
fn constructor_pattern(pat: &Expr) -> Result<(String, Vec<Expr>), String> {
    let mut sub_patterns = Vec::new();
    let mut head = pat;
    let ctor_name = loop {
        match &*head.e {
            ENode::Application(f, arg) => {
                sub_patterns.push((**arg).clone());
                head = f;
            }
            ENode::Variable(n) => break n.clone(),
            _ => return Err(format!("codegen: unsupported match pattern {:?}", *pat.e)),
        }
    };
    Ok((ctor_name, sub_patterns))
}

/// The bindings a non-catch-all case pattern produces (loaded in the branch
/// body). Constructor patterns, tuples, records, and lists all recurse through
/// [`pattern_bind`].
pub(crate) fn case_pattern<'c>(
    case: &MatchCase,
    scrut_typ: &Monotype,
    module: &mut Module<'c>,
) -> Result<PatternBind<'c>, String> {
    pattern_bind(&case.val, scrut_typ, module)
}

/// The element `Monotype`s of a tuple scrutinee type, or an error if it is
/// not a tuple.
fn tuple_element_types(scrut_typ: &Monotype) -> Result<Vec<Monotype>, String> {
    match default_free_vars(scrut_typ) {
        Monotype::TypeFuncApplication(f, args) if matches!(*f, TypeFunc::Tuple) => Ok(args),
        other => Err(format!(
            "codegen: tuple pattern requires a tuple scrutinee, got {other:?}"
        )),
    }
}

/// Build the [`PatternBind`] that destructures `pat` against a value of type
/// `ty`, recursing into nested patterns (tuples, cons, records, constructors,
/// list literals). `Nil` means the pattern binds nothing (a literal, `[]`, or
/// a nullary constructor).
fn pattern_bind<'c>(
    pat: &Expr,
    ty: &Monotype,
    module: &mut Module<'c>,
) -> Result<PatternBind<'c>, String> {
    match &*pat.e {
        ENode::Variable(name) => {
            // A bare constructor name (e.g. `None`) is a nullary constructor
            // pattern and binds nothing; the discriminant test lives in
            // `sub_condition`. Any other variable is a binding.
            if module.constructors.contains_key(name) {
                Ok(PatternBind::Nil)
            } else {
                Ok(PatternBind::Var { name: name.clone() })
            }
        }
        ENode::Literal(_) => Ok(PatternBind::Nil),
        ENode::List(es) if es.is_empty() => Ok(PatternBind::Nil),
        // `[a, b, ..]` desugars to nested cons patterns.
        ENode::List(es) => {
            let elem = list_elem(ty)
                .ok_or_else(|| "codegen: list pattern requires a list scrutinee".to_string())?;
            let elem_ty = default_free_vars(&elem);
            let elem_mlir = lower_type(&elem_ty, module)?;
            let list_mlir = lower_type(&default_free_vars(ty), module)?;
            let head = pattern_bind(&es[0], &elem_ty, module)?;
            let tail = if es.len() > 1 {
                pattern_bind(&Expr::from(ENode::List(es[1..].to_vec())), ty, module)?
            } else {
                PatternBind::Nil
            };
            Ok(PatternBind::Cons {
                head: Box::new(head),
                head_type: elem_mlir,
                tail: Box::new(tail),
                tail_type: list_mlir,
            })
        }
        ENode::Cons(hd, tl) => {
            let elem = list_elem(ty)
                .ok_or_else(|| "codegen: cons pattern requires a list scrutinee".to_string())?;
            let elem_ty = default_free_vars(&elem);
            let elem_mlir = lower_type(&elem_ty, module)?;
            let list_mlir = lower_type(&default_free_vars(ty), module)?;
            let head = pattern_bind(hd, &elem_ty, module)?;
            let tail = pattern_bind(tl, &default_free_vars(ty), module)?;
            Ok(PatternBind::Cons {
                head: Box::new(head),
                head_type: elem_mlir,
                tail: Box::new(tail),
                tail_type: list_mlir,
            })
        }
        ENode::Application(_, _) => {
            let (ctor_name, sub_patterns) = constructor_pattern(pat)?;
            let &(ref enum_name, variant_index, arity) = module
                .constructors
                .get(&ctor_name)
                .ok_or_else(|| format!("codegen: unsupported match pattern {:?}", *pat.e))?;
            if sub_patterns.len() != arity {
                return Err(format!(
                    "codegen: constructor pattern `{ctor_name}` with arity {arity} applied to {} arguments",
                    sub_patterns.len()
                ));
            }
            let fields = enum_variant_fields(module, ty, enum_name, variant_index)?;
            let field_mlirs: Vec<Type<'c>> = fields
                .iter()
                .map(|m| lower_type(&default_free_vars(m), module))
                .collect::<Result<_, _>>()?;
            let mut binds = Vec::with_capacity(arity);
            for (i, sub) in sub_patterns.iter().rev().enumerate() {
                let sub_ty = default_free_vars(&fields[i]);
                binds.push(pattern_bind(sub, &sub_ty, module)?);
            }
            Ok(PatternBind::Enum {
                field_types: field_mlirs,
                binds,
            })
        }
        ENode::Record(_, fields) => {
            let rec_fields = record_fields(ty)?;
            let mut bindings = Vec::new();
            for fa in fields {
                let index = field_index(&rec_fields, &fa.field)?;
                let field_mono = default_free_vars(&rec_fields[index].1);
                let field_ty = lower_type(&field_mono, module)?;
                let sub = pattern_bind(&fa.exp, &field_mono, module)?;
                bindings.push((fa.field.clone(), field_ty, index, sub));
            }
            Ok(PatternBind::Record { fields: bindings })
        }
        ENode::Tuple(elems) => {
            let field_monos = tuple_element_types(ty)?;
            let field_mlirs: Vec<Type<'c>> = field_monos
                .iter()
                .map(|m| lower_type(&default_free_vars(m), module))
                .collect::<Result<_, _>>()?;
            let mut elements = Vec::with_capacity(elems.len());
            for (e, mono) in elems.iter().zip(field_monos.iter()) {
                let sub_ty = default_free_vars(mono);
                let sub = pattern_bind(e, &sub_ty, module)?;
                elements.push((field_mlirs[elements.len()], sub));
            }
            Ok(PatternBind::Tuple { elements })
        }
        _ => Err(format!("codegen: unsupported match pattern {:?}", *pat.e)),
    }
}

/// The `i1` condition under which a non-catch-all case matches. Delegates to
/// [`sub_condition`], which recurses through every pattern kind.
pub(crate) fn case_condition<'c, 'a>(
    case: &MatchCase,
    scrut: Value<'c, 'a>,
    scrut_typ: &Monotype,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    sub_condition(&case.val, scrut_typ, scrut, block, module, location)
}

/// The `i1` condition under which the sub-pattern `pat` (of type `ty`) matches
/// a value `val` of that type. Recurses so nested tuples, records, lists, and
/// constructor patterns are handled. A bare variable always matches; a bare
/// constructor name (e.g. `None`) matches on the discriminant.
///
/// List and constructor sub-patterns guard their pointer loads behind an
/// `scf.if`: an eager `arith.andi` would otherwise dereference a null list or
/// a nullary variant's null `data` pointer when the guard fails.
fn sub_condition<'c, 'a>(
    pat: &Expr,
    ty: &Monotype,
    val: Value<'c, 'a>,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let bool_type: Type<'c> = IntegerType::new(module.context, 1).into();
    match &*pat.e {
        ENode::Literal(lit) => {
            let pattern = lower_literal(lit, block, module, location)?;
            lower_equality(
                &default_free_vars(ty),
                val,
                pattern,
                block,
                module,
                location,
            )
        }
        // A nullary constructor pattern `None` tests the discriminant; a plain
        // variable always matches.
        ENode::Variable(name) => {
            if let Some(&(_, variant_index, _)) = module.constructors.get(name) {
                enum_disc_eq(module, block, val, variant_index, location)
            } else {
                bool_constant(module, block, true, location)
            }
        }
        // `[]` matches when the list is empty (null).
        ENode::List(es) if es.is_empty() => list_is_null(val, block, module, location),
        // `[a, b, ..]` desugars to nested cons conditions.
        ENode::List(es) => {
            let head = es[0].clone();
            let tail = if es.len() > 1 {
                Expr::from(ENode::List(es[1..].to_vec()))
            } else {
                Expr::from(ENode::List(vec![]))
            };
            let cons = Expr::from(ENode::Cons(Box::new(head), Box::new(tail)));
            sub_condition(&cons, ty, val, block, module, location)
        }
        ENode::Cons(hd, tl) => {
            let elem = list_elem(ty)
                .ok_or_else(|| "codegen: cons pattern requires a list scrutinee".to_string())?;
            let elem_ty = default_free_vars(&elem);
            let elem_mlir = lower_type(&elem_ty, module)?;
            let ptr: Type = Type::parse(module.context, "!llvm.ptr")
                .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
            let cell = cell_struct_type(module, elem_mlir)?;
            let list_ty = default_free_vars(ty);

            let is_null = list_is_null(val, block, module, location)?;
            let one = bool_constant(module, block, true, location)?;
            let nonnull = block
                .append_operation(arith_binop("arith.xori", is_null, one, location)?)
                .result(0)
                .map_err(|e| e.to_string())?
                .into();

            let then_block = Block::new(&[]);
            let head_val = load_field(module, &then_block, val, cell, 0, elem_mlir, location)?;
            let tail_val = load_field(module, &then_block, val, cell, 1, ptr, location)?;
            let mut cond = sub_condition(hd, &elem_ty, head_val, &then_block, module, location)?;
            let sub = sub_condition(tl, &list_ty, tail_val, &then_block, module, location)?;
            cond = and_i1(&then_block, cond, sub, location)?;
            then_block.append_operation(scf::r#yield(&[cond], location));
            let then_region = Region::new();
            then_region.append_block(then_block);

            let else_block = Block::new(&[]);
            let f = bool_constant(module, &else_block, false, location)?;
            else_block.append_operation(scf::r#yield(&[f], location));
            let else_region = Region::new();
            else_region.append_block(else_block);

            block
                .append_operation(scf::r#if(
                    nonnull,
                    &[bool_type],
                    then_region,
                    else_region,
                    location,
                ))
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
        ENode::Tuple(elems) => {
            let field_monos = tuple_element_types(ty)?;
            let mut cond = bool_constant(module, block, true, location)?;
            for (i, e) in elems.iter().enumerate() {
                let elem_ty = default_free_vars(&field_monos[i]);
                let elem_mlir = lower_type(&elem_ty, module)?;
                let elem_val = extract_field(module, block, val, i as i32, elem_mlir, location)?;
                let sub = sub_condition(e, &elem_ty, elem_val, block, module, location)?;
                cond = and_i1(block, cond, sub, location)?;
            }
            Ok(cond)
        }
        ENode::Record(_, fields) => {
            let rec_fields = record_fields(ty)?;
            let mut cond = bool_constant(module, block, true, location)?;
            for fa in fields {
                let index = field_index(&rec_fields, &fa.field)?;
                let field_mono = default_free_vars(&rec_fields[index].1);
                let field_ty = lower_type(&field_mono, module)?;
                let field_val = extract_field(module, block, val, index as i32, field_ty, location)?;
                let sub = sub_condition(&fa.exp, &field_mono, field_val, block, module, location)?;
                cond = and_i1(block, cond, sub, location)?;
            }
            Ok(cond)
        }
        ENode::Application(_, _) => {
            let (ctor_name, sub_patterns) = constructor_pattern(pat)?;
            let &(ref enum_name, variant_index, _) = module.constructors.get(&ctor_name).ok_or_else(
                || format!("codegen: unsupported match pattern {:?}", *pat.e),
            )?;
            let fields = enum_variant_fields(module, ty, enum_name, variant_index)?;
            let field_mlirs: Vec<Type<'c>> = fields
                .iter()
                .map(|m| lower_type(&default_free_vars(m), module))
                .collect::<Result<_, _>>()?;

            let disc_ok = enum_disc_eq(module, block, val, variant_index, location)?;

            let then_block = Block::new(&[]);
            let mut cond = bool_constant(module, &then_block, true, location)?;
            for (i, sub) in sub_patterns.iter().rev().enumerate() {
                let field_mono = default_free_vars(&fields[i]);
                let field_val = load_enum_payload_field(
                    module, &then_block, val, &field_mlirs, i as i32, location,
                )?;
                let sub_cond = sub_condition(sub, &field_mono, field_val, &then_block, module, location)?;
                cond = and_i1(&then_block, cond, sub_cond, location)?;
            }
            then_block.append_operation(scf::r#yield(&[cond], location));
            let then_region = Region::new();
            then_region.append_block(then_block);

            let else_block = Block::new(&[]);
            let f = bool_constant(module, &else_block, false, location)?;
            else_block.append_operation(scf::r#yield(&[f], location));
            let else_region = Region::new();
            else_region.append_block(else_block);

            block
                .append_operation(scf::r#if(
                    disc_ok,
                    &[bool_type],
                    then_region,
                    else_region,
                    location,
                ))
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
        _ => Ok(bool_constant(module, block, true, location)?),
    }
}

/// Logical AND of two `i1` values (`arith.andi`).
fn and_i1<'c, 'a>(
    block: &'a Block<'c>,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    block
        .append_operation(arith_binop("arith.andi", lhs, rhs, location)?)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}
