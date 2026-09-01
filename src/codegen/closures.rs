//! Free-variable analysis and closure conversion.

use crate::ast::*;
use melior::dialect::llvm;
use melior::dialect::llvm::LoadStoreOptions;
use melior::ir::{
    attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute},
    operation::OperationBuilder,
    r#type::IntegerType,
    Block, BlockLike, Identifier, Location, Type, Value,
};
use std::collections::HashSet;

use super::Module;
use super::lists::{empty_list, malloc_call};

// ----------------------------------------------------------------------------
// Free variables (for closure capture)
// ----------------------------------------------------------------------------

pub(crate) fn union_into(a: &mut HashSet<String>, b: HashSet<String>) {
    a.extend(b);
}

/// Variables referenced by `expr` but not bound inside it.
pub(crate) fn free_variables(expr: &Expr) -> HashSet<String> {
    match &*expr.e {
        ENode::Variable(n) => HashSet::from([n.clone()]),
        ENode::Literal(_) => HashSet::new(),
        ENode::Abstraction(binding, body) => {
            let mut fv = free_variables(body);
            fv.remove(&binding.0);
            fv
        }
        ENode::Application(f, x) => {
            let mut fv = free_variables(f);
            union_into(&mut fv, free_variables(x));
            fv
        }
        ENode::Let(name, e1, e2) => {
            let mut fv = free_variables(e1);
            union_into(&mut fv, free_variables(e2));
            fv.remove(name);
            fv
        }
        ENode::IfElse(c, t, e) => {
            let mut fv = free_variables(c);
            union_into(&mut fv, free_variables(t));
            union_into(&mut fv, free_variables(e));
            fv
        }
        ENode::Block(stmts, e) => {
            let mut fv = free_variables(e);
            for s in stmts.iter().rev() {
                match &*s.s {
                    SNode::Decl(e1, _, e2) => {
                        union_into(&mut fv, free_variables(e2));
                        if let ENode::Variable(n) = &*e1.e {
                            fv.remove(n);
                        }
                    }
                    SNode::Expr(e1) => union_into(&mut fv, free_variables(e1)),
                    SNode::TypeDecl(..) => {}
                }
            }
            fv
        }
        ENode::Comparison(_, a, b)
        | ENode::Arithmetic(_, a, b)
        | ENode::Logical(_, a, b) => {
            let mut fv = free_variables(a);
            union_into(&mut fv, free_variables(b));
            fv
        }
        ENode::Unary(_, e) => free_variables(e),
        ENode::List(es) => {
            let mut fv = HashSet::new();
            for e in es {
                union_into(&mut fv, free_variables(e));
            }
            fv
        }
        ENode::Cons(h, t) => {
            let mut fv = free_variables(h);
            union_into(&mut fv, free_variables(t));
            fv
        }
        ENode::Match(scrut, cases) => {
            let mut fv = free_variables(scrut);
            for c in cases {
                let mut cv = free_variables(&c.exp);
                for name in pattern_bound_vars(&c.val) {
                    cv.remove(&name);
                }
                union_into(&mut fv, cv);
            }
            fv
        }
        ENode::FieldAccess(e, _) => free_variables(e),
        ENode::Record(_, field_assns) => {
            let mut fv = HashSet::new();
            for fa in field_assns {
                union_into(&mut fv, free_variables(&fa.exp));
            }
            fv
        }
        ENode::With(e, field_assns) => {
            let mut fv = free_variables(e);
            for fa in field_assns {
                union_into(&mut fv, free_variables(&fa.exp));
            }
            fv
        }
        ENode::Tuple(exprs) => {
            let mut fv = HashSet::new();
            for e in exprs {
                union_into(&mut fv, free_variables(e));
            }
            fv
        },
    }
}

/// Variables a pattern binds (e.g. `x::xs` binds `x` and `xs`; `Some val`
/// binds `val`; literals bind nothing).
pub(crate) fn pattern_bound_vars(pat: &Expr) -> Vec<String> {
    match &*pat.e {
        ENode::Variable(n) => vec![n.clone()],
        ENode::Literal(_) => vec![],
        ENode::Cons(h, t) => {
            let mut v = pattern_bound_vars(h);
            v.extend(pattern_bound_vars(t));
            v
        }
        ENode::List(es) => {
            let mut v = Vec::new();
            for e in es {
                v.extend(pattern_bound_vars(e));
            }
            v
        }
        ENode::Application(_, _) => {
            // A constructor pattern `Ctor p1 ... pn` is parsed as left-nested
            // applications; the head is the constructor name (never a
            // binding), while each argument is a sub-pattern.
            let mut pats = Vec::new();
            let mut head = pat;
            while let ENode::Application(f, arg) = &*head.e {
                pats.push(arg);
                head = f;
            }
            let mut v = Vec::new();
            for p in pats {
                v.extend(pattern_bound_vars(p));
            }
            v
        }
        ENode::Record(_, fields) => {
            let mut v = Vec::new();
            for fa in fields {
                v.extend(pattern_bound_vars(&fa.exp));
            }
            v
        }
        ENode::Tuple(es) => {
            let mut v = Vec::new();
            for e in es {
                v.extend(pattern_bound_vars(e));
            }
            v
        }
        _ => vec![],
    }
}

// ----------------------------------------------------------------------------
// Self tail calls (for tail call optimization)
// ----------------------------------------------------------------------------

/// Whether `expr` contains a *self tail call*: an application of `self_name`
/// to exactly `arity` arguments in tail position.
///
/// Tail positions follow OCaml's rules: the expression itself, the branches
/// of an `if`/`match` in tail position, the continuation of a `let` in tail
/// position, and the final expression of a block in tail position. Nothing
/// else is scanned (call arguments, operands, `let` right-hand sides, and
/// nested lambda bodies are never in tail position of this function).
///
/// The result is a cheap filter used to decide whether a specialization body
/// needs the loop skeleton; the lowering in [`super::tail`] re-verifies each
/// candidate (resolving to the same specialization symbol) before emitting a
/// backedge, so a false positive here only costs an unused loop header.
pub(crate) fn has_self_tail_call(
    expr: &Expr,
    self_name: &str,
    arity: usize,
    inlineable: &std::collections::HashMap<String, Expr>,
) -> bool {
    has_self_tail_call_at(expr, self_name, arity, inlineable, false)
}

/// [`has_self_tail_call`] with `shadowed` tracking whether `self_name` has
/// been rebound between the function root and `expr` (a shadowed reference
/// resolves to another binding and is not a self call).
fn has_self_tail_call_at(
    expr: &Expr,
    self_name: &str,
    arity: usize,
    inlineable: &std::collections::HashMap<String, Expr>,
    shadowed: bool,
) -> bool {
    match &*expr.e {
        ENode::IfElse(_, then_branch, else_branch) => {
            has_self_tail_call_at(then_branch, self_name, arity, inlineable, shadowed)
                || has_self_tail_call_at(else_branch, self_name, arity, inlineable, shadowed)
        }
        ENode::Let(name, _, e2) => {
            has_self_tail_call_at(e2, self_name, arity, inlineable, shadowed || name == self_name)
        }
        ENode::Block(stmts, e) => {
            let mut shadowed = shadowed;
            for s in stmts {
                if let SNode::Decl(e1, _, _) = &*s.s
                && let ENode::Variable(n) = &*e1.e {
                    shadowed = shadowed || n == self_name;
                }
            }
            has_self_tail_call_at(e, self_name, arity, inlineable, shadowed)
        }
        ENode::Match(_, cases) => cases.iter().any(|c| {
            let shadowed =
                shadowed || pattern_bound_vars(&c.val).iter().any(|n| n == self_name);
            has_self_tail_call_at(&c.exp, self_name, arity, inlineable, shadowed)
        }),
        ENode::Application(..) => {
            if shadowed {
                return false;
            }
            let (root, args) = application_root(expr, inlineable);
            root == Some(self_name) && args == arity
        }
        _ => false,
    }
}

/// The root variable of a flattened application chain `((f a) b)` and the
/// number of applied arguments, expanding inlineable function-valued `let`
/// bindings like [`super::apply::collect_application_root`] does.
fn application_root<'a>(
    expr: &'a Expr,
    inlineable: &'a std::collections::HashMap<String, Expr>,
) -> (Option<&'a str>, usize) {
    let mut args = 0;
    let mut current = expr;
    loop {
        match &*current.e {
            ENode::Application(f, _) => {
                args += 1;
                current = f;
            }
            ENode::Variable(name) => {
                if let Some(rhs) = inlineable.get(name) {
                    current = rhs;
                } else {
                    return (Some(name), args);
                }
            }
            _ => return (None, args),
        }
    }
}

// ----------------------------------------------------------------------------
// Closures
// ----------------------------------------------------------------------------

/// A closure is a heap struct `{ fn_ptr: !llvm.ptr, env: !llvm.ptr }`; all
/// function values are closures.
pub(crate) fn closure_struct_type<'c>(module: &Module<'c>) -> Result<Type<'c>, String> {
    Type::parse(module.context, "!llvm.struct<(!llvm.ptr, !llvm.ptr)>").ok_or_else(|| {
        "codegen: failed to create closure struct type `!llvm.struct<(!llvm.ptr, !llvm.ptr)>`"
            .to_string()
    })
}

/// The heap struct type holding the captured variables of a closure.
pub(crate) fn env_struct_type<'c>(
    module: &Module<'c>,
    captures: &[(String, Value<'c, '_>, Type<'c>)],
) -> Result<Type<'c>, String> {
    let fields: Vec<String> = captures.iter().map(|(_, _, t)| t.to_string()).collect();
    Type::parse(module.context, &format!("!llvm.struct<({})>", fields.join(", "))).ok_or_else(|| {
        format!(
            "codegen: failed to create environment struct type `!llvm.struct<({})>`",
            fields.join(", ")
        )
    })
}

/// Address of a `func.func` symbol as a raw `!llvm.ptr`.
pub(crate) fn fn_address<'c, 'a>(
    module: &Module<'c>,
    block: &'a Block<'c>,
    symbol: &str,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let op = OperationBuilder::new("llvm.mlir.addressof", location)
        .add_attributes(&[(
            Identifier::new(module.context, "global_name"),
            FlatSymbolRefAttribute::new(module.context, symbol).into(),
        )])
        .add_results(&[ptr])
        .build()
        .map_err(|e| e.to_string())?;
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Store `value` into field `index` of the struct at `struct_ptr`.
pub(crate) fn store_field<'c, 'a>(
    module: &Module<'c>,
    block: &'a Block<'c>,
    struct_ptr: Value<'c, 'a>,
    elem_type: Type<'c>,
    index: i32,
    value: Value<'c, 'a>,
    location: Location<'c>,
) -> Result<(), String> {
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let op = llvm::get_element_ptr(
        module.context,
        struct_ptr,
        DenseI32ArrayAttribute::new(module.context, &[0, index]),
        elem_type,
        ptr,
        location,
    );
    let addr: Value<'c, 'a> = block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(llvm::store(
        module.context,
        value,
        addr,
        location,
        LoadStoreOptions::new(),
    ));
    Ok(())
}

/// Load field `index` of the struct at `struct_ptr` as `result_type`.
pub(crate) fn load_field<'c, 'a>(
    module: &Module<'c>,
    block: &'a Block<'c>,
    struct_ptr: Value<'c, 'a>,
    elem_type: Type<'c>,
    index: i32,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let op = llvm::get_element_ptr(
        module.context,
        struct_ptr,
        DenseI32ArrayAttribute::new(module.context, &[0, index]),
        elem_type,
        ptr,
        location,
    );
    let addr: Value<'c, 'a> = block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block
        .append_operation(llvm::load(
            module.context,
            addr,
            result_type,
            location,
            LoadStoreOptions::new(),
        ))
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Allocate a closure `{ fn_ptr: @symbol, env }` on the heap.
pub(crate) fn build_closure<'c, 'a>(
    module: &mut Module<'c>,
    block: &'a Block<'c>,
    fn_symbol: &str,
    captures: &[(String, Value<'c, 'a>, Type<'c>)],
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let closure_struct = closure_struct_type(module)?;
    let closure = malloc_call(module, block, 16, location)?;

    let fn_ptr = fn_address(module, block, fn_symbol, location)?;
    store_field(module, block, closure, closure_struct, 0, fn_ptr, location)?;

    let env = if captures.is_empty() {
        empty_list(block, module, location)?
    } else {
        let env_struct = env_struct_type(module, captures)?;
        let env_ptr = malloc_call(module, block, 8 * captures.len() as i64, location)?;
        for (i, (_, value, _)) in captures.iter().enumerate() {
            store_field(module, block, env_ptr, env_struct, i as i32, *value, location)?;
        }
        env_ptr
    };
    store_field(module, block, closure, closure_struct, 1, env, location)?;
    Ok(closure)
}

/// Call a closure value with `args`, loading `fn_ptr`/`env` and calling
/// `fn_ptr(args..., env)`.
pub(crate) fn closure_call<'c, 'a>(
    module: &mut Module<'c>,
    block: &'a Block<'c>,
    closure: Value<'c, '_>,
    args: &[Value<'c, '_>],
    ret_type: Type<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let closure_struct = closure_struct_type(module)?;
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let fn_ptr = load_field(module, block, closure, closure_struct, 0, ptr, location)?;
    let env_ptr = load_field(module, block, closure, closure_struct, 1, ptr, location)?;
    let env_i64: Value<'c, 'a> = block
        .append_operation(
            OperationBuilder::new("llvm.ptrtoint", location)
                .add_operands(&[env_ptr])
                .add_results(&[IntegerType::new(module.context, 64).into()])
                .build()
                .map_err(|e| e.to_string())?,
        )
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    let mut operands = vec![fn_ptr];
    operands.extend(args.iter().copied());
    operands.push(env_i64);
    let call = OperationBuilder::new("llvm.call_indirect", location)
        .add_operands(&operands)
        .add_results(&[ret_type])
        .build()
        .map_err(|e| e.to_string())?;
    block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}
