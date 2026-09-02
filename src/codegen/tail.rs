//! Tail-position lowering for tail call optimization.
//!
//! [`lower_tail`] lowers an expression in tail position of a tail-recursive
//! specialization (see [`super::apply::specialize_binding`]), branching to a
//! shared loop header block instead of returning a value:
//!
//! - a *self tail call* — a full application that specializes to the active
//!   [`TailCtx`] symbol — lowers its arguments and branches back to the
//!   header, `cf.br ^header(new_args...)`, so the recursion runs in constant
//!   stack space;
//! - any other expression lowers normally and is returned directly with
//!   `func.return`.
//!
//! `if`/`match` in tail position branch with `cf.cond_br` into fresh blocks
//! (a backedge cannot pass through an `scf.if`'s `scf.yield`); the fresh
//! blocks are queued in [`TailCtx::pending`] and appended to the function
//! region by the caller. Tail positions follow OCaml's rules: branches of
//! `if`/`match`, the continuation of a `let`, and the final expression of a
//! block inherit tail position; nothing else does.

use crate::ast::*;
use crate::types::Monotype;
use melior::dialect::{cf, func};
use melior::ir::{Block, BlockLike, Location, Value};

use super::{Env, EnvEntry, Module};
use super::apply::{
    abstraction_params, bind_in_env, collect_application_root, default_free_vars, lambda_captures,
    specialize_binding,
};
use super::enums::destructure_pattern;
use super::expr::{case_condition, case_pattern, lower_block_stmt, lower_expr};

/// Loop-header state for lowering one tail-recursive specialization body.
pub(crate) struct TailCtx<'b, 'c> {
    /// The specialization symbol being emitted; a tail-position application
    /// that specializes to this same symbol is a self tail call.
    pub symbol: String,
    /// The loop header block, whose arguments are the specialization's
    /// `(captures..., params...)` inputs.
    pub header: &'b Block<'c>,
    /// Branch blocks created while lowering `if`/`match` in tail position,
    /// appended to the function region after the header block.
    pub pending: Vec<Block<'c>>,
}

/// Lower `expr` in tail position, terminating `block` (and every branch
/// block created for it) with either a `cf.br` backedge to the loop header
/// or a `func.return`.
pub(crate) fn lower_tail<'c, 'a>(
    expr: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    tail: &mut TailCtx<'_, 'c>,
) -> Result<(), String> {
    let location = module.location(&expr.pos);
    match &*expr.e {
        ENode::IfElse(c, t, e) => {
            let condition = lower_expr(c, block, module, env)?;
            let then_block = Block::new(&[]);
            let else_block = Block::new(&[]);
            block.append_operation(cf::cond_br(
                module.context,
                condition,
                &then_block,
                &else_block,
                &[],
                &[],
                location,
            ));
            let mut then_env = env.clone();
            lower_tail(t, &then_block, module, &mut then_env, tail)?;
            let mut else_env = env.clone();
            lower_tail(e, &else_block, module, &mut else_env, tail)?;
            tail.pending.push(then_block);
            tail.pending.push(else_block);
            Ok(())
        }
        ENode::Let(name, e1, e2) => {
            // Mirrors lower_let, with the continuation in tail position.
            let previous = env.get(name).cloned();
            bind_in_env(name, e1, block, module, env)?;
            let result = lower_tail(e2, block, module, env, tail);
            match previous {
                Some(old) => {
                    env.insert(name.to_string(), old);
                }
                None => {
                    env.remove(name);
                }
            }
            result
        }
        ENode::Block(stmts, e) => {
            let mut block_env = env.clone();
            for stmt in stmts {
                lower_block_stmt(stmt, block, module, &mut block_env)?;
            }
            lower_tail(e, block, module, &mut block_env, tail)
        }
        ENode::Match(scrutinee, cases) => {
            let scrut = lower_expr(scrutinee, block, module, env)?;
            let scrut_typ = default_free_vars(&scrutinee.typ);
            lower_tail_match_cases(scrut, &scrut_typ, cases, 0, location, block, module, env, tail)
        }
        ENode::Application(f, x) => {
            if !try_self_tail_call(f, x, block, module, env, tail, location)? {
                return_value(expr, block, module, env)?;
            }
            Ok(())
        }
        _ => return_value(expr, block, module, env),
    }
}

/// Lower `expr` with the ordinary value lowering and `func.return` it.
fn return_value<'c, 'a>(
    expr: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<(), String> {
    let location = module.location(&expr.pos);
    let value = lower_expr(expr, block, module, env)?;
    block.append_operation(func::r#return(&[value], location));
    Ok(())
}

/// If `f x` is a self tail call, lower the arguments and branch back to the
/// loop header, returning `true`; otherwise return `false` so the caller can
/// fall back to an ordinary call.
///
/// A self tail call is a full application whose callee specializes to the
/// active [`TailCtx`] symbol — the same resolution
/// [`super::apply::lower_application`] performs, so shadowed or differently
/// instantiated references fall through to the ordinary path.
fn try_self_tail_call<'c, 'a>(
    f: &Expr,
    x: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    tail: &TailCtx<'_, 'c>,
    location: Location<'c>,
) -> Result<bool, String> {
    // Flatten `(((f a) b) c)` into a root and argument list, exactly like
    // lower_application.
    let mut args = Vec::new();
    let root = collect_application_root(f, &mut args, module);
    args.push((*x).clone());

    let ENode::Variable(name) = &*root.e else {
        return Ok(false);
    };
    // The same resolution order as lower_application: the local environment
    // (where a shadowing binding lives) before top-level abstractions.
    let sym = if let Some(EnvEntry::Abstraction(s)) = env.get(name) {
        Some(s.clone())
    } else if module.abstractions.contains_key(name) {
        Some(name.clone())
    } else {
        None
    };
    let Some(sym) = sym else {
        return Ok(false);
    };
    let Some(info) = module.abstractions.get(&sym).cloned() else {
        return Ok(false);
    };
    let params = abstraction_params(&info.param, &info.body);
    if args.len() != params.len() {
        // A partial self application produces a function value: a real call.
        return Ok(false);
    }
    let captures = lambda_captures(&info, name, env);
    let symbol = specialize_binding(&sym, name, &root.typ, &captures, module, location)?;
    if symbol != tail.symbol {
        // Another function (or this one at a different instantiation): not a
        // loop backedge.
        return Ok(false);
    }

    // Branch to the loop header with `(captures..., args...)` — the layout
    // of the specialization's parameter list.
    let mut operands: Vec<Value<'c, 'a>> = Vec::with_capacity(captures.len() + args.len());
    for (cname, _) in &captures {
        match env.get(cname) {
            Some(EnvEntry::Value(v)) => operands.push(*v),
            _ => return Err(format!("codegen: missing capture `{cname}`")),
        }
    }
    for a in &args {
        operands.push(lower_expr(a, block, module, env)?);
    }
    block.append_operation(cf::br(tail.header, &operands, location));
    Ok(true)
}

/// [`super::expr::lower_match`] for tail position: the case-analysis `if`
/// chain becomes `cf.cond_br` branches so case bodies can end in backedges.
fn lower_tail_match_cases<'c, 'a: 'b, 'b>(
    scrut: Value<'c, 'a>,
    scrut_typ: &Monotype,
    cases: &[MatchCase],
    index: usize,
    location: Location<'c>,
    block: &'b Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    tail: &mut TailCtx<'_, 'c>,
) -> Result<(), String> {
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
            return lower_tail(&case.exp, block, module, &mut case_env, tail);
    }

    let binding = case_pattern(case, scrut_typ, module)?;

    // The last case is guaranteed to match (exhaustiveness), so lower it
    // directly instead of emitting one more branch.
    if last {
        let mut case_env = env.clone();
        for (name, value) in destructure_pattern(binding, scrut, block, module, location)? {
            case_env.insert(name, EnvEntry::Value(value));
        }
        return lower_tail(&case.exp, block, module, &mut case_env, tail);
    }

    let cond = case_condition(case, scrut, scrut_typ, block, module, location)?;
    let then_block = Block::new(&[]);
    let else_block = Block::new(&[]);
    block.append_operation(cf::cond_br(
        module.context,
        cond,
        &then_block,
        &else_block,
        &[],
        &[],
        location,
    ));

    let mut then_env = env.clone();
    for (name, value) in destructure_pattern(binding, scrut, &then_block, module, location)? {
        then_env.insert(name, EnvEntry::Value(value));
    }
    lower_tail(&case.exp, &then_block, module, &mut then_env, tail)?;
    tail.pending.push(then_block);
    lower_tail_match_cases(
        scrut,
        scrut_typ,
        cases,
        index + 1,
        location,
        &else_block,
        module,
        env,
        tail,
    )?;
    tail.pending.push(else_block);
    Ok(())
}
