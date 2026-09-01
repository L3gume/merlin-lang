//! Closure/application machinery: lambdas, calls, variable references, and
//! per-type specialization.

use crate::ast::*;
use crate::types::{Monotype, TypeContext, TypeFunc};
use melior::dialect::{arith, func, llvm};
use melior::ir::{
    Attribute,
    attribute::{
        BoolAttribute, DenseI32ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute,
        IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    r#type::{FunctionType, IntegerType},
    Block, BlockLike, Identifier, Location, Region, RegionLike, Type, Value, ValueLike,
};
use std::collections::{HashMap, HashSet};

use super::{AbstractionInfo, Env, EnvEntry, Module};
use super::closures::{
    build_closure, closure_call, env_struct_type, free_variables, has_self_tail_call, load_field,
};
use super::enums::{build_enum_value, build_payload};
use super::expr::lower_expr;
use super::lists::empty_list;
use super::tail::{TailCtx, lower_tail};
use super::types::{lower_type, tuple_size};

pub(crate) fn lower_let<'c, 'a>(
    name: &str,
    e1: &Expr,
    e2: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<Value<'c, 'a>, String> {
    let previous = env.get(name).cloned();
    bind_in_env(name, e1, block, module, env)?;
    let result = lower_expr(e2, block, module, env);
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

/// All parameter names in a curried abstraction chain `\a => \b => body`,
/// starting from the first bound name.
pub(crate) fn abstraction_params(first: &str, body: &Expr) -> Vec<String> {
    let mut params = vec![first.to_string()];
    let mut cur = body;
    while let ENode::Abstraction(binding, b) = &*cur.e {
        params.push(binding.0.clone());
        cur = b;
    }
    params
}

/// The innermost body after unwrapping all curried abstractions.
fn peel_inner(body: &Expr) -> Expr {
    let mut cur = body.clone();
    while let ENode::Abstraction(_, b) = &*cur.e {
        cur = (**b).clone();
    }
    cur
}

/// Free variables of a lambda (beyond its own parameters and the recursive
/// self-name) that are bound to runtime values in the current environment.
/// These must be threaded as leading capture parameters to the lifted
/// specialization. Returns `(name, MLIR type)` pairs, sorted for determinism.
pub(crate) fn lambda_captures<'c, 'a>(
    info: &AbstractionInfo,
    self_name: &str,
    env: &Env<'c, 'a>,
) -> Vec<(String, Type<'c>)> {
    let params = abstraction_params(&info.param, &info.body);
    let body = peel_inner(&info.body);
    let free = free_variables(&body);
    let mut caps: Vec<(String, Type<'c>)> = Vec::new();
    for fv in free {
        if fv == self_name || params.contains(&fv) {
            continue;
        }
        if let Some(EnvEntry::Value(v)) = env.get(&fv) 
        && !caps.iter().any(|(n, _)| n == &fv) {
            caps.push((fv.clone(), v.r#type()));
        }
    }
    caps.sort_by(|a, b| a.0.cmp(&b.0));
    caps
}

/// Flatten a left-nested application `((f a) b)` into the root expression and
/// an ordered argument list, expanding function-valued `let` bindings.
pub(crate) fn collect_application_root(
    expr: &Expr,
    args: &mut Vec<Expr>,
    module: &Module,
) -> Expr {
    match &*expr.e {
        ENode::Application(f, arg) => {
            let root = collect_application_root(f, args, module);
            args.push((**arg).clone());
            root
        }
        ENode::Variable(name) => {
            if let Some(rhs) = module.inlineable.get(name) {
                collect_application_root(rhs, args, module)
            } else {
                (*expr).clone()
            }
        }
        _ => (*expr).clone(),
    }
}

/// Bind `name` to the lowered value of `e2` in `env`.
///
/// A lambda initializer is registered in [`Module::abstractions`] (bound as
/// an [`EnvEntry::Abstraction`]) so it stays polymorphic and is specialized on
/// demand at each use of `name`.
pub(crate) fn bind_in_env<'c, 'a>(
    name: &str,
    e2: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
) -> Result<(), String> {
    if let ENode::Abstraction(binding, body) = &*e2.e {
        let sym = format!("let_{}", module.let_counter);
        module.let_counter += 1;
        module.abstractions.insert(
            sym.clone(),
            AbstractionInfo {
                param: binding.0.clone(),
                param_type: binding.1.t.clone(),
                body: (**body).clone(),
                abs_type: e2.typ.clone(),
            },
        );
        env.insert(name.to_string(), EnvEntry::Abstraction(sym));
    } else {
        let value = lower_expr(e2, block, module, env)?;
        env.insert(name.to_string(), EnvEntry::Value(value));
    }
    Ok(())
}

/// Emit a `func.constant` reference to the specialization of `sym` at the
/// concrete type `typ`, returning a func-typed SSA value.
fn reference_specialization<'c, 'a>(
    sym: &str,
    self_name: &str,
    typ: &Monotype,
    captures: &[(String, Type<'c>)],
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    if !captures.is_empty() {
        return Err(format!(
            "codegen: cannot use `{self_name}` (a closure with captures) as a value yet"
        ));
    }
    let symbol = specialize_binding(sym, self_name, typ, captures, module, location)?;
    let func_type = specialization_function_type(sym, typ, captures, module)?;
    block
        .append_operation(func::constant(
            module.context,
            FlatSymbolRefAttribute::new(module.context, &symbol),
            func_type,
            location,
        ))
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Lower a variable reference: a bound value/abstraction if it is in `env`, a
/// specialized closure if it names a registered lambda binding, otherwise a
/// `func.call` on the top-level symbol of the same name.
pub(crate) fn lower_variable<'c, 'a>(
    expr: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let ENode::Variable(name) = &*expr.e else {
        unreachable!()
    };
    match env.get(name) {
        Some(EnvEntry::Value(value)) => return Ok(*value),
        Some(EnvEntry::Abstraction(sym)) => {
            let info = module.abstractions.get(sym).ok_or_else(|| {
                format!("codegen: `{sym}` is not a registered lambda binding")
            })?;
            let captures = lambda_captures(info, name, env);
            return reference_specialization(sym, name, &expr.typ, &captures, block, module, location);
        }
        None => {}
    }

    // An inlineable function-valued `let` (e.g. `let sum = lfold add 0;`).
    if let Some(rhs) = module.inlineable.get(name) {
        let rhs = rhs.clone();
        let mut inline_env = env.clone();
        return lower_expr(&rhs, block, module, &mut inline_env);
    }

    // A top-level lambda binding is specialized at the concrete type this use
    // site resolved to, and referenced by `func.constant` on that
    // specialization.
    if module.abstractions.contains_key(name) {
        let info = module.abstractions.get(name).unwrap();
        let captures = lambda_captures(info, name, env);
        return reference_specialization(name, name, &expr.typ, &captures, block, module, location);
    }

    // A nullary enum constructor (`None`) builds a tagged value with no
    // payload.
    if let Some(&(_, variant_index, arity)) = module.constructors.get(name) && arity == 0 {
        let payload = empty_list(block, module, location)?;
        return build_enum_value(module, block, variant_index, payload, location);
    }

    // A payload constructor used as a bare value (`Some` without its argument)
    // would be a partial application, which is not supported.
    if let Some(&(_, _, arity)) = module.constructors.get(name) && arity > 0 {
        return Err(format!(
            "codegen: partial application of constructor `{name}` is not supported"
        ));
    }

    let function_type = module.symbols.get(name).ok_or_else(|| {
        if TypeContext::is_builtin(name) {
            format!("codegen: builtin `{name}` is not implemented yet")
        } else {
            format!("codegen: undefined variable `{name}` (not a bound parameter or symbol)")
        }
    })?;

    // A symbol that takes parameters is a builtin function value: reference it
    // with a `func.constant` so ordinary application can `call_indirect` it.
    // Nullary top-level bindings (`let x = ...` symbols) are called directly.
    if function_type.input_count() > 0 {
        return block
            .append_operation(func::constant(
                module.context,
                FlatSymbolRefAttribute::new(module.context, name),
                *function_type,
                location,
            ))
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into);
    }

    let mut results = Vec::new();
    for i in 0..function_type.result_count() {
        results.push(function_type.result(i).map_err(|e| e.to_string())?);
    }
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, name),
        &[],
        &results,
        location,
    );
    block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Emit (or reuse) a `func.func` for `name` specialized at the concrete
/// function type `typ`, returning its symbol.
///
/// A curried lambda `\a => \b => body` is compiled to a single function
/// `func.func @name_spec(a, b) -> ret` taking all of its parameters at once.
/// The specialization is cached by `(name, typ)` so each instantiation is
/// emitted exactly once; the cache is populated *before* the body is lowered
/// so recursive uses of `name` at the same type resolve to the in-progress
/// symbol.
///
/// When the body contains a self tail call (see
/// [`super::closures::has_self_tail_call`]), it is lowered in tail position
/// by [`super::tail::lower_tail`]: the entry block doubles as the loop
/// header and self tail calls become `cf.br` backedges, so the recursion
/// runs in constant stack space.
pub(crate) fn specialize_binding<'c>(
    name: &str,
    self_name: &str,
    typ: &Monotype,
    captures: &[(String, Type<'c>)],
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<String, String> {
    let info = module.abstractions.get(name).ok_or_else(|| {
        format!("codegen: `{name}` is not a registered lambda binding")
    })?;
    let params = abstraction_params(&info.param, &info.body);

    let concrete = default_free_vars(typ);
    let (param_monos, ret_mono) = concrete_func_parts(typ, params.len())?;

    let capture_types: Vec<String> = captures.iter().map(|(_, t)| t.to_string()).collect();
    let key = (
        name.to_string(),
        format!("{concrete:?}"),
        capture_types.join("|"),
    );
    if let Some(symbol) = module.specializations.get(&key) {
        return Ok(symbol.clone());
    }

    let substitution = crate::types::unify(&mut TypeContext::new(), &info.abs_type, &concrete)
        .map_err(|e| format!("codegen: cannot specialize `{name}`: {}", e.message))?;
    let mut body = info.body.clone();
    crate::types::apply_substitution(&mut body, &substitution);
    let inner_body = peel_inner(&body);

    let symbol = format!("{name}_spec_{}", module.spec_counter);
    module.spec_counter += 1;
    module.specializations.insert(key, symbol.clone());

    let param_mlirs: Vec<Type> = param_monos
        .iter()
        .map(|m| lower_type(m, module))
        .collect::<Result<_, _>>()?;
    let ret_mlir = lower_type(&ret_mono, module)?;

    // Captured variables are threaded as leading parameters, so the lifted
    // function is `@spec(captures..., params...) -> ret`.
    let mut all_inputs: Vec<Type> = captures.iter().map(|(_, t)| *t).collect();
    all_inputs.extend(param_mlirs.iter().copied());

    let block = Block::new(&all_inputs.iter().map(|t| (*t, location)).collect::<Vec<_>>());

    // A body with a self tail call is lowered in tail position (see
    // [`super::tail`]). The region entry block may not have predecessors, so
    // it trampolines to a separate loop header block with the same argument
    // list; the parameters and captures bind to the *header's* arguments.
    let header = if has_self_tail_call(&inner_body, self_name, params.len(), &module.inlineable) {
        Some(Block::new(
            &all_inputs.iter().map(|t| (*t, location)).collect::<Vec<_>>(),
        ))
    } else {
        None
    };
    let body_block = header.as_ref().unwrap_or(&block);

    let mut env = HashMap::new();
    for (i, (cname, _)) in captures.iter().enumerate() {
        let arg: Value<'c, '_> = body_block.argument(i).map_err(|e| e.to_string())?.into();
        env.insert(cname.clone(), EnvEntry::Value(arg));
    }
    for (j, p) in params.iter().enumerate() {
        let arg: Value<'c, '_> = body_block
            .argument(captures.len() + j)
            .map_err(|e| e.to_string())?
            .into();
        env.insert(p.clone(), EnvEntry::Value(arg));
    }
    if self_name != name {
        env.insert(self_name.to_string(), EnvEntry::Abstraction(name.to_string()));
    }

    let pending = if let Some(header) = &header {
        let mut tail = TailCtx {
            symbol: symbol.clone(),
            header,
            pending: Vec::new(),
        };
        lower_tail(&inner_body, header, module, &mut env, &mut tail)?;
        let entry_args: Vec<Value<'c, '_>> = (0..all_inputs.len())
            .map(|i| {
                block
                    .argument(i)
                    .map(Into::into)
                    .map_err(|e: melior::Error| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        block.append_operation(melior::dialect::cf::br(header, &entry_args, location));
        let TailCtx { pending, .. } = tail;
        pending
    } else {
        let body_value = lower_expr(&inner_body, &block, module, &mut env)?;
        block.append_operation(func::r#return(&[body_value], location));
        Vec::new()
    };

    let function_type = FunctionType::new(module.context, &all_inputs, &[ret_mlir]);
    let region = Region::new();
    region.append_block(block);
    if let Some(header) = header {
        region.append_block(header);
    }
    for pending_block in pending {
        region.append_block(pending_block);
    }

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

/// Lower a function application `f x`, flattening curried application chains
/// into a single multi-argument call when the function is fully applied.
pub(crate) fn lower_application<'c, 'a>(
    f: &Expr,
    x: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {

    // Flatten `(((f a) b) c)` into a root and an ordered argument list,
    // expanding any inlineable function-valued `let` bindings.
    let mut args = Vec::new();
    let root = collect_application_root(f, &mut args, module);
    args.push((*x).clone());

    // An enum constructor applied to all of its arguments: allocate the enum
    // value with a payload struct holding every field.
    if let ENode::Variable(name) = &*root.e
    && let Some(&(_, variant_index, arity)) = module.constructors.get(name) {
        if args.len() != arity {
            return Err(format!(
                "codegen: constructor `{name}` applied to the wrong number of arguments"
            ));
        }
        let mut fields = Vec::with_capacity(arity);
        let mut field_monos = Vec::with_capacity(arity);
        for arg in &args {
            let value = lower_expr(arg, block, module, env)?;
            let mono = default_free_vars(&arg.typ);
            let typ = lower_type(&mono, module)?;
            field_monos.push(mono);
            fields.push((value, typ));
        }
        let size = tuple_size(&field_monos) as i64;
        let payload = build_payload(module, block, &fields, size, location)?;
        return build_enum_value(module, block, variant_index, payload, location);
    }

    // A known lambda applied to exactly as many arguments as it takes: emit a
    // multi-argument specialization and call it once. The local environment
    // takes precedence over top-level bindings so a local binding can shadow
    // a top-level abstraction name.
    if let ENode::Variable(name) = &*root.e {
        let sym = if let Some(EnvEntry::Abstraction(s)) = env.get(name) {
            Some(s.clone())
        } else if module.abstractions.contains_key(name) {
            Some(name.clone())
        } else {
            None
        };
        if let Some(sym) = sym && let Some(info) = module.abstractions.get(&sym) {
            let info = info.clone();
            let params = abstraction_params(&info.param, &info.body);
            if args.len() == params.len() {
                return lower_full_application(
                    &sym,
                    name,
                    &info,
                    &root.typ,
                    &params,
                    &args,
                    block,
                    module,
                    env,
                    location,
                );
            }
            return lower_partial_application(
                &sym,
                name,
                &info,
                &root.typ,
                &params,
                &args,
                block,
                module,
                env,
                location,
            );
        }
    }

    // A general function value: fully apply it when the number of arguments
    // matches its arity.
    let function = lower_expr(&root, block, module, env)?;
    if let Ok(func_type) = FunctionType::try_from(function.r#type()) {
        if args.len() == func_type.input_count() {
            let arg_values: Vec<Value<'c, 'a>> = args
                .iter()
                .map(|a| lower_expr(a, block, module, env))
                .collect::<Result<_, _>>()?;
            let mut ret_types = Vec::with_capacity(func_type.result_count());
            for i in 0..func_type.result_count() {
                ret_types.push(func_type.result(i).map_err(|e| e.to_string())?);
            }
            return block
                .append_operation(func::call_indirect(
                    function,
                    &arg_values,
                    &ret_types,
                    location,
                ))
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into);
        }
        return Err(
            "codegen: partial application of a function value is not supported yet".to_string(),
        );
    }

    // Fallback: single-argument application through a closure struct.
    let argument = lower_expr(x, block, module, env)?;
    let (_, ret_mono) = concrete_parts(&f.typ).ok_or_else(|| {
        format!(
            "codegen: cannot apply {:?}: expected a single-argument function type",
            *f.e
        )
    })?;
    let ret_mlir = lower_type(&ret_mono, module)?;
    closure_call(module, block, function, &[argument], ret_mlir, location)
}

/// Lower a full application of the lambda `sym` to all of its arguments.
fn lower_full_application<'c, 'a>(
    sym: &str,
    self_name: &str,
    info: &AbstractionInfo,
    root_typ: &Monotype,
    params: &[String],
    args: &[Expr],
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &mut Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let captures = lambda_captures(info, self_name, env);
    let symbol = specialize_binding(sym, self_name, root_typ, &captures, module, location)?;

    let (_, ret_mono) = concrete_func_parts(root_typ, params.len())?;
    let ret_mlir = lower_type(&ret_mono, module)?;

    let mut call_args: Vec<Value<'c, 'a>> = Vec::with_capacity(captures.len() + args.len());
    for (cname, _) in &captures {
        match env.get(cname) {
            Some(EnvEntry::Value(v)) => call_args.push(*v),
            _ => return Err(format!("codegen: missing capture `{cname}`")),
        }
    }
    for a in args {
        call_args.push(lower_expr(a, block, module, env)?);
    }

    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, &symbol),
        &call_args,
        &[ret_mlir],
        location,
    );
    block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Lower a partial application of the lambda `sym`: it has more parameters
/// than the arguments supplied, so the result is itself a function value.
///
/// Emits a wrapper `func.func @sym_partial_N(remaining...) -> ret` whose body
/// calls the fully specialized `@sym_spec_...` with the already-supplied
/// arguments followed by the remaining parameters, and references it with a
/// `func.constant` of the flat function type `(remaining...) -> ret`.
fn lower_partial_application<'c, 'a>(
    sym: &str,
    self_name: &str,
    info: &AbstractionInfo,
    root_typ: &Monotype,
    params: &[String],
    args: &[Expr],
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let captures = lambda_captures(info, self_name, env);
    if !captures.is_empty() {
        return Err(format!(
            "codegen: partial application of `{self_name}` (a closure with captures) is not supported yet"
        ));
    }
    let full_symbol = specialize_binding(sym, self_name, root_typ, &captures, module, location)?;
    let (param_monos, ret_mono) = concrete_func_parts(root_typ, params.len())?;
    let ret_mlir = lower_type(&ret_mono, module)?;

    let remaining_types: Vec<Type<'c>> = param_monos[args.len()..]
        .iter()
        .map(|m| lower_type(m, module))
        .collect::<Result<_, _>>()?;

    let symbol = format!("{sym}_partial_{}", module.spec_counter);
    module.spec_counter += 1;

    let partial_block =
        Block::new(&remaining_types.iter().map(|t| (*t, location)).collect::<Vec<_>>());
    let mut partial_env = HashMap::new();
    for (j, p) in params[args.len()..].iter().enumerate() {
        let arg: Value<'c, 'a> = partial_block
            .argument(j)
            .map_err(|e| e.to_string())?
            .into();
        partial_env.insert(p.clone(), EnvEntry::Value(arg));
    }

    // The already-supplied arguments are lowered inside the wrapper (they may
    // reference top-level symbols); the remaining parameters come from the
    // wrapper's own block arguments.
    let mut call_args: Vec<Value<'c, '_>> = Vec::new();
    for a in args {
        call_args.push(lower_expr(a, &partial_block, module, &mut partial_env)?);
    }
    for j in 0..remaining_types.len() {
        let arg: Value<'c, '_> = partial_block
            .argument(j)
            .map_err(|e| e.to_string())?
            .into();
        call_args.push(arg);
    }

    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, &full_symbol),
        &call_args,
        &[ret_mlir],
        location,
    );
    let result: Value<'c, '_> = partial_block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    partial_block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &remaining_types, &[ret_mlir]);
    let region = Region::new();
    region.append_block(partial_block);

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

    block
        .append_operation(func::constant(
            module.context,
            FlatSymbolRefAttribute::new(module.context, &symbol),
            function_type,
            location,
        ))
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Split a single-argument function type `A => B` into `(A, B)`.
fn function_parts(typ: &Monotype) -> Option<(Monotype, Monotype)> {
    match typ {
        Monotype::TypeFuncApplication(f, args)
            if matches!(**f, TypeFunc::Fn) && args.len() == 2 =>
        {
            Some((args[0].clone(), args[1].clone()))
        }
        _ => None,
    }
}

/// Replace any remaining type variables with `int`, monomorphizing types the
/// type checker left unconstrained (e.g. the discarded result of applying a
/// polymorphic function). MLIR needs a concrete type, and a free variable has
/// no other constraint to satisfy.
pub(crate) fn default_free_vars(typ: &Monotype) -> Monotype {
    match typ {
        Monotype::TypeVariable(_) => Monotype::int(),
        Monotype::TypeFuncApplication(f, args) => Monotype::TypeFuncApplication(
            f.clone(),
            args.iter().map(default_free_vars).collect(),
        ),
    }
}

/// [`function_parts`] with free type variables defaulted to `int`.
fn concrete_parts(typ: &Monotype) -> Option<(Monotype, Monotype)> {
    function_parts(typ).map(|(a, b)| (default_free_vars(&a), default_free_vars(&b)))
}

/// Split a curried function type of `arity` parameters into the flat parameter
/// list and the final result type, with free type variables defaulted.
fn concrete_func_parts(typ: &Monotype, arity: usize) -> Result<(Vec<Monotype>, Monotype), String> {
    let mut params = Vec::with_capacity(arity);
    let mut ret = default_free_vars(typ);
    for _ in 0..arity {
        match function_parts(&ret) {
            Some((p, r)) => {
                params.push(default_free_vars(&p));
                ret = r;
            }
            None => return Err("codegen: not a function type with enough parameters".to_string()),
        }
    }
    Ok((params, default_free_vars(&ret)))
}

/// The flat `FunctionType` (all parameters at once) for a specialization of
/// `name` at `typ` — the same shape `specialize_binding` emits.
fn specialization_function_type<'a>(
    name: &str,
    typ: &Monotype,
    captures: &[(String, Type<'a>)],
    module: &Module<'a>,
) -> Result<FunctionType<'a>, String> {
    let info = module
        .abstractions
        .get(name)
        .ok_or_else(|| format!("codegen: `{name}` is not a registered lambda binding"))?;
    let arity = abstraction_params(&info.param, &info.body).len();
    let (params, ret) = concrete_func_parts(typ, arity)?;
    let mut param_mlirs: Vec<Type<'a>> = captures.iter().map(|(_, t)| *t).collect();
    for m in &params {
        param_mlirs.push(lower_type(m, module)?);
    }
    let ret_mlir = lower_type(&ret, module)?;
    Ok(FunctionType::new(module.context, &param_mlirs, &[ret_mlir]))
}

pub(crate) fn is_scalar_type(typ: &Monotype) -> bool {
    !matches!(typ, Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Fn))
}

/// Lower a bare abstraction `\x1 x2 ... => e` to a closure.
///
/// Curried lambdas are un-curried into a single function taking all of their
/// parameters at once, matching the flat `FunctionType` that function types
/// lower to (see [`super::types::lower_type`]). The abstraction compiles to
/// `func.func @closure_N(x1, ..., xn [, env]) -> ret`; the closure value
/// emitted in the current block is a `func.constant` on that function when
/// there are no captures, otherwise a heap closure struct `{ fn_ptr, env }`
/// holding the captures.
pub(crate) fn lower_abstraction<'c, 'a>(
    expr: &Expr,
    binding: &Binding,
    body: &Expr,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    env: &Env<'c, 'a>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    // Un-curry: `\x => \y => ... => body` becomes `(x, y, ...)` and the
    // innermost body.
    let mut params = vec![binding.clone()];
    let mut inner = body;
    while let ENode::Abstraction(b, bd) = &*inner.e {
        params.push((**b).clone());
        inner = bd;
    }

    // The concrete parameter types come from the lambda's (flattened)
    // resolved type, so every parameter's MLIR type matches the function
    // type expected at the lambda's use sites.
    let (param_monos, _) = concrete_func_parts(&expr.typ, params.len())?;
    let param_mlirs: Vec<Type> = param_monos
        .iter()
        .map(|m| lower_type(m, module))
        .collect::<Result<_, _>>()?;
    let env_i64 = IntegerType::new(module.context, 64).into();

    // Captures: free variables of the innermost body (beyond all parameters)
    // that are bound in the enclosing environment. Their values are available
    // here (at closure creation) and are loaded from the `env` pointer inside
    // the compiled function.
    let free = free_variables(inner);
    let param_names: HashSet<String> = params.iter().map(|b| b.0.clone()).collect();
    let mut captures: Vec<(String, Value<'c, 'a>, Type<'c>)> = Vec::new();
    for (name, entry) in env.iter() {
        if free.contains(name) && !param_names.contains(name) {
            match entry {
                EnvEntry::Value(value) => captures.push((name.clone(), *value, value.r#type())),
                EnvEntry::Abstraction(sym) => {
                    if let Some(info) = module.abstractions.get(sym) {
                        let concrete = default_free_vars(&info.abs_type);
                        let mlir_type = lower_type(&concrete, module)?;
                        let ref_val = reference_specialization(sym, name, &concrete, &[], block, module, location)?;
                        captures.push((name.clone(), ref_val, mlir_type));
                    } else if let Some(info) = module.abstractions.get(name) {
                        let concrete = default_free_vars(&info.abs_type);
                        let mlir_type = lower_type(&concrete, module)?;
                        let ref_val = reference_specialization(name, name, &concrete, &[], block, module, location)?;
                        captures.push((name.clone(), ref_val, mlir_type));
                    } else {
                        return Err(format!(
                            "codegen: cannot capture lambda binding `{name}` in a closure yet"
                        ));
                    }
                }
            }
        }
    }

    let symbol = format!("closure_{}", module.closures);
    module.closures += 1;

    if captures.is_empty() {
        let closure_block = Block::new(
            &param_mlirs.iter().map(|t| (*t, location)).collect::<Vec<_>>(),
        );
        let mut closure_env = HashMap::new();
        for (i, Binding(param, _)) in params.iter().enumerate() {
            let arg: Value<'c, 'a> = closure_block
                .argument(i)
                .map_err(|e| e.to_string())?
                .into();
            closure_env.insert(param.clone(), EnvEntry::Value(arg));
        }

        let body_value = lower_expr(inner, &closure_block, module, &mut closure_env)?;
        closure_block.append_operation(func::r#return(&[body_value], location));

        let function_type = FunctionType::new(module.context, &param_mlirs, &[body_value.r#type()]);
        let region = Region::new();
        region.append_block(closure_block);

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

        block
            .append_operation(func::constant(
                module.context,
                FlatSymbolRefAttribute::new(module.context, &symbol),
                function_type,
                location,
            ))
            .result(0)
            .map_err(|e| e.to_string())
            .map(Into::into)
    } else {
        let mut all_inputs: Vec<Type> = param_mlirs.clone();
        all_inputs.push(env_i64);
        let closure_block = Block::new(
            &all_inputs.iter().map(|t| (*t, location)).collect::<Vec<_>>(),
        );
        let mut closure_env = HashMap::new();
        for (i, Binding(param, _)) in params.iter().enumerate() {
            let arg: Value<'c, 'a> = closure_block
                .argument(i)
                .map_err(|e| e.to_string())?
                .into();
            closure_env.insert(param.clone(), EnvEntry::Value(arg));
        }
        let env_arg_i64: Value<'c, 'a> = closure_block
            .argument(params.len())
            .map_err(|e| e.to_string())?
            .into();

        let env_ptr = Type::parse(module.context, "!llvm.ptr")
            .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
        let inttoptr = OperationBuilder::new("llvm.inttoptr", location)
            .add_operands(&[env_arg_i64])
            .add_results(&[env_ptr])
            .build()
            .map_err(|e| e.to_string())?;
        let env_arg: Value<'c, 'a> = closure_block
            .append_operation(inttoptr)
            .result(0)
            .map_err(|e| e.to_string())?
            .into();
        let env_struct = env_struct_type(module, &captures)?;
        for (i, (capture, _, typ)) in captures.iter().enumerate() {
            let value = load_field(
                module,
                &closure_block,
                env_arg,
                env_struct,
                i as i32,
                *typ,
                location,
            )?;
            closure_env.insert(capture.clone(), EnvEntry::Value(value));
        }

        let body_value = lower_expr(inner, &closure_block, module, &mut closure_env)?;
        closure_block.append_operation(func::r#return(&[body_value], location));

        let function_type =
            FunctionType::new(module.context, &all_inputs, &[body_value.r#type()]);
        let region = Region::new();
        region.append_block(closure_block);

        let function = func::func(
            module.context,
            StringAttribute::new(module.context, &symbol),
            TypeAttribute::new(function_type.into()),
            region,
            &[(
                Identifier::new(module.context, "llvm.emit_c_interface"),
                Attribute::unit(module.context),
            )],
            location,
        );
        module.module.body().append_operation(function);
        module.functions += 1;

        build_closure(module, block, &symbol, &captures, location)
    }
}

/// SSA => Static Single Assignment
/// Lower a literal to an `arith.constant`, returning its SSA value.
pub(crate) fn lower_literal<'c, 'a>(
    lit: &Lit,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    match lit {
        // Strings live in a module-level global and need several ops.
        Lit::Str(value) => lower_string(value, block, module, location),
        _ => {
            let operation = match lit {
                Lit::Int(value) => arith::constant(
                    module.context,
                    IntegerAttribute::new(
                        IntegerType::new(module.context, 32).into(),
                        *value as i64,
                    )
                    .into(),
                    location,
                ),
                Lit::Float(value) => arith::constant(
                    module.context,
                    FloatAttribute::new(module.context, Type::float32(module.context), *value as f64)
                        .into(),
                    location,
                ),
                Lit::Bool(value) => arith::constant(
                    module.context,
                    BoolAttribute::new(module.context, *value).into(),
                    location,
                ),
                // Unit is treated as i32
                Lit::Unit => arith::constant(
                    module.context,
                    IntegerAttribute::new(IntegerType::new(module.context, 32).into(), 0).into(),
                    location,
                ),
                Lit::Char(value) => arith::constant(
                    module.context,
                    IntegerAttribute::new(
                        IntegerType::new(module.context, 32).into(),
                        *value as i64).into(),
                    location
                ),
                Lit::Str(_) => unreachable!(),
            };
            // The op must be appended to the block before its results are used,
            // otherwise it is destroyed when `operation` drops and `value` dangles.
            block
                .append_operation(operation)
                .result(0)
                .map_err(|e| e.to_string())
                .map(Into::into)
        }
    }
}

/// Lower a string literal to a module-level `llvm.mlir.global` plus
/// `llvm.mlir.addressof` and `llvm.getelementptr`, returning `!llvm.ptr`.
pub(crate) fn lower_string<'c, 'a>(
    value: &str,
    block: &'a Block<'c>,
    module: &mut Module<'c>,
    location: Location<'c>,
) -> Result<Value<'c, 'a>, String> {
    let context = module.context;

    let symbol = format!("str_{}", module.strings);
    module.strings += 1;

    let bytes = value.len() + 1; // trailing NUL
    let array_type = Type::parse(context, &format!("!llvm.array<{bytes} x i8>"))
        .ok_or_else(|| format!("codegen: failed to create `!llvm.array<{bytes} x i8>`"))?;
    let ptr_type = Type::parse(context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;

    // `llvm.mlir.global private @str_N = "value\0" : !llvm.array<N x i8>`
    let global = OperationBuilder::new("llvm.mlir.global", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, &symbol).into(),
            ),
            (
                Identifier::new(context, "value"),
                StringAttribute::new(context, &format!("{value}\0")).into(),
            ),
            (
                Identifier::new(context, "global_type"),
                TypeAttribute::new(array_type).into(),
            ),
            (
                Identifier::new(context, "linkage"),
                llvm_private_linkage(context),
            ),
        ])
        .add_regions([Region::new()])
        .build()
        .map_err(|e| e.to_string())?;
    module.module.body().append_operation(global);

    // `llvm.mlir.addressof @str_N : !llvm.ptr`
    let addressof = OperationBuilder::new("llvm.mlir.addressof", location)
        .add_attributes(&[(
            Identifier::new(context, "global_name"),
            FlatSymbolRefAttribute::new(context, &symbol).into(),
        )])
        .add_results(&[ptr_type])
        .build()
        .map_err(|e| e.to_string())?;
    let array_ptr = block.append_operation(addressof).result(0).map_err(|e| e.to_string())?;

    // `llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr`
    let gep = llvm::get_element_ptr(
        context,
        array_ptr.into(),
        DenseI32ArrayAttribute::new(context, &[0, 0]),
        array_type,
        ptr_type,
        location,
    );
    block
        .append_operation(gep)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// `#llvm.linkage<private>` attribute, required by `llvm.mlir.global`.
fn llvm_private_linkage(context: &melior::Context) -> melior::ir::Attribute<'_> {
    melior::dialect::llvm::attributes::linkage(
        context,
        melior::dialect::llvm::attributes::Linkage::Private,
    )
}
