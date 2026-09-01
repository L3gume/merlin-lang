//! Top-level statements and declarations.

use crate::ast::*;
use crate::types::{Monotype, TypeFunc};
use melior::dialect::llvm::LoadStoreOptions;
use melior::dialect::{arith, func, llvm};
use melior::ir::{
    Attribute,
    attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
    operation::OperationBuilder,
    r#type::{FunctionType, IntegerType},
    Block, BlockLike, Identifier, Location, Region, RegionLike, Type, Value, ValueLike,
};
use std::collections::HashMap;

use super::{AbstractionInfo, EnumLayout, Module, RecordLayout};
use super::apply::{default_free_vars, lower_string};
use super::expr::lower_expr;
use super::lists::{integer_constant, malloc_call};
use super::types::lower_type;

/// Lower a type-checked program to an MLIR module.
///
/// `context` is borrowed and must outlive the returned module; the REPL owns
/// it so bindings can be appended across input lines.
///
/// Returns an error string if any statement cannot be lowered (e.g. an AST
/// node with no dialect mapping yet).
pub fn lower<'a>(prog: &Program, context: &'a melior::Context) -> Result<Module<'a>, String> {
    let mut module = Module::new(context);
    module.set_source_name(prog.source_name.clone());
    register_runtime_builtins(&mut module)?;

    let entry_block = Block::new(&[]);
    let mut last_value: Option<Value<'a, '_>> = None;
    let mut last_monotype: Option<Monotype> = None;
    for stmt in &prog.stmts {
        match lower_stmt(stmt, &mut module, &entry_block)
            .map_err(|e| with_stmt_pos(e, &stmt.pos, &prog.source_name))?
        {
            Some(value) => {
                last_value = Some(value);
                last_monotype = if let SNode::Expr(e) = &*stmt.s {
                    Some(e.typ.clone())
                } else {
                    None
                };
            }
            // A trailing non-expression statement (e.g. a `let`) produces no
            // value: `__main` must return unit, not a stale earlier value.
            None => {
                last_value = None;
                last_monotype = None;
            }
        }
    }

    let location = Location::unknown(context);
    let outputs: Vec<Type<'a>> = match last_value {
        Some(value) => {
            let typ = value.r#type();
            entry_block.append_operation(func::r#return(&[value], location));
            vec![typ]
        }
        None => {
            entry_block.append_operation(func::r#return(&[], location));
            vec![]
        }
    };

    module.entry_return_monotype = last_monotype;
    emit_entry_function(&mut module, entry_block, &outputs)?;
    Ok(module)
}

// ----------------------------------------------------------------------------
// Statements
// ----------------------------------------------------------------------------

/// Prefix a codegen error with the source position of the statement being
/// lowered, when known.
fn with_stmt_pos(msg: String, pos: &crate::ast::Pos, source_name: &str) -> String {
    if pos.is_nil() || source_name.is_empty() {
        return msg;
    }
    format!("{source_name}:{}:{}: {}", pos.start_line, pos.start_col, msg)
}

fn lower_stmt<'a, 'b>(
    stmt: &Stmt,
    module: &mut Module<'a>,
    entry: &'b Block<'a>,
) -> Result<Option<Value<'a, 'b>>, String> {
    match &*stmt.s {
        SNode::TypeDecl(h, t) => {
            lower_type_decl(h, t, module)?;
            Ok(None)
        }
        SNode::Decl(e1, t, e2) => {
            lower_decl(e1, t, e2, module)?;
            Ok(None)
        }
        SNode::Expr(e1) => lower_expr_stmt(e1, module, entry).map(Some),
    }
}

/// Lower a top-level expression statement `e;` into the entry function body.
///
/// The produced value is tracked by [`lower`] and becomes the `func.return`
/// of `@__main` if this is the last statement.
fn lower_expr_stmt<'a, 'b>(
    expr: &Expr,
    module: &mut Module<'a>,
    entry: &'b Block<'a>,
) -> Result<Value<'a, 'b>, String> {
    let mut env = HashMap::new();
    lower_expr(expr, entry, module, &mut env)
}

/// Emit the runtime builtin functions and register their signatures in
/// [`Module::symbols`], so `print "hi"` — and even `map print xs` — lower
/// through the ordinary application path (`lower_variable` -> `call_indirect`)
/// instead of special dispatch.
///
/// Each builtin is a real `func.func` with a body. The remaining builtins are
/// seeded in the type context (see [`crate::types::builtins`]); their emitters
/// are still stubs that report "not implemented yet", so a use site fails in
/// `lower_variable` with a clear error until each gets a body here.
pub(crate) fn register_runtime_builtins<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let i32_type: Type = IntegerType::new(module.context, 32).into();
    let i64_type: Type = IntegerType::new(module.context, 64).into();
    ensure_extern(module, "printf", &[i64_type], &[i32_type])?;
    ensure_extern(module, "puts", &[i64_type], &[i32_type])?;

    // `register_builtin` runs every emitter: implemented ones emit their
    // `func.func` and register a symbol, while stubs report the
    // "not implemented yet" error, which is skipped so compiling a program
    // that never mentions them still succeeds.
    register_builtin(module, emit_print)?;
    register_builtin(module, emit_println)?;
    register_builtin(module, emit_itostr)?;
    register_builtin(module, emit_ftostr)?;
    register_builtin(module, emit_btostr)?;
    register_builtin(module, emit_strtoi)?;
    register_builtin(module, emit_strtof)?;
    register_builtin(module, emit_strtob)?;
    register_builtin(module, emit_itof)?;
    register_builtin(module, emit_ftoi)?;
    register_builtin(module, emit_readin)?;
    Ok(())
}

/// Run a builtin emitter, treating the "not implemented yet" stub error as a
/// skip: the builtin simply stays out of [`Module::symbols`] until it gets a
/// body. Any other error is propagated.
fn register_builtin<'a>(
    module: &mut Module<'a>,
    emit: impl FnOnce(&mut Module<'a>) -> Result<(), String>,
) -> Result<(), String> {
    match emit(module) {
        Err(e) if e.ends_with("not implemented yet") => Ok(()),
        result => result,
    }
}

// ----------------------------------------------------------------------------
// Builtin type mapping
// ----------------------------------------------------------------------------

/// The MLIR types the runtime builtins operate on.
///
/// Unit is lowered to `i32` because LLVM has no unit type: `str -> unit`
/// becomes `(!llvm.ptr) -> i32` and `unit -> str` becomes `(i32) -> !llvm.ptr`.
struct BuiltinTypes<'a> {
    int: Type<'a>,
    float: Type<'a>,
    bool: Type<'a>,
    string: Type<'a>,
    unit: Type<'a>,
}

impl<'a> BuiltinTypes<'a> {
    fn new(module: &Module<'a>) -> Result<BuiltinTypes<'a>, String> {
        Ok(BuiltinTypes {
            int: IntegerType::new(module.context, 32).into(),
            float: Type::float32(module.context),
            bool: IntegerType::new(module.context, 1).into(),
            string: Type::parse(module.context, "!llvm.ptr")
                .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?,
            unit: IntegerType::new(module.context, 32).into(),
        })
    }
}

/// Append `llvm.ptrtoint %value : !llvm.ptr to i64` to `block`, matching the
/// `i64` argument convention used for every external libc call.
pub(crate) fn ptrtoint_i64<'a, 'b>(
    module: &Module<'a>,
    block: &'b Block<'a>,
    value: Value<'a, 'b>,
    location: Location<'a>,
) -> Result<Value<'a, 'b>, String> {
    let op = OperationBuilder::new("llvm.ptrtoint", location)
        .add_operands(&[value])
        .add_results(&[IntegerType::new(module.context, 64).into()])
        .build()
        .map_err(|e| e.to_string())?;
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

/// Append `llvm.inttoptr %value : i64 to !llvm.ptr` to `block`, converting a
/// byte address (e.g. `buf + offset`) back into a pointer for a store.
pub(crate) fn inttoptr_ptr<'a, 'b>(
    module: &Module<'a>,
    block: &'b Block<'a>,
    value: Value<'a, 'b>,
    location: Location<'a>,
) -> Result<Value<'a, 'b>, String> {
    let ptr = Type::parse(module.context, "!llvm.ptr")
        .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())?;
    let op = OperationBuilder::new("llvm.inttoptr", location)
        .add_operands(&[value])
        .add_results(&[ptr])
        .build()
        .map_err(|e| e.to_string())?;
    block
        .append_operation(op)
        .result(0)
        .map_err(|e| e.to_string())
        .map(Into::into)
}

// ----------------------------------------------------------------------------
// Individual builtins
// ----------------------------------------------------------------------------

/// `print : str -> unit`, lowered as `func.func @print(!llvm.ptr) -> i32`.
///
/// The body forwards to `@printf`:
///
/// ```text
/// @print(%arg0: !llvm.ptr) -> i32 {
///   %0 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
///   %1 = func.call @printf(%0) : (i64) -> i32
///   return %1 : i32
/// }
/// ```
fn emit_print<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();

    let block = Block::new(&[(t.string, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let ptrtoint = OperationBuilder::new("llvm.ptrtoint", location)
        .add_operands(&[arg])
        .add_results(&[i64_type])
        .build()
        .map_err(|e| e.to_string())?;
    let arg_i64: Value<'_, '_> = block
        .append_operation(ptrtoint)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "printf"),
        &[arg_i64],
        &[t.unit],
        location,
    );
    let result: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.string], &[t.unit]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "print"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("print".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `println : str -> unit`, lowered as `func.func @println(!llvm.ptr) -> i32`.
///
/// The body forwards to `@puts`, which appends a trailing newline itself:
///
/// ```text
/// @println(%arg0: !llvm.ptr) -> i32 {
///   %0 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
///   %1 = func.call @puts(%0) : (i64) -> i32
///   return %1 : i32
/// }
/// ```
fn emit_println<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();

    let block = Block::new(&[(t.string, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let ptrtoint = OperationBuilder::new("llvm.ptrtoint", location)
        .add_operands(&[arg])
        .add_results(&[i64_type])
        .build()
        .map_err(|e| e.to_string())?;
    let arg_i64: Value<'_, '_> = block
        .append_operation(ptrtoint)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "puts"),
        &[arg_i64],
        &[t.unit],
        location,
    );
    let result: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.string], &[t.unit]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "println"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("println".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `itostr : int -> str`, lowered as `func.func @itostr(i32) -> !llvm.ptr`.
///
/// The body formats the integer into a fresh heap buffer with `@sprintf`:
///
/// ```text
/// @itostr(%arg0: i32) -> !llvm.ptr {
///   %0 = arith.sitofp %arg0 : i32 to f64
///   %1 = func.call @malloc(12) : (i64) -> i64
///   %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
///   %fmt = llvm.mlir.addressof @str_N   // "%.0f"
///   %3 = llvm.ptrtoint %2 : !llvm.ptr to i64
///   %4 = llvm.ptrtoint %fmt : !llvm.ptr to i64
///   %5 = func.call @sprintf(%3, %4, %0) : (i64, i64, f64) -> i32
///   return %2 : !llvm.ptr
/// }
/// ```
///
/// The int is widened to `f64` so a single `@sprintf(i64, i64, f64)` declaration
/// serves both `itostr` and `ftostr`; every `i32` is exact in a `f64`, and
/// `"%.0f"` renders it without a decimal point.
fn emit_itostr<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();
    let f64_type: Type = Type::float64(module.context);
    let i32_type: Type = IntegerType::new(module.context, 32).into();

    ensure_extern(module, "sprintf", &[i64_type, i64_type, f64_type], &[i32_type])?;

    let block = Block::new(&[(t.int, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();

    let sitofp = OperationBuilder::new("arith.sitofp", location)
        .add_operands(&[arg])
        .add_results(&[f64_type])
        .build()
        .map_err(|e| e.to_string())?;
    let f64val: Value<'_, '_> = block
        .append_operation(sitofp)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    // `i32` needs at most 11 characters ("-2147483648") plus a trailing NUL.
    let buf = malloc_call(module, &block, 12, location)?;

    // The `"%.0f"` format string lives in a module-level global.
    let fmt = lower_string("%.0f", &block, module, location)?;

    let buf_i64 = ptrtoint_i64(module, &block, buf, location)?;
    let fmt_i64 = ptrtoint_i64(module, &block, fmt, location)?;
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "sprintf"),
        &[buf_i64, fmt_i64, f64val],
        &[i32_type],
        location,
    );
    block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?;
    block.append_operation(func::r#return(&[buf], location));

    let function_type = FunctionType::new(module.context, &[t.int], &[t.string]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "itostr"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("itostr".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `ftostr : float -> str`, lowered as `func.func @ftostr(f32) -> !llvm.ptr`.
///
/// The float is widened to `f64` before the call: C variadic promotion passes
/// `float` as `double`, so `@sprintf` must receive a `f64` to read back the
/// right value. The body then formats it into a fresh heap buffer:
///
/// ```text
/// @ftostr(%arg0: f32) -> !llvm.ptr {
///   %0 = arith.extf %arg0 : f32 to f64
///   %1 = func.call @malloc(32) : (i64) -> i64
///   %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
///   %fmt = llvm.mlir.addressof @str_N   // "%.7g"
///   %3 = llvm.ptrtoint %2 : !llvm.ptr to i64
///   %4 = llvm.ptrtoint %fmt : !llvm.ptr to i64
///   %5 = func.call @sprintf(%3, %4, %0) : (i64, i64, f64) -> i32
///   return %2 : !llvm.ptr
/// }
/// ```
///
/// `%.7g` renders the full precision of an `f32` (7 significant digits) while
/// trimming trailing zeros, e.g. `3.14`.
fn emit_ftostr<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();
    let f64_type: Type = Type::float64(module.context);
    let i32_type: Type = IntegerType::new(module.context, 32).into();

    ensure_extern(module, "sprintf", &[i64_type, i64_type, f64_type], &[i32_type])?;

    let block = Block::new(&[(t.float, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();

    let extf = OperationBuilder::new("arith.extf", location)
        .add_operands(&[arg])
        .add_results(&[f64_type])
        .build()
        .map_err(|e| e.to_string())?;
    let f64val: Value<'_, '_> = block
        .append_operation(extf)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    // 32 bytes comfortably fits any `%.7g` rendering of an `f32`-range value.
    let buf = malloc_call(module, &block, 32, location)?;

    // The `"%.7g"` format string lives in a module-level global.
    let fmt = lower_string("%.7g", &block, module, location)?;

    let buf_i64 = ptrtoint_i64(module, &block, buf, location)?;
    let fmt_i64 = ptrtoint_i64(module, &block, fmt, location)?;
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "sprintf"),
        &[buf_i64, fmt_i64, f64val],
        &[i32_type],
        location,
    );
    block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?;
    block.append_operation(func::r#return(&[buf], location));

    let function_type = FunctionType::new(module.context, &[t.float], &[t.string]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "ftostr"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("ftostr".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `btostr : bool -> str`, lowered as `func.func @btostr(i1) -> !llvm.ptr`.
///
/// Returns one of two static string globals selected on the argument — no heap
/// allocation, so nothing is owned or leaked:
///
/// ```text
/// @btostr(%arg0: i1) -> !llvm.ptr {
///   %t = llvm.mlir.addressof @str_N   // "true"
///   %f = llvm.mlir.addressof @str_M   // "false"
///   %0 = arith.select %arg0, %t, %f : i1, !llvm.ptr
///   return %0 : !llvm.ptr
/// }
/// ```
fn emit_btostr<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;

    let block = Block::new(&[(t.bool, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();

    let true_str = lower_string("true", &block, module, location)?;
    let false_str = lower_string("false", &block, module, location)?;
    let select = arith::select(arg, true_str, false_str, location);
    let result: Value<'_, '_> = block
        .append_operation(select)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.bool], &[t.string]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "btostr"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("btostr".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `strtoi : str -> int`, lowered as `func.func @strtoi(!llvm.ptr) -> i32`.
///
/// Parses a decimal string with `@atoi`. The C function has no error signal —
/// `atoi` returns `0` when no conversion happens — so malformed input yields
/// `0` rather than an error.
fn emit_strtoi<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();
    let i32_type: Type = IntegerType::new(module.context, 32).into();

    ensure_extern(module, "atoi", &[i64_type], &[i32_type])?;

    let block = Block::new(&[(t.string, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let arg_i64 = ptrtoint_i64(module, &block, arg, location)?;
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "atoi"),
        &[arg_i64],
        &[i32_type],
        location,
    );
    let result: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.string], &[t.int]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "strtoi"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("strtoi".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `strtof : str -> float`, lowered as `func.func @strtof(!llvm.ptr) -> f32`.
///
/// Parses with `@atof` (which returns a double) and narrows to `f32`. The C
/// function `strtof` shares the builtin's symbol name, so `atof` avoids a
/// collision with the language builtin.
fn emit_strtof<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();
    let f64_type: Type = Type::float64(module.context);

    ensure_extern(module, "atof", &[i64_type], &[f64_type])?;

    let block = Block::new(&[(t.string, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let arg_i64 = ptrtoint_i64(module, &block, arg, location)?;
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "atof"),
        &[arg_i64],
        &[f64_type],
        location,
    );
    let d: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let truncf = OperationBuilder::new("arith.truncf", location)
        .add_operands(&[d])
        .add_results(&[t.float])
        .build()
        .map_err(|e| e.to_string())?;
    let result: Value<'_, '_> = block
        .append_operation(truncf)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.string], &[t.float]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "strtof"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("strtof".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `strtob : str -> bool`, lowered as `func.func @strtob(!llvm.ptr) -> i1`.
///
/// `true` iff the string equals `"true"` (case-sensitive), compared with
/// `@strcmp`.
fn emit_strtob<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i64_type: Type = IntegerType::new(module.context, 64).into();
    let i32_type: Type = IntegerType::new(module.context, 32).into();

    ensure_extern(module, "strcmp", &[i64_type, i64_type], &[i32_type])?;

    let block = Block::new(&[(t.string, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let arg_i64 = ptrtoint_i64(module, &block, arg, location)?;
    let true_str = lower_string("true", &block, module, location)?;
    let true_i64 = ptrtoint_i64(module, &block, true_str, location)?;
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "strcmp"),
        &[arg_i64, true_i64],
        &[i32_type],
        location,
    );
    let cmp: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let zero = arith::constant(
        module.context,
        IntegerAttribute::new(IntegerType::new(module.context, 32).into(), 0).into(),
        location,
    );
    let zero_v: Value<'_, '_> = block
        .append_operation(zero)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let eq = arith::cmpi(module.context, arith::CmpiPredicate::Eq, cmp, zero_v, location);
    let result: Value<'_, '_> = block
        .append_operation(eq)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.string], &[t.bool]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "strtob"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("strtob".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `itof : int -> float`, lowered as `func.func @itof(i32) -> f32`.
///
/// Widens with `arith.sitofp`; no libc call needed.
fn emit_itof<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;

    let block = Block::new(&[(t.int, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let sitofp = OperationBuilder::new("arith.sitofp", location)
        .add_operands(&[arg])
        .add_results(&[t.float])
        .build()
        .map_err(|e| e.to_string())?;
    let result: Value<'_, '_> = block
        .append_operation(sitofp)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.int], &[t.float]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "itof"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("itof".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `ftoi : float -> int`, lowered as `func.func @ftoi(f32) -> i32`.
///
/// Narrows with `arith.fptosi`, truncating toward zero; no libc call needed.
fn emit_ftoi<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;

    let block = Block::new(&[(t.float, location)]);
    let arg: Value<'_, '_> = block.argument(0).map_err(|e| e.to_string())?.into();
    let fptosi = OperationBuilder::new("arith.fptosi", location)
        .add_operands(&[arg])
        .add_results(&[t.int])
        .build()
        .map_err(|e| e.to_string())?;
    let result: Value<'_, '_> = block
        .append_operation(fptosi)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(func::r#return(&[result], location));

    let function_type = FunctionType::new(module.context, &[t.float], &[t.int]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "ftoi"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("ftoi".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// `readin : unit -> str`, lowered as `func.func @readin(i32) -> !llvm.ptr`.
///
/// Reads one line from stdin into a fresh heap buffer via POSIX `read`, then
/// strips the trailing newline with `strcspn`. The `unit` argument is the
/// `i32` stand-in (applications need an argument).
///
/// ```text
/// @readin(%arg0: i32) -> !llvm.ptr {
///   %buf = func.call @malloc(1024)          // 1 KiB line buffer
///   %n   = func.call @read(0, %buf, 1023)   // cap below buffer size
///   %n   = max(%n, 0)                       // errors/EOF read nothing
///   buf[%n] = 0                             // NUL-terminate
///   %end = func.call @strcspn(%buf, "\n")
///   buf[%end] = 0                           // drop the newline
///   return %buf
/// }
/// ```
fn emit_readin<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let t = BuiltinTypes::new(module)?;
    let i8_type: Type = IntegerType::new(module.context, 8).into();
    let i32_type: Type = IntegerType::new(module.context, 32).into();
    let i64_type: Type = IntegerType::new(module.context, 64).into();

    // Ensure extern symbols are loaded and available
    ensure_extern(module, "read", &[i32_type, i64_type, i64_type], &[i64_type])?;
    ensure_extern(module, "strcspn", &[i64_type, i64_type], &[i64_type])?;

    let block = Block::new(&[(t.unit, location)]);

    // Allocate buffer for read (1024 bytes)
    let buf = malloc_call(module, &block, 1024, location)?;
    let buf_i64 = ptrtoint_i64(module, &block, buf, location)?;

    // Specify file descriptor (stdin = 0) and buffer size (1023) arguments
    let fd = integer_constant(module, &block, 32, 0, location)?;
    let count = integer_constant(module, &block, 64, 1023, location)?;
    // Emit call to read external symbol with file descriptor, buffer, and size args
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "read"),
        &[fd, buf_i64, count],
        &[i64_type],
        location,
    );
    // Map result of call to value n
    let n: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    // Clamp to >= 0 so an error (-1) still NUL-terminates within the buffer.
    let zero64 = integer_constant(module, &block, 64, 0, location)?;
    let maxsi = arith::maxsi(n, zero64, location);
    // Overwrite n as result of Max(n, 0)
    let n: Value<'_, '_> = block
        .append_operation(maxsi)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    // Set buf[n] to 0 to ensure buffer is always null-terminated
    let n_addr = arith::addi(buf_i64, n, location);
    let n_addr: Value<'_, '_> = block
        .append_operation(n_addr)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let n_ptr = inttoptr_ptr(module, &block, n_addr, location)?;
    let zero_i8 = arith::constant(
        module.context,
        IntegerAttribute::new(i8_type, 0).into(),
        location,
    );
    let zero_i8: Value<'_, '_> = block
        .append_operation(zero_i8)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    block.append_operation(llvm::store(
        module.context,
        zero_i8,
        n_ptr,
        location,
        LoadStoreOptions::new(),
    ));

    // Use strcspn to get length of string up to LF (\n)
    let nl = lower_string("\n", &block, module, location)?;
    let nl_i64 = ptrtoint_i64(module, &block, nl, location)?;
    let call = func::call(
        module.context,
        FlatSymbolRefAttribute::new(module.context, "strcspn"),
        &[buf_i64, nl_i64],
        &[i64_type],
        location,
    );
    let end: Value<'_, '_> = block
        .append_operation(call)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();

    // Store string without \n to memory
    // Do this by overwriting \n char with \0
    let end_addr = arith::addi(buf_i64, end, location);
    let end_addr: Value<'_, '_> = block
        .append_operation(end_addr)
        .result(0)
        .map_err(|e| e.to_string())?
        .into();
    let end_ptr = inttoptr_ptr(module, &block, end_addr, location)?;
    block.append_operation(llvm::store(
        module.context,
        zero_i8,
        end_ptr,
        location,
        LoadStoreOptions::new(),
    ));

    // Emit return of the read buffer
    block.append_operation(func::r#return(&[buf], location));

    let function_type = FunctionType::new(module.context, &[t.unit], &[t.string]);
    let region = Region::new();
    region.append_block(block);

    // Create builtin readin function and add to module object for later use
    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "readin"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.module.body().append_operation(function);
    module.symbols.insert("readin".to_string(), function_type);
    module.functions += 1;
    Ok(())
}

/// Emit the external declaration `func.func @name(inputs) -> results` once.
///
/// Externs are the C-library functions behind the builtins (`printf`, `puts`,
/// `sprintf`, `atoi`, ...). Each is declared at most once per module, marked
/// `private`, and uses only built-in types so the `func_to_llvm` pass converts
/// it cleanly. Pointer arguments are passed as `i64` (via `llvm.ptrtoint`).
pub(crate) fn ensure_extern<'a>(
    module: &mut Module<'a>,
    name: &str,
    inputs: &[Type<'a>],
    results: &[Type<'a>],
) -> Result<(), String> {
    if module.externs.contains(name) {
        return Ok(());
    }
    let location = Location::unknown(module.context);
    let function_type = FunctionType::new(module.context, inputs, results);
    let function = func::func(
        module.context,
        StringAttribute::new(module.context, name),
        TypeAttribute::new(function_type.into()),
        Region::new(),
        &[(
            Identifier::new(module.context, "sym_visibility"),
            StringAttribute::new(module.context, "private").into(),
        )],
        location,
    );
    module.module.body().append_operation(function);
    module.externs.insert(name.to_string());
    Ok(())
}

/// Materialize the entry function `@__main` with the given result types.
///
/// Its body is the accumulated top-level statements; an empty result list
/// means the function returns unit.
fn emit_entry_function<'a>(
    module: &mut Module<'a>,
    entry: Block<'a>,
    outputs: &[Type<'a>],
) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let function_type = FunctionType::new(module.context, &[], outputs);
    let region = Region::new();
    region.append_block(entry);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "__main"),
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
    Ok(())
}

// ----------------------------------------------------------------------------
// Declarations
// ----------------------------------------------------------------------------

/// Lower a top-level `let x : T = e;` binding to a `func.func`.
///
/// A binding becomes a nullary symbol `@x` whose result type is `T` (or, for
/// an unannotated binding, the type of the lowered initializer) and whose
/// body lowers `e` and `func.return`s it.
///
/// When the initializer is a lambda, no symbol is emitted: the abstraction is
/// registered in [`Module::abstractions`] and specialized on demand for every
/// concrete type it is used at, so a polymorphic function like `let id =
/// \x => x;` can be applied at `int` and `bool` independently.
pub fn lower_decl<'a>(
    e1: &Expr,
    typ: &crate::ast::Type,
    e2: &Expr,
    module: &mut Module<'a>,
) -> Result<(), String> {
    let name = match &*e1.e {
        ENode::Variable(n) => n.clone(),
        _ => {
            return Err(format!(
                "codegen: expected a variable name in declaration, got {:?}",
                *e1.e
            ))
        }
    };

    if let ENode::Abstraction(binding, body) = &*e2.e {
        module.abstractions.insert(
            name.clone(),
            AbstractionInfo {
                param: binding.0.clone(),
                param_type: binding.1.t.clone(),
                body: (**body).clone(),
                abs_type: e2.typ.clone(),
            },
        );
        return Ok(());
    }

    // A function-valued application (e.g. `let sum = lfold add 0;`): keep the
    // expression and inline it at use sites, so partial applications become
    // full applications instead of first-class closure values.
    if let ENode::Application(..) = &*e2.e
        && !super::apply::is_scalar_type(&e2.typ)
    {
        module.inlineable.insert(name, e2.clone());
        return Ok(());
    }

    // An unannotated binding carries `infer`; its result type is taken from
    // the initializer's lowered value instead.
    let declared_type = match &typ.t {
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Infer) => None,
        _ => Some(lower_type(&typ.t, module)?),
    };

    // Build the body first: `func.func` is constructed with the final return
    // type, which for unannotated bindings only exists after lowering `e2`.
    let block = Block::new(&[]);
    let mut env = HashMap::new();
    let value = lower_expr(e2, &block, module, &mut env)?;
    block.append_operation(func::r#return(&[value], module.location(&e2.pos)));

    let result_type = match declared_type {
        Some(t) => t,
        None => value.r#type(),
    };
    let function_type = FunctionType::new(module.context, &[], &[result_type]);

    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, &name),
        TypeAttribute::new(function_type.into()),
        region,
        &[(
            Identifier::new(module.context, "llvm.emit_c_interface"),
            Attribute::unit(module.context),
        )],
        module.location(&e1.pos),
    );
    module.module.body().append_operation(function);
    module.symbols.insert(name.clone(), function_type);
    module.functions += 1;
    Ok(())
}

// ----------------------------------------------------------------------------
// Type declarations
// ----------------------------------------------------------------------------

/// Resolve record aliases inside a stored field type to their structural
/// `Rec(RowExt(..))` form, mirroring the type checker's `expand`.
///
/// The type checker registers a `record` declaration as a *type alias* whose
/// RHS is a `Rec` row, but a `TypeDecl`'s AST carries the raw field types,
/// where a record name is still spelled `Enum(name)`. If such a type reaches
/// `lower_type` unexpanded, its `Enum(_)` arm maps it to `!llvm.ptr` and the
/// record payload is never materialized as a struct, so field access on it
/// fails. Enum variant field types are the one raw type the codegen registry
/// actually consumes (via `enum_variant_fields`), so they are expanded here.
fn expand_stored_type(module: &Module, typ: &Monotype, seen: &mut Vec<String>) -> Monotype {
    match typ {
        Monotype::TypeVariable(_) => typ.clone(),
        Monotype::TypeFuncApplication(func, args) => match &**func {
            TypeFunc::Enum(name) if module.records.contains_key(name) => {
                if seen.iter().any(|n| n == name) {
                    return typ.clone();
                }
                seen.push(name.clone());
                let rec = &module.records[name];
                let mut map = HashMap::new();
                for (p, a) in rec.params.iter().zip(args.iter()) {
                    map.insert(p.clone(), a.clone());
                }
                let mut row = Monotype::empty_row();
                for (label, field) in rec.fields.iter().rev() {
                    let inst = field.instantiate(&mut map);
                    row = Monotype::row_ext(
                        label.clone(),
                        expand_stored_type(module, &inst, seen),
                        row,
                    );
                }
                seen.pop();
                Monotype::rec(row)
            }
            _ => {
                let expanded: Vec<Monotype> = args
                    .iter()
                    .map(|a| expand_stored_type(module, a, seen))
                    .collect();
                Monotype::TypeFuncApplication(func.clone(), expanded)
            }
        },
    }
}

/// Lower a type declaration by registering it in `module`.
///
/// A `TypeDecl` emits no MLIR operations; it only records the type so that
/// later uses of it in `lower_type` can be resolved:
///   - `enum E <tvars> = ...`  registers [`EnumLayout`] under the enum name.
///   - `type E <tvars> = T`    registers the alias' expanded right-hand side
///     under the alias name.
///
/// The type checker has already rejected duplicate/conflicting declarations,
/// so the name collisions checked here are defensive only.
pub fn lower_type_decl<'a>(
    header: &TypeHeader,
    dec: &TypeDec,
    module: &mut Module<'a>,
) -> Result<(), String> {
    match dec {
        TypeDec::Enum(variants) => {
            if module.enums.contains_key(&header.n) || module.aliases.contains_key(&header.n) {
                return Err(format!("codegen: type `{}` is already declared", header.n));
            }
            let layout = EnumLayout {
                params: header.tvars.clone(),
                variants: variants
                    .iter()
                    .map(|v| {
                        (
                            v.n.clone(),
                            v.tparams
                                .iter()
                                .map(|t| expand_stored_type(module, &t.t, &mut Vec::new()))
                                .collect(),
                        )
                    })
                    .collect(),
            };
            for (index, variant) in variants.iter().enumerate() {
                module.constructors.insert(
                    variant.n.clone(),
                    (header.n.clone(), index, variant.tparams.len()),
                );
            }
            module.enums.insert(header.n.clone(), layout);
        },
        TypeDec::Record(fields) => {
            if module.records.contains_key(&header.n)
                || module.aliases.contains_key(&header.n)
                || module.enums.contains_key(&header.n)
            {
                return Err(format!("codegen: type `{}` is already declared", header.n));
            }
            let layout = RecordLayout {
                params: header.tvars.clone(),
                fields: fields
                    .iter()
                    .map(|b| (b.0.clone(), b.1.t.clone()))
                    .collect(),
            };
            module.records.insert(header.n.clone(), layout);
        }
        TypeDec::Alias(rhs) => {
            if module.aliases.contains_key(&header.n) || module.enums.contains_key(&header.n) {
                return Err(format!("codegen: type `{}` is already declared", header.n));
            }
            module.aliases.insert(header.n.clone(), rhs.t.clone());
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Executable entry point
// ----------------------------------------------------------------------------

/// Add a C-callable `func.func @main() -> i32` that invokes `@__main` and
/// maps its result to the process exit code (0 unless the program ends with
/// an `int` expression, which is used directly).
pub(crate) fn emit_main_wrapper<'a>(module: &mut Module<'a>) -> Result<(), String> {
    let location = Location::unknown(module.context);
    let block = Block::new(&[]);
    let i32_type = IntegerType::new(module.context, 32).into();

    let ret_mono = module.entry_return_monotype().cloned();
    let is_int_exit = matches!(
        ret_mono,
        Some(ref m) if matches!(
            default_free_vars(m),
            Monotype::TypeFuncApplication(ref f, ref args)
                if args.is_empty() && matches!(**f, TypeFunc::Int)
        )
    );

    let ret_value: Value<'a, '_> = if is_int_exit {
        let call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "__main"),
            &[],
            &[i32_type],
            location,
        );
        block
            .append_operation(call)
            .result(0)
            .map_err(|e| e.to_string())?
            .into()
    } else {
        let outputs: Vec<Type<'a>> = match ret_mono {
            Some(ref m) => vec![lower_type(&default_free_vars(m), module)?],
            None => vec![],
        };
        let call = func::call(
            module.context,
            FlatSymbolRefAttribute::new(module.context, "__main"),
            &[],
            &outputs,
            location,
        );
        block.append_operation(call);
        let zero = arith::constant(
            module.context,
            IntegerAttribute::new(IntegerType::new(module.context, 32).into(), 0).into(),
            location,
        );
        block
            .append_operation(zero)
            .result(0)
            .map_err(|e| e.to_string())?
            .into()
    };

    block.append_operation(func::r#return(&[ret_value], location));

    let function_type = FunctionType::new(module.context, &[], &[i32_type]);
    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        module.context,
        StringAttribute::new(module.context, "main"),
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
    Ok(())
}
