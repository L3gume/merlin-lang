//! JIT execution (for the REPL).

use crate::types::{Monotype, TypeFunc};
use super::Module;
use super::apply::default_free_vars;
use melior::{ExecutionEngine, pass};

#[derive(PartialEq)]
pub enum ExecutionResult {
    Int(i32),
    Float(f32),
    Bool(bool),
    String(String),
    Char(char),
    List(Vec<ExecutionResult>),
    Unit,
}

impl std::fmt::Debug for ExecutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionResult::Int(n) => write!(f, "{}", n),
            ExecutionResult::Float(n) => write!(f, "{}", n),
            ExecutionResult::Bool(b) => write!(f, "{}", b),
            ExecutionResult::String(s) => write!(f, "\"{}\"", s),
            ExecutionResult::Char(c) => write!(f, "\'{}\'", c),
            ExecutionResult::List(items) => {
                let rendered: Vec<String> = items.iter().map(|i| format!("{:?}", i)).collect();
                write!(f, "[{}]", rendered.join(", "))
            }
            ExecutionResult::Unit => write!(f, "()"),
        }
    }
}

/// Run the compiled module through the LLVM JIT and read back the result.
///
/// `target` selects which symbol to invoke:
///   - `Some((name, type))` invokes the nullary binding `@name` (used to show
///     the value of a `let` declaration, e.g. a list).
///   - `None` invokes `@__main` with the module's recorded entry type.
/// Lower the module to the LLVM dialect (the common step before JIT or
/// object emission).
fn run_passes(module: &mut Module) -> Result<(), String> {
    let context = module.context;
    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(module.as_mlir_module_mut())
        .map_err(|e| format!("codegen: pass manager failed: {}", e))
}

/// Compile the module to a native object file. Adds a `@main` wrapper (so the
/// object can be linked into an executable), lowers to LLVM, and dumps the
/// object to `obj_path`.
pub fn compile(module: &mut Module, obj_path: &str) -> Result<(), String> {
    super::stmt::emit_main_wrapper(module)?;
    run_passes(module)?;
    let engine = ExecutionEngine::new(module.as_mlir_module_mut(), 2, &[], true, true);
    engine.dump_to_object_file(obj_path);
    Ok(())
}

/// Run the compiled module through the LLVM JIT and read back the result.
///
/// `target` selects which symbol to invoke:
///   - `Some((name, type))` invokes the nullary binding `@name` (used to show
///     the value of a `let` declaration, e.g. a list).
///   - `None` invokes `@__main` with the module's recorded entry type.
pub fn execute(
    module: &mut Module,
    target: Option<(String, Monotype)>,
) -> Result<ExecutionResult, String> {
    run_passes(module)?;
    let engine = ExecutionEngine::new(module.as_mlir_module_mut(), 2, &[], false, false);

    let (symbol, return_type) = match target {
        Some((name, mono)) => (name, Some(mono)),
        None => ("__main".to_string(), module.entry_return_monotype().cloned()),
    };

    invoke(&engine, &symbol, return_type.as_ref())
}

fn invoke(
    engine: &ExecutionEngine,
    symbol: &str,
    mono: Option<&Monotype>,
) -> Result<ExecutionResult, String> {
    let Some(mono) = mono else {
        // `__main` returns nothing (trailing `print`/`let`): truly void.
        unsafe {
            engine
                .invoke_packed(symbol, &mut [])
                .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
        }
        return Ok(ExecutionResult::Unit);
    };
    match default_free_vars(mono) {
        Monotype::TypeFuncApplication(ref f, ref args) if args.is_empty() => match **f {
            TypeFunc::Int => {
                let mut result: i32 = 0;
                unsafe {
                    engine
                        .invoke_packed(symbol, &mut [&mut result as *mut i32 as *mut ()])
                        .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                }
                Ok(ExecutionResult::Int(result))
            }
            TypeFunc::Float => {
                let mut result: f32 = 0.0;
                unsafe {
                    engine
                        .invoke_packed(symbol, &mut [&mut result as *mut f32 as *mut ()])
                        .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                }
                Ok(ExecutionResult::Float(result))
            }
            TypeFunc::Bool => {
                let mut result: u8 = 0;
                unsafe {
                    engine
                        .invoke_packed(symbol, &mut [&mut result as *mut u8 as *mut ()])
                        .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                }
                Ok(ExecutionResult::Bool(result != 0))
            }
            TypeFunc::Str => {
                let mut result: *const std::ffi::c_char = std::ptr::null();
                unsafe {
                    engine
                        .invoke_packed(
                            symbol,
                            &mut [&mut result as *mut *const std::ffi::c_char as *mut ()],
                        )
                        .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                    let s = if result.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(result)
                            .to_string_lossy()
                            .into_owned()
                    };
                    Ok(ExecutionResult::String(s))
                }
            }
            TypeFunc::Char => {
                let mut result: i32 = 0;
                unsafe {
                    engine
                        .invoke_packed(symbol, &mut [&mut result as *mut i32 as *mut ()])
                        .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                }
                Ok(ExecutionResult::Char(char::from_u32(result as u32).unwrap_or('\u{FFFD}')))
            }
            TypeFunc::Unit => {
                // Unit expressions are materialized as `i32` in MLIR, so the
                // invoked function does return a value and needs a result slot.
                let mut result: i32 = 0;
                unsafe {
                    engine
                        .invoke_packed(symbol, &mut [&mut result as *mut i32 as *mut ()])
                        .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                }
                Ok(ExecutionResult::Unit)
            }
            _ => Err(format!("codegen: cannot JIT-execute type {:?}", mono)),
        },
        Monotype::TypeFuncApplication(ref f, ref args)
            if matches!(**f, TypeFunc::List) && args.len() == 1 =>
        {
            let mut result: *const u8 = std::ptr::null();
            unsafe {
                engine
                    .invoke_packed(symbol, &mut [&mut result as *mut *const u8 as *mut ()])
                    .map_err(|e| format!("codegen: jit invocation failed: {}", e))?;
                let items = read_list(result, &args[0]);
                Ok(ExecutionResult::List(items))
            }
        }
        _ => Err(format!("codegen: cannot JIT-execute type {:?}", mono)),
    }
}

/// Walk a heap-allocated cons-cell list (`{ head: T, tail: !llvm.ptr }`,
/// `null` for `[]`) reading each element's value.
unsafe fn read_list(ptr: *const u8, elem: &Monotype) -> Vec<ExecutionResult> {
    let mut items = Vec::new();
    let mut cur = ptr;
    while !cur.is_null() {
        items.push(unsafe { read_value(cur, elem) });
        // Tail pointer is the second field of the cons-cell struct; with the
        // element types supported here it always sits at offset 8.
        cur = unsafe { *(cur.add(8) as *const *const u8) };
    }
    items
}

/// Read a single value of type `mono` from the memory at `ptr`.
unsafe fn read_value(ptr: *const u8, mono: &Monotype) -> ExecutionResult {
    unsafe {
        match default_free_vars(mono) {
            Monotype::TypeFuncApplication(ref f, ref args) if args.is_empty() => match **f {
                TypeFunc::Int => ExecutionResult::Int(*(ptr as *const i32)),
                TypeFunc::Float => ExecutionResult::Float(*(ptr as *const f32)),
                TypeFunc::Bool => ExecutionResult::Bool(*(ptr as *const u8) != 0),
                TypeFunc::Str => {
                    let s = *(ptr as *const *const std::ffi::c_char);
                    if s.is_null() {
                        ExecutionResult::String(String::new())
                    } else {
                        ExecutionResult::String(
                            std::ffi::CStr::from_ptr(s)
                                .to_string_lossy()
                                .into_owned(),
                        )
                    }
                }
                TypeFunc::Unit => ExecutionResult::Unit,
                TypeFunc::Char => {
                    let c = *(ptr as *const i32);
                    ExecutionResult::Char(char::from_u32(c as u32).unwrap_or('\u{FFFD}'))
                }
                _ => ExecutionResult::Unit,
            },
            Monotype::TypeFuncApplication(ref f, ref args)
                if matches!(**f, TypeFunc::List) && args.len() == 1 =>
            {
                let sub = *(ptr as *const *const u8);
                ExecutionResult::List(read_list(sub, &args[0]))
            }
            _ => ExecutionResult::Unit,
        }
    }
}
