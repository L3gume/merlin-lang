//! MLIR code generation.
//!
//! Lowers a type-checked [`Program`] into an MLIR module that can be fed to
//! the LLVM backend and JIT-compiled for the REPL.
//!
//! Pipeline: parse -> typecheck -> [this module] -> LLVM backend -> JIT.
//!
//! Layout:
//! - [`Module`], the type registries, and the shared environment live here.
//! - [`stmt`] lowers top-level statements and declarations.
//! - [`expr`] lowers expressions.
//! - [`closures`], [`lists`], [`enums`], [`types`] provide the pieces those
//!   use (closure conversion, cons cells, tagged enum values, type mapping).
//! - [`tail`] lowers self tail calls to loop backedges (tail call
//!   optimization).
//! - [`execute`] runs the compiled module through the JIT.

mod apply;
mod closures;
mod enums;
mod equality;
mod execute;
mod expr;
mod lists;
mod records;
mod stmt;
mod tail;
mod types;

pub use execute::{ExecutionResult, compile, execute};
pub use stmt::lower;

use crate::ast::*;
use crate::types::Monotype;
use melior::ir::{
    operation::OperationLike,
    r#type::FunctionType,
    Location, Value,
};
use std::collections::{HashMap, HashSet};

/// A binding in the current expression scope.
#[derive(Clone)]
pub enum EnvEntry<'c, 'a> {
    /// A lowered SSA value (e.g. `let x = 42 in ...`).
    Value(Value<'c, 'a>),
    /// A lambda registered in [`Module::abstractions`]; specialized on demand
    /// at each use (e.g. `let x = \y => y in ...`).
    Abstraction(String),
}

pub(crate) type Env<'c, 'a> = HashMap<String, EnvEntry<'c, 'a>>;

// ----------------------------------------------------------------------------
// Context
// ----------------------------------------------------------------------------

/// Create an MLIR context with all dialects registered.
///
/// `arith.constant` (and every other op we emit) fails to verify unless its
/// dialect is loaded into the context. The REPL owns one such context for the
/// whole session and reuses it across input lines.
pub fn new_context() -> melior::Context {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    context
}

// ----------------------------------------------------------------------------
// Module (top-level MLIR container)
// ----------------------------------------------------------------------------

/// Layout of a declared enum: its variants and their payload field types.
///
/// A parametric enum's field types may reference the header's type variables
/// (e.g. `enum option a = None | Some(a)`); those are resolved when the enum
/// is *applied* in `lower_type`.
pub struct EnumLayout {
    /// The header's type parameter names (e.g. `a` in `option a`).
    pub params: Vec<String>,
    /// `(variant name, payload field types)` in declaration order.
    pub variants: Vec<(String, Vec<Monotype>)>,
}

/// Layout of a declared record: its fields in declaration order.
///
/// Field order is the source of truth for the LLVM struct layout; the row in
/// a record's type is unordered up to commutation, so codegen consults this
/// registry (keyed by record name) rather than trusting the row order.
pub struct RecordLayout {
    /// The header's type parameter names (e.g. `a` in `Poly a`).
    pub params: Vec<String>,
    /// `(field name, field type)` in declaration order.
    pub fields: Vec<(String, Monotype)>,
}

/// A polymorphic (or monomorphic) lambda binding `name = \p => body`, kept so
/// a specialized `func.func` can be emitted on demand for every concrete type
/// the binding is used at.
#[derive(Clone)]
pub struct AbstractionInfo {
    /// The bound parameter name.
    pub param: String,
    /// The declared parameter type (may be `infer`).
    #[allow(dead_code)] // informational
    pub param_type: Monotype,
    /// The body expression (owned clone).
    pub body: Expr,
    /// The abstraction's resolved type, used to compute the substitution
    /// that specializes the body for a concrete instantiation.
    pub abs_type: Monotype,
}

/// An MLIR module under construction.
///
/// TODO(melior): hold `melior::ir::Module`, created from a `melior::Context`
/// that owns dialect registration. Keep the `Context` alive for the whole
/// REPL session so bindings can be appended across input lines.
pub struct Module<'a> {
    context: &'a melior::Context,
    module:  melior::ir::Module<'a>,
    functions: usize,
    /// Declared enum layouts, keyed by type name.
    enums: HashMap<String, EnumLayout>,
    /// Declared record layouts, keyed by type name.
    records: HashMap<String, RecordLayout>,
    /// Declared type aliases, keyed by alias name; the value is the expanded
    /// right-hand side, which may reference the header's type variables.
    aliases: HashMap<String, Monotype>,
    /// Number of string globals emitted, for unique symbol names.
    strings: usize,
    /// Whether the external `malloc` declaration has been emitted.
    malloc_declared: bool,
    /// Names of the external libc `func.func` declarations already emitted, so
    /// each is declared at most once per module.
    externs: HashSet<String>,
    /// Types of the top-level `func.func` symbols, keyed by name; a
    /// `Variable` that is not a bound parameter lowers to `func.call` on it.
    symbols: HashMap<String, FunctionType<'a>>,
    /// Number of closures emitted, for unique symbol names.
    closures: usize,
    /// Lambda bindings awaiting per-type specialization, keyed by name.
    abstractions: HashMap<String, AbstractionInfo>,
    /// Cache of emitted specializations: `(binding name, canonical
    /// instantiation type, capture types) -> closure symbol`.
    specializations: HashMap<(String, String, String), String>,
    /// Cache of emitted equality helpers: canonical (defaulted) type ->
    /// function symbol.
    eq_functions: HashMap<String, String>,
    /// Number of equality helpers emitted, for unique symbol names.
    eq_counter: usize,
    /// Function-valued `let` bindings whose right-hand side is an application
    /// (e.g. `let sum = lfold add 0;`), kept as expressions and inlined at
    /// use sites rather than evaluated to a closure value eagerly.
    inlineable: HashMap<String, Expr>,
    /// Number of specialization symbols emitted.
    spec_counter: usize,
    /// Number of let-bound abstractions registered, for unique registry names.
    let_counter: usize,
    /// Enum constructors: constructor name → `(enum name, variant index,
    /// arity)`. Built when an enum is declared.
    constructors: HashMap<String, (String, usize, usize)>,
    /// The resolved Merlin type of the `@__main` entry function's return
    /// value, used by the JIT to interpret the result slot.
    entry_return_monotype: Option<Monotype>,
    /// Name of the source being compiled (file path or `"<repl>"`), attached
    /// to generated MLIR locations.
    source_name: String,
}

impl<'a> Module<'a> {
    /// Create an empty module inside `context`.
    pub fn new(context: &'a melior::Context) -> Module<'a> {
        Module {
            context,
            module: melior::ir::Module::new(melior::ir::Location::unknown(context)),
            functions: 0,
            enums: HashMap::new(),
            records: HashMap::new(),
            aliases: HashMap::new(),
            strings: 0,
            malloc_declared: false,
            externs: HashSet::new(),
            symbols: HashMap::new(),
            closures: 0,
            abstractions: HashMap::new(),
            specializations: HashMap::new(),
            eq_functions: HashMap::new(),
            eq_counter: 0,
            inlineable: HashMap::new(),
            spec_counter: 0,
            let_counter: 0,
            constructors: HashMap::new(),
            entry_return_monotype: None,
            source_name: String::new(),
        }
    }

    /// Number of top-level `func.func` operations emitted so far.
    pub fn function_count(&self) -> usize {
        self.functions
    }

    /// Set the source name used for generated MLIR locations.
    pub fn set_source_name(&mut self, name: String) {
        self.source_name = name;
    }

    /// The MLIR `Location` for an AST span, or `unknown` when the position is
    /// nil or the module has no source name.
    pub fn location(&self, pos: &crate::ast::Pos) -> Location<'a> {
        if pos.is_nil() || self.source_name.is_empty() {
            return Location::unknown(self.context);
        }
        Location::file_line_col_range(
            self.context,
            &self.source_name,
            pos.start_line as usize,
            pos.start_col as usize,
            pos.end_line as usize,
            pos.end_col as usize,
        )
    }

    /// Print the module in MLIR textual form, including source locations.
    pub fn dump(&self) -> String {
        self.module
            .as_operation()
            .to_string_with_flags(
                melior::ir::operation::OperationPrintingFlags::new().enable_debug_info(true, true),
            )
            .unwrap_or_else(|_| self.module.as_operation().to_string())
    }

    /// Mutable access to the inner MLIR module (for running passes).
    pub fn as_mlir_module_mut(&mut self) -> &mut melior::ir::Module<'a> {
        &mut self.module
    }

    /// The resolved Merlin type of the entry function's return value, if
    /// any. `None` means the entry function returns unit.
    pub fn entry_return_monotype(&self) -> Option<&Monotype> {
        self.entry_return_monotype.as_ref()
    }
}