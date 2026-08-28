//! Merlin compiler library.

pub mod ast;
#[cfg(feature = "codegen")]
pub mod codegen;
pub mod display;
pub mod prelude;
#[cfg(feature = "codegen")]
pub mod repl;
pub mod types;

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub grammar);
