//! Rendering of types for human-readable output.

use crate::types::{Monotype, TypeFunc};
use std::collections::HashMap;

/// Render a [`Monotype`] to a human-readable string.
///
/// Type variables (internal names like `t0`, `t1`) are renamed to readable
/// fresh names (`'a`, `'b`, ...). The mapping is created fresh on every call
/// and threaded through the recursion so the same type variable always renders
/// with the same name within one type.
pub fn render_type(t: &Monotype) -> String {
    let mut vars = HashMap::new();
    render_type_inner(t, &mut vars)
}

fn render_type_inner(t: &Monotype, vars: &mut HashMap<String, String>) -> String {
    match t {
        Monotype::TypeVariable(v) => match vars.get(v) {
            Some(name) => name.clone(),
            None => {
                let name = fresh_var_name(vars.len());
                vars.insert(v.clone(), name.clone());
                name
            }
        },
        Monotype::TypeFuncApplication(f, args) => match **f {
            TypeFunc::Infer => "_".to_string(),
            TypeFunc::Unit => "()".to_string(),
            TypeFunc::Int => "int".to_string(),
            TypeFunc::Float => "float".to_string(),
            TypeFunc::Bool => "bool".to_string(),
            TypeFunc::Str => "str".to_string(),
            TypeFunc::Char => "char".to_string(),
            TypeFunc::Fn => args
                .iter()
                .map(|a| render_type_inner(a, vars))
                .collect::<Vec<_>>()
                .join(" -> "),
            TypeFunc::List => match args.first() {
                Some(elem) => format!("[{}]", render_type_inner(elem, vars)),
                None => "list".to_string(),
            },
            TypeFunc::Enum(ref name) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let rendered: Vec<String> = args.iter().map(|a| render_type_inner(a, vars)).collect();
                    format!("{}({})", name, rendered.join(", "))
                }
            }
            TypeFunc::Rec => todo!(),
            TypeFunc::RowExt(_) => todo!(),
            TypeFunc::EmptyRow => todo!(),
            TypeFunc::Tuple => todo!(),
        },
    }
}

/// The `n`-th readable type-variable name: `'a`, `'b`, ..., `'z`, `'aa`, ...
fn fresh_var_name(index: usize) -> String {
    let mut suffix = String::new();
    let mut n = index;
    loop {
        suffix.insert(0, (b'a' + (n % 26) as u8) as char);
        n /= 26;
        if n == 0 {
            break;
        }
        n -= 1;
    }
    format!("'{suffix}")
}
