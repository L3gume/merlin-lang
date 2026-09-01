//! Type mapping to MLIR types.

use crate::types::{Monotype, TypeFunc};
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::Type;

use super::Module;
use super::apply::default_free_vars;
use super::records::record_fields;

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

/// Map a [`Monotype`] to an `mlir::ir::Type`.
///
pub(crate) fn lower_type<'a>(typ: &Monotype, module: &Module<'a>) -> Result<Type<'a>, String> {
    match typ {
        Monotype::TypeFuncApplication(f, args) if args.is_empty() => match **f {
            TypeFunc::Int => Ok(IntegerType::new(module.context, 32).into()),
            TypeFunc::Float => Ok(Type::float32(module.context)),
            TypeFunc::Bool => Ok(IntegerType::new(module.context, 1).into()),
            TypeFunc::Str => Type::parse(module.context, "!llvm.ptr")
                .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string()),
            TypeFunc::Char => Ok(IntegerType::new(module.context, 32).into()),
            TypeFunc::Unit => Ok(IntegerType::new(module.context, 32).into()),
            TypeFunc::Infer => Err(
                "codegen: cannot lower an unresolved (inferred) type".to_string(),
            ),
            TypeFunc::Fn => Err("codegen: function type lowering not implemented".to_string()),
            TypeFunc::List => Err("codegen: list type lowering not implemented".to_string()),
            TypeFunc::Enum(_) => Type::parse(module.context, "!llvm.ptr")
                .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string()),
            TypeFunc::Rec | TypeFunc::RowExt(_) | TypeFunc::EmptyRow => {
                Err("codegen: cannot lower a bare row constructor".to_string())
            }
        },
        // A record type lowers to a struct of its fields in row order
        // (canonicalized to declaration order by the type checker).
        Monotype::TypeFuncApplication(f, args)
            if matches!(**f, TypeFunc::Rec) && args.len() == 1 =>
        {
            let fields = record_fields(typ)?;
            let mut field_types: Vec<String> = Vec::new();
            for (_, field) in &fields {
                field_types.push(lower_type(&default_free_vars(field), module)?.to_string());
            }
            let struct_str = format!("!llvm.struct<({})>", field_types.join(", "));
            Type::parse(module.context, &struct_str)
                .ok_or_else(|| format!("codegen: failed to create record struct type `{struct_str}`"))
        }
        // Curried function types lower to a single flat `FunctionType` taking
        // all parameters at once, matching the multi-argument specialization
        // shape emitted by codegen.
        Monotype::TypeFuncApplication(f, args) if matches!(**f, TypeFunc::Fn) => {
            let mut params: Vec<Type> = Vec::new();
            let mut cur = args;
            loop {
                if cur.len() < 2 {
                    return Err(format!(
                        "codegen: cannot lower a function type with {} arguments",
                        cur.len()
                    ));
                }
                params.push(lower_type(&cur[0], module)?);
                match &cur[1] {
                    Monotype::TypeFuncApplication(f2, args2)
                        if matches!(**f2, TypeFunc::Fn) && args2.len() == 2 =>
                    {
                        cur = args2;
                    }
                    rest => {
                        let ret = lower_type(rest, module)?;
                        return Ok(FunctionType::new(module.context, &params, &[ret]).into());
                    }
                }
            }
        }
        Monotype::TypeFuncApplication(f, args)
            if matches!(**f, TypeFunc::List) && args.len() == 1 =>
        {
            Type::parse(module.context, "!llvm.ptr")
                .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())
        }
        Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::Enum(_)) => {
            Type::parse(module.context, "!llvm.ptr")
                .ok_or_else(|| "codegen: failed to create `!llvm.ptr`".to_string())
        }
        Monotype::TypeFuncApplication(_, args) => Err(format!(
            "codegen: type lowering not implemented for application with {} argument(s)",
            args.len()
        )),
        Monotype::TypeVariable(v) => {
            Err(format!("codegen: cannot lower type variable `{}`", v))
        }
    }
}

// ----------------------------------------------------------------------------
// Size computation (for sizing heap allocations of lowered values)
// ----------------------------------------------------------------------------

/// Byte size of a [`Monotype`] under LLVM's default data layout.
///
/// Used to size the heap allocation that backs an enum payload: a payload
/// field that is a record is a struct value, which can be larger than the
/// `8` bytes `build_payload` used to assume per field.
pub(crate) fn monotype_size(typ: &Monotype) -> usize {
    match typ {
        Monotype::TypeVariable(_) => 8,
        Monotype::TypeFuncApplication(f, _) => match &**f {
            TypeFunc::Bool => 1,
            TypeFunc::Int | TypeFunc::Float | TypeFunc::Char | TypeFunc::Unit => 4,
            TypeFunc::Str | TypeFunc::List | TypeFunc::Enum(_) | TypeFunc::Fn => 8,
            TypeFunc::Rec => struct_size(&record_fields(typ).unwrap_or_default()),
            TypeFunc::RowExt(_) | TypeFunc::EmptyRow | TypeFunc::Infer => 8,
        },
    }
}

/// Alignment of a [`Monotype`] under LLVM's default data layout.
fn monotype_align(typ: &Monotype) -> usize {
    match typ {
        Monotype::TypeVariable(_) => 8,
        Monotype::TypeFuncApplication(f, _) => match &**f {
            TypeFunc::Bool => 1,
            TypeFunc::Int | TypeFunc::Float | TypeFunc::Char | TypeFunc::Unit => 4,
            TypeFunc::Rec => record_fields(typ)
                .unwrap_or_default()
                .iter()
                .map(|(_, t)| monotype_align(t))
                .max()
                .unwrap_or(1),
            _ => 8,
        },
    }
}

/// Size of a record struct laid out in field order (LLVM default layout:
/// each field aligned to its natural alignment, the struct rounded up to its
/// largest member's alignment).
fn struct_size(fields: &[(String, Monotype)]) -> usize {
    tuple_size(&fields.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>())
}

/// Byte size of a struct laid out from `types` in declaration order (LLVM
/// default layout: each field aligned to its natural alignment, the struct
/// rounded up to its largest member's alignment). Used to size an enum
/// variant payload holding multiple fields.
pub(crate) fn tuple_size(types: &[Monotype]) -> usize {
    let mut offset = 0usize;
    let mut max_align = 1usize;
    for t in types {
        let align = monotype_align(t);
        offset = offset.div_ceil(align) * align;
        offset += monotype_size(t);
        max_align = max_align.max(align);
    }
    offset.div_ceil(max_align) * max_align
}
