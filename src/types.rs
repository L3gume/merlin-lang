use std::fmt::Display;
use std::sync::OnceLock;
use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::types::Monotype::TypeFuncApplication;


#[derive(Debug, Clone, PartialEq)]
pub enum TypeFunc {
    Infer,
    Unit,
    Int,
    Float,
    Bool,
    Str,
    Char,
    Fn, // ->
    List,
    Enum(String),
    Rec,
    RowExt(String),
    EmptyRow,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Monotype {
    TypeVariable(String),
    TypeFuncApplication(Box<TypeFunc>, Vec<Monotype>),
}

impl Display for Monotype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeVariable(var) => write!(f, "TypeVar({})", var),
            Self::TypeFuncApplication(type_func, monotypes) => {
                match &**type_func {
                    TypeFunc::Infer => write!(f, "<infer>"),
                    TypeFunc::Unit => write!(f, "()"),
                    TypeFunc::Int => write!(f, "Int"),
                    TypeFunc::Float => write!(f, "Float"),
                    TypeFunc::Bool => write!(f, "Bool"),
                    TypeFunc::Str => write!(f, "Str"),
                    TypeFunc::Char => write!(f, "Char"),
                    TypeFunc::Fn => write!(f, "{} -> {}", monotypes[0], monotypes[1]),
                    TypeFunc::List => write!(f, "list {}", monotypes[0]),
                    TypeFunc::Enum(n) => {
                        if monotypes.is_empty() {
                            write!(f, "{}", n)
                        } else {
                            write!(f, "{}({})", n,
                                monotypes.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", "))
                        }
                    }
                    TypeFunc::Rec => {
                        write!(f, "{{")?;
                        fmt_row(f, &monotypes[0])?;
                        write!(f, "}}")
                    },
                    TypeFunc::RowExt(n) => write!(f, "({}: {}; {})", n, monotypes[0], monotypes[1]),
                    TypeFunc::EmptyRow => write!(f, "∅"),
                }
            }
        }
    }
}

fn fmt_row(f: &mut std::fmt::Formatter<'_>, row: &Monotype) -> std::fmt::Result {
    match row {
        Monotype::TypeVariable(v) => write!(f, "{}", v),
        Monotype::TypeFuncApplication(func, args) => match &**func {
            TypeFunc::EmptyRow => Ok(()),
            TypeFunc::RowExt(label) => {
                write!(f, "{}: {}", label, args[0])?;
                match &args[1] {
                    Monotype::TypeFuncApplication(f2, _) if **f2 == TypeFunc::EmptyRow => Ok(()),
                    Monotype::TypeVariable(v) => write!(f, " | {}", v),
                    rest => {
                        write!(f, ", ")?;
                        fmt_row(f, rest)
                    }
                }
            }
            _ => write!(f, "{}", row),
        },
    }
}

impl Default for Monotype {
    fn default() -> Self {
        Self::TypeVariable(String::new())
    }
}

impl Monotype {
    pub fn apply(&self, sub : &Substitution) -> Monotype {
        match self.clone() {
            Self::TypeVariable(name) =>
                match sub.variables.get(&name) {
                    Some(monotype) => monotype.apply(sub),
                    _ => Self::TypeVariable(name),
                },
            Self::TypeFuncApplication(typ_fn, types) =>
                Self::TypeFuncApplication(typ_fn, types.iter().map(|typ| typ.apply(sub)).collect())
        }
    }

    pub fn instantiate(&self, mappings : &mut HashMap<String, Monotype>) -> Monotype {
        match self {
            Self::TypeVariable(var) => match mappings.get(var) {
                Some(monotype) => monotype.clone(),
                _ => self.clone()
            },
            Self::TypeFuncApplication(func, types) =>
                Self::TypeFuncApplication(func.clone(), types.iter().map(|typ| typ.instantiate(mappings)).collect())
        }
    }

    pub fn free_variables(&self) -> Vec<String> {
        match self {
            Self::TypeVariable(v) => vec![v.clone()],
            Self::TypeFuncApplication(_, ts) => ts.iter().flat_map(|t| t.free_variables()).collect()
        }
    }

    pub fn contains(&self, typ : &Monotype) -> bool {
        match typ {
            Self::TypeVariable(v) => match self {
                Self::TypeVariable(v2) => v == v2,
                Self::TypeFuncApplication(_, ts) => ts.iter().any(|t| t.contains(typ))
            },
            Self::TypeFuncApplication(_, _) => false
        }
    }

    pub fn var(name : String) -> Monotype {
        Self::TypeVariable(name)
    }

    pub fn infer() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Infer), vec![])
    }

    pub fn bool() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Bool), vec![])
    }

    pub fn int() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Int), vec![])
    }

    pub fn float() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Float), vec![])
    }

    pub fn string() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Str), vec![])
    }

    pub fn char() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Char), vec![])
    }

    pub fn unit() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Unit), vec![])
    }

    pub fn func(vars : Vec<Monotype>) -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Fn), vars)
    }

    pub fn list(var : Monotype) -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::List), vec![var])
    }

    pub fn enum_type(name : String) -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Enum(name)), vec![])
    }

    pub fn enum_app(name : String, vars : Vec<Monotype>) -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Enum(name)), vars)
    }

    pub fn rec(row: Monotype) -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::Rec), vec![row])
    }

    pub fn row_ext(label: String, field: Monotype, rest: Monotype) -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::RowExt(label)), vec![field, rest])
    }

    pub fn empty_row() -> Monotype {
        Monotype::TypeFuncApplication(Box::new(TypeFunc::EmptyRow), vec![])
    }

}

#[derive(Debug, Clone, PartialEq)]
pub enum Polytype {
    Mono(Box<Monotype>),
    TypeQuantifier(String, Box<Polytype>),
}

impl Polytype {
    pub fn apply(&self, sub : &Substitution) -> Polytype {
        match self.clone() {
            Self::Mono(mono) => Self::Mono(Box::new(mono.apply(sub))),
            Self::TypeQuantifier(s, poly) => Self::TypeQuantifier(s, Box::new(poly.apply(sub))),
        }
    }

    pub fn instantiate(&self, ctx : &mut TypeContext, mappings : Option<HashMap<String, Monotype>>) -> Monotype {
        let mut maps = mappings.unwrap_or_default();
        match self {
            Self::Mono(mon) => mon.instantiate(&mut maps),
            Self::TypeQuantifier(quant, typ) => {
                maps.insert(quant.clone(), Monotype::TypeVariable(ctx.new_typevar()));
                typ.instantiate(ctx, Some(maps))
            }
        }
    }

    pub fn free_variables(&self) -> Vec<String> {
        match self {
            Self::Mono(mon) => mon.free_variables(),
            Self::TypeQuantifier(quant, typ) =>
                typ.free_variables().into_iter().filter(|n| n != quant).collect()
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Substitution {
    pub variables : HashMap<String, Monotype>
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

impl Substitution {
    pub fn new() -> Substitution {
        Substitution { variables: HashMap::new() }
    }

    pub fn make(map: HashMap<String, Monotype>) -> Substitution {
        Substitution { variables : map }
    }

    pub fn combine(&self, s2 : Substitution) -> Substitution {
        let mut applied: HashMap<String, Monotype> = s2.variables.iter()
            .map(|(k, mon)| (k.clone(), mon.apply(self)))
            .collect();
        for (k, mon) in &self.variables {
            if !s2.variables.contains_key(k) {
                applied.insert(k.clone(), mon.clone());
            }
        }
        Substitution::make(applied)
    }
}

pub(crate) fn builtins() -> &'static [(String, Polytype)] {
    static BUILTINS: OnceLock<Vec<(String, Polytype)>> = OnceLock::new();
    BUILTINS.get_or_init(|| {
        let mono = |m: Monotype| Polytype::Mono(Box::new(m));
        vec![
            ("print".to_string(),       mono(Monotype::func(vec![Monotype::string(), Monotype::unit()]))),
            ("println".to_string(),     mono(Monotype::func(vec![Monotype::string(), Monotype::unit()]))),
            ("itostr".to_string(),      mono(Monotype::func(vec![Monotype::int(), Monotype::string()]))),
            ("ftostr".to_string(),      mono(Monotype::func(vec![Monotype::float(), Monotype::string()]))),
            ("btostr".to_string(),      mono(Monotype::func(vec![Monotype::bool(), Monotype::string()]))),
            ("strtoi".to_string(),      mono(Monotype::func(vec![Monotype::string(), Monotype::int()]))),
            ("strtof".to_string(),      mono(Monotype::func(vec![Monotype::string(), Monotype::float()]))),
            ("strtob".to_string(),      mono(Monotype::func(vec![Monotype::string(), Monotype::bool()]))),
            ("itof".to_string(),        mono(Monotype::func(vec![Monotype::int(), Monotype::float()]))),
            ("ftoi".to_string(),        mono(Monotype::func(vec![Monotype::float(), Monotype::int()]))),
            ("readin".to_string(),      mono(Monotype::func(vec![Monotype::unit(), Monotype::string()]))),
        ]
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeAlias {
    pub params : Vec<String>,
    pub rhs : Monotype,
}


#[derive(Debug, Clone, PartialEq)]
pub struct TypeContext {
    type_var_ctr : u32,
    pub variables : HashMap<String, Polytype>,
    type_aliases : HashMap<String, TypeAlias>,
    enum_names : HashSet<String>,
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeContext {
    pub fn new() -> TypeContext {
        let mut ctx = TypeContext {
            type_var_ctr : 0,
            variables : HashMap::new(),
            type_aliases : HashMap::new(),
            enum_names : HashSet::new(),
        };
        for (name, poly) in builtins() {
            ctx.variables.insert(name.clone(), poly.clone());
        }
        ctx
    }

    pub fn make(map: HashMap<String, Polytype>) -> TypeContext {
        TypeContext { 
            type_var_ctr : 0,
            variables : map,
            type_aliases : HashMap::new(),
            enum_names : HashSet::new(),
        }
    }

    pub fn get(&self, name : &String) -> Option<Polytype> {
        self.variables.get(name).cloned()
    }

    pub fn add(&mut self, name : String, typ : Polytype) {
        self.variables.insert(name, typ);
    }

    pub fn remove(&mut self, name : &str) {
        self.variables.remove(name);
    }

    pub fn add_alias(&mut self, name : String, alias : TypeAlias) {
        self.type_aliases.insert(name, alias);
    }

    pub fn get_alias(&self, name : &str) -> Option<&TypeAlias> {
        self.type_aliases.get(name)
    }

    pub fn add_enum_name(&mut self, name : String) {
        self.enum_names.insert(name);
    }

    pub fn has_enum_name(&self, name : &str) -> bool {
        self.enum_names.contains(name)
    }

    pub fn apply(&self, sub : &Substitution) -> TypeContext {
        TypeContext {
            type_var_ctr: self.type_var_ctr,
            variables: self.variables.iter().map(|(k, t)| (k.clone(), t.apply(sub))).collect(),
            type_aliases: self.type_aliases.clone(),
            enum_names: self.enum_names.clone(),
        }
    }

    pub fn new_typevar(&mut self) -> String {
        let ret = format!("t{}", self.type_var_ctr);
        self.type_var_ctr += 1;
        ret
    }

    pub fn free_variables(&self) -> Vec<String> {
        self.variables.values().flat_map(|t| t.free_variables()).collect()
    }

    pub fn is_builtin(name: &str) -> bool {
        builtins().iter().any(|(n, _)| n == name)
    }

    // Generalise free variables (that aren't free in the context) by renaming
    // each to a fresh, unique name. The fresh names come from the monotonic
    // type-variable counter, so a stored polytype's bound variables can never
    // collide with type variables allocated by later statements — otherwise
    // `context.apply` would silently corrupt already-generalised types.
    pub fn generalise(&mut self, typ : &Monotype) -> Polytype {
        let mut quants = diff(typ.free_variables(), self.free_variables());
        quants.sort_unstable();
        quants.dedup();
        let mut sub = Substitution::new();
        let mut poly = Polytype::Mono(Box::new(typ.clone()));
        for q in quants {
            let fresh_name = self.new_typevar();
            sub.variables.insert(q.clone(), Monotype::var(fresh_name.clone()));
            poly = Polytype::TypeQuantifier(fresh_name, Box::new(poly));
        }
        poly.apply(&sub)
    }
}

pub fn unify(context: &mut TypeContext, typ1 : &Monotype, typ2 : &Monotype) -> Result<Substitution, UnificationError> {
    match (typ1, typ2) {
        (Monotype::TypeVariable(v1), Monotype::TypeVariable(v2)) => {
            if v1 == v2 {
                Ok(Substitution::new())
            } else {
                if typ2.contains(typ1) {
                    Err(UnificationError { pos: None, message: "Infinite recursive type".to_string() })
                } else {
                    Ok(Substitution::make(HashMap::from([(v1.clone(), typ2.clone())])))
                }
            }
        },
        (Monotype::TypeVariable(v1), _) => { 
            if typ2.contains(typ1) {
                Err(UnificationError { pos: None, message: "Infinite recursive type".to_string() })
            } else {
                Ok(Substitution::make(HashMap::from([(v1.clone(), typ2.clone())])))
            }
        }
        (Monotype::TypeFuncApplication(_, _), Monotype::TypeVariable(_)) => unify(context, typ2, typ1),
        // Row constructors (`Rec`, `RowExt`, `EmptyRow`) need dedicated
        // unification (label commutation); the generic pointwise case below
        // would wrongly reject or mis-unify them.
        (Monotype::TypeFuncApplication(f1, _), Monotype::TypeFuncApplication(f2, _))
            if is_row_ctor(&**f1) || is_row_ctor(&**f2) =>
            unify_row(context, typ1, typ2),
        (Monotype::TypeFuncApplication(f1, ts1),
            Monotype::TypeFuncApplication(f2, ts2 )) => {
            if f1 != f2 {
                Err(UnificationError { pos: None, message: format!("Type function application mismatch: {:?} != {:?} (full: {:?} vs {:?})", f1, f2, typ1, typ2) })
            } else {
                if ts1.len() != ts2.len() {
                    Err(UnificationError { pos: None, message: format!("Type functions have different number of args: {:?}, {:?}", ts1, ts2) })
                } else {
                    let mut sub = Substitution::new();
                    for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                        sub = sub.combine(unify(context,&t1.apply(&sub), &t2.apply(&sub))?);
                    }
                    Ok(sub)
                }
            }
        }
    }
}


fn is_row_ctor(f: &TypeFunc) -> bool {
    matches!(f, TypeFunc::Rec | TypeFunc::RowExt(_) | TypeFunc::EmptyRow)
}

// A `Rec` constructor wraps a single row; unwrap it so `unify_row` only ever
// sees rows (type variables, `EmptyRow`, or `RowExt` chains).
fn peel_rec(m: &Monotype) -> &Monotype {
    match m {
        Monotype::TypeFuncApplication(f, ts) if **f == TypeFunc::Rec => &ts[0],
        _ => m,
    }
}

fn unify_row(context : &mut TypeContext, r1 : &Monotype, r2 : &Monotype) -> Result<Substitution, UnificationError> {
    let row1 = peel_rec(r1);
    let row2 = peel_rec(r2);
    match (row1, row2) {
        (Monotype::TypeVariable(v), _) => {
            if row2.contains(row1) {
                Err(UnificationError { pos: None, message: "Infinite recursive type".to_string() })
            } else {
                Ok(Substitution::make(HashMap::from([(v.clone(), row2.clone())])))
            }
        },
        (_, Monotype::TypeVariable(v)) => {
            if row1.contains(row2) {
                Err(UnificationError { pos: None, message: "Infinite recursive type".to_string() })
            } else {
                Ok(Substitution::make(HashMap::from([(v.clone(), row1.clone())])))
            }
        },
        (Monotype::TypeFuncApplication(f1, ts1), Monotype::TypeFuncApplication(f2, ts2)) => {
            match (&**f1, &**f2) {
                (TypeFunc::EmptyRow, TypeFunc::EmptyRow) => Ok(Substitution::new()),
                (TypeFunc::EmptyRow, TypeFunc::RowExt(_)) | (TypeFunc::RowExt(_), TypeFunc::EmptyRow) =>
                    Err(UnificationError { pos: None, message: "Row width mismatch: cannot unify a closed row with a field row".to_string() }),
                (TypeFunc::RowExt(l1), TypeFunc::RowExt(l2)) => {
                    let (t1, rest1) = (&ts1[0], &ts1[1]);
                    let (t2, rest2) = (&ts2[0], &ts2[1]);
                    if l1 == l2 {
                        let s = unify(context, t1, t2)?;
                        let s2 = unify_row(context, &rest1.apply(&s), &rest2.apply(&s))?;
                        Ok(s.combine(s2))
                    } else {
                        // Commutation: fold each missing field into the other
                        // row's tail, sharing a fresh row variable.
                        let fresh = Monotype::var(context.new_typevar());
                        let s1 = unify_row(context, rest1, &Monotype::row_ext(l2.clone(), t2.clone(), fresh.clone()))?;
                        let s2 = unify_row(context, &rest2.apply(&s1), &Monotype::row_ext(l1.clone(), t1.apply(&s1), fresh.apply(&s1)))?;
                        Ok(s1.combine(s2))
                    }
                },
                _ => Err(UnificationError { pos: None, message: format!("Type function application mismatch: {:?} != {:?} (full: {:?} vs {:?})", f1, f2, r1, r2) }),
            }
        },
    }
}

#[derive(Debug, PartialEq)]
pub struct UnificationError {
    pub message : String,
    /// The source position of the offending expression/statement, if known.
    pub pos : Option<crate::ast::Pos>,
}

impl UnificationError {
    /// Attach a source position, keeping an existing one if already set.
    pub fn with_pos(mut self, pos: crate::ast::Pos) -> Self {
        if self.pos.is_none() && !pos.is_nil() {
            self.pos = Some(pos);
        }
        self
    }
}

impl std::fmt::Display for UnificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}


fn diff<T>(v1 : Vec<T>, v2 : Vec<T>) -> Vec<T> where T : PartialEq + Clone {
    v1.into_iter().filter(|x| !v2.contains(x)).collect()
}

fn expand(typ : &Monotype, context : &mut TypeContext, visited : &mut Vec<String>) -> Result<Monotype, UnificationError> {
    match typ {
        Monotype::TypeVariable(_) => Ok(typ.clone()),
        Monotype::TypeFuncApplication(func, args) => {
            let mut expanded_args : Vec<Monotype> = Vec::new();
            for arg in args {
                expanded_args.push(expand(arg, context, visited)?);
            }
            match &**func {
                TypeFunc::Infer => Ok(Monotype::TypeVariable(context.new_typevar())),
                TypeFunc::Enum(name) => match context.get_alias(name).cloned() {
                    None => Ok(Monotype::TypeFuncApplication(func.clone(), expanded_args)),
                    Some(alias) => {
                        if visited.contains(name) {
                            return Err(UnificationError { pos: None, message: format!("Recursive type alias: {}", name) });
                        }
                        if expanded_args.len() != alias.params.len() {
                            return Err(UnificationError { pos: None, message: format!("Type alias `{}` expects {} argument(s), got {}", name, alias.params.len(), expanded_args.len()) });
                        }
                        let mut sub : HashMap<String, Monotype> = HashMap::new();
                        for (p, a) in alias.params.iter().zip(expanded_args.iter()) {
                            sub.insert(p.clone(), a.clone());
                        }
                        let instantiated = alias.rhs.instantiate(&mut sub);
                        visited.push(name.clone());
                        let result = expand(&instantiated, context, visited);
                        visited.pop();
                        result
                    }
                },
                _ => Ok(Monotype::TypeFuncApplication(func.clone(), expanded_args)),
            }
        }
    }
}

pub fn type_to_typefn(typ : &Type, context : &mut TypeContext) -> Result<Monotype, UnificationError> {
    expand(&typ.t, context, &mut Vec::new())
}

fn check_undeclared(typ : &Monotype, declared : &[String]) -> Result<(), UnificationError> {
    let undeclared = diff(typ.free_variables(), declared.to_vec());
    if undeclared.is_empty() {
        Ok(())
    } else {
        Err(UnificationError { pos: None, message: format!("Undeclared type variable(s): {:?}", undeclared) })
    }
}

pub fn handle_type_decl(header : &TypeHeader, dec : &TypeDec, context : &mut TypeContext) -> Result<(), UnificationError> {
    let mut mapping : HashMap<String, Monotype> = HashMap::new();
    let mut fresh_vars : Vec<Monotype> = Vec::new();
    let mut fresh_names : Vec<String> = Vec::new();
    for name in &header.tvars {
        let fresh_name = context.new_typevar();
        let fresh = Monotype::var(fresh_name.clone());
        mapping.insert(name.clone(), fresh.clone());
        fresh_vars.push(fresh);
        fresh_names.push(fresh_name);
    }
    if context.get_alias(&header.n).is_some() {
        return Err(UnificationError { pos: None, message: format!("Type alias `{}` is already declared", header.n) });
    }
    if context.has_enum_name(&header.n) {
        return Err(UnificationError { pos: None, message: format!("`{}` is already declared as an enum", header.n) });
    }
    match dec {
        TypeDec::Enum(variants) => {
            context.add_enum_name(header.n.clone());
            let enum_typ = Monotype::enum_app(header.n.clone(), fresh_vars);
            for variant in variants {
                let mut ctor = enum_typ.clone();
                for field in variant.tparams.iter().rev() {
                    let inst = field.t.instantiate(&mut mapping);
                    let expanded = expand(&inst, context, &mut Vec::new())?;
                    ctor = Monotype::func(vec![expanded, ctor]);
                }
                check_undeclared(&ctor, &fresh_names)?;
                let generalized = context.generalise(&ctor);
                context.add(variant.n.clone(), generalized);
            }
        },
        TypeDec::Record(fields) => {
            let mut row = Monotype::empty_row();
            for field in fields.iter().rev() {
                let inst = field.1.t.instantiate(&mut mapping);
                let expanded = expand(&inst, context, &mut Vec::new())?;
                check_undeclared(&expanded, &fresh_names)?;
                row = Monotype::row_ext(field.0.clone(), expanded, row);
            }
            context.add_alias(header.n.clone(), TypeAlias { params: fresh_names, rhs: Monotype::rec(row) });
        }
        TypeDec::Alias(rhs) => {
            let elaborated = rhs.t.instantiate(&mut mapping);
            check_undeclared(&elaborated, &fresh_names)?;
            context.add_alias(header.n.clone(), TypeAlias { params : fresh_names, rhs : elaborated });
        },
    }
    Ok(())
}

/// Reject non-exhaustive `match`es.
///
/// A variable pattern covers everything. Otherwise the scrutinee type decides
/// what must be covered: a `bool` needs both `true` and `false`, and a `list`
/// needs both `[]` and a cons pattern. Every other scrutinee type has
/// infinitely many (or unenumerable) values, so a catch-all is required.
fn check_exhaustive(match_t : &Monotype, cases : &[MatchCase]) -> Result<(), UnificationError> {
    if cases.iter().any(|c| matches!(&*c.val.e, ENode::Variable(_))) {
        return Ok(());
    }
    match match_t {
        Monotype::TypeFuncApplication(f, _) if **f == TypeFunc::Bool => {
            let has_true = cases.iter().any(|c| matches!(&*c.val.e, ENode::Literal(l) if matches!(&**l, Lit::Bool(true))));
            let has_false = cases.iter().any(|c| matches!(&*c.val.e, ENode::Literal(l) if matches!(&**l, Lit::Bool(false))));
            if has_true && has_false {
                Ok(())
            } else {
                Err(UnificationError { pos: None, message: "Match on `bool` is not exhaustive: cover both `true` and `false`".to_string() })
            }
        },
        Monotype::TypeFuncApplication(f, _) if **f == TypeFunc::List => {
            let covers_empty = cases.iter().any(|c| matches!(&*c.val.e, ENode::List(es) if es.is_empty()));
            let covers_nonempty = cases.iter().any(|c| matches!(&*c.val.e, ENode::Cons(..)));
            if covers_empty && covers_nonempty {
                Ok(())
            } else {
                Err(UnificationError { pos: None, message: "Match on a list is not exhaustive: cover both `[]` and a `x::xs` pattern".to_string() })
            }
        },
        _ => Err(UnificationError { pos: None, message: format!("Match on {:?} is not exhaustive: add a catch-all variable pattern", match_t) }),
    }
}

/*
* Bottom-Up algo
*/
pub fn algo_w(context : &mut TypeContext, expr : &mut Expr) -> Result<(Substitution, Monotype), UnificationError> {
    let result = algo_w_inner(context, expr);
    if let Ok((_, typ)) = &result {
        expr.typ = typ.clone();
    }
    result
}

fn algo_w_inner(context : &mut TypeContext, expr : &mut Expr) -> Result<(Substitution, Monotype), UnificationError> {
    match &mut *expr.e {
        ENode::Variable(name) => infer_variable(context, name),
        ENode::Abstraction(bind, exp) => infer_abstraction(context, bind, exp),
        ENode::Application(e1, e2) => infer_application(context, e1, e2),
        ENode::Let(name, e1, e2) => infer_let(context, name, e1, e2),
        ENode::IfElse(cond, e1, e2) => infer_if_else(context, cond, e1, e2),
        ENode::Block(stmts, exp) => infer_block(context, stmts, exp),
        ENode::List(exps) => infer_list(context, exps),
        ENode::Cons(e1, e2) => infer_cons(context, e1, e2),
        ENode::Match(e, cases) => infer_match(context, e, cases),
        ENode::Arithmetic(op, e1, e2) => infer_arithmetic(context, op, e1, e2),
        ENode::Comparison(op, e1, e2) => infer_comparison(context, op, e1, e2),
        ENode::Logical(_, e1, e2) => infer_logical(context, e1, e2),
        ENode::Unary(op, e) => infer_unary(context, op, e),
        ENode::Literal(lit) => infer_literal(lit),
        ENode::FieldAccess(e, f) => infer_field_access(context, e, f),
        ENode::Record(name, fs) => infer_record(context, name, fs),
        ENode::With(e, fs) => infer_with(context, e, fs),
    }
}

fn infer_variable(context : &mut TypeContext, name : &mut String) -> Result<(Substitution, Monotype), UnificationError> {
    match context.get(name) {
        Some(poly) => Ok((Substitution::new(), poly.instantiate(context, None))),
        _ => Err(UnificationError { pos: None, message: format!("Undefined variable {}!", name) }),
    }
}

fn infer_abstraction(context : &mut TypeContext, bind : &mut Box<Binding>, exp : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let Binding(name, typp) = &**bind;
    let beta_mon = type_to_typefn(typp, context)?;
    let beta_poly = Polytype::Mono(Box::new(beta_mon.clone()));
    let old_binding = context.get(name);
    context.add(name.clone(), beta_poly);
    let (sub1, t1) = algo_w(context, exp)?;
    match old_binding {
        Some(poly) => context.add(name.clone(), poly),
        None => context.remove(name),
    }
    let beta = Monotype::TypeFuncApplication(Box::new(TypeFunc::Fn), vec![beta_mon, t1]).apply(&sub1);
    Ok((sub1, beta))
}

fn infer_application(context : &mut TypeContext, exp1 : &mut Box<Expr>, exp2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, exp1)?;
    *context = context.apply(&s1);
    let (s2, t2) = algo_w(context, exp2)?;
    let ret_var = Monotype::var(context.new_typevar());
    let beta = TypeFuncApplication(Box::new(TypeFunc::Fn), vec![t2, ret_var.clone()]);
    let s3 = unify(context, &t1.apply(&s2), &beta)?;
    Ok((s1.combine(s2).combine(s3.clone()), ret_var.apply(&s3)))
}

fn infer_let(context : &mut TypeContext, name : &mut String, exp1 : &mut Box<Expr>, exp2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    if TypeContext::is_builtin(name) {
        return Err(UnificationError { pos: None, message: format!("Redefinition of builtin function '{}' not allowed", name) });
    }
    let rec_var = Monotype::var(context.new_typevar());
    let old_binding = context.get(name);
    context.add(name.clone(), Polytype::Mono(Box::new(rec_var.clone())));
    let (s1, t1) = algo_w(context, exp1)?;
    *context = context.apply(&s1);
    let s_rec = unify(context, &t1, &rec_var.apply(&s1))?;
    let combined = s1.combine(s_rec.clone());
    *context = context.apply(&combined);
    match old_binding {
        Some(poly) => context.add(name.clone(), poly),
        None => context.remove(name),
    }
    let generalized = context.generalise(&t1.apply(&s_rec));
    context.add(name.clone(), generalized);
    let (s2, t2) = algo_w(context, exp2)?;
    Ok((combined.combine(s2), t2))
}

fn infer_if_else(context : &mut TypeContext, cond : &mut Box<Expr>, exp1 : &mut Box<Expr>, exp2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, cond)?;
    let s2 = unify(context, &t1, &Monotype::bool())?;
    *context = context.apply(&s1).apply(&s2);
    let (s3, t3) = algo_w(context, exp1)?;
    *context = context.apply(&s3);
    let (s4, t4) = algo_w(context, exp2)?;
    let s5 = unify(context, &t3.apply(&s4), &t4)?;
    Ok((
        s1.combine(s2).combine(s3).combine(s4.clone()).combine(s5.clone()),
        t3.apply(&s4).apply(&s5)
    ))
}

fn infer_block(context : &mut TypeContext, stmts : &mut Vec<Stmt>, exp : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let mut combined = Substitution::new();
    for s in stmts {
        match &mut *s.s {
            SNode::Decl(e1, t1, e2) => {
                let var_name = match &*e1.e {
                    ENode::Variable(name) => name.clone(),
                    _ => return Err(UnificationError { pos: None, message: "Expected a variable name in declaration".to_string() }),
                };
                if TypeContext::is_builtin(&var_name) {
                    return Err(UnificationError { pos: None, message: format!("Redefinition of builtin function '{}' not allowed", var_name) });
                }
                let binding_type = type_to_typefn(t1, context)?;
                let old_binding = context.get(&var_name);
                context.add(var_name.clone(), Polytype::Mono(Box::new(binding_type.clone())));
                let (s1, inferred_type) = algo_w(context, e2)?;
                *context = context.apply(&s1);
                combined = combined.combine(s1);
                let s2 = unify(context, &binding_type.apply(&combined), &inferred_type)?;
                *context = context.apply(&s2);
                combined = combined.combine(s2);
                match old_binding {
                    Some(poly) => context.add(var_name.clone(), poly),
                    None => context.remove(&var_name),
                }
                let resolved = binding_type.apply(&combined);
                let generalized = context.generalise(&resolved);
                context.add(var_name, generalized);
            },
            SNode::Expr(e1) => {
                let (s1, _) = algo_w(context, e1)?;
                *context = context.apply(&s1);
                combined = combined.combine(s1);
            },
            SNode::TypeDecl(_, _) => return Err(UnificationError {
                pos: None,
                message: "Type declarations are not allowed inside block expressions".to_string()
            }),
        }
    }
    let (s_exp, t_exp) = algo_w(context, exp)?;
    combined = combined.combine(s_exp);
    Ok((combined, t_exp))
}

fn infer_list(context : &mut TypeContext, exps : &mut Vec<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    if exps.is_empty() {
        let tv = Monotype::var(context.new_typevar());
        Ok((Substitution::new(), Monotype::list(tv)))
    } else {
        let (s0, t0) = algo_w(context, &mut exps[0])?;
        *context = context.apply(&s0);
        let mut combined = s0;
        let mut elem_type = t0;
        for e in exps[1..].iter_mut() {
            let (s_i, t_i) = algo_w(context, e)?;
            *context = context.apply(&s_i);
            combined = combined.combine(s_i);
            let s_u = unify(context, &elem_type, &t_i)?;
            combined = combined.combine(s_u.clone());
            elem_type = elem_type.apply(&s_u);
        }
        Ok((combined, Monotype::list(elem_type)))
    }
}

fn infer_cons(context : &mut TypeContext, e1 : &mut Box<Expr>, e2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, e1)?;
    *context = context.apply(&s1);
    let (s2, t2) = algo_w(context, e2)?;
    let elem = t1.apply(&s2);
    let s3 = unify(context, &t2, &Monotype::list(elem.clone()))?;
    let result = Monotype::list(elem.apply(&s3));
    Ok((s1.combine(s2).combine(s3), result))
}

fn infer_match(context : &mut TypeContext, e : &mut Box<Expr>, cases : &mut Vec<MatchCase>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s0, t0) = algo_w(context, e)?;
    *context = context.apply(&s0);
    let mut match_t = t0;
    let mut combined = s0;
    let ret = Monotype::var(context.new_typevar());
    for MatchCase { val: e1, exp: e2 } in cases.iter_mut() {
        let mut case_ctx = context.apply(&combined);
        let s1 = type_pattern(&mut case_ctx, e1, &match_t)?;
        combined = combined.combine(s1);
        let (s2, t2) = algo_w(&mut case_ctx, e2)?;
        combined = combined.combine(s2);
        // case_ctx is a clone with its own type-var counter; vars allocated
        // there leak into `combined`/`match_t`, so pull the counter back up to
        // keep fresh names globally unique.
        context.type_var_ctr = context.type_var_ctr.max(case_ctx.type_var_ctr);
        let s_u = unify(context, &ret.apply(&combined), &t2.apply(&combined))?;
        combined = combined.combine(s_u);
        match_t = match_t.apply(&combined);
    }
    check_exhaustive(&match_t, cases)?;
    let resolved_ret = ret.apply(&combined);
    Ok((combined, resolved_ret))
}

fn infer_arithmetic(context : &mut TypeContext, op : &mut ArithOp, e1 : &mut Box<Expr>, e2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, e1)?;
    *context = context.apply(&s1);
    let (s2, t2) = algo_w(context, e2)?;
    let s3 = unify(context, &t1.apply(&s2), &t2)?;
    let unified = t1.apply(&s2).apply(&s3);
    if !matches!(unified, Monotype::TypeVariable(_)) {
        match op {
            ArithOp::Plus => {
                unify(context, &unified, &Monotype::int())
                    .or_else(|_| unify(context, &unified, &Monotype::float()))
                    .or_else(|_| unify(context, &unified, &Monotype::string()))
                    .map_err(|_| UnificationError { pos: None, message: format!("'+' requires int, float, or string operands, got {:?}", unified) })?;
            },
            _ => {
                unify(context, &unified, &Monotype::int())
                    .or_else(|_| unify(context, &unified, &Monotype::float()))
                    .map_err(|_| UnificationError { pos: None, message: format!("{:?} requires int or float operands, got {:?}", op, unified) })?;
            },
        }
    }
    Ok((s1.combine(s2).combine(s3), unified))
}

fn infer_comparison(context : &mut TypeContext, op : &mut CompOp, e1 : &mut Box<Expr>, e2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, e1)?;
    *context = context.apply(&s1);
    let (s2, t2) = algo_w(context, e2)?;
    let s3 = unify(context, &t1.apply(&s2), &t2)?;
    let unified = t1.apply(&s2).apply(&s3);
    if !matches!(unified, Monotype::TypeVariable(_)) {
        match op {
            CompOp::Eq | CompOp::NotEq => {
                if let Monotype::TypeFuncApplication(f, _) = &unified && **f == TypeFunc::Fn {
                    return Err(UnificationError { pos: None, message: "Cannot compare function types".to_string() });
                }
                //let op_name = if *op == compop::eq { "==" } else { "!=" };
                //unify(context, &unified, &Monotype::int())
                //    .or_else(|_| unify(context, &unified, &Monotype::float()))
                //    .or_else(|_| unify(context, &unified, &Monotype::string()))
                //    .or_else(|_| unify(context, &unified, &Monotype::bool()))
                //    .map_err(|_| UnificationError { pos: None, message: format!("'{}' requires int, float, string, or bool operands", op_name) })?;
            },
            _ => {
                unify(context, &unified, &Monotype::int())
                    .or_else(|_| unify(context, &unified, &Monotype::float()))
                    .map_err(|_| UnificationError { pos: None, message: "Comparison requires int or float operands".to_string() })?;
            },
        }
    }
    Ok((s1.combine(s2).combine(s3), Monotype::bool()))
}

fn infer_logical(context : &mut TypeContext, e1 : &mut Box<Expr>, e2 : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, e1)?;
    *context = context.apply(&s1);
    let (s2, t2) = algo_w(context, e2)?;
    let s3 = unify(context, &t1.apply(&s2), &t2)?;
    let unified = t1.apply(&s2).apply(&s3);
    let s4 = unify(context, &unified, &Monotype::bool())
        .map_err(|_| UnificationError { pos: None, message: format!("Logical operations require bool operands, got {:?}", unified) })?;
    Ok((s1.combine(s2).combine(s3).combine(s4), Monotype::bool()))
}

fn infer_unary(context : &mut TypeContext, op : &mut UnaryOp, e : &mut Box<Expr>) -> Result<(Substitution, Monotype), UnificationError> {
    match op {
        UnaryOp::Negate => {
            let (s1, t1) = algo_w(context, e)?;
            if matches!(t1, Monotype::TypeVariable(_)) {
                *context = context.apply(&s1);
                return Ok((s1, t1));
            }
            let s2 = unify(context, &t1, &Monotype::int())
                .or_else(|_| unify(context, &t1, &Monotype::float()))
                .map_err(|_| UnificationError { pos: None, message: format!("Unary negation requires int or float operand, got {:?}", t1) })?;
            let s3 = s1.combine(s2);
            *context = context.apply(&s3);
            Ok((s3.clone(), t1.apply(&s3)))
        },
        UnaryOp::Not => {
            let (s1, t1) = algo_w(context, e)?;
            let s2 = unify(context, &t1, &Monotype::bool())?;
            let s3 = s1.combine(s2);
            *context = context.apply(&s3);
            Ok((s3.clone(), Monotype::bool()))
        },
    }
}

fn literal_monotype(lit : &Lit) -> Monotype {
    match lit {
        Lit::Int(_) => Monotype::int(),
        Lit::Bool(_) => Monotype::bool(),
        Lit::Str(_) => Monotype::string(),
        Lit::Float(_) => Monotype::float(),
        Lit::Char(_) => Monotype::char(),
        Lit::Unit => Monotype::unit(),
    }
}

fn infer_literal(lit : &mut Box<Lit>) -> Result<(Substitution, Monotype), UnificationError> {
    Ok((Substitution::new(), literal_monotype(lit)))
}

fn infer_field_access(context : &mut TypeContext, expr : &mut Box<Expr>, field : &mut String) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, expr)?;
    *context = context.apply(&s1);
    let (alpha, rho) = (
        Monotype::var(context.new_typevar()),
        Monotype::var(context.new_typevar())
    );
    let rowtype = Monotype::rec(Monotype::row_ext(field.clone(), alpha.clone(), rho));
    let s2 = unify(context, &t1.apply(&s1), &rowtype)?;
    let s_unified = s1.combine(s2);
    Ok((s_unified.clone(), alpha.apply(&s_unified)))
}

fn infer_record(context : &mut TypeContext, name : &mut Option<String>, field_assns : &mut Vec<FieldAssn>) -> Result<(Substitution, Monotype), UnificationError> {
    match name {
        Some(n) => {
            let n = n.clone();
            infer_named_record(context, &n, field_assns)
        },
        None => infer_anonymous_record(context, field_assns),
    }
}

/// Infer an unnamed record literal: build a closed row from the fields in the
/// order they are written.
fn infer_anonymous_record(context : &mut TypeContext, field_assns : &mut Vec<FieldAssn>) -> Result<(Substitution, Monotype), UnificationError> {
    let mut combined = Substitution::new();
    let mut acc_row = Monotype::empty_row();
    for FieldAssn { field, exp } in field_assns {
        let (s1, t1) = algo_w(context, exp)?;
        combined = combined.combine(s1.clone());
        *context = context.apply(&s1);
        acc_row = acc_row.apply(&s1);
        acc_row = Monotype::row_ext(field.clone(), t1.apply(&s1), acc_row);
    }
    Ok((combined, Monotype::rec(acc_row)))
}

/// Infer `Name { field = value, ... }`: instantiate the declared record's type
/// alias and unify each field value against the corresponding declared field
/// type. The result is the declaration-ordered row, and missing or unknown
/// fields are rejected.
fn infer_named_record(context : &mut TypeContext, name : &str, field_assns : &mut Vec<FieldAssn>) -> Result<(Substitution, Monotype), UnificationError> {
    let alias = context.get_alias(name).cloned()
        .ok_or_else(|| UnificationError { pos: None, message: format!("Unknown record type `{}`", name) })?;

    let mut sub = HashMap::new();
    for p in &alias.params {
        sub.insert(p.clone(), Monotype::var(context.new_typevar()));
    }
    let rec_type = alias.rhs.instantiate(&mut sub);

    let declared = record_row_fields(&rec_type)?;

    for fa in field_assns.iter() {
        if !declared.iter().any(|(label, _)| label == &fa.field) {
            return Err(UnificationError { pos: None, message: format!("Unknown field `{}` in record `{}`", fa.field, name) });
        }
    }

    let mut combined = Substitution::new();
    for (label, declared_ty) in &declared {
        let fa = field_assns.iter_mut().find(|fa| &fa.field == label)
            .ok_or_else(|| UnificationError { pos: None, message: format!("Missing field `{}` in record `{}`", label, name) })?;
        let (s1, t1) = algo_w(context, &mut fa.exp)?;
        combined = combined.combine(s1.clone());
        *context = context.apply(&s1);
        let s2 = unify(context, &declared_ty.apply(&combined), &t1)?;
        *context = context.apply(&s2);
        combined = combined.combine(s2);
    }

    Ok((combined.clone(), rec_type.apply(&combined)))
}

/// Walk a `Rec` record type into its `(label, field type)` list in row order.
/// Errors if the row is not closed (an open row variable tail means the record
/// was not fully resolved).
fn record_row_fields(typ : &Monotype) -> Result<Vec<(String, Monotype)>, UnificationError> {
    let row = match typ {
        Monotype::TypeFuncApplication(f, args) if matches!(**f, TypeFunc::Rec) && args.len() == 1 => &args[0],
        _ => return Err(UnificationError { pos: None, message: format!("Expected a record type, got {:?}", typ) }),
    };
    let mut fields = Vec::new();
    let mut cur = row;
    loop {
        match cur {
            Monotype::TypeFuncApplication(f, _) if matches!(**f, TypeFunc::EmptyRow) => break,
            Monotype::TypeFuncApplication(f, args) if matches!(**f, TypeFunc::RowExt(_)) && args.len() == 2 => {
                let label = match &**f { TypeFunc::RowExt(l) => l.clone(), _ => unreachable!() };
                fields.push((label, args[0].clone()));
                cur = &args[1];
            },
            _ => return Err(UnificationError { pos: None, message: "Record row is not closed".to_string() }),
        }
    }
    Ok(fields)
}

fn infer_with(context : &mut TypeContext, expr : &mut Box<Expr>, field_assns : &mut Vec<FieldAssn>) -> Result<(Substitution, Monotype), UnificationError> {
    let (s1, t1) = algo_w(context, expr)?;
    *context = context.apply(&s1);
    let mut combined = s1;
    for FieldAssn { field, exp } in field_assns {
        let (tau_l, rho) = (
            Monotype::var(context.new_typevar()),
            Monotype::var(context.new_typevar())
        );
        let s_rec = unify(context, &t1.apply(&combined), &Monotype::rec(Monotype::row_ext(field.clone(), tau_l.clone(), rho)))?;
        *context = context.apply(&s_rec);
        combined = combined.combine(s_rec);

        let (s_val, t_val) = algo_w(context, exp)?;
        *context = context.apply(&s_val);
        combined = combined.combine(s_val);

        let s_field = unify(context, &tau_l.apply(&combined), &t_val)?;
        *context = context.apply(&s_field);
        combined = combined.combine(s_field);
    }
    Ok((combined.clone(), t1.apply(&combined)))
}

pub fn type_pattern(context : &mut TypeContext, expr : &Expr, typ : &Monotype) -> Result<Substitution,UnificationError> {
    match &*expr.e {
        ENode::Literal(lit) => {
            let s = unify(context, typ, &literal_monotype(lit))?;
            *context = context.apply(&s);
            Ok(s)
        },
        ENode::Variable(name) => {
            context.add(name.clone(), Polytype::Mono(Box::new(typ.clone())));
            Ok(Substitution::new())
        },
        ENode::Cons(hd, tl) => {
            let alpha = Monotype::var(context.new_typevar());
            let s0 = unify(context, typ, &Monotype::list(alpha.clone()))?;
            *context = context.apply(&s0);
            let elem = alpha.apply(&s0);
            let s1 = type_pattern(context, hd, &elem)?;
            let s2 = type_pattern(context, tl, &Monotype::list(elem.apply(&s1)))?;
            Ok(s0.combine(s1).combine(s2))
        },
        ENode::Application(f, arg) => {
            let mut args = vec![arg];
            let mut head = f;
            loop {
                match &*head.e {
                    ENode::Application(f2, a2) => {
                        args.push(a2);
                        head = f2;
                    },
                    ENode::Variable(name) => {
                        let poly = context.get(name).ok_or_else(||
                            UnificationError { pos: None, message: format!("Undefined constructor {}", name) })?;
                        let mut ctor = poly.instantiate(context, None);
                        let mut combined = Substitution::new();
                        for a in args.iter().rev() {
                            if let Monotype::TypeFuncApplication(typ_fn, ts) = &ctor
                                && **typ_fn == TypeFunc::Fn && ts.len() == 2 {
                                    let field = ts[0].clone();
                                    let rest = ts[1].clone();
                                    let s = type_pattern(context, a, &field.apply(&combined))?;
                                    *context = context.apply(&s);
                                    combined = combined.combine(s);
                                    ctor = rest.apply(&combined);
                                    continue;
                            }
                            return Err(UnificationError {
                                pos: None,
                                message: format!("Constructor `{}` applied to too many arguments", name)
                            });
                        }
                        let s_last = unify(context, typ, &ctor)?;
                        return Ok(combined.combine(s_last));
                    },
                    _ => return Err(UnificationError {
                        pos: None,
                        message: "Constructor pattern must be a constructor name applied to arguments".to_string()
                    }),
                }
            }
        },
        ENode::List(exprs) => {
            let alpha = Monotype::var(context.new_typevar());
            let s0 = unify(context, typ, &Monotype::list(alpha.clone()))?;
            *context = context.apply(&s0);
            let elem = alpha.apply(&s0);
            let mut combined = s0;
            for e in exprs {
                let s = type_pattern(context, e, &elem)?;
                *context = context.apply(&s);
                combined = combined.combine(s);
            }
            Ok(combined)
        },
        ENode::Record(_, fields) => {
            let mut alphas : Vec<Monotype> = Vec::new();
            let mut row = Monotype::var(context.new_typevar());
            for FieldAssn { field, .. } in fields.iter().rev() {
                let alpha = Monotype::var(context.new_typevar());
                row = Monotype::row_ext(field.clone(), alpha.clone(), row);
                alphas.push(alpha);
            }
            alphas.reverse();
            let s0 = unify(context, typ, &Monotype::rec(row))?;
            *context = context.apply(&s0);
            let mut combined = s0;
            for (FieldAssn { exp, .. }, alpha) in fields.iter().zip(alphas.iter()) {
                let s = type_pattern(context, exp, &alpha.apply(&combined))?;
                *context = context.apply(&s);
                combined = combined.combine(s);
            }
            Ok(combined)
        },
        _ => Err(UnificationError { pos: None, message: format!("Unsupported pattern {:?}", expr) }),
    }
}

// ----------------------------------------------------------------------------
// Statement / program typechecking
// ----------------------------------------------------------------------------

impl Stmt {
    pub fn typecheck(&mut self, ctx : &TypeContext) -> Result<(Substitution, Monotype), UnificationError> {
        let result = (|| -> Result<(Substitution, Monotype), UnificationError> {
            let mut context = ctx.clone();
            let (combined, typ) = match &mut *self.s {
                SNode::Decl(e1, t1, e2) => {
                    let var_name = match &*e1.e {
                        ENode::Variable(name) => name.clone(),
                        _ => return Err(UnificationError { pos: Some(self.pos.clone()), message: format!("Expected a variable name in declaration, got {:?}", *e1.e) }),
                    };
                    if TypeContext::is_builtin(&var_name) {
                        return Err(UnificationError { pos: Some(self.pos.clone()), message: format!("Redefinition of builtin function '{}' not allowed", var_name) });
                    }
                    let binding_type = type_to_typefn(t1, &mut context)?;
                    let old_binding = context.get(&var_name);
                    context.add(var_name.clone(), Polytype::Mono(Box::new(binding_type.clone())));
                    let (s1, inferred_type) = algo_w(&mut context, e2)?;
                    let s2 = unify(&mut context, &binding_type.apply(&s1), &inferred_type)?;
                    let combined = s1.combine(s2);
                    context = context.apply(&combined);
                    match old_binding {
                        Some(poly) => context.add(var_name.clone(), poly),
                        None => context.remove(&var_name),
                    }
                    let resolved_typ = binding_type.apply(&combined);
                    let generalized = context.generalise(&resolved_typ);
                    context.add(var_name, generalized);
                    self.ctx = context;
                    (combined, resolved_typ)
                },
                SNode::Expr(e1) => {
                    let (sub, typ) = algo_w(&mut context, e1)?;
                    self.ctx = context.apply(&sub);
                    (sub, typ)
                },
                SNode::TypeDecl(header, dec) => {
                    handle_type_decl(header, dec, &mut context)?;
                    self.ctx = context;
                    (Substitution::new(), Monotype::unit())
                }
            };
            resolve_stmt_types(self, &combined);
            Ok((combined, typ))
        })();
        result.map_err(|e| e.with_pos(self.pos.clone()))
    }
}

impl Program {
    pub fn typecheck(prog : &mut Program) -> Result<(), UnificationError> {
        for stmt in prog.stmts.iter_mut() {
            stmt.typecheck(&prog.ctx)?;
            prog.ctx = stmt.ctx.clone();
        }
        Ok(())
    }
}

/// Apply `sub` to the recorded (`algo_w`-annotated) type of every expression
/// reachable from `stmt`. Runs after a statement is type-checked, once the
/// statement's full substitution is known, resolving inferred types into
/// concrete ones. Type variables bound by a generalized `let` are resolved
/// too: codegen targets monomorphic MLIR, so each instantiation is specialized
/// at its use site rather than kept polymorphic.
pub fn resolve_stmt_types(stmt : &mut Stmt, sub : &Substitution) {
    match &mut *stmt.s {
        SNode::Decl(e1, _, e2) => {
            resolve_expr_types(e1, sub);
            resolve_expr_types(e2, sub);
        },
        SNode::Expr(e1) => resolve_expr_types(e1, sub),
        SNode::TypeDecl(_, _) => {}
    }
}

/// Apply `sub` to the recorded type of `expr` and everything reachable from
/// it. Used by codegen to specialize a lambda body: the definition statement
/// may leave free type variables (e.g. a recursive use), which the
/// instantiation's substitution replaces with concrete types.
pub fn apply_substitution(expr : &mut Expr, sub : &Substitution) {
    resolve_expr_types(expr, sub);
}

fn resolve_expr_types(expr : &mut Expr, sub : &Substitution) {
    expr.typ = expr.typ.apply(sub);
    match &mut *expr.e {
        ENode::Variable(_) | ENode::Literal(_) => {}
        ENode::Abstraction(_, body) => resolve_expr_types(body, sub),
        ENode::Application(f, x) => {
            resolve_expr_types(f, sub);
            resolve_expr_types(x, sub);
        },
        ENode::Let(_, e1, e2) => {
            resolve_expr_types(e1, sub);
            resolve_expr_types(e2, sub);
        },
        ENode::IfElse(c, t, e) => {
            resolve_expr_types(c, sub);
            resolve_expr_types(t, sub);
            resolve_expr_types(e, sub);
        },
        ENode::Block(stmts, e) => {
            for s in stmts.iter_mut() {
                resolve_stmt_types(s, sub);
            }
            resolve_expr_types(e, sub);
        },
        ENode::Comparison(_, a, b) => {
            resolve_expr_types(a, sub);
            resolve_expr_types(b, sub);
        },
        ENode::Arithmetic(_, a, b) => {
            resolve_expr_types(a, sub);
            resolve_expr_types(b, sub);
        },
        ENode::Logical(_, a, b) => {
            resolve_expr_types(a, sub);
            resolve_expr_types(b, sub);
        },
        ENode::Unary(_, e) => resolve_expr_types(e, sub),
        ENode::List(es) => {
            for e in es.iter_mut() {
                resolve_expr_types(e, sub);
            }
        },
        ENode::Cons(h, t) => {
            resolve_expr_types(h, sub);
            resolve_expr_types(t, sub);
        },
        ENode::Match(scrut, cases) => {
            resolve_expr_types(scrut, sub);
            for c in cases.iter_mut() {
                resolve_expr_types(&mut c.val, sub);
                resolve_expr_types(&mut c.exp, sub);
            }
        },
        ENode::FieldAccess(e, _) => resolve_expr_types(e, sub),
        ENode::Record(_, fields) => {
            for fa in fields.iter_mut() {
                resolve_expr_types(&mut fa.exp, sub);
            }
        },
        ENode::With(e, fields) => {
            resolve_expr_types(e, sub);
            for fa in fields.iter_mut() {
                resolve_expr_types(&mut fa.exp, sub);
            }
        },
    }
}

