use std::fmt::Display;
use crate::prelude::get_prelude;
use crate::types::*;
use crate::grammar;

#[derive(Debug, Clone, PartialEq)]
pub struct Pos {
    /// 1-based source span. Filled by `Program::parse` from the byte offsets
    /// that the grammar records; all-zero for synthetic/nil positions.
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
    /// Raw byte offsets into the parsed buffer (the values the grammar's
    /// `@L`/`@R` produce). Kept so positions can be re-indexed; unused by
    /// codegen once the line/column span is filled.
    pub start: u32,
    pub end: u32,
}

impl Display for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[L:{}, C:{}]", self.start_line, self.start_col)
    }
}

impl Pos {
    pub fn nil() -> Pos {
        Pos { start_line: 0, start_col: 0, end_line: 0, end_col: 0, start: 0, end: 0 }
    }

    /// A raw byte-offset span, as produced by the parser's `@L`/`@R`. The
    /// line/column fields are filled in by [`Program::parse`].
    pub fn bytes(start: u32, end: u32) -> Pos {
        Pos { start_line: 0, start_col: 0, end_line: 0, end_col: 0, start, end }
    }

    pub fn is_nil(&self) -> bool {
        self.start == 0 && self.end == 0
    }

    /// Convert the raw byte offsets into a 1-based (line, column) span using
    /// the buffer's line index.
    pub fn fill(&mut self, index: &LineIndex) {
        let (sl, sc) = index.line_col(self.start);
        let (el, ec) = index.line_col(self.end);
        self.start_line = sl;
        self.start_col = sc;
        self.end_line = el;
        self.end_col = ec;
    }

    /// The span covering both `self` and `other` (min start, max end),
    /// re-filling the line/column fields for the combined byte range.
    pub fn merge(&mut self, other: &Pos) {
        self.start = self.start.min(other.start);
        self.end = self.end.max(other.end);
        self.start_line = 0;
        self.start_col = 0;
        self.end_line = 0;
        self.end_col = 0;
    }
}

/// A line index over a parsed buffer, used to convert byte offsets to
/// 1-based (line, column) positions.
#[derive(Debug, Clone)]
pub struct LineIndex {
    line_starts: Vec<u32>,
}

impl LineIndex {
    pub fn new(buf: &str) -> LineIndex {
        let mut line_starts = vec![0];
        for (i, b) in buf.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        LineIndex { line_starts }
    }

    /// 1-based (line, column) for a byte offset.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (line as u32 + 1, offset - self.line_starts[line] + 1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(i32),
    Float(f32),
    Bool(bool),
    Str(String),
    Char(char),
    Unit,
}

impl Display for Lit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lit::Int(i) => write!(f, "Int({})", i),
            Lit::Float(r) => write!(f, "Float({})", r),
            Lit::Bool(b) => write!(f, "Bool({})", b),
            Lit::Str(s) => write!(f, "Str(\"{}\")", s),
            Lit::Char(c) => write!(f, "Char(\'{}\')", c),
            Lit::Unit => write!(f, "Unit()"),
        }
    }
}

/// Decode the escape sequence beginning at `\` (with the iterator already past
/// the backslash), returning the character it denotes. Supported escapes are
/// `\n`, `\t`, `\r`, `\0`, `\a`, `\b`, `\f`, `\v`, `\\`, `\'`, `\"`, `\xHH`,
/// and `\u{...}`. An unknown escape yields the escaped character itself (so
/// `\q` is `q`).
fn decode_escape(chars: &mut std::str::Chars) -> char {
    match chars.next() {
        Some('n') => '\n',
        Some('t') => '\t',
        Some('r') => '\r',
        Some('0') => '\0',
        Some('a') => '\x07',
        Some('b') => '\x08',
        Some('f') => '\x0c',
        Some('v') => '\x0b',
        Some('\\') => '\\',
        Some('\'') => '\'',
        Some('"') => '"',
        Some('x') => {
            let hex: String = chars.take(2).collect();
            let code = u32::from_str_radix(&hex, 16).unwrap_or(0);
            char::from_u32(code).unwrap_or('\0')
        }
        Some('u') => {
            chars.next(); // '{'
            let digits: String = chars.take_while(|c| *c != '}').collect();
            let code = u32::from_str_radix(&digits, 16).unwrap_or(0);
            char::from_u32(code).unwrap_or('\0')
        }
        Some(other) => other,
        None => '\0',
    }
}

/// Decode a single-quoted char literal (including its quotes), resolving the
/// escape sequences handled by [`decode_escape`].
pub(crate) fn decode_char_literal(raw: &str) -> char {
    let inner = &raw[1..raw.len().saturating_sub(1)];
    let mut chars = inner.chars();
    match chars.next() {
        None => '\0',
        Some('\\') => decode_escape(&mut chars),
        Some(c) => c,
    }
}

/// Decode a double-quoted string literal (including its quotes), resolving
/// escape sequences with the same rules as [`decode_char_literal`].
pub(crate) fn decode_string_literal(raw: &str) -> String {
    let inner = &raw[1..raw.len().saturating_sub(1)];
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            out.push(decode_escape(&mut chars));
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub t: Monotype
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.t)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding(pub String, pub Box<Type>);

impl Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} : {})", self.0, self.1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDec {
    Alias(Box<Type>),
    Enum(Vec<Variant>),
    Record(Vec<Binding>)
}

impl Display for TypeDec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Alias(t) => write!(f, "Alias({})", t),
            Self::Enum(variants) => write!(f, "Enum({})",
                variants.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n")),
            Self::Record(fields) => write!(f, "Record({})",
                fields.iter().map(|b| b.to_string()).collect::<Vec<_>>().join("\n"))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub n : String,
    pub tparams : Vec<Type>
}

impl Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.n, self.tparams.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeHeader {
    pub n : String,
    pub tvars : Vec<String>
}

impl Display for TypeHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.n, self.tvars.join(", "))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SNode {
    Decl(Box<Expr>, Box<Type>, Box<Expr>),  // let x [: Type] = e;
    Expr(Box<Expr>),                        // e; special case, not always ()
    TypeDecl(TypeHeader, Box<TypeDec>) // name <type vars> = <type>
}

impl Display for SNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decl(expr, typ, expr1) => write!(f, "Decl({}: {}, {})", expr, typ, expr1),
            Self::Expr(expr) => write!(f, "Expr({})", expr),
            Self::TypeDecl(type_header, type_dec) => write!(f, "TypeDecl({}, {}", type_header, type_dec),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub s : Box<SNode>,
    pub ctx : TypeContext,
    pub pos : Pos
}

impl Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stmt({},\n{})", self.pos, self.s)
    }
}

impl PartialEq for Stmt {
    /// Structural equality: source positions are metadata and deliberately
    /// excluded so parsed trees compare equal to hand-built expected trees.
    fn eq(&self, other: &Self) -> bool {
        self.s == other.s && self.ctx == other.ctx
    }
}

impl Stmt {
    pub fn from(node : SNode) -> Stmt {
        Self::at(Pos::nil(), node)
    }

    pub fn at(pos : Pos, node : SNode) -> Stmt {
        Stmt {
            s : Box::new(node),
            ctx : TypeContext::new(),
            pos
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ENode {
    Variable(String),
    Literal(Box<Lit>),
    Abstraction(Box<Binding>, Box<Expr>),
    Application(Box<Expr>, Box<Expr>),
    Let(String,Box<Expr>,Box<Expr>),
    IfElse(Box<Expr>,Box<Expr>,Box<Expr>),
    Block(Vec<Stmt>, Box<Expr>),
    Comparison(CompOp, Box<Expr>, Box<Expr>),
    Arithmetic(ArithOp, Box<Expr>, Box<Expr>),
    Logical(LogicalOp, Box<Expr>, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    List(Vec<Expr>),
    Cons(Box<Expr>, Box<Expr>),
    Match(Box<Expr>, Vec<MatchCase>),
    FieldAccess(Box<Expr>, String),
    Record(Option<String>, Vec<FieldAssn>),
    With(Box<Expr>, Vec<FieldAssn>),
}

impl Display for ENode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ENode::Variable(n) => write!(f, "Var({})", n),
            ENode::Literal(lit) => write!(f, "Lit({})", lit),
            ENode::Abstraction(binding, expr) => write!(f, "Abs({} -> {})", binding, expr),
            ENode::Application(expr, expr1) => write!(f, "App({}, {})", expr, expr1),
            ENode::Let(n, expr, expr1) => write!(f, "Let({} = {}, {})", n, expr, expr1),
            ENode::IfElse(expr, expr1, expr2) => write!(f, "IfThenElse({}, {}, {})", expr, expr1, expr2),
            ENode::Block(stmts, expr) => write!(f, "Block({}\n{})", stmts.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n"), expr),
            ENode::Comparison(comp_op, expr, expr1) => write!(f, "Comp({}, {}, {})", comp_op, expr, expr1),
            ENode::Arithmetic(arith_op, expr, expr1) => write!(f, "Arith({}, {}, {})", arith_op, expr, expr1),
            ENode::Logical(logical_op, expr, expr1) => write!(f, "Logic({}, {}, {})", logical_op, expr, expr1),
            ENode::Unary(unary_op, expr) => write!(f, "Unary({}, {})", unary_op, expr),
            ENode::List(exprs) => write!(f, "List({})", exprs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ")),
            ENode::Cons(expr, expr1) => write!(f, "Cons({}, {})", expr, expr1),
            ENode::Match(expr, match_cases) => write!(f, "Match({}, {})", expr, match_cases.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")),
            ENode::FieldAccess(expr, field) => write!(f, "Field({}, {})", expr, field),
            ENode::Record(name, field_assns) => write!(f, "Record({}, {})",
                name.clone().unwrap_or_else(|| "_".to_string()),
                field_assns.iter().map(|fa| format!("{}: {}", fa.field, fa.exp)).collect::<Vec<_>>().join(", ")),
            ENode::With(expr, field_assns) => write!(f, "With({}, {})", expr,
                field_assns.iter().map(|fa| format!("{}: {}", fa.field, fa.exp)).collect::<Vec<_>>().join(", ")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCase {
    pub val : Box<Expr>,
    pub exp : Box<Expr>
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldAssn {
    pub field: String,
    pub exp : Box<Expr>
}

impl Display for MatchCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Case({} => {})", self.val, self.exp)
    }
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub e : Box<ENode>,
    pub ctx : TypeContext,
    pub pos : Pos,
    /// The inferred type of this expression: filled with the raw type during
    /// typechecking (`algo_w`) and resolved by the post-typecheck pass
    /// ([`crate::types::resolve_stmt_types`]) once the statement's full
    /// substitution is known.
    pub typ : Monotype,
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Expr({},\n{})", self.pos, self.e)
    }
}

impl PartialEq for Expr {
    /// Structural equality; `pos` is metadata and deliberately excluded.
    fn eq(&self, other: &Self) -> bool {
        self.e == other.e && self.ctx == other.ctx && self.typ == other.typ
    }
}

impl Expr {
    pub fn from(node : ENode) -> Expr {
        Self::at(Pos::nil(), node)
    }

    pub fn at(pos : Pos, node : ENode) -> Expr {
        Expr {
            e : Box::new(node),
            ctx : TypeContext::new(),
            pos,
            typ : Monotype::infer(),
        }
    }
}

/// Fill every `Pos` reachable from `stmt` with line/column data derived from
/// the buffer's line index (the grammar records raw byte offsets).
pub fn fill_stmt_positions(stmt : &mut Stmt, index : &LineIndex) {
    if !stmt.pos.is_nil() {
        stmt.pos.fill(index);
    }
    match &mut *stmt.s {
        SNode::Decl(e1, _, e2) => {
            fill_expr_positions(e1, index);
            fill_expr_positions(e2, index);
        },
        SNode::Expr(e1) => fill_expr_positions(e1, index),
        SNode::TypeDecl(_, _) => {}
    }
}

fn fill_expr_positions(expr : &mut Expr, index : &LineIndex) {
    if !expr.pos.is_nil() {
        expr.pos.fill(index);
    }
    match &mut *expr.e {
        ENode::Variable(_) | ENode::Literal(_) => {}
        ENode::Abstraction(_, body) => fill_expr_positions(body, index),
        ENode::Application(f, x) => {
            fill_expr_positions(f, index);
            fill_expr_positions(x, index);
        },
        ENode::Let(_, e1, e2) => {
            fill_expr_positions(e1, index);
            fill_expr_positions(e2, index);
        },
        ENode::IfElse(c, t, e) => {
            fill_expr_positions(c, index);
            fill_expr_positions(t, index);
            fill_expr_positions(e, index);
        },
        ENode::Block(stmts, e) => {
            for s in stmts.iter_mut() {
                fill_stmt_positions(s, index);
            }
            fill_expr_positions(e, index);
        },
        ENode::Comparison(_, a, b) => {
            fill_expr_positions(a, index);
            fill_expr_positions(b, index);
        },
        ENode::Arithmetic(_, a, b) => {
            fill_expr_positions(a, index);
            fill_expr_positions(b, index);
        },
        ENode::Logical(_, a, b) => {
            fill_expr_positions(a, index);
            fill_expr_positions(b, index);
        },
        ENode::Unary(_, e) => fill_expr_positions(e, index),
        ENode::List(es) => {
            for e in es.iter_mut() {
                fill_expr_positions(e, index);
            }
        },
        ENode::Cons(h, t) => {
            fill_expr_positions(h, index);
            fill_expr_positions(t, index);
        },
        ENode::Match(scrut, cases) => {
            fill_expr_positions(scrut, index);
            for c in cases.iter_mut() {
                fill_expr_positions(&mut c.val, index);
                fill_expr_positions(&mut c.exp, index);
            }
        },
        ENode::FieldAccess(e, _) => fill_expr_positions(e, index),
        ENode::Record(_, fields) => {
            for fa in fields.iter_mut() {
                fill_expr_positions(&mut fa.exp, index);
            }
        },
        ENode::With(e, fields) => {
            fill_expr_positions(e, index);
            for fa in fields.iter_mut() {
                fill_expr_positions(&mut fa.exp, index);
            }
        },
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompOp {
    Eq,
    NotEq,
    Less,
    Greater,
    LessEq,
    GreatEq,
}

impl Display for CompOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompOp::Eq => write!(f, "Eq"),
            CompOp::NotEq => write!(f, "NotEq"),
            CompOp::Less => write!(f, "Less"),
            CompOp::Greater => write!(f, "Greater"),
            CompOp::LessEq => write!(f, "LessEq"),
            CompOp::GreatEq => write!(f, "GreatEq"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArithOp {
    Plus,
    Minus,
    Div,
    Times,
    Mod,
}

impl Display for ArithOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArithOp::Plus => write!(f, "Plus"),
            ArithOp::Minus => write!(f, "Minus"),
            ArithOp::Div => write!(f, "Div"),
            ArithOp::Times => write!(f, "Times"),
            ArithOp::Mod => write!(f, "Mod"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOp {
    And,
    Or,
    Xor,
}

impl Display for LogicalOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogicalOp::And => write!(f, "And"),
            LogicalOp::Or => write!(f, "Or"),
            LogicalOp::Xor => write!(f, "Xor"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Negate,
    Not
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Negate => write!(f, "Negate"),
            UnaryOp::Not => write!(f, "Not"),
        }
    }
}

#[derive(Debug)]
pub struct Program {
    pub stmts : Vec<Stmt>,
    pub ctx : TypeContext,
    /// The name of the source this program was parsed from (file path, or
    /// `"<repl>"`), used when attaching locations to generated MLIR.
    pub source_name : String
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Program({},\n{})", self.source_name, self.stmts.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n"))
    }
}

/// Format a LALRPOP parse error, converting its raw byte offsets to 1-based
/// (line, column) positions via `index`. `T` is the generated token type,
/// which implements `Display`.
fn format_parse_error<L, T, E>(
    error: &lalrpop_util::ParseError<L, T, E>,
    index: &LineIndex,
) -> String
where
    L: Copy + Into<usize>,
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    use lalrpop_util::ParseError;
    let at = |byte: L| {
        let (line, col) = index.line_col(byte.into() as u32);
        format!("{line}:{col}")
    };
    match error {
        ParseError::InvalidToken { location } => {
            format!("parse error at {}: invalid token", at(*location))
        }
        ParseError::UnrecognizedEof { location, expected } => format!(
            "parse error at {}: unexpected end of input (expected {})",
            at(*location),
            expected.join(" or ")
        ),
        ParseError::UnrecognizedToken { token: (start, tok, _), expected } => format!(
            "parse error at {}: unexpected token `{tok}` (expected {})",
            at(*start),
            expected.join(" or ")
        ),
        ParseError::ExtraToken { token: (start, tok, _) } => {
            format!("parse error at {}: extra token `{tok}`", at(*start))
        }
        ParseError::User { error } => format!("parse error: {error}"),
    }
}

impl Program {
    pub fn parse(buf : &str) -> Result<Box<Program>, String> {
        let index = LineIndex::new(buf);
        let mut program = grammar::ProgParser::new()
            .parse(buf)
            .map_err(|e| format_parse_error(&e, &index))?;
        for stmt in program.stmts.iter_mut() {
            fill_stmt_positions(stmt, &index);
        }
        Ok(program)
    }

    pub fn parse_with_prelude(buf : &str) -> Result<Box<Program>, String> {
        let mut program = Self::parse(buf)?;
        let prelude = get_prelude();
        program.stmts.splice(0..0, prelude.iter().cloned());
        Ok(program)
    }
}
