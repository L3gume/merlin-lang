/**
 * Treesitter grammar for Merlin, mirroring the LALR grammar in
 * src/grammar.lalrpop.
 *
 * The precedence cascade (lowest to highest) matches grammar.lalrpop:
 *   ConsExpr < LogExpr < CmpExpr < AddExpr < MulExpr < UnaryExpr < AppExpr < Atom
 * `::` is right-associative and its left operand is restricted to
 * unary/app/atom/block expressions, exactly as in the LALR grammar.
 */

const PREC = {
  TYPE_ARROW: 0,
  LOG: 1,
  CMP: 2,
  ADD: 3,
  MUL: 4,
  UNARY: 5,
  APP: 6,
  CONS: 7,
};

export default grammar({
  name: 'merlin',

  word: $ => $.identifier,

  extras: $ => [
    /[\s\uFEFF\u00A0]/,
    $.comment,
  ],

  rules: {
    // ------------------------------------------------------------
    // Program
    // ------------------------------------------------------------

    // Prog: SemiCol<Stmt> — statements separated by `;`, optional trailing.
    source_file: $ => seq(
      repeat(seq($._statement, ';')),
      optional($._statement),
    ),

    comment: $ => token(seq('#', /.*/)),

    // ------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------

    // Stmt: TypeDecl | BlockStmt  (BlockStmt carries the bare-Expr statements)
    _statement: $ => choice(
      $._type_declaration,
      $._block_statement,
    ),

    // BlockStmt: `let` decls and bare expression statements. Expression
    // statements must live here (and not in a top-level `_statement`
    // alternative) to mirror the LALR grammar, which keeps them only in
    // `BlockStmt` so its lookahead sets stay LR(1)-clean.
    _block_statement: $ => choice(
      $.let_statement,
      $._expression,
    ),

    let_statement: $ => choice(
      seq(
        'let',
        field('name', $.identifier),
        ':',
        field('type', $._type),
        '=',
        field('value', $._expression),
      ),
      seq(
        'let',
        field('name', $.identifier),
        '=',
        field('value', $._expression),
      ),
    ),

    // ------------------------------------------------------------
    // Type declarations
    // ------------------------------------------------------------

    _type_declaration: $ => choice(
      $.type_declaration,
      $.enum_declaration,
    ),

    // TypeDecl: "type" TypeHeader Type | "enum " TypeHeader Pipe<Variant>
    type_declaration: $ => seq(
      'type',
      field('header', $.type_header),
      field('type', $._type),
    ),

    enum_declaration: $ => seq(
      'enum',
      field('header', $.type_header),
      field('variants', $.variants),
    ),

    // TypeHeader: Name OptTypeVars "="
    type_header: $ => seq(
      field('name', $.identifier),
      optional($.type_parameter_list),
      '=',
    ),

    // OptTypeVars: ("(" CommaOne<TypeVar> ")")?
    type_parameter_list: $ => seq(
      '(',
      comma_one($, $.type_var),
      ')',
    ),

    // TypeVar: "'" Name
    type_var: $ => seq("'", $.identifier),

    // Pipe<Variant> (at least one variant)
    variants: $ => seq(
      $.variant,
      repeat(seq('|', $.variant)),
    ),

    // Variant: Name | Name "(" CommaOne<Type> ")"
    variant: $ => seq(
      field('name', $.identifier),
      optional(seq('(', comma_one($, $._type), ')')),
    ),

    // ------------------------------------------------------------
    // Types
    // ------------------------------------------------------------

    // Type: TypeBase | TypeBase "=>" Type   (right-associative)
    _type: $ => choice(
      prec.right(PREC.TYPE_ARROW, seq($._type_base, '=>', $._type)),
      $._type_base,
    ),

    // TypeBase: BuiltinType | AppType | TypeVar | "list" TypeBase | "(" Type ")"
    _type_base: $ => choice(
      $.builtin_type,
      $.app_type,
      $.type_var,
      $.list_type,
      $.parenthesized_type,
    ),

    builtin_type: $ => choice(
      'int',
      'bool',
      'float',
      'str',
      'char',
      '()',
    ),

    // AppType: Name | AppType AppArg   (greedy, left-associative)
    app_type: $ => choice(
      $.identifier,
      prec.left(PREC.APP, seq(
        field('constructor', $.app_type),
        field('arg', $.type_arg),
      )),
    ),

    // AppArg: BuiltinType | Name | TypeVar | "list" AppArg | "(" Type ")"
    type_arg: $ => choice(
      $.builtin_type,
      $.identifier,
      $.type_var,
      seq('list', $.type_arg),
      $.parenthesized_type,
    ),

    list_type: $ => seq('list', $._type_base),

    parenthesized_type: $ => seq('(', $._type, ')'),

    // ------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------

    // Expr: Abs_Expr | let-in | if-then-else | match | ConsExpr
    _expression: $ => choice(
      $.abstraction,
      $.let_expression,
      $.if_expression,
      $.match_expression,
      $.cons_expression,
    ),

    // Abs_Expr: "\" Binding+ "=>" Expr
    abstraction: $ => seq(
      '\\',
      repeat1(field('binding', $.binding)),
      '=>',
      field('body', $._expression),
    ),

    // Binding: Name | "(" Name ":" Type ")"
    binding: $ => choice(
      $.identifier,
      seq(
        '(',
        field('name', $.identifier),
        ':',
        field('type', $._type),
        ')',
      ),
    ),

    let_expression: $ => seq(
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._expression),
      'in',
      field('body', $._expression),
    ),

    if_expression: $ => seq(
      'if',
      field('condition', $._expression),
      'then',
      field('consequence', $._expression),
      'else',
      field('alternative', $._expression),
    ),

    // Match: "match" ConsExpr "|" Pipe<MatchCase>
    match_expression: $ => seq(
      'match',
      field('value', $.cons_expression),
      '|',
      field('cases', $.match_cases),
    ),

    match_cases: $ => seq(
      $.match_case,
      repeat(seq('|', $.match_case)),
    ),

    // MatchCase: ConsExpr "=>" ConsExpr
    match_case: $ => seq(
      field('value', $.cons_expression),
      '=>',
      field('result', $.cons_expression),
    ),

    // ConsExpr: ExprBase | ConsOperand "::" ConsExpr   (right-associative)
    cons_expression: $ => choice(
      $._cons_base,
      prec.right(PREC.CONS, seq($._cons_operand, '::', $.cons_expression)),
    ),

    // ExprBase: LogExpr | "{" BlkBody Expr "}"
    _cons_base: $ => choice(
      $.logical_expression,
      $.block_expression,
    ),

    // ConsOperand: UnaryExpr | "{" BlkBody Expr "}"
    _cons_operand: $ => choice(
      $.unary_expression,
      $.block_expression,
    ),

    // --- LogExpr: ||, &&, ^ (lowest precedence) ---
    logical_expression: $ => choice(
      prec.left(PREC.LOG, seq(
        field('left', $.logical_expression),
        field('operator', choice('||', '&&', '^')),
        field('right', $.comparison_expression),
      )),
      $.comparison_expression,
    ),

    // --- CmpExpr: ==, !=, <, >, <=, >= ---
    comparison_expression: $ => choice(
      prec.left(PREC.CMP, seq(
        field('left', $.comparison_expression),
        field('operator', choice('==', '!=', '<', '>', '<=', '>=')),
        field('right', $.additive_expression),
      )),
      $.additive_expression,
    ),

    // --- AddExpr: +, - ---
    additive_expression: $ => choice(
      prec.left(PREC.ADD, seq(
        field('left', $.additive_expression),
        field('operator', choice('+', '-')),
        field('right', $.multiplicative_expression),
      )),
      $.multiplicative_expression,
    ),

    // --- MulExpr: *, /, % ---
    multiplicative_expression: $ => choice(
      prec.left(PREC.MUL, seq(
        field('left', $.multiplicative_expression),
        field('operator', choice('*', '/', '%')),
        field('right', $.unary_expression),
      )),
      $.unary_expression,
    ),

    // --- UnaryExpr: prefix -, ! ---
    unary_expression: $ => choice(
      prec(PREC.UNARY, seq('-', field('argument', $.unary_expression))),
      prec(PREC.UNARY, seq('!', field('argument', $.unary_expression))),
      $.application_expression,
    ),

    // --- AppExpr: function application (highest binary precedence) ---
    application_expression: $ => choice(
      prec.left(PREC.APP, seq(
        field('function', $.application_expression),
        field('argument', $._atom),
      )),
      $._atom,
    ),

    // "{" BlkBody Expr "}" — statements are `;`-terminated, and the block's
    // value is the final bare expression (a `;` before `}` is a parse error,
    // matching the LALR grammar).
    block_expression: $ => seq(
      '{',
      repeat(seq($._block_statement, ';')),
      $._expression,
      '}',
    ),

    // Atom: Name | Literal | "[" "]" | "[" CommaOne<Expr> "]" | "(" Expr ")"
    _atom: $ => choice(
      $.variable,
      $._literal,
      $.list_literal,
      $.parenthesized_expression,
    ),

    variable: $ => $.identifier,

    list_literal: $ => choice(
      seq('[', ']'),
      seq('[', comma_one($, $._expression), ']'),
    ),

    parenthesized_expression: $ => seq('(', $._expression, ')'),

    // ------------------------------------------------------------
    // Literals
    // ------------------------------------------------------------

    _literal: $ => choice(
      $.float_literal,
      $.integer_literal,
      $.string_literal,
      $.boolean_literal,
      $.unit,
    ),

    float_literal: $ => token(/[0-9]+\.[0-9]+/),
    integer_literal: $ => token(/[0-9]+/),
    string_literal: $ => token(/"[^"]*"/),
    boolean_literal: $ => choice('true', 'false'),
    unit: $ => seq('(', ')'),

    // ------------------------------------------------------------
    // Tokens
    // ------------------------------------------------------------

    // Name: [a-zA-Z_][a-zA-Z0-9_]*
    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,
  },
});

function comma_one($, rule) {
  return seq(rule, repeat(seq(',', rule)));
}
