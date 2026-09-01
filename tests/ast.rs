use merlin_lang::ast::*;
use merlin_lang::types::*;


fn parse(src: &str) -> Box<Program> {
    Program::parse(src).unwrap()
}

fn mono(t: Monotype) -> Box<Type> {
    Box::new(Type { t })
}

fn first(p: &Program) -> &Stmt {
    &p.stmts[0]
}

// ---- Literals ----

#[test]
fn int_literal() {
    let p = parse("42;");
    assert_eq!(p.stmts.len(), 1);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42)))))));
}

#[test]
fn negative_int_literal() {
    let p = parse("-7;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Unary(
        UnaryOp::Negate,
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(7))))),
    )))));
}

#[test]
fn float_literal() {
    let p = parse("3.14;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Float(3.14)))))));
}

#[test]
fn negative_float_literal() {
    let p = parse("-2.5;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Unary(
        UnaryOp::Negate,
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Float(2.5))))),
    )))));
}

#[test]
fn bool_true_literal() {
    let p = parse("true;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true)))))));
}

#[test]
fn bool_false_literal() {
    let p = parse("false;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(false)))))));
}

#[test]
fn string_literal() {
    let p = parse(r#""hello";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("hello".to_string())))))));
}

#[test]
fn empty_string_literal() {
    let p = parse(r#""";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("".to_string())))))));
}

#[test]
fn string_escape_newline() {
    let p = parse(r#""a\nb";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("a\nb".to_string())))))));
}

#[test]
fn string_escape_tab() {
    let p = parse(r#""a\tb";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("a\tb".to_string())))))));
}

#[test]
fn string_escape_null() {
    let p = parse(r#""a\0b";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("a\0b".to_string())))))));
}

#[test]
fn string_escape_quote() {
    let p = parse(r#""\"hi\"";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("\"hi\"".to_string())))))));
}

#[test]
fn string_escape_backslash() {
    let p = parse(r#""a\\b";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("a\\b".to_string())))))));
}

#[test]
fn string_escape_hex() {
    let p = parse(r#""\x41\x42";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("AB".to_string())))))));
}

#[test]
fn string_escape_unicode() {
    let p = parse(r#""hi \u{1F600}";"#);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("hi 😀".to_string())))))));
}

#[test]
fn unit_literal() {
    let p = parse("();");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Unit))))));
}

#[test]
fn char_literal() {
    let p = parse("'a';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('a')))))));
}

#[test]
fn char_escape_newline() {
    let p = parse(r"'\n';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('\n')))))));
}

#[test]
fn char_escape_null() {
    let p = parse(r"'\0';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('\0')))))));
}

#[test]
fn char_escape_tab() {
    let p = parse(r"'\t';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('\t')))))));
}

#[test]
fn char_escape_single_quote() {
    let p = parse(r"'\'';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('\'')))))));
}

#[test]
fn char_escape_backslash() {
    let p = parse(r"'\\';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('\\')))))));
}

#[test]
fn char_escape_hex() {
    let p = parse(r"'\x41';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('A')))))));
}

#[test]
fn char_escape_unicode() {
    let p = parse(r"'\u{1F600}';");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Char('😀')))))));
}

// ---- Variables ----

#[test]
fn simple_variable() {
    let p = parse("x;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Variable("x".to_string())))));
}

#[test]
fn underscore_variable() {
    let p = parse("_foo;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Variable("_foo".to_string())))));
}

#[test]
fn alphanumeric_variable() {
    let p = parse("x2y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Variable("x2y".to_string())))));
}

// ---- Application ----

#[test]
fn single_application() {
    let p = parse("f x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("f".to_string()))),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn left_associative_application() {
    let p = parse("f x y;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Application(
                Box::new(Expr::from(ENode::Variable("f".to_string()))),
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
            Box::new(Expr::from(ENode::Variable("y".to_string()))),
        ))))
    );
}

#[test]
fn application_with_literal_arg() {
    let p = parse("f 42;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("f".to_string()))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
        ))))
    );
}

#[test]
fn application_with_parenthesized_expr() {
    let p = parse("f (g x);");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("f".to_string()))),
            Box::new(Expr::from(ENode::Application(
                Box::new(Expr::from(ENode::Variable("g".to_string()))),
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
        ))))
    );
}

// ---- Block expressions ----

#[test]
fn block_only_expression() {
    let p = parse("{ 42 };");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Block(
            vec![],
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
        ))))
    );
}

#[test]
fn block_with_one_let() {
    let p = parse("{ let x = 1; x };");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Block(
            vec![Stmt::from(SNode::Decl(
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
                Box::new(Type { t: Monotype::infer() }),
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
            ))],
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn block_with_multiple_stmts() {
    let p = parse("{ let x = 1; let y = 2; x };");
    let block = match &*first(&p).s {
        SNode::Expr(e) => &*e.e,
        _ => panic!("expected Expr"),
    };
    let ENode::Block(stmts, expr) = block else {
        panic!("expected Block");
    };
    assert_eq!(stmts.len(), 2);
    assert!(matches!(&*stmts[0].s, SNode::Decl(..)));
    assert!(matches!(&*stmts[1].s, SNode::Decl(..)));
    assert_eq!(&*expr.e, &ENode::Variable("x".to_string()));
}

#[test]
fn block_in_let_rhs() {
    let p = parse("let x = { 42 };");
    assert_eq!(
        &*first(&p).s,
        &SNode::Decl(
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
            Box::new(Type { t: Monotype::infer() }),
            Box::new(Expr::from(ENode::Block(
                vec![],
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
            ))),
        )
    );
}

#[test]
fn block_in_if_else_branches() {
    let p = parse("if true then { 1 } else { 2 };");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::IfElse(
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
            Box::new(Expr::from(ENode::Block(
                vec![],
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
            ))),
            Box::new(Expr::from(ENode::Block(
                vec![],
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
            ))),
        ))))
    );
}

#[test]
fn nested_block() {
    let p = parse("{ { 42 } };");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Block(
            vec![],
            Box::new(Expr::from(ENode::Block(
                vec![],
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
            ))),
        ))))
    );
}

#[test]
fn block_with_print() {
    let p = parse(r#"{ print "hi" };"#);
    let block = match &*first(&p).s {
        SNode::Expr(e) => &*e.e,
        _ => panic!("expected Expr"),
    };
    let ENode::Block(stmts, expr) = block else {
        panic!("expected Block");
    };
    assert!(stmts.is_empty());
    assert!(matches!(&*expr.e, ENode::Application(..)));
}

// ---- Abstraction ----

#[test]
fn lambda_without_type_annotation() {
    let p = parse("\\x => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding("x".to_string(), mono(Monotype::infer()))),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn lambda_with_type_annotation() {
    let p = parse("\\(x : int) => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding("x".to_string(), mono(Monotype::int()))),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn nested_lambda() {
    let p = parse("\\x => \\y => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding("x".to_string(), mono(Monotype::infer()))),
            Box::new(Expr::from(ENode::Abstraction(
                Box::new(Binding("y".to_string(), mono(Monotype::infer()))),
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
        ))))
    );
}

#[test]
fn lambda_with_function_type_annotation() {
    let p = parse("\\(f : int => bool) => f;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding("f".to_string(), mono(Monotype::func(vec![Monotype::int(), Monotype::bool()])))),
            Box::new(Expr::from(ENode::Variable("f".to_string()))),
        ))))
    );
}

#[test]
fn lambda_with_enum_type_annotation() {
    let p = parse("\\(x : Option(int)) => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding(
                "x".to_string(),
                mono(Monotype::enum_app("Option".to_string(), vec![Monotype::int()])),
            )),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn lambda_with_multi_arg_enum_type_annotation() {
    let p = parse("\\(x : Result(int, bool)) => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding(
                "x".to_string(),
                mono(Monotype::enum_app(
                    "Result".to_string(),
                    vec![Monotype::int(), Monotype::bool()],
                )),
            )),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn lambda_with_parenthesized_enum_type_annotation() {
    let p = parse("\\(x : Option(list int)) => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding(
                "x".to_string(),
                mono(Monotype::enum_app(
                    "Option".to_string(),
                    vec![Monotype::list(Monotype::int())],
                )),
            )),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn lambda_with_nullary_enum_type_annotation() {
    let p = parse("\\(x : Maybe) => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding(
                "x".to_string(),
                mono(Monotype::enum_app("Maybe".to_string(), vec![])),
            )),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn lambda_with_nested_enum_type_annotation() {
    let p = parse("\\(x : Option(Result(int, bool))) => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding(
                "x".to_string(),
                mono(Monotype::enum_app(
                    "Option".to_string(),
                    vec![Monotype::enum_app(
                        "Result".to_string(),
                        vec![Monotype::int(), Monotype::bool()],
                    )],
                )),
            )),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn enum_constructor_desugars_to_application() {
    let p = parse("Some(42);");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("Some".to_string()))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
        ))))
    );
}

#[test]
fn multi_arg_enum_constructor_desugars_to_nested_application() {
    let p = parse("Pair(1, 2);");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Application(
                Box::new(Expr::from(ENode::Variable("Pair".to_string()))),
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
            ))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
        ))))
    );
}

#[test]
fn enum_constructor_pattern_desugars_to_application() {
    let p = parse("match x | Some(n) => n | None => 0;");
    let SNode::Expr(e) = &*first(&p).s else { panic!("expected Expr") };
    let ENode::Match(_, cases) = &*e.e else { panic!("expected Match") };
    assert_eq!(cases.len(), 2);
    let ENode::Application(f, arg) = &*cases[0].val.e else { panic!("expected Application pattern") };
    assert_eq!(f.e, Box::new(ENode::Variable("Some".to_string())));
    assert_eq!(arg.e, Box::new(ENode::Variable("n".to_string())));
    let ENode::Variable(name) = &*cases[1].val.e else { panic!("expected Variable pattern") };
    assert_eq!(name, "None");
}

// ---- Let-in expression ----

#[test]
fn let_in() {
    let p = parse("let x = 1 in x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Let(
            "x".to_string(),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn let_in_with_complex_body() {
    let p = parse("let x = 1 in let y = 2 in x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Let(
            "x".to_string(),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
            Box::new(Expr::from(ENode::Let(
                "y".to_string(),
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
        ))))
    );
}

// ---- If-else ----

#[test]
fn if_else() {
    let p = parse("if true then 1 else 2;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::IfElse(
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
        ))))
    );
}

#[test]
fn if_else_with_variable_condition() {
    let p = parse("if x then y else z;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::IfElse(
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
            Box::new(Expr::from(ENode::Variable("y".to_string()))),
            Box::new(Expr::from(ENode::Variable("z".to_string()))),
        ))))
    );
}

#[test]
fn nested_if_else() {
    let p = parse("if true then if false then 1 else 2 else 3;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::IfElse(
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
            Box::new(Expr::from(ENode::IfElse(
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(false))))),
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
                Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
            ))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(3))))),
        ))))
    );
}

// ---- Parenthesized expressions ----

#[test]
fn parenthesized_variable() {
    let p = parse("(x);");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Variable("x".to_string())))));
}

#[test]
fn parenthesized_application() {
    let p = parse("(f x);");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("f".to_string()))),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

// ---- Statements ----

#[test]
fn let_decl_without_type() {
    let p = parse("let x = 42;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Decl(
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
            mono(Monotype::infer()),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
        )
    );
}

#[test]
fn let_decl_with_type() {
    let p = parse("let x : int = 42;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Decl(
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
            mono(Monotype::int()),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
        )
    );
}

#[test]
fn let_decl_with_function_type() {
    let p = parse("let f : int => int = \\x => x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Decl(
            Box::new(Expr::from(ENode::Variable("f".to_string()))),
            mono(Monotype::func(vec![Monotype::int(), Monotype::int()])),
            Box::new(Expr::from(ENode::Abstraction(
                Box::new(Binding("x".to_string(), mono(Monotype::infer()))),
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
        )
    );
}

#[test]
fn let_decl_with_bool_type() {
    let p = parse("let b : bool = true;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Decl(
            Box::new(Expr::from(ENode::Variable("b".to_string()))),
            mono(Monotype::bool()),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
        )
    );
}

#[test]
fn print_statement() {
    let p = parse("print x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("print".to_string()))),
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn print_literal() {
    let p = parse("print 42;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Application(
            Box::new(Expr::from(ENode::Variable("print".to_string()))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42))))),
        ))))
    );
}

// ---- Types ----

#[test]
fn simple_type_int() {
    let p = parse("let x : int = 0;");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => assert_eq!(typ.t, Monotype::int()),
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn simple_type_bool() {
    let p = parse("let x : bool = true;");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => assert_eq!(typ.t, Monotype::bool()),
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn simple_type_float() {
    let p = parse("let x : float = 1.0;");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => assert_eq!(typ.t, Monotype::float()),
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn simple_type_str() {
    let p = parse(r#"let x : str = "hi";"#);
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => assert_eq!(typ.t, Monotype::string()),
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn simple_type_unit() {
    let p = parse("let x : () = ();");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => assert_eq!(typ.t, Monotype::unit()),
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn function_type() {
    let p = parse("let f : int => bool = true;");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => {
            assert_eq!(typ.t, Monotype::func(vec![Monotype::int(), Monotype::bool()]))
        }
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn nested_function_type() {
    let p = parse("let f : int => bool => str = true;");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => {
            assert_eq!(
                typ.t,
                Monotype::func(vec![
                    Monotype::int(),
                    Monotype::func(vec![Monotype::bool(), Monotype::string()])
                ])
            )
        }
        other => panic!("expected Decl, got {:?}", other),
    }
}

// ---- Programs ----

#[test]
fn empty_program() {
    let p = parse("");
    assert!(p.stmts.is_empty());
}

#[test]
fn single_statement_no_semicolon() {
    let p = parse("42");
    assert_eq!(p.stmts.len(), 1);
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(42)))))));
}

#[test]
fn multiple_statements() {
    let p = parse("let x = 1; let y = 2;");
    assert_eq!(p.stmts.len(), 2);
    match &*p.stmts[0].s {
        SNode::Decl(name, _, _) => assert_eq!(**name, Expr::from(ENode::Variable("x".to_string()))),
        other => panic!("expected Decl, got {:?}", other),
    }
    match &*p.stmts[1].s {
        SNode::Decl(name, _, _) => assert_eq!(**name, Expr::from(ENode::Variable("y".to_string()))),
        other => panic!("expected Decl, got {:?}", other),
    }
}

#[test]
fn mixed_statement_types() {
    let p = parse("let x = 1; print x; x;");
    assert_eq!(p.stmts.len(), 3);
    assert!(matches!(&*first(&p).s, SNode::Decl(..)));
    assert!(matches!(&*p.stmts[1].s, SNode::Expr(..)));
    assert!(matches!(&*p.stmts[2].s, SNode::Expr(..)));
}

#[test]
fn last_statement_needs_no_semicolon() {
    let p = parse("let x = 1; let y = 2");
    assert_eq!(p.stmts.len(), 2);
}

// ---- Complex / integration ----

#[test]
fn complex_nested_expression() {
    let p = parse("if true then let x = \\(a : int) => a in x 1 else 0;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::IfElse(
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
            Box::new(Expr::from(ENode::Let(
                "x".to_string(),
                Box::new(Expr::from(ENode::Abstraction(
                    Box::new(Binding("a".to_string(), mono(Monotype::int()))),
                    Box::new(Expr::from(ENode::Variable("a".to_string()))),
                ))),
                Box::new(Expr::from(ENode::Application(
                    Box::new(Expr::from(ENode::Variable("x".to_string()))),
                    Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
                ))),
            ))),
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(0))))),
        ))))
    );
}

#[test]
fn identity_function_applied() {
    let p = parse("\\x => x y;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Abstraction(
            Box::new(Binding("x".to_string(), mono(Monotype::infer()))),
            Box::new(Expr::from(ENode::Application(
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
                Box::new(Expr::from(ENode::Variable("y".to_string()))),
            ))),
        ))))
    );
}

#[test]
fn multi_arg_function_type() {
    let p = parse("let f : int => bool => str = 0;");
    match &*first(&p).s {
        SNode::Decl(_, typ, _) => {
            assert_eq!(
                typ.t,
                Monotype::func(vec![
                    Monotype::int(),
                    Monotype::func(vec![Monotype::bool(), Monotype::string()])
                ])
            )
        }
        other => panic!("expected Decl, got {:?}", other),
    }
}

// ---- Unary expressions ---- 

#[test]
fn negate_variable() {
    let p = parse("-x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Unary(
            UnaryOp::Negate,
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn not_variable() {
    let p = parse("!x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Unary(
            UnaryOp::Not,
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
        ))))
    );
}

#[test]
fn not_true() {
    let p = parse("!true;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Unary(
            UnaryOp::Not,
            Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
        ))))
    );
}

#[test]
fn double_negation() {
    let p = parse("--x;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Unary(
            UnaryOp::Negate,
            Box::new(Expr::from(ENode::Unary(
                UnaryOp::Negate,
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
        ))))
    );
}

#[test]
fn negate_precedence_over_mul() {
    let p = parse("-x * y;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Arithmetic(
            ArithOp::Times,
            Box::new(Expr::from(ENode::Unary(
                UnaryOp::Negate,
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
            Box::new(Expr::from(ENode::Variable("y".to_string()))),
        ))))
    );
}

#[test]
fn not_precedence_over_and() {
    let p = parse("!x && y;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Logical(
            LogicalOp::And,
            Box::new(Expr::from(ENode::Unary(
                UnaryOp::Not,
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
            ))),
            Box::new(Expr::from(ENode::Variable("y".to_string()))),
        ))))
    );
}

// ---- Whole program typechecking ----

#[test]
fn typecheck_empty_program() {
    let mut p = parse("");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn resolved_types_annotated_lambda() {
    let mut p = parse("let id = \\(x : int) => x;");
    Program::typecheck(&mut p).unwrap();
    let SNode::Decl(_, _, lambda) = &*p.stmts[0].s else {
        panic!("expected Decl");
    };
    // The abstraction's resolved type, recorded by the post-pass.
    assert_eq!(
        lambda.typ,
        Monotype::func(vec![Monotype::int(), Monotype::int()])
    );
    // Its body `x` resolves to the parameter type.
    let ENode::Abstraction(_, body) = &*lambda.e else {
        panic!("expected Abstraction");
    };
    assert_eq!(body.typ, Monotype::int());
}

#[test]
fn resolved_types_arithmetic() {
    let mut p = parse("let y = 1 + 2;");
    Program::typecheck(&mut p).unwrap();
    let SNode::Decl(_, _, rhs) = &*p.stmts[0].s else {
        panic!("expected Decl");
    };
    assert_eq!(rhs.typ, Monotype::int());
}

#[test]
fn resolved_types_within_statement() {
    // `\x => x` is unannotated, but the use site fixes its type to int.
    let mut p = parse("let apply = \\(f : int => int) => f;");
    Program::typecheck(&mut p).unwrap();
    let SNode::Decl(_, _, lambda) = &*p.stmts[0].s else {
        panic!("expected Decl");
    };
    assert_eq!(
        lambda.typ,
        Monotype::func(vec![
            Monotype::func(vec![Monotype::int(), Monotype::int()]),
            Monotype::func(vec![Monotype::int(), Monotype::int()]),
        ])
    );
}

#[test]
fn typecheck_int_literal() {
    let mut p = parse("42;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_let_decl() {
    let mut p = parse("let x = 42;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_enum_constructor_pattern() {
    let mut p = parse(
        "enum Maybe('a) = Just('a) | Nothing; \
         let get = \\(m : Maybe(int)) => match m | Just(n) => n | Nothing => 0;",
    );
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_let_and_use() {
    let mut p = parse("let x = 42; x;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_let_annotated() {
    let mut p = parse("let x : int = 42; x;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_let_wrong_annotation() {
    let mut p = parse("let x : bool = 42;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_function_application() {
    let mut p = parse("let f = \\x => x; f 42;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_print_string() {
    let mut p = parse(r#"print "hi";"#);
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_print_non_string() {
    let mut p = parse("print 42;");
    assert!(Program::typecheck(&mut p).is_err());
}

// ---- Match exhaustiveness ----

#[test]
fn match_non_exhaustive_int_rejected() {
    let mut p = parse("match 1 | 0 => 1 | 1 => 2;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn match_int_with_catch_all_accepted() {
    let mut p = parse("match 1 | 0 => 1 | x => x;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn match_bool_exhaustive() {
    let mut p = parse("match true | true => 1 | false => 2;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn match_bool_incomplete_rejected() {
    let mut p = parse("match true | true => 1;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn match_list_exhaustive() {
    let mut p = parse("match [1] | [] => 0 | x::xs => 1;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn match_list_incomplete_rejected() {
    let mut p = parse("match [1] | [] => 0;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_undefined_variable() {
    let mut p = parse("x;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_multiple_decls() {
    let mut p = parse("let x = 1; let y = 2; x; y;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_polymorphic_let() {
    let mut p = parse("let id = \\x => x; id 42; id true;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_recursive_let_simple() {
    let src = r"let loop = \(x : int) => if x > 0 then loop 0 else 0;";
    let mut p = parse(src);
    match Program::typecheck(&mut p) {
        Ok(_) => {},
        Err(e) => panic!("type error: {}", e),
    }
}

#[test]
fn typecheck_recursive_let() {
    let src = r"let rec = \(x : int) => if x > 0 then rec (x * 1) else 0;";
    let mut p = parse(src);
    match Program::typecheck(&mut p) {
        Ok(_) => {},
        Err(e) => panic!("type error: {}", e),
    }
}

#[test]
fn typecheck_if_else() {
    let mut p = parse("if true then 1 else 2;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_if_else_non_bool_cond() {
    let mut p = parse("if 1 then 2 else 3;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_if_else_branch_mismatch() {
    let mut p = parse("if true then 1 else true;");
    assert!(Program::typecheck(&mut p).is_err());
}

// ---- Binary expressions ----

#[test]
fn arith_plus() {
    let p = parse("1 + 2;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Arithmetic(
        ArithOp::Plus,
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
    )))));
}

#[test]
fn arith_minus() {
    let p = parse("x - y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Arithmetic(
        ArithOp::Minus,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn arith_times() {
    let p = parse("a * b;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Arithmetic(
        ArithOp::Times,
        Box::new(Expr::from(ENode::Variable("a".to_string()))),
        Box::new(Expr::from(ENode::Variable("b".to_string()))),
    )))));
}

#[test]
fn arith_div() {
    let p = parse("a / b;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Arithmetic(
        ArithOp::Div,
        Box::new(Expr::from(ENode::Variable("a".to_string()))),
        Box::new(Expr::from(ENode::Variable("b".to_string()))),
    )))));
}

#[test]
fn arith_mod() {
    let p = parse("x % y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Arithmetic(
        ArithOp::Mod,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn comp_eq() {
    let p = parse("x == y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Comparison(
        CompOp::Eq,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn comp_not_eq() {
    let p = parse("x != y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Comparison(
        CompOp::NotEq,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn comp_less() {
    let p = parse("1 < 2;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Comparison(
        CompOp::Less,
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))),
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(2))))),
    )))));
}

#[test]
fn comp_greater() {
    let p = parse("x > y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Comparison(
        CompOp::Greater,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn comp_less_eq() {
    let p = parse("x <= y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Comparison(
        CompOp::LessEq,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn comp_great_eq() {
    let p = parse("x >= y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Comparison(
        CompOp::GreatEq,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn logic_and() {
    let p = parse("true && false;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Logical(
        LogicalOp::And,
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(true))))),
        Box::new(Expr::from(ENode::Literal(Box::new(Lit::Bool(false))))),
    )))));
}

#[test]
fn logic_or() {
    let p = parse("x || y;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Logical(
        LogicalOp::Or,
        Box::new(Expr::from(ENode::Variable("x".to_string()))),
        Box::new(Expr::from(ENode::Variable("y".to_string()))),
    )))));
}

#[test]
fn logic_xor() {
    let p = parse("a ^ b;");
    assert_eq!(&*first(&p).s, &SNode::Expr(Box::new(Expr::from(ENode::Logical(
        LogicalOp::Xor,
        Box::new(Expr::from(ENode::Variable("a".to_string()))),
        Box::new(Expr::from(ENode::Variable("b".to_string()))),
    )))));
}

// ---- Unary typechecking ----

#[test]
fn typecheck_negate_int() {
    let mut p = parse("-5;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_negate_float() {
    let mut p = parse("-3.14;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_negate_string_error() {
    let mut p = parse(r#"-"hi";"#);
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_not_bool() {
    let mut p = parse("!true;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_not_int_error() {
    let mut p = parse("!5;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_not_string_error() {
    let mut p = parse(r#"!"hi";"#);
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_negate_in_let() {
    let mut p = parse("let x : int = -5; x;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_not_in_if_cond() {
    let mut p = parse("if !false then 1 else 2;");
    assert!(Program::typecheck(&mut p).is_ok());
}

// ---- Error cases ----

#[test]
fn empty_input_is_valid() {
    let p = parse("");
    assert!(p.stmts.is_empty());
}

#[test]
fn syntax_error_returns_err() {
    let result = Program::parse("===");
    assert!(result.is_err());
}

#[test]
fn incomplete_let_returns_err() {
    let result = Program::parse("let");
    assert!(result.is_err());
}

#[test]
fn unmatched_paren_returns_err() {
    let result = Program::parse("(x;");
    assert!(result.is_err());
}

// ---- Positions ----

#[test]
fn stmt_positions_filled() {
    let p = parse("let x = 1;\nx + 2;");
    assert_eq!((p.stmts[0].pos.start_line, p.stmts[0].pos.start_col), (1, 1));
    assert_eq!((p.stmts[0].pos.end_line, p.stmts[0].pos.end_col), (1, 10));
    assert_eq!((p.stmts[1].pos.start_line, p.stmts[1].pos.start_col), (2, 1));
    assert_eq!((p.stmts[1].pos.end_line, p.stmts[1].pos.end_col), (2, 6));
}

#[test]
fn nested_expr_positions_filled() {
    let p = parse("1 + 2 * 3;");
    let SNode::Expr(e) = &*first(&p).s else { panic!("expected Expr") };
    assert_eq!((e.pos.start_col, e.pos.end_col), (1, 10));
    let ENode::Arithmetic(_, a, b) = &*e.e else { panic!("expected Arith") };
    assert_eq!((a.pos.start_col, a.pos.end_col), (1, 2));       // `1`
    assert_eq!((b.pos.start_col, b.pos.end_col), (5, 10));      // `2 * 3`
    let ENode::Arithmetic(_, b1, b2) = &*b.e else { panic!("expected Arith") };
    assert_eq!((b1.pos.start_col, b1.pos.end_col), (5, 6));     // `2`
    assert_eq!((b2.pos.start_col, b2.pos.end_col), (9, 10));    // `3`
}

#[test]
fn typecheck_error_carries_position() {
    let mut p = parse("let x = 1;\nlet y = true;\nx + y;");
    let err = Program::typecheck(&mut p).unwrap_err();
    let pos = err.pos.expect("typecheck error should carry a position");
    assert_eq!((pos.start_line, pos.start_col), (3, 1));
}

// ---- Records: parsing ----

fn fa(name: &str, e: Box<Expr>) -> FieldAssn {
    FieldAssn { field: name.to_string(), exp: e }
}

#[test]
fn record_declaration() {
    let p = parse("record Foo = { bar: int, baz: str };");
    assert_eq!(
        &*first(&p).s,
        &SNode::TypeDecl(
            TypeHeader { n: "Foo".to_string(), tvars: vec![] },
            Box::new(TypeDec::Record(vec![
                Binding("bar".to_string(), mono(Monotype::int())),
                Binding("baz".to_string(), mono(Monotype::string())),
            ]))
        )
    );
}

#[test]
fn record_declaration_with_type_var() {
    let p = parse("record Poly('a) = { bar: 'a };");
    assert_eq!(
        &*first(&p).s,
        &SNode::TypeDecl(
            TypeHeader { n: "Poly".to_string(), tvars: vec!["a".to_string()] },
            Box::new(TypeDec::Record(vec![
                Binding("bar".to_string(), mono(Monotype::var("a".to_string()))),
            ]))
        )
    );
}

#[test]
fn record_literal() {
    let p = parse(r#"Foo { bar: 1, baz: "x" };"#);
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::Record(Some("Foo".to_string()), vec![
            fa("bar", Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1)))))),
            fa("baz", Box::new(Expr::from(ENode::Literal(Box::new(Lit::Str("x".to_string())))))),
        ]))))
    );
}

#[test]
fn field_access() {
    let p = parse("x.bar;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::FieldAccess(
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
            "bar".to_string(),
        ))))
    );
}

#[test]
fn nested_field_access() {
    let p = parse("x.a.b;");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::FieldAccess(
            Box::new(Expr::from(ENode::FieldAccess(
                Box::new(Expr::from(ENode::Variable("x".to_string()))),
                "a".to_string(),
            ))),
            "b".to_string(),
        ))))
    );
}

#[test]
fn with_update() {
    let p = parse("x with { bar: 1 };");
    assert_eq!(
        &*first(&p).s,
        &SNode::Expr(Box::new(Expr::from(ENode::With(
            Box::new(Expr::from(ENode::Variable("x".to_string()))),
            vec![fa("bar", Box::new(Expr::from(ENode::Literal(Box::new(Lit::Int(1))))))],
        ))))
    );
}

#[test]
fn record_pattern_in_match() {
    let p = parse("match x | Foo { bar: n, baz: opt } => n | _ => 0;");
    let SNode::Expr(e) = &*first(&p).s else { panic!("expected Expr") };
    let ENode::Match(_, cases) = &*e.e else { panic!("expected Match") };
    assert_eq!(cases.len(), 2);
    let ENode::Record(_, fields) = &*cases[0].val.e else { panic!("expected Record pattern") };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].field, "bar");
    assert_eq!(fields[1].field, "baz");
}

// ---- Records: typechecking ----

#[test]
fn typecheck_record_decl_and_literal() {
    let mut p = parse("record Foo = { bar: int }; let x = Foo { bar: 1 }; x.bar;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_record_field_access() {
    let mut p = parse(r#"record Foo = { bar: int, baz: str }; let x = Foo { bar: 1, baz: "hi" }; x.bar;"#);
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_record_with_update() {
    let mut p = parse(r#"record Foo = { bar: int, baz: str }; let x = Foo { bar: 1, baz: "hi" }; let y = x with { bar: 2 }; y.bar;"#);
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_record_pattern() {
    let mut p = parse(r#"record Foo = { bar: int, baz: str }; let x = Foo { bar: 1, baz: "hi" }; match x | Foo { bar: n, baz: opt } => n | _ => 0;"#);
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_field_access_on_variable() {
    // Structural: `\x => x.bar` is typable without knowing the record name.
    let mut p = parse("\\x => x.bar;");
    assert!(Program::typecheck(&mut p).is_ok());
}

#[test]
fn typecheck_missing_field_error() {
    let mut p = parse("record Foo = { bar: int }; let x = Foo { bar: 1 }; x.baz;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_field_access_on_int_error() {
    let mut p = parse("let x = 1; x.bar;");
    assert!(Program::typecheck(&mut p).is_err());
}

#[test]
fn typecheck_with_on_int_error() {
    let mut p = parse("let x = 1; x with { bar: 2 };");
    assert!(Program::typecheck(&mut p).is_err());
}
