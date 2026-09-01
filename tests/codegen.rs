//! End-to-end code generation tests: parse -> typecheck -> lower -> JIT.
//!
//! These exercise the MLIR codegen backend (including the LLVM JIT), so they
//! cover the closure/application machinery that the AST and type-checker unit
//! tests cannot reach.

#![cfg(feature = "codegen")]

use merlin_lang::ast;
use merlin_lang::codegen;

/// Parse (optionally with the prelude prepended), typecheck, lower, and JIT
/// run `source`, returning the value of the trailing expression statement.
fn run(source: &str, prelude: bool) -> Result<codegen::ExecutionResult, String> {
    let mut prog = if prelude {
        ast::Program::parse_with_prelude(source)?
    } else {
        ast::Program::parse(source)?
    };
    ast::Program::typecheck(&mut prog).map_err(|e| e.to_string())?;
    let context = codegen::new_context();
    let mut module = codegen::lower(&prog, &context)?;
    codegen::execute(&mut module, None)
}

fn expect_int(source: &str) -> i32 {
    match run(source, false) {
        Ok(codegen::ExecutionResult::Int(n)) => n,
        other => panic!("expected Int, got {other:?}"),
    }
}

fn expect_int_prelude(source: &str) -> i32 {
    match run(source, true) {
        Ok(codegen::ExecutionResult::Int(n)) => n,
        other => panic!("expected Int, got {other:?}"),
    }
}

fn expect_list(source: &str) -> Vec<codegen::ExecutionResult> {
    match run(source, true) {
        Ok(codegen::ExecutionResult::List(items)) => items,
        other => panic!("expected List, got {other:?}"),
    }
}

fn expect_bool(source: &str) -> bool {
    match run(source, false) {
        Ok(codegen::ExecutionResult::Bool(b)) => b,
        other => panic!("expected Bool, got {other:?}"),
    }
}

fn expect_bool_prelude(source: &str) -> bool {
    match run(source, true) {
        Ok(codegen::ExecutionResult::Bool(b)) => b,
        other => panic!("expected Bool, got {other:?}"),
    }
}

// ----------------------------------------------------------------------------
// Basic expressions
// ----------------------------------------------------------------------------

#[test]
fn int_arithmetic() {
    assert_eq!(expect_int("1 + 2 * 3;"), 7);
}

#[test]
fn nested_arithmetic() {
    assert_eq!(expect_int("(1 + 2) * (3 + 4);"), 21);
}

#[test]
fn if_else() {
    assert_eq!(expect_int("if true then 1 else 2;"), 1);
    assert_eq!(expect_int("if false then 1 else 2;"), 2);
}

#[test]
fn recursion() {
    assert_eq!(
        expect_int(
            "let fib = \\(n : int) => match n | 0 => 1 | 1 => 1 | x => fib (x - 1) + fib (x - 2);
             fib 10;"
        ),
        89
    );
}

// ----------------------------------------------------------------------------
// Inline lambdas
// ----------------------------------------------------------------------------

#[test]
fn inline_lambda_full_application() {
    assert_eq!(expect_int("(\\x y => x + y) 3 4;"), 7);
}

#[test]
fn inline_curried_lambda_three_params() {
    assert_eq!(expect_int("(\\x y z => x * y + z) 2 3 4;"), 10);
}

#[test]
fn inline_lambda_in_application_argument() {
    // Passing a curried inline lambda as an argument used to produce a
    // single-parameter closure and fail verification.
    assert_eq!(expect_int_prelude("lfold (\\x y => x + y) 0 [1,2,3,4,5];"), 15);
}

#[test]
fn inline_cons_lambda_as_argument() {
    assert_eq!(
        expect_list("lfold (\\acc x => x::acc) [] [1,2,3];"),
        vec![
            codegen::ExecutionResult::Int(3),
            codegen::ExecutionResult::Int(2),
            codegen::ExecutionResult::Int(1),
        ]
    );
}

#[test]
fn map_with_inline_lambda() {
    assert_eq!(
        expect_list("map (\\x => x * 2) [1,2,3];"),
        vec![
            codegen::ExecutionResult::Int(2),
            codegen::ExecutionResult::Int(4),
            codegen::ExecutionResult::Int(6),
        ]
    );
}

#[test]
fn filter_with_inline_lambda() {
    assert_eq!(
        expect_list("filter (\\x => x > 2) [1,2,3,4];"),
        vec![
            codegen::ExecutionResult::Int(3),
            codegen::ExecutionResult::Int(4),
        ]
    );
}

// ----------------------------------------------------------------------------
// Partial application
// ----------------------------------------------------------------------------

#[test]
fn partial_application_of_named_lambda() {
    assert_eq!(
        expect_int("let add = \\x y => x + y; let inc = add 1; inc 41;"),
        42
    );
}

#[test]
fn partial_application_in_expression() {
    // `add 1` is partially applied and passed directly as an argument.
    assert_eq!(
        expect_list("let add = \\x y => x + y; map (add 1) [10,20,30];"),
        vec![
            codegen::ExecutionResult::Int(11),
            codegen::ExecutionResult::Int(21),
            codegen::ExecutionResult::Int(31),
        ]
    );
}

#[test]
fn partial_application_two_steps() {
    assert_eq!(
        expect_int(
            "let add3 = \\x y z => x + y + z;
             let add_two = add3 1;
             let inc = add_two 2;
             inc 39;"
        ),
        42
    );
}

#[test]
fn partial_application_then_apply() {
    assert_eq!(
        expect_int("let lfold = \\fn acc ls => match ls | [] => acc | x::xs => lfold fn (fn acc x) xs; lfold (\\x y => x + y) 0 [1,2,3];"),
        6
    );
}

#[test]
fn partial_application_via_inlineable_let() {
    assert_eq!(
        expect_int_prelude("let sum = lfold (\\x y => x + y) 0; sum [1,2,3,4];"),
        10
    );
}

// ----------------------------------------------------------------------------
// Prelude functions
// ----------------------------------------------------------------------------

#[test]
fn prelude_map() {
    assert_eq!(
        expect_list("map (\\x => x * 3) [1,2];"),
        vec![
            codegen::ExecutionResult::Int(3),
            codegen::ExecutionResult::Int(6),
        ]
    );
}

#[test]
fn prelude_lfold_sum() {
    assert_eq!(expect_int_prelude("lfold (\\a b => a + b) 0 [1,2,3,4,5,6];"), 21);
}

#[test]
fn prelude_filter_odd() {
    assert_eq!(
        expect_list("filter odd [1,2,3,4,5];"),
        vec![
            codegen::ExecutionResult::Int(1),
            codegen::ExecutionResult::Int(3),
            codegen::ExecutionResult::Int(5),
        ]
    );
}

#[test]
fn prelude_boolean_logic() {
    assert!(expect_bool_prelude("even 4;"));
    assert!(!expect_bool_prelude("odd 4;"));
}

// ----------------------------------------------------------------------------
// Builtin functions
// ----------------------------------------------------------------------------

#[test]
fn print_builtin_runs() {
    // `print "hi";` is an application of the seeded `print : str -> unit`
    // builtin; it must lower through the ordinary application path and run.
    match run(r#"print "hello";"#, false) {
        Ok(codegen::ExecutionResult::Unit) => {}
        other => panic!("expected Unit, got {other:?}"),
    }
}

#[test]
fn println_builtin_runs() {
    // `println` prints a string and a trailing newline via `@puts`.
    match run(r#"println "hello";"#, false) {
        Ok(codegen::ExecutionResult::Unit) => {}
        other => panic!("expected Unit, got {other:?}"),
    }
}

#[test]
fn print_is_first_class() {
    // `map print xs` passes the builtin as a function value.
    match run(r#"map print ["a", "b", "c"];"#, true) {
        Ok(codegen::ExecutionResult::List(_)) => {}
        other => panic!("expected List, got {other:?}"),
    }
}

#[test]
fn print_requires_string_argument() {
    // `print : str -> unit`; applying it to an `int` is a type error.
    assert!(run("print 42;", false).is_err());
}

#[test]
fn itostr_builtin_runs() {
    // `itostr` formats an int into a heap string via `@sprintf`.
    match run(r#"itostr 42;"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "42"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn itostr_handles_negatives() {
    match run(r#"let x = 0 - 42; itostr x;"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "-42"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn ftostr_builtin_runs() {
    // `ftostr` widens the float to f64 and formats it via `@sprintf`.
    match run("ftostr 3.14;", false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "3.14"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn btostr_builtin_runs() {
    // `btostr` selects between the static "true"/"false" globals.
    match run("btostr true;", false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "true"),
        other => panic!("expected String, got {other:?}"),
    }
    match run("btostr false;", false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "false"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn strtoi_builtin_runs() {
    match run(r#"strtoi "42";"#, false) {
        Ok(codegen::ExecutionResult::Int(n)) => assert_eq!(n, 42),
        other => panic!("expected Int, got {other:?}"),
    }
}

#[test]
fn strtof_builtin_runs() {
    match run(r#"strtof "1.5";"#, false) {
        Ok(codegen::ExecutionResult::Float(n)) => assert_eq!(n, 1.5),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn strtob_builtin_runs() {
    match run(r#"strtob "true";"#, false) {
        Ok(codegen::ExecutionResult::Bool(b)) => assert!(b),
        other => panic!("expected Bool, got {other:?}"),
    }
    match run(r#"strtob "false";"#, false) {
        Ok(codegen::ExecutionResult::Bool(b)) => assert!(!b),
        other => panic!("expected Bool, got {other:?}"),
    }
}

#[test]
fn itof_builtin_runs() {
    match run("itof 3;", false) {
        Ok(codegen::ExecutionResult::Float(n)) => assert_eq!(n, 3.0),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn char_literal_runs() {
    // A char literal is its Unicode code point, lowered as an `i32` and read
    // back as a `char`.
    match run("'a';", false) {
        Ok(codegen::ExecutionResult::Char(c)) => assert_eq!(c, 'a'),
        other => panic!("expected Char, got {other:?}"),
    }
    match run(r"'\n';", false) {
        Ok(codegen::ExecutionResult::Char(c)) => assert_eq!(c, '\n'),
        other => panic!("expected Char, got {other:?}"),
    }
}

#[test]
fn escaped_string_literal_runs() {
    // Escape sequences are decoded at parse time, so the runtime string
    // contains the real newline.
    match run(r#""a\nb";"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "a\nb"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn string_concat_basic() {
    match run(r#""hello" + " world";"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "hello world"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn string_concat_empty_lhs() {
    match run(r#"let empty = ""; empty + "world";"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "world"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn string_concat_empty_rhs() {
    match run(r#"let empty = ""; "hello" + empty;"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, "hello"),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn string_concat_both_empty() {
    match run(r#"let e = ""; e + e;"#, false) {
        Ok(codegen::ExecutionResult::String(s)) => assert_eq!(s, ""),
        other => panic!("expected String, got {other:?}"),
    }
}

#[test]
fn ftoi_builtin_runs() {
    // `ftoi` truncates toward zero.
    match run("ftoi 3.9;", false) {
        Ok(codegen::ExecutionResult::Int(n)) => assert_eq!(n, 3),
        other => panic!("expected Int, got {other:?}"),
    }
}

#[test]
fn readin_lowers() {
    // `readin` reads stdin at runtime, so just check it lowers cleanly.
    let mut prog = ast::Program::parse("readin ();").unwrap();
    ast::Program::typecheck(&mut prog).unwrap();
    let context = codegen::new_context();
    let module = codegen::lower(&prog, &context).unwrap();
    assert!(module.dump().contains("@readin"));
}

#[test]
fn codegen_ops_carry_source_locations() {
    let mut prog = ast::Program::parse("let x = 1;\nx + 2;").unwrap();
    prog.source_name = "test.mln".to_string();
    ast::Program::typecheck(&mut prog).unwrap();
    let context = codegen::new_context();
    let module = codegen::lower(&prog, &context).unwrap();
    let dump = module.dump();
    // The `1` literal and the `x + 2` statement must carry real locations.
    assert!(dump.contains("test.mln:1:9"), "missing stmt-1 loc in:\n{dump}");
    assert!(dump.contains("test.mln:2:1"), "missing stmt-2 loc in:\n{dump}");
}

// ----------------------------------------------------------------------------
// Tail call optimization
// ----------------------------------------------------------------------------

#[test]
fn tail_recursion_if_else_deep() {
    // 1M-deep self tail recursion: a stack overflow without TCO.
    assert_eq!(
        expect_int("let count = \\n acc => if n == 0 then acc else count (n - 1) (acc + 1); count 1000000 0;"),
        1000000
    );
}

#[test]
fn tail_recursion_through_match_deep() {
    // The self call sits in a match branch in tail position.
    assert_eq!(
        expect_int("let down = \\(n : int) => match n | 0 => 0 | x => down (x - 1); down 1000000;"),
        0
    );
}

#[test]
fn tail_recursion_through_let_and_block() {
    // The self call sits in a block tail in tail position.
    assert_eq!(
        expect_int(
            "let down = \\(n : int) => if n == 0 then 0 else { let m = n - 1; down m }; down 1000000;"
        ),
        0
    );
}

#[test]
fn tail_recursive_local_loop() {
    // The tl_fib shape from programs/fib.mln: a let-bound loop whose self
    // call is in tail position of the `else` branch.
    assert_eq!(
        expect_int(
            "let tl_fib = \\(n : int) =>
                let loop = \\i a b =>
                    if i == n then a
                    else loop (i + 1) (b) (a + b)
                in loop 0 1 1;
             tl_fib 10;"
        ),
        89
    );
}

#[test]
fn tail_recursion_with_capture() {
    // `step` captures `start`; the backedge passes the capture through.
    assert_eq!(
        expect_int(
            "let make = \\(start : int) =>
                let step = \\n acc => if n == 0 then acc else step (n - 1) (acc + start)
                in step 2 0;
             make 5;"
        ),
        10
    );
}

#[test]
fn shadowed_self_name_is_not_a_self_tail_call() {
    // The inner `f` shadows the recursive binding; the tail-position call is
    // to the inner `f`, so no backedge may be emitted.
    assert_eq!(
        expect_int(
            "let f = \\(n : int) => if n == 0 then 0 else (let f = \\(x : int) => 99 in f (n - 1)); f 5;"
        ),
        99
    );
}

#[test]
fn non_tail_recursion_still_works() {
    // `f (x - 1) + 1` is not in tail position and stays a real call.
    assert_eq!(
        expect_int("let depth = \\(n : int) => if n == 0 then 0 else depth (n - 1) + 1; depth 1000;"),
        1000
    );
}

// ----------------------------------------------------------------------------
// Structural equality (== / !=) on lists, enums, and records
// ----------------------------------------------------------------------------

#[test]
fn list_equality() {
    assert!(expect_bool("[1,2,3] == [1,2,3];"));
    assert!(!expect_bool("[1,2] == [1,2,3];"));
    assert!(!expect_bool("[1,2,3] != [1,2,3];"));
    assert!(expect_bool("[1,2] != [1,2,3];"));
    assert!(expect_bool("[] == [];"));
    assert!(!expect_bool("[1,2] == [1,3];"));
}

#[test]
fn nested_list_equality() {
    assert!(expect_bool("[[1],[2,3]] == [[1],[2,3]];"));
    assert!(!expect_bool("[[1],[2,3]] == [[1],[2]];"));
    assert!(expect_bool("[[],[1]] != [[]];"));
}

#[test]
fn enum_equality() {
    assert!(expect_bool_prelude("Some 5 == Some 5;"));
    assert!(!expect_bool_prelude("Some 5 == None;"));
    assert!(expect_bool_prelude("Some 5 != None;"));
    assert!(expect_bool_prelude("None == None;"));
    assert!(expect_bool_prelude("Some 5 != Some 6;"));
}

#[test]
fn record_equality() {
    assert!(expect_bool(
        "record Point = { x: int, y: int };
         let p1 = Point { x: 1, y: 2 };
         let p2 = Point { x: 1, y: 2 };
         p1 == p2;"
    ));
    assert!(!expect_bool(
        "record Point = { x: int, y: int };
         let p1 = Point { x: 1, y: 2 };
         let p2 = Point { x: 3, y: 2 };
         p1 == p2;"
    ));
}

#[test]
fn record_equality_with_string_field() {
    assert!(expect_bool(
        "record P = { x: int, s: str };
         let p1 = P { x: 1, s: \"a\" };
         let p2 = P { x: 1, s: \"a\" };
         p1 == p2;"
    ));
    assert!(!expect_bool(
        "record P = { x: int, s: str };
         let p1 = P { x: 1, s: \"a\" };
         let p2 = P { x: 1, s: \"b\" };
         p1 == p2;"
    ));
}

#[test]
fn record_equality_with_nested_option() {
    assert!(expect_bool_prelude(
        "record P = { x: int, o: Option str };
         let p1 = P { x: 1, o: Some \"hi\" };
         let p2 = P { x: 1, o: Some \"hi\" };
         p1 == p2;"
    ));
    assert!(!expect_bool_prelude(
        "record P = { x: int, o: Option str };
         let p1 = P { x: 1, o: Some \"hi\" };
         let p2 = P { x: 1, o: None };
         p1 == p2;"
    ));
}

#[test]
fn recursive_enum_equality() {
    // A self-referential enum must not infinitely recurse during codegen; the
    // equality helper is cached before its body is lowered.
    assert!(expect_bool(
        "enum Nat = Zero | Succ(Nat);
         let two = Succ (Succ Zero);
         two == Succ (Succ Zero);"
    ));
    assert!(!expect_bool(
        "enum Nat = Zero | Succ(Nat);
         let two = Succ (Succ Zero);
         two == Succ Zero;"
    ));
}

#[test]
fn list_of_options_equality() {
    assert!(expect_bool_prelude("[Some 1, None] == [Some 1, None];"));
    assert!(!expect_bool_prelude("[Some 1, None] == [Some 1, Some 2];"));
}

// ----------------------------------------------------------------------------
// n-ary enum constructors and patterns
// ----------------------------------------------------------------------------

#[test]
fn n_ary_enum_construct_and_match() {
    assert_eq!(
        expect_int(
            "enum Trio = T(int, int, int);
             let sum = \\t => match t | T a b c => a + b + c | _ => 0;
             sum (T 1 2 3);"
        ),
        6
    );
}

#[test]
fn n_ary_enum_pattern_with_literal_field() {
    assert_eq!(
        expect_int(
            "enum Trio = T(int, int, int);
             let mid = \\t => match t | T a 2 c => a + c | _ => 0;
             mid (T 1 2 3);"
        ),
        4
    );
}

#[test]
fn n_ary_enum_pattern_all_literals() {
    assert_eq!(
        expect_int(
            "enum Trio = T(int, int, int);
             let is_it = \\t => match t | T 1 2 3 => 42 | _ => 0;
             is_it (T 1 2 3);"
        ),
        42
    );
}
