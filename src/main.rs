use std::process;
use merlin_lang::ast::Program;

fn main() {
    let mut dump_ast = false;
    let mut dump_mlir = false;
    let mut start_repl = false;
    let mut include_prelude = false;
    let mut file: Option<String> = None;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--ast" => dump_ast = true,
            "--mlir" => dump_mlir = true,
            "--repl" => start_repl = true,
            "--prelude" => include_prelude = true,
            "--help" | "-h" => {
                println!("usage: merlin [--ast] [--repl] <file.mln>");
                println!("  --ast    dump the program's AST after it completes");
                println!("  --repl   start the REPL with the program already in the context");
                return;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown option `{}`", s);
                eprintln!("usage: merlin [--ast] [--repl] <file.mln>");
                process::exit(1);
            }
            s => file = Some(s.to_string()),
        }
    }

    match file {
        None => {
            #[cfg(feature = "codegen")]
            merlin_lang::repl::repl_loop(None);
            #[cfg(not(feature = "codegen"))]
            {
                eprintln!("error: no input file and REPL is unavailable (built without codegen)");
                process::exit(1);
            }
        }
        Some(path) => {
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: failed to read `{}`: {}", path, e);
                    process::exit(1);
                }
            };

            let mut prog = match
                if start_repl && !include_prelude {Program::parse(&source)} else {Program::parse_with_prelude(&source)} {
                Ok(p) => {
                    println!("parse: ok");
                    p
                }
                Err(e) => {
                    eprintln!("parse: error: {}", e);
                    process::exit(1);
                }
            };
            prog.source_name = path.clone();

            if dump_ast {
                println!("{}", *prog);
            }

            if let Err(e) = Program::typecheck(&mut prog) {
                match e.pos {
                    Some(ref pos) => eprintln!(
                        "typecheck: error: {}:{}:{}: {}",
                        prog.source_name, pos.start_line, pos.start_col, e.message
                    ),
                    None => eprintln!("typecheck: error: {}", e.message),
                }
                process::exit(2);
            }
            println!("typecheck: ok");

            #[cfg(feature = "codegen")]
            {
                let context = merlin_lang::codegen::new_context();
                match merlin_lang::codegen::lower(&prog, &context) {
                    Ok(mut module) => {
                        println!(
                            "codegen: ok ({} top-level functions)",
                            module.function_count()
                        );

                        if dump_mlir {
                            println!("{}", module.dump());
                        }

                        if !start_repl {
                            compile_executable(&path, &mut module);
                        }
                    },
                    Err(e) => {
                        eprintln!("codegen: error: {}", e);
                        process::exit(3);
                    }
                }

                if start_repl {
                    merlin_lang::repl::repl_loop(Some(prog));
                }
            }

            #[cfg(not(feature = "codegen"))]
            {
                if dump_mlir || start_repl {
                    eprintln!("error: `--mlir` and `--repl` require the `codegen` feature");
                    process::exit(3);
                }
            }
        }
    }
}

/// Emit a native object file and link it into an executable named after the
/// source file (with the `.mln` extension stripped).
#[cfg(feature = "codegen")]
fn compile_executable(source_path: &str, module: &mut merlin_lang::codegen::Module) {
    let output_path = std::path::Path::new(source_path).with_extension("");
    let obj_path = std::env::temp_dir().join(format!("merlin_{}.o", process::id()));

    if let Err(e) = merlin_lang::codegen::compile(module, obj_path.to_str().unwrap()) {
        eprintln!("codegen: error: {}", e);
        process::exit(3);
    }

    let link = process::Command::new("cc")
        .arg("-o")
        .arg(&output_path)
        .arg(&obj_path)
        .status();
    let _ = std::fs::remove_file(&obj_path);

    match link {
        Ok(status) if status.success() => {
            println!("compiled: {}", output_path.display());
        }
        Ok(_) => {
            eprintln!("error: failed to link `{}`", output_path.display());
            process::exit(3);
        }
        Err(e) => {
            eprintln!("error: could not run `cc` to link the executable: {}", e);
            process::exit(3);
        }
    }
}
