# Bugs to investigate:

## lambdas as application arguments? (FIXED)

```
❯ cargo run src/prelude/prelude.mln --repl
warning: unused import: `crate::prelude`
 --> src/ast.rs:3:5
  |
3 | use crate::prelude;
  |     ^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `ExecutionResult`
  --> src/codegen/mod.rs:25:19
   |
25 | pub use execute::{ExecutionResult, compile, execute};
   |                   ^^^^^^^^^^^^^^^

warning: unused variable: `scrutinee`
    --> src/types.rs:1048:22
     |
1048 |         ENode::Match(scrutinee, cases) => {
     |                      ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_scrutinee`
     |
     = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default

warning: unused variable: `cases`
    --> src/types.rs:1048:33
     |
1048 |         ENode::Match(scrutinee, cases) => {
     |                                 ^^^^^ help: if this is intentional, prefix it with an underscore: `_cases`

warning: `merlin-lang` (bin "merlin-lang") generated 4 warnings (run `cargo fix --bin "merlin-lang" -p merlin-lang` to apply 4 suggestions)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s
     Running `target/debug/merlin-lang src/prelude/prelude.mln --repl`
parse: ok
typecheck: ok
codegen: ok (1 top-level functions)
> lfold (\acc x => x::acc) [] [1,2,3];
error: 'func.call' op operand type mismatch: expected operand type '(!llvm.ptr, i32) -> !llvm.ptr', but provided '(!llvm.ptr) -> !llvm.ptr' for operand number 0
execution error: codegen: pass manager failed: failed to run pass
> 
```

Other example:
```
> let add = \x y => x + y;
  : t84 -> t84 -> t84 : <fn>
> let sym = lfold add 0 [1,2,3,4,5]
  : int = 15
> lfold (\x y => x + y) 0 [1,2,3,4,5]
error: 'func.call' op operand type mismatch: expected operand type '(i32, i32) -> i32', but provided '(i32) -> !llvm.ptr' for operand number 0
execution error: codegen: pass manager failed: failed to run pass
> 
```

## Enum ctors not working

Actually works fine in compiled code, interesting

```
parse: ok
typecheck: ok
codegen: ok (1 top-level functions)
> opt 1;
execution error: codegen: cannot JIT-execute type TypeFuncApplication(Enum("Option"), [TypeFuncApplication(Int, [])])
>  
> get_i 0 [1,2,3]
execution error: codegen: cannot JIT-execute type TypeFuncApplication(Enum("Option"), [TypeFuncApplication(Int, [])])
>   
> let get_res = \opt => match opt | Some val => Ok val | None => Err "no value";
  : Option t198 -> Result t198 str : <fn>
> get_res (Some 0);
execution error: codegen: cannot JIT-execute type TypeFuncApplication(Enum("Result"), [TypeFuncApplication(Int, []), TypeFuncApplication(Str, [])])
> get_res None; 
execution error: codegen: cannot JIT-execute type TypeFuncApplication(Enum("Result"), [TypeVariable("t206"), TypeFuncApplication(Str, [])])
> 
> 
```

## Partial application not supported (FIXED)

```
> let add = \x y => x+1;
  : int -> t2 -> int : <fn>
> add 1;
codegen error: codegen: partial application of `add` is not supported yet
>  
```

## Enum constructors with arity > 1 cannot be built

Only nullary (`None`) and single-argument (`Some x`) constructor *applications* are
lowered. A constructor with two or more fields falls through to the ordinary
variable path and is reported as undefined.

```
enum Pair('a, 'b) = Mk('a, 'b);
let p = Mk 42 true;
```

```
parse: ok
typecheck: ok
codegen: error: codegen: undefined variable `Mk` (not a bound parameter or symbol)
```

Cause: `lower_application` only recognizes a constructor call when the head is a
bare `Variable` applied to a single argument (`src/codegen/apply.rs:427`); a
2-arg application's head is itself an `Application`, so it is never matched.
Nullary constructors are handled separately at `apply.rs:227`.

## Enum constructor patterns with arity > 1 unsupported

Matching a multi-field constructor fails at codegen (the typechecker accepts it).

```
enum Pair('a, 'b) = Mk('a, 'b);
let fst = \p => match p
    | Mk x _ => x
    | _ => 0;
```

```
parse: ok
typecheck: ok
codegen: error: codegen: unsupported match pattern Application(Application(Variable("Mk"), ...), ...)
```

Cause: `case_pattern` only handles nullary constructors (`ENode::Variable`) and
single-argument constructors whose payload is a single variable
(`src/codegen/expr.rs:685`). It rejects `arity != 1` at `expr.rs:701` and
non-single-variable payloads at `expr.rs:707`; a `Node p l r c v` pattern's
outer application has a non-`Variable` head, so it hits the "unsupported match
pattern" arm.

## `==`/`!=` only works on int/float/str/bool

No equality on enums, records, or lists — comparing enum values (e.g. to find a
child's direction) is a type error.

```
enum Color = Black | Red;
let eq = \c => c == Red;
```

```
typecheck: error: '==' requires int, float, string, or bool operands
```

Cause: `infer_comparison` explicitly restricts `Eq`/`NotEq` to the four primitive
types (`src/types.rs:906-915`).

### Notes

* Infer_comparison: could simply remove the restriction on unifying with primitive types (l.911).
* This means that code gen will have to be adjusted to deal with enums, records, and lists.
    * Enum: Same constructor with same value(s) -> Also look into having enums stack-allocated (problematic)
    * Record: Same record name with same field values
    * Lists: All elements equal (same length implied) -> potential optimization here by comparing words like strcmp

___TODO___:

- [x] Figure out how to express operations on:
    - [x] Enums (discriminant + operands)
    - [x] structs (LLVM/MLIR probably has something)
    - [x] lists

## Recursive record types do not unify

A record whose field type refers back to itself fails on *nested* construction —
the self-reference is stored as an unexpanded alias and never unified with the
actual `Rec` type.

```
enum Color = Black | Red;
record Node('a) = {
    color : Color,
    value : 'a,
    left  : Option (Node 'a),
    right : Option (Node 'a)
};
let n = Node { color: Red, value: 1,
    left: Some (Node { color: Black, value: 2, left: None, right: None }),
    right: None };
```

```
typecheck: error: Type function application mismatch: Enum("Node") != RowExt("color")
```

Cause: `handle_type_decl` registers the record alias only *after* expanding its
field types (`src/types.rs:620-628`), so the self-reference `Node 'a` stays as
`Enum("Node")`; `infer_named_record` (`types.rs:1020`) instantiates the alias rhs
without re-expanding, so that `Enum("Node")` never meets the `Rec(...)` produced
by construction. `expand` also rejects recursive aliases outright
(`types.rs:552`). Result: recursive data can only be expressed through *enums*
(nominal), not records.

## Field access on a record-valued enum payload fails (FIXED)

Binding a record out of a unary enum constructor (`Node d => d.value`) used to
emit `llvm.extractvalue` on a pointer and fail MLIR verification.

```
enum Color = Black | Red;
record NodeData('a) = {
    color : Color,
    value : 'a,
    left  : RbTree('a),
    right : RbTree('a)
};
enum RbTree('a) = Leaf | Node(NodeData('a));
let get = \x => match x | Leaf => 0 | Node d => d.value;
println (itostr (get (Node (NodeData { color: Red, value: 42, left: Leaf, right: Leaf }))));
```

```
codegen: error: 'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got '!llvm.ptr'
```

Cause: `lower_type_decl` stored the *raw* variant field type
(`v.tparams.iter().map(|t| t.t.clone())`, `src/codegen/stmt.rs:1153`), so the
record alias `NodeData` stayed `Enum("NodeData")`; `enum_variant_fields` returned
it unexpanded and `lower_type` mapped `Enum(_)` to `!llvm.ptr`
(`src/codegen/types.rs:85`), binding the payload as a pointer.

Fix: `lower_type_decl` now expands record aliases to their `Rec(RowExt(..))` form
when storing enum variant field types (`expand_stored_type` in
`src/codegen/stmt.rs`), so the payload lowers to a struct and `extract_field`
works on it.

This exposed a second bug: `build_payload` sized its heap allocation as
`8 * fields.len()` bytes, assuming every field is pointer-sized. A record payload
(e.g. `{ i32, i32, ptr, ptr }`, 24 bytes) overflowed that buffer and corrupted the
heap (`malloc(): corrupted top size`). `build_payload` now takes the total size,
computed by `monotype_size` (`src/codegen/types.rs`), which `lower_application`
passes.

## Typechecker stack overflow on recursive types + nested construction

Typechecking a function that constructs a nested record against a *polymorphic*
recursive enum/record type overflows the stack (after `parse: ok`). A red-black
tree `balance` written against the arity-1 enum + record encoding reproduces it.
`insert`/`contains` without `balance` typecheck and run fine, and the same nested
construction at the top level (`let t = Node (NodeData { left: Node (…) })`)
typechecks fine — so it is specifically the combination of a polymorphic
recursive type and nested record construction inside a function body.

```
parse: ok
thread 'main' (NNN) has overflowed its stack
fatal runtime error: stack overflow, aborting
```

Not yet root-caused; likely unbounded recursion in `unify`/`generalise` over the
`Rec`/`Enum("RbTree")` cycle with a free type variable. Minimal standalone repro
still needed.

