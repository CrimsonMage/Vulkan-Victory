//! Per-function complexity: cyclomatic (McCabe) and cognitive (SonarSource).
//!
//! Every function with a body is scored: free functions, inherent and trait
//! impl methods, trait default methods and functions nested in other
//! functions (scored on their own). Closures count toward the function that
//! holds them.
//!
//! Cyclomatic: 1, plus 1 for each `if` (and `else if`), `while`, `for`,
//! `loop`, `let ... else`, `&&`, `||` and `?`, plus arms - 1 per `match`.
//!
//! Cognitive (SonarSource, "Cognitive Complexity", G. Ann Campbell):
//! - +1, plus the nesting level, for `if`, `match`, `for`, `while`, `loop`
//!   and `let ... else`;
//! - +1 for `else if` and `else`, and for a labelled `break` / `continue`;
//! - +1 for each run of like operators in a `&&` / `||` sequence
//!   (`a && b && c` is 1, `a && b || c` is 2);
//! - the bodies of those constructs, and closures, nest one level deeper;
//! - +1 once if the function calls itself directly (`name(..)` in a free
//!   function, `Self::name(..)` or `self.name(..)` in a method). Indirect
//!   recursion is not detected.
//!
//! Macro bodies are scored when they parse as comma-separated expressions
//! (`assert!`, `format!`, `ensure!`, ...); other macros are opaque.

use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    BinOp, Block, Expr, ExprBinary, ImplItem, Item, Local, Macro, Path, Signature, Token,
    TraitItem, Type,
};

/// Scores of one function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnScore {
    /// `name`, `Type::name`, `<Type as Trait>::name`, `module::name` or
    /// `outer::inner`; a repeated name in one file gets `#2`, `#3`, ...
    pub name: String,
    pub line: usize,
    pub cognitive: u32,
    pub cyclomatic: u32,
}

/// Scores every function in one source file.
pub fn analyze(src: &str) -> syn::Result<Vec<FnScore>> {
    let file = syn::parse_file(src)?;
    let mut out = Vec::new();
    for item in &file.items {
        item_fns(item, "", &mut out);
    }
    number_duplicates(&mut out);
    Ok(out)
}

fn item_fns(item: &Item, prefix: &str, out: &mut Vec<FnScore>) {
    match item {
        Item::Fn(f) => function(
            format!("{prefix}{}", f.sig.ident),
            &f.sig,
            &f.block,
            false,
            out,
        ),
        Item::Impl(i) => impl_fns(i, prefix, out),
        Item::Trait(t) => {
            for it in &t.items {
                if let TraitItem::Fn(f) = it
                    && let Some(body) = &f.default
                {
                    let name = format!("{prefix}{}::{}", t.ident, f.sig.ident);
                    function(name, &f.sig, body, true, out);
                }
            }
        }
        Item::Mod(syn::ItemMod {
            ident,
            content: Some((_, items)),
            ..
        }) => {
            let prefix = format!("{prefix}{ident}::");
            for item in items {
                item_fns(item, &prefix, out);
            }
        }
        _ => {}
    }
}

fn impl_fns(i: &syn::ItemImpl, prefix: &str, out: &mut Vec<FnScore>) {
    let ty = type_name(&i.self_ty);
    let owner = match &i.trait_ {
        Some((_, path, _)) => format!("<{ty} as {}>", last_ident(path)),
        None => ty,
    };
    for it in &i.items {
        if let ImplItem::Fn(f) = it {
            let name = format!("{prefix}{owner}::{}", f.sig.ident);
            function(name, &f.sig, &f.block, true, out);
        }
    }
}

fn function(name: String, sig: &Signature, body: &Block, method: bool, out: &mut Vec<FnScore>) {
    let mut s = Scorer {
        fn_name: sig.ident.to_string(),
        method,
        nesting: 0,
        cognitive: 0,
        cyclomatic: 1,
        recursive: false,
        nested_items: Vec::new(),
    };
    s.visit_block(body);
    out.push(FnScore {
        line: sig.ident.span().start().line,
        cognitive: s.cognitive + u32::from(s.recursive),
        cyclomatic: s.cyclomatic,
        name: name.clone(),
    });
    let prefix = format!("{name}::");
    for item in &s.nested_items {
        item_fns(item, &prefix, out);
    }
}

fn type_name(ty: &Type) -> String {
    match ty {
        Type::Path(p) => last_ident(&p.path),
        Type::Reference(r) => type_name(&r.elem),
        _ => "_".to_owned(),
    }
}

fn last_ident(path: &Path) -> String {
    path.segments
        .last()
        .map_or_else(|| "_".to_owned(), |s| s.ident.to_string())
}

fn number_duplicates(fns: &mut [FnScore]) {
    let mut seen = std::collections::HashMap::<String, u32>::new();
    for f in fns {
        let n = seen.entry(f.name.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            f.name = format!("{}#{n}", f.name);
        }
    }
}

/// Walks one function body. Holds no references into the syntax tree, so it
/// can also walk expressions parsed out of macro bodies.
struct Scorer {
    fn_name: String,
    method: bool,
    nesting: u32,
    cognitive: u32,
    cyclomatic: u32,
    recursive: bool,
    /// Items declared inside the body; scored separately afterwards.
    nested_items: Vec<Item>,
}

impl Scorer {
    /// A structural increment: +1 plus the nesting level.
    fn structural(&mut self) {
        self.cognitive += 1 + self.nesting;
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.nesting += 1;
        f(self);
        self.nesting -= 1;
    }

    fn calls_itself(&self, path: &Path) -> bool {
        let segments: Vec<_> = path.segments.iter().map(|s| &s.ident).collect();
        match (self.method, segments.as_slice()) {
            (false, [name]) => **name == self.fn_name,
            (true, [ty, name]) => *ty == "Self" && **name == self.fn_name,
            _ => false,
        }
    }
}

fn is_logical(op: &BinOp) -> bool {
    matches!(op, BinOp::And(_) | BinOp::Or(_))
}

/// Flattens a `&&` / `||` tree (through parentheses) into its operators in
/// source order (true = `&&`) and its operands.
fn flatten<'a>(e: &'a Expr, ops: &mut Vec<bool>, leaves: &mut Vec<&'a Expr>) {
    match e {
        Expr::Binary(b) if is_logical(&b.op) => {
            flatten(&b.left, ops, leaves);
            ops.push(matches!(b.op, BinOp::And(_)));
            flatten(&b.right, ops, leaves);
        }
        Expr::Paren(p) if matches!(&*p.expr, Expr::Binary(b) if is_logical(&b.op)) => {
            flatten(&p.expr, ops, leaves);
        }
        _ => leaves.push(e),
    }
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

impl<'ast> Visit<'ast> for Scorer {
    fn visit_item(&mut self, i: &'ast Item) {
        self.nested_items.push(i.clone());
    }

    fn visit_expr_if(&mut self, e: &'ast syn::ExprIf) {
        self.structural();
        self.cyclomatic += 1;
        self.visit_expr(&e.cond);
        self.nested(|s| s.visit_block(&e.then_branch));
        let mut tail = e.else_branch.as_ref();
        while let Some((_, expr)) = tail {
            self.cognitive += 1;
            if let Expr::If(elif) = &**expr {
                self.cyclomatic += 1;
                self.visit_expr(&elif.cond);
                self.nested(|s| s.visit_block(&elif.then_branch));
                tail = elif.else_branch.as_ref();
            } else {
                self.nested(|s| s.visit_expr(expr));
                tail = None;
            }
        }
    }

    fn visit_expr_match(&mut self, e: &'ast syn::ExprMatch) {
        self.structural();
        self.cyclomatic += count(e.arms.len().saturating_sub(1));
        self.visit_expr(&e.expr);
        self.nested(|s| e.arms.iter().for_each(|arm| s.visit_arm(arm)));
    }

    fn visit_expr_for_loop(&mut self, e: &'ast syn::ExprForLoop) {
        self.structural();
        self.cyclomatic += 1;
        self.visit_expr(&e.expr);
        self.nested(|s| s.visit_block(&e.body));
    }

    fn visit_expr_while(&mut self, e: &'ast syn::ExprWhile) {
        self.structural();
        self.cyclomatic += 1;
        self.visit_expr(&e.cond);
        self.nested(|s| s.visit_block(&e.body));
    }

    fn visit_expr_loop(&mut self, e: &'ast syn::ExprLoop) {
        self.structural();
        self.cyclomatic += 1;
        self.nested(|s| s.visit_block(&e.body));
    }

    fn visit_local(&mut self, l: &'ast Local) {
        if let Some(init) = &l.init
            && let Some((_, diverge)) = &init.diverge
        {
            self.structural();
            self.cyclomatic += 1;
            self.visit_expr(&init.expr);
            self.nested(|s| s.visit_expr(diverge));
        } else {
            visit::visit_local(self, l);
        }
    }

    fn visit_expr_closure(&mut self, e: &'ast syn::ExprClosure) {
        self.nested(|s| s.visit_expr(&e.body));
    }

    fn visit_expr_break(&mut self, e: &'ast syn::ExprBreak) {
        if e.label.is_some() {
            self.cognitive += 1;
        }
        visit::visit_expr_break(self, e);
    }

    fn visit_expr_continue(&mut self, e: &'ast syn::ExprContinue) {
        if e.label.is_some() {
            self.cognitive += 1;
        }
    }

    fn visit_expr_try(&mut self, e: &'ast syn::ExprTry) {
        self.cyclomatic += 1;
        visit::visit_expr_try(self, e);
    }

    fn visit_expr_binary(&mut self, e: &'ast ExprBinary) {
        if !is_logical(&e.op) {
            return visit::visit_expr_binary(self, e);
        }
        let (mut ops, mut leaves) = (Vec::new(), Vec::new());
        flatten(&e.left, &mut ops, &mut leaves);
        ops.push(matches!(e.op, BinOp::And(_)));
        flatten(&e.right, &mut ops, &mut leaves);
        self.cyclomatic += count(ops.len());
        self.cognitive += 1 + count(ops.windows(2).filter(|w| w[0] != w[1]).count());
        for leaf in leaves {
            self.visit_expr(leaf);
        }
    }

    fn visit_expr_call(&mut self, e: &'ast syn::ExprCall) {
        if let Expr::Path(p) = &*e.func
            && self.calls_itself(&p.path)
        {
            self.recursive = true;
        }
        visit::visit_expr_call(self, e);
    }

    fn visit_expr_method_call(&mut self, e: &'ast syn::ExprMethodCall) {
        if self.method
            && e.method == self.fn_name
            && matches!(&*e.receiver, Expr::Path(p) if p.path.is_ident("self"))
        {
            self.recursive = true;
        }
        visit::visit_expr_method_call(self, e);
    }

    fn visit_macro(&mut self, m: &'ast Macro) {
        if let Ok(args) = m.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
            for arg in &args {
                self.visit_expr(arg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::analyze;

    /// (name, cognitive, cyclomatic) of every function in `src`.
    fn scores(src: &str) -> Vec<(String, u32, u32)> {
        analyze(src)
            .unwrap()
            .into_iter()
            .map(|f| (f.name, f.cognitive, f.cyclomatic))
            .collect()
    }

    fn one(src: &str) -> (u32, u32) {
        let s = scores(src);
        assert_eq!(s.len(), 1, "{s:?}");
        (s[0].1, s[0].2)
    }

    #[test]
    fn empty_function() {
        assert_eq!(one("fn a() {}"), (0, 1));
    }

    /// `sumOfPrimes` from the Cognitive Complexity paper: cognitive 7,
    /// cyclomatic 4.
    #[test]
    fn sum_of_primes() {
        let src = "
            fn sum_of_primes(max: u32) -> u32 {
                let mut total = 0;
                'outer: for i in 1..=max {      // +1
                    for j in 2..i {             // +2 (nesting 1)
                        if i % j == 0 {         // +3 (nesting 2)
                            continue 'outer;    // +1
                        }
                    }
                    total += i;
                }
                total
            }";
        assert_eq!(one(src), (7, 4));
    }

    /// `getWords` from the paper: a switch is cognitive 1, cyclomatic 4.
    #[test]
    fn match_counts_once_for_cognitive_and_per_arm_for_cyclomatic() {
        let src = r#"
            fn get_words(n: u32) -> &'static str {
                match n {
                    1 => "one",
                    2 => "a couple",
                    3 => "a few",
                    _ => "lots",
                }
            }"#;
        assert_eq!(one(src), (1, 4));
    }

    #[test]
    fn logical_operator_sequences() {
        // One change of operator: two runs. Three operators.
        assert_eq!(
            one("fn f(a: bool, b: bool, c: bool, d: bool) -> bool { a && b && c || d }"),
            (2, 4)
        );
        // Parentheses do not break a run; negation does.
        assert_eq!(
            one("fn f(a: bool, b: bool, c: bool) -> bool { (a && b) && c }"),
            (1, 3)
        );
        assert_eq!(
            one("fn f(a: bool, b: bool, c: bool) { if a && !(b || c) {} }"),
            (3, 4)
        );
    }

    #[test]
    fn else_if_else_nesting_and_closures() {
        let src = "
            fn g(x: i32, v: &[i32]) -> i32 {
                if x > 0 {                  // +1
                    for y in v {            // +2
                        if *y == x {        // +3
                            return 1;
                        }
                    }
                    0
                } else if x < 0 {           // +1
                    -1
                } else {                    // +1
                    // closure body at nesting 2: if +3, else +1
                    let f = |z: i32| if z > 0 { z } else { -z };
                    f(x)
                }
            }";
        assert_eq!(one(src), (12, 6));
    }

    #[test]
    fn let_else_try_and_loops() {
        let src = "
            fn h(s: &str) -> Option<u32> {
                let Some(first) = s.chars().next() else { return None }; // +1, cyc +1
                let n: u32 = s.parse().ok()?;                            // cyc +1
                let mut i = 0;
                while i < n {                                            // +1
                    loop {                                               // +2
                        break;                                           // unlabelled: 0
                    }
                    i += 1;
                }
                Some(first as u32 + i)
            }";
        assert_eq!(one(src), (4, 5));
    }

    #[test]
    fn direct_recursion() {
        let src = "fn fact(n: u64) -> u64 { if n == 0 { 1 } else { n * fact(n - 1) } }";
        assert_eq!(one(src), (3, 2));
        let src = "struct S; impl S { fn walk(&self, d: u32) { if d > 0 { self.walk(d - 1) } } }";
        assert_eq!(scores(src), [("S::walk".to_owned(), 2, 2)]);
        // A method calling a free function of the same name is not recursion.
        let src = "struct S; impl S { fn walk(&self) { walk() } }";
        assert_eq!(scores(src), [("S::walk".to_owned(), 0, 1)]);
    }

    #[test]
    fn nested_functions_are_scored_on_their_own() {
        let src = "fn outer() { fn inner(x: bool) { if x {} } if true {} }";
        assert_eq!(
            scores(src),
            [
                ("outer".to_owned(), 1, 2),
                ("outer::inner".to_owned(), 1, 2)
            ]
        );
    }

    #[test]
    fn macro_arguments_are_scored() {
        assert_eq!(
            one(r#"fn m(a: bool, b: bool) { assert!(a && b, "x"); }"#),
            (1, 2)
        );
        // Not an expression list: opaque, but harmless.
        assert_eq!(
            one("fn m(x: Option<u8>) -> bool { matches!(x, Some(y) if y > 1) }"),
            (0, 1)
        );
    }

    #[test]
    fn names() {
        let src = "
            use std::fmt;
            struct S;
            impl fmt::Display for S {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { Ok(()) }
            }
            trait T { fn d(&self) {} fn r(&self); }
            mod m { fn f() {} }
            #[cfg(unix)] fn os() {}
            #[cfg(not(unix))] fn os() {}";
        let names: Vec<_> = scores(src).into_iter().map(|s| s.0).collect();
        assert_eq!(names, ["<S as Display>::fmt", "T::d", "m::f", "os", "os#2"]);
    }

    #[test]
    fn line_numbers() {
        let f = analyze("\n\nfn a() {}\n").unwrap();
        assert_eq!(f[0].line, 3);
    }
}
