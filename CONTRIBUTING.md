# Contributing

Use Rust 1.94.1 or newer. Before opening a pull request, run formatting, workspace Clippy,
`cargo nextest run --workspace --all-targets --all-features`, rustdoc with warnings denied, and
`cargo deny check --all-features`.

Changes spanning workspace crates must keep their versions synchronized and preserve the
dependency order `typesayer-types` → `typesayer-parser` / `typesayer`.
