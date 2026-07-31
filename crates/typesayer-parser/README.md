# typesayer-parser

Incrementally parses Typesayer's structured language-model output into typed events.

Most users should start with [`typesayer`](https://crates.io/crates/typesayer).

```toml
[dependencies]
typesayer-parser = "0.1.1"
```

The default build provides the production streaming parser. The `test-utils` feature adds bounded
property-test strategies and Modelplease stream helpers for downstream parity testing:

```toml
[dev-dependencies]
typesayer-parser = { version = "0.1.1", features = ["test-utils"] }
```

The minimum supported Rust version is 1.88. Licensed under either MIT or Apache-2.0 at your option.
