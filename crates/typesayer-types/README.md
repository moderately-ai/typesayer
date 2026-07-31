# typesayer-types

Shared field, signature, media, and error types for the `typesayer` structured-prediction family.

Most users should start with [`typesayer`](https://crates.io/crates/typesayer).

```toml
[dependencies]
typesayer-types = "0.1.1"
```

This crate contains the serializable field, signature, media, and prediction error vocabulary used
by Typesayer and its streaming parser. It depends on Modelplease's core-only API and enables no
network provider.

The minimum supported Rust version is 1.88. Licensed under either MIT or Apache-2.0 at your option.
