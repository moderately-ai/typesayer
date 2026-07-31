// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! Native Rust predict engine for structured LLM prediction.
//!
//! Provides core types for building structured input/output contracts for language
//! model calls, formatting prompts, parsing responses, and composing multi-step
//! LLM programs with introspectable, serializable state.
//!
//! ## Prompt visibility for debugging
//!
//! `ChatAdapter::format` emits the fully
//! assembled prompt at `TRACE` on the target
//! `typesayer::adapter::chat::messages` — a `signature` shorthand
//! (`"inputs -> outputs"`) plus `messages_json`, the provider-agnostic
//! `Vec<Message>` serialized to JSON (the `[[ ## field ## ]]` envelope, demo
//! turns, and cache-breakpoint markers). Off by default; an operator opts in:
//!
//! - Standalone binaries / examples (env filter reads `RUST_LOG`):
//!   `RUST_LOG=typesayer::adapter::chat::messages=trace`
//! - Applications with a broader filter can pass a complete directive:
//!   `RUST_LOG="warn,typesayer=info,typesayer::adapter::chat::messages=trace"`
//!
//! Prompts can contain sensitive user data, so
//! this is `TRACE`-gated and intended for `… 2>&1 | tee` inspection — never
//! enable it in production log shipping.
//!
//! ## Quick Start
//!
//! ```rust
//! use std::{collections::BTreeMap, sync::Arc};
//!
//! use modelplease::{DummyLM, ModelId};
//! use typesayer::{ChatAdapter, Context, Predict};
//! use typesayer_types::{FieldDef, FieldType, FieldValue, Signature};
//!
//! # async fn example() -> typesayer_types::Result<()> {
//! let sig = Signature::builder("Answer the question.")
//!     .input(FieldDef::input("question", FieldType::String, "The question"))
//!     .output(FieldDef::output("answer", FieldType::String, "The answer"))
//!     .build()?;
//!
//! let lm = DummyLM::sequential(vec!["[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into()]);
//! let ctx = Context {
//!     provider: Arc::new(lm),
//!     model: ModelId::new("test"),
//!     adapter: Arc::new(ChatAdapter::default()),
//! };
//!
//! let prediction = Predict::new(sig)
//!     .call(
//!         &BTreeMap::from([("question".into(), FieldValue::Str("Capital of France?".into()))]),
//!         &ctx,
//!     )
//!     .await?;
//!
//! assert_eq!(prediction.get::<String>("answer")?, "Paris");
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! Core types (`PredictError`, `Result`, `FieldType`, `FieldDef`, `FieldKind`,
//! `FieldValue`, `ObjectField`, `Signature`, `SignatureBuilder`) live in the
//! `typesayer-types` crate and are re-exported here.
//!
//! - [`format`] — `FieldSerializer`, `FieldDeserializer`, `JsonFieldSerializer`,
//!   `JsonFieldDeserializer`
//! - `adapter` — `Adapter` trait, `ChatAdapter`, `Demo`
//! - `prediction` — `Prediction`, `TryFromFieldValue`
//! - `context` — `Context`
//! - `predict` — `Predict`
//! - `example` — `Example` (flat fields + input key separation)
//! - `module` — `Module` trait (composition, introspection, state persistence)
//! - `state` — module save/load helpers

pub(crate) mod adapter;
pub(crate) mod context;
pub(crate) mod evaluate;
pub(crate) mod example;
pub(crate) mod format;
pub(crate) mod module;
pub(crate) mod optimizer;
pub(crate) mod predict;
pub(crate) mod prediction;
pub(crate) mod propose;
pub(crate) mod schema;
pub(crate) mod state;
pub(crate) mod trace;

pub use adapter::{Adapter, ChatAdapter, Demo, chat::CachePlacement};
pub use context::Context;
pub use evaluate::{EvaluateConfig, EvaluationResult, evaluate};
pub use example::Example;
pub use format::{FieldDeserializer, FieldSerializer, JsonFieldDeserializer, JsonFieldSerializer};
pub use modelplease::{
    AwsAccountId, CapabilityError, ContentPart, HttpsUrl, MediaKind, MediaSource, MediaType,
    Message, ModelId, ProviderFileId, Role, S3Uri, SourceKind,
};
pub use module::Module;
pub use optimizer::{
    AutoMode, BootstrapFewShot, CompileRequest, LabeledFewShot, MIPROv2, MetricFn,
    MiproCompileRequest, MiproConfig, MiproDeps, Optimizer, Progress, ProgressFn,
};
pub use predict::Predict;
pub use prediction::{Prediction, TryFromFieldValue};
pub use typesayer_types::{
    FieldDef, FieldKind, FieldType, FieldValue, ObjectField, OneOfDiscriminator, PredictError,
    Result, Signature, SignatureBuilder, VariantArm,
};
// `is_field_required` is still `pub` in `schema` for any out-of-tree
// consumer that imports it directly via `typesayer::schema::*`,
// but it is no longer re-exported at the crate root — new code should
// rely on standard JSON Schema nullability (parent `required: [...]`
// array, `anyOf` with null, or `type: [..., "null"]`). The function
// itself carries a `#[deprecated]` attribute that fires on use.
#[expect(
    deprecated,
    reason = "explicit re-export retained for any out-of-tree caller that still imports it from the crate root; the deprecation warning will fire at each actual usage site"
)]
pub use schema::is_field_required;
pub use schema::{
    JsonSchemaDefinition, extract_description, field_type_from_schema, signature_from_json_schema,
};
pub use trace::{ExecutionTrace, TraceEntry};
