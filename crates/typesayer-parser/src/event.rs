// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed events emitted by [`super::parser::ChatStreamParser`].
//!
//! Each variant corresponds to a step in progressively building the
//! structured output tree from a streaming LM response. Paths are
//! JSON-Pointer strings (RFC 6901) addressing nodes within the schema
//! tree, so clients can route events to UI components per path.
//!
//! The variant set is matched 1:1 to [`FieldType`]
//! taxonomy — adding a new `FieldType` variant grows this enum (and
//! the corresponding [`super::field_parsers`] dispatch); existing
//! variants stay stable.

use typesayer_types::field::{FieldType, FieldValue};

/// One event produced by the streaming parser.
///
/// Variants are emitted in the order content arrives from the LM.
/// Consumers can rebuild the full structured output from the sequence,
/// OR react to specific events (`FieldComplete` for "field X is done",
/// `StreamComplete` for "everything is done", `ValueAppend` for chat-UI
/// typing).
#[derive(Debug, Clone, PartialEq)]
pub enum ParseEvent {
    /// First event, always. Carries an opaque hash of the output
    /// signature so clients can sanity-check they were rendering
    /// against the right schema. The full schema is fetched separately
    /// via the existing application endpoints (kept out of the
    /// stream-start payload to bound the on-open size).
    StreamStart { output_schema_hash: String },

    /// A declared output field's `[[ ## name ## ]]` marker arrived.
    /// Path is `/<field_name>`; field_type is the declared type for
    /// the field so clients know what value-event variants to expect.
    FieldStart { path: String, field_type: FieldType },

    /// Incremental text content for a [`FieldType::String`] field.
    /// `delta` is the new content since the previous event for this
    /// path; clients concatenate to render progressively (chat-UI
    /// typing). Only emitted for `String` fields; structured fields
    /// use `ElementAdded` / `EntryAdded` instead.
    ValueAppend { path: String, delta: String },

    /// Atomic value emitted when a non-string scalar finishes
    /// parsing (`Int`, `Float`, `Bool`, `Enum`, `Nullable<scalar>`).
    /// Reserved for future per-FieldType-variant expansion — not
    /// emitted by the initial-step parser (which supports `String`
    /// and `List<String>` only).
    ValueSet { path: String, value: FieldValue },

    /// A [`FieldType::List`] field's content opened — the LM emitted
    /// `[`. Path identifies the field; clients can render an empty
    /// list shell before elements arrive.
    ListOpen { path: String },

    /// One element of a [`FieldType::List`] finished parsing.
    /// `index` is the element's position in the list; `value` is its
    /// typed value.
    ElementAdded {
        path: String,
        index: usize,
        value: FieldValue,
    },

    /// A [`FieldType::List`] field's content closed — the LM emitted
    /// `]`. `value` is the complete parsed list for clients that
    /// didn't accumulate from `ElementAdded` events.
    ListClose { path: String, value: FieldValue },

    /// A [`FieldType::Map`] or [`FieldType::Object`] field's content
    /// opened — the LM emitted `{`. Reserved for future expansion.
    ObjectOpen { path: String },

    /// One key-value pair of a [`FieldType::Map`] / [`FieldType::Object`]
    /// finished parsing. Reserved for future expansion.
    EntryAdded {
        path: String,
        key: String,
        value: FieldValue,
    },

    /// A [`FieldType::Map`] / [`FieldType::Object`] field's content
    /// closed — the LM emitted `}`. Reserved for future expansion.
    ObjectClose { path: String, value: FieldValue },

    /// A field's marker boundary arrived; the prior field is fully
    /// done. Carries the full parsed value for clients that didn't
    /// accumulate from per-value events. Always emitted at field-
    /// completion, after any `ListClose` / `ObjectClose` /
    /// `ValueAppend` events for the field.
    FieldComplete { path: String, value: FieldValue },

    /// The `[[ ## completed ## ]]` sentinel arrived. `output` is the
    /// full structured output as a [`FieldValue::Object`] map of
    /// field-name → field-value, mirroring `Adapter::parse`.
    StreamComplete { output: FieldValue },

    /// Mid-stream parse or deserialization failure. `path` is the
    /// node where the error occurred (may be `None` for errors
    /// between fields). Streaming halts after this event; the
    /// consumer returns a non-retryable error.
    StreamError {
        path: Option<String>,
        message: String,
    },
}

impl ParseEvent {
    /// Stable event-type identifier for wire serialization. Mirrors
    /// the snake-case identifier the streaming consumer uses
    /// when declaring schemas via `StreamingEventDef`.
    #[must_use]
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::StreamStart { .. } => "stream_start",
            Self::FieldStart { .. } => "field_start",
            Self::ValueAppend { .. } => "value_append",
            Self::ValueSet { .. } => "value_set",
            Self::ListOpen { .. } => "list_open",
            Self::ElementAdded { .. } => "element_added",
            Self::ListClose { .. } => "list_close",
            Self::ObjectOpen { .. } => "object_open",
            Self::EntryAdded { .. } => "entry_added",
            Self::ObjectClose { .. } => "object_close",
            Self::FieldComplete { .. } => "field_complete",
            Self::StreamComplete { .. } => "stream_complete",
            Self::StreamError { .. } => "stream_error",
        }
    }

    /// Whether this event variant should be durable (WAL-persisted)
    /// for late-subscriber replay. Boundary events (start / open /
    /// close / complete / error) are durable because they reconstruct
    /// the schema's structure; per-value events (`value_append`,
    /// `value_set`, `element_added`, `entry_added`) are volatile —
    /// late subscribers fetch the final structured output via the
    /// existing execution endpoint and only need structural events
    /// for skeleton rendering.
    #[must_use]
    pub const fn is_durable(&self) -> bool {
        match self {
            Self::ValueAppend { .. }
            | Self::ValueSet { .. }
            | Self::ElementAdded { .. }
            | Self::EntryAdded { .. } => false,
            Self::StreamStart { .. }
            | Self::FieldStart { .. }
            | Self::ListOpen { .. }
            | Self::ListClose { .. }
            | Self::ObjectOpen { .. }
            | Self::ObjectClose { .. }
            | Self::FieldComplete { .. }
            | Self::StreamComplete { .. }
            | Self::StreamError { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_type_distinct_per_variant() {
        let events = [
            ParseEvent::StreamStart {
                output_schema_hash: String::new(),
            },
            ParseEvent::FieldStart {
                path: "/x".into(),
                field_type: FieldType::String,
            },
            ParseEvent::ValueAppend {
                path: "/x".into(),
                delta: String::new(),
            },
            ParseEvent::ValueSet {
                path: "/x".into(),
                value: FieldValue::Null,
            },
            ParseEvent::ListOpen { path: "/x".into() },
            ParseEvent::ElementAdded {
                path: "/x".into(),
                index: 0,
                value: FieldValue::Null,
            },
            ParseEvent::ListClose {
                path: "/x".into(),
                value: FieldValue::List(vec![]),
            },
            ParseEvent::ObjectOpen { path: "/x".into() },
            ParseEvent::EntryAdded {
                path: "/x".into(),
                key: String::new(),
                value: FieldValue::Null,
            },
            ParseEvent::ObjectClose {
                path: "/x".into(),
                value: FieldValue::Object(std::collections::BTreeMap::new()),
            },
            ParseEvent::FieldComplete {
                path: "/x".into(),
                value: FieldValue::Null,
            },
            ParseEvent::StreamComplete {
                output: FieldValue::Object(std::collections::BTreeMap::new()),
            },
            ParseEvent::StreamError {
                path: None,
                message: String::new(),
            },
        ];
        let types: std::collections::HashSet<_> =
            events.iter().map(ParseEvent::event_type).collect();
        assert_eq!(
            types.len(),
            events.len(),
            "every variant must have a unique event_type"
        );
    }

    #[test]
    fn durability_split_matches_design() {
        // Value events are volatile.
        assert!(
            !ParseEvent::ValueAppend {
                path: "/x".into(),
                delta: String::new()
            }
            .is_durable()
        );
        assert!(
            !ParseEvent::ValueSet {
                path: "/x".into(),
                value: FieldValue::Null
            }
            .is_durable()
        );
        assert!(
            !ParseEvent::ElementAdded {
                path: "/x".into(),
                index: 0,
                value: FieldValue::Null
            }
            .is_durable()
        );
        assert!(
            !ParseEvent::EntryAdded {
                path: "/x".into(),
                key: String::new(),
                value: FieldValue::Null,
            }
            .is_durable()
        );

        // Boundary events are durable.
        for ev in [
            ParseEvent::StreamStart {
                output_schema_hash: String::new(),
            },
            ParseEvent::FieldStart {
                path: "/x".into(),
                field_type: FieldType::String,
            },
            ParseEvent::ListOpen { path: "/x".into() },
            ParseEvent::ListClose {
                path: "/x".into(),
                value: FieldValue::List(vec![]),
            },
            ParseEvent::ObjectOpen { path: "/x".into() },
            ParseEvent::ObjectClose {
                path: "/x".into(),
                value: FieldValue::Object(std::collections::BTreeMap::new()),
            },
            ParseEvent::FieldComplete {
                path: "/x".into(),
                value: FieldValue::Null,
            },
            ParseEvent::StreamComplete {
                output: FieldValue::Object(std::collections::BTreeMap::new()),
            },
            ParseEvent::StreamError {
                path: None,
                message: String::new(),
            },
        ] {
            assert!(ev.is_durable(), "expected durable: {ev:?}");
        }
    }
}
