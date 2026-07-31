// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-FieldType sub-parsers driven by the top-level
//! [`super::parser::ChatStreamParser`].
//!
//! Two layers of dispatch:
//!
//! - **Marker-delimited** (`FieldParser`): handles content between `[[ ## name ## ]]` markers.
//!   The top-level parser calls `parser_for` to pick a sub-parser per output field. Strings are
//!   the special case (raw passthrough for chat-UI typing); everything else wraps a
//!   `JsonValueParser` via `JsonFieldParser`.
//!
//! - **Value-delimited** (`JsonValueParser`): parses one complete JSON value,
//!   self-terminating on the value's syntactic close. Generic over the expected
//!   [`FieldType`] — arrays and objects recurse via `json_value_parser_for` for their inner types,
//!   so arbitrary nesting
//!   works through the same dispatch.
//!
//! The split exists because top-level strings are raw text (the chat
//! adapter's `output_format_hint` returns `None` for `String`, so the
//! LM emits the value verbatim between markers), while strings inside
//! structured fields are JSON strings (`"..."` with escapes).

pub(super) mod bool_top;
pub(super) mod enum_;
pub(super) mod json;
pub(super) mod json_field;
pub(super) mod nullable_raw;
pub(super) mod string;
pub(super) mod unsupported;

use typesayer_types::field::{FieldType, FieldValue};

use super::event::ParseEvent;

/// Sub-parser for one field of a known FieldType. Lifetime of the
/// instance is one field — instantiated on `FieldStart`, fed chunks
/// via `push`, finalized via `finish` when the next marker arrives.
pub(super) trait FieldParser: Send {
    /// Feed one chunk of text content (the bytes between two markers,
    /// or a slice thereof when the chunk straddles a marker boundary).
    /// Returns any events the parser can emit from this input.
    fn push(&mut self, chunk: &str) -> Vec<ParseEvent>;

    /// Finalize at field-end (next marker arrived) or stream-end.
    /// Returns trailing events PLUS the full parsed `FieldValue` for
    /// the wrapping `FieldComplete` event.
    fn finish(self: Box<Self>) -> (Vec<ParseEvent>, FieldValue);
}

/// Construct the appropriate sub-parser for a given top-level field
/// type. Strings get raw-text passthrough; everything else funnels
/// through the JSON value-parser dispatch. `Media` falls through to
/// `UnsupportedParser` since media output is out of scope for streaming
/// (it'd arrive as base64 / URLs through a different content channel).
pub(super) fn parser_for(field_type: &FieldType, path: String) -> Box<dyn FieldParser> {
    match field_type {
        FieldType::String => Box::new(string::StringParser::new(path)),
        FieldType::Enum(variants) => Box::new(enum_::EnumParser::new(path, variants.clone())),
        // Top-level Bool gets the permissive parser (yes/no/1/0
        // case-insensitive, matching buffered's format.rs:224-228).
        // Nested Bool inside JSON envelopes stays strict via JsonBoolParser.
        FieldType::Bool => Box::new(bool_top::BoolTopLevelParser::new(path)),
        // Top-level Nullable<Bool>: permissive bool + null sentinel.
        FieldType::Nullable(inner) if matches!(**inner, FieldType::Bool) => {
            Box::new(bool_top::NullableBoolTopLevelParser::new(path))
        }
        // Top-level Nullable<String> / Nullable<Enum> get the raw-text
        // path that mirrors buffered's null-sentinel handling
        // (`empty || eq_ignore_ascii_case("null")` → Null). Other inner
        // types (`Nullable<Int>`, `Nullable<List<...>>`, etc.) route
        // through the JSON layer below — JsonFieldParser → JsonNullableParser
        // → inner type's strict-JSON parser.
        FieldType::Nullable(inner)
            if nullable_raw::NullableRawTextParser::can_handle_inner(inner) =>
        {
            Box::new(nullable_raw::NullableRawTextParser::new(
                path,
                (**inner).clone(),
            ))
        }
        FieldType::Media { .. } => Box::new(unsupported::UnsupportedParser::new(
            path,
            field_type.clone(),
        )),
        other => Box::new(json_field::JsonFieldParser::new(path, other.clone())),
    }
}
