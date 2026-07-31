// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level `Nullable<String>` and `Nullable<Enum>` field parser.
//!
//! Matches the buffered `format.rs:288-294` semantics: at top level,
//! the LM emits either the bare value (raw string content, or a bare
//! enum variant name) OR the literal `null` / empty string. The
//! buffered parser checks `trimmed.is_empty() || eq_ignore_ascii_case("null")`
//! → `FieldValue::Null`; otherwise delegates to the inner type's
//! top-level form (raw passthrough for `String`, variant validation
//! for `Enum`).
//!
//! Streaming mirrors that: accumulate raw text + emit `ValueAppend`
//! deltas per chunk (progressive UI), then at `finish` decide the
//! final value from the trimmed buffer.
//!
//! `Nullable<Int>`, `Nullable<Float>`, `Nullable<Bool>` etc. take a
//! different path: top-level dispatch routes those to the JSON-value
//! layer via `JsonFieldParser`, which composes `JsonNullableParser`
//! over the inner type's `Json*Parser`. This file only handles the
//! raw-text top-level cases the LM emits unquoted.

use typesayer_types::field::{FieldType, FieldValue};

use super::{super::event::ParseEvent, FieldParser};

pub(super) struct NullableRawTextParser {
    path: String,
    inner: FieldType,
    accumulated: String,
}

impl NullableRawTextParser {
    pub(super) const fn new(path: String, inner: FieldType) -> Self {
        Self {
            path,
            inner,
            accumulated: String::new(),
        }
    }

    /// True for inner types this parser can handle at the top level —
    /// `String` and `Enum`, the raw-text-emitted top-level types.
    pub(super) const fn can_handle_inner(inner: &FieldType) -> bool {
        matches!(inner, FieldType::String | FieldType::Enum(_))
    }
}

impl FieldParser for NullableRawTextParser {
    fn push(&mut self, chunk: &str) -> Vec<ParseEvent> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.accumulated.push_str(chunk);
        vec![ParseEvent::ValueAppend {
            path: self.path.clone(),
            delta: chunk.to_owned(),
        }]
    }

    fn finish(self: Box<Self>) -> (Vec<ParseEvent>, FieldValue) {
        let trimmed = self.accumulated.trim();

        // Null-sentinel check — matches buffered exactly: empty OR
        // case-insensitive "null" → Null.
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
            return (
                vec![ParseEvent::ValueSet {
                    path: self.path.clone(),
                    value: FieldValue::Null,
                }],
                FieldValue::Null,
            );
        }

        match &self.inner {
            FieldType::String => {
                let value = FieldValue::Str(trimmed.to_owned());
                (
                    vec![ParseEvent::ValueSet {
                        path: self.path.clone(),
                        value: value.clone(),
                    }],
                    value,
                )
            }
            FieldType::Enum(variants) => {
                if variants.iter().any(|v| v == trimmed) {
                    let value = FieldValue::Str(trimmed.to_owned());
                    (
                        vec![ParseEvent::ValueSet {
                            path: self.path.clone(),
                            value: value.clone(),
                        }],
                        value,
                    )
                } else {
                    let msg = format!(
                        "value {trimmed:?} is not one of the declared enum variants: {variants:?}"
                    );
                    (
                        vec![ParseEvent::StreamError {
                            path: Some(self.path),
                            message: msg,
                        }],
                        FieldValue::Null,
                    )
                }
            }
            other => {
                // Defensive: dispatch should never construct this
                // parser for non-raw-text inner types.
                let msg = format!(
                    "NullableRawTextParser constructed with non-raw-text inner type {other:?}"
                );
                (
                    vec![ParseEvent::StreamError {
                        path: Some(self.path),
                        message: msg,
                    }],
                    FieldValue::Null,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nullable_string() -> NullableRawTextParser {
        NullableRawTextParser::new("/answer".into(), FieldType::String)
    }

    fn nullable_enum() -> NullableRawTextParser {
        NullableRawTextParser::new(
            "/answer".into(),
            FieldType::Enum(vec!["yes".into(), "no".into()]),
        )
    }

    #[test]
    fn empty_buffer_yields_null() {
        let p = nullable_string();
        let (events, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Null);
        match &events[0] {
            ParseEvent::ValueSet { value, .. } => assert_eq!(value, &FieldValue::Null),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn null_word_lowercase_yields_null() {
        let mut p = nullable_string();
        p.push("null");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Null);
    }

    #[test]
    fn null_word_uppercase_yields_null() {
        let mut p = nullable_string();
        p.push("NULL");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Null);
    }

    #[test]
    fn null_word_with_whitespace_yields_null() {
        let mut p = nullable_string();
        p.push("  Null  ");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Null);
    }

    #[test]
    fn non_null_string_value() {
        let mut p = nullable_string();
        p.push("hello world");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str("hello world".into()));
    }

    #[test]
    fn whitespace_trimmed_around_string_value() {
        let mut p = nullable_string();
        p.push("\n  hello  \n");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str("hello".into()));
    }

    #[test]
    fn enum_variant_validated() {
        let mut p = nullable_enum();
        p.push("yes");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str("yes".into()));
    }

    #[test]
    fn enum_invalid_errors() {
        let mut p = nullable_enum();
        p.push("maybe");
        let (events, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Null);
        assert!(matches!(events[0], ParseEvent::StreamError { .. }));
    }

    #[test]
    fn enum_null_word_yields_null() {
        let mut p = nullable_enum();
        p.push("null");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Null);
    }
}
