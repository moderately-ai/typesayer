// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level enum field parser.
//!
//! Mirrors [`super::string::StringParser`]'s raw-passthrough behavior
//! (emit `ValueAppend` per chunk, accumulate the buffer, trim at
//! finish) but validates the final value against the declared variant
//! list. At top level the LM emits a bare variant name — no quotes,
//! per `FieldType::output_format_hint` for `Enum`, which produces
//! `"exactly one of: red, green, blue"`. Matches the buffered
//! `format.rs:276-285` semantics: case-sensitive exact match against
//! the variants list, mismatch → error.
//!
//! Nested-in-JSON Enum (inside arrays/objects) takes a different
//! parser — [`super::json::enum_::JsonEnumParser`] — that wraps
//! `JsonStringParser` to handle the quoted form `"variant"`.

use typesayer_types::field::FieldValue;

use super::{super::event::ParseEvent, FieldParser};

pub(super) struct EnumParser {
    path: String,
    variants: Vec<String>,
    accumulated: String,
}

impl EnumParser {
    pub(super) const fn new(path: String, variants: Vec<String>) -> Self {
        Self {
            path,
            variants,
            accumulated: String::new(),
        }
    }
}

impl FieldParser for EnumParser {
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
        if self.variants.iter().any(|v| v == trimmed) {
            let value = FieldValue::Str(trimmed.to_owned());
            // Emit the typed ValueSet here too — event-aware clients
            // get a typed terminal signal symmetric with how the JSON
            // layer fires ValueSet via JsonFieldParser.
            (
                vec![ParseEvent::ValueSet {
                    path: self.path.clone(),
                    value: value.clone(),
                }],
                value,
            )
        } else {
            let msg = format!(
                "value {trimmed:?} is not one of the declared enum variants: {:?}",
                self.variants
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> EnumParser {
        EnumParser::new(
            "/color".into(),
            vec!["red".into(), "green".into(), "blue".into()],
        )
    }

    #[test]
    fn single_chunk_valid_variant() {
        let mut p = parser();
        let events = p.push("red");
        assert_eq!(events.len(), 1); // ValueAppend
        let (trailing, value) = Box::new(p).finish();
        // Trailing carries ValueSet for the typed terminal signal.
        assert_eq!(trailing.len(), 1);
        assert!(matches!(trailing[0], ParseEvent::ValueSet { .. }));
        assert_eq!(value, FieldValue::Str("red".into()));
    }

    #[test]
    fn surrounding_whitespace_trimmed() {
        let mut p = parser();
        p.push("\n  blue  \n");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str("blue".into()));
    }

    #[test]
    fn invalid_variant_errors_with_message() {
        let mut p = parser();
        p.push("yellow");
        let (events, value) = Box::new(p).finish();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ParseEvent::StreamError { message, .. } => {
                assert!(message.contains("yellow"));
                assert!(message.contains("red"));
            }
            other => panic!("got {other:?}"),
        }
        assert_eq!(value, FieldValue::Null);
    }

    #[test]
    fn case_sensitive_match() {
        let mut p = parser();
        p.push("Red");
        let (events, _) = Box::new(p).finish();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ParseEvent::StreamError { .. }));
    }

    #[test]
    fn chunk_boundary_mid_variant() {
        let mut p = parser();
        p.push("gre");
        p.push("en");
        let (trailing, value) = Box::new(p).finish();
        assert!(matches!(trailing[0], ParseEvent::ValueSet { .. }));
        assert_eq!(value, FieldValue::Str("green".into()));
    }
}
