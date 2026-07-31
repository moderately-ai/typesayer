// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level Bool field parser with buffered-parity permissiveness.
//!
//! At the top level the LM emits the value bare, per `output_format_hint`
//! returning `"`true` or `false`"`. The buffered `format.rs:224-228`
//! parser also accepts `yes/1` and `no/0`, case-insensitive, because
//! prompt drift in the wild commonly produces those forms. The
//! streaming codec matches that tolerance verbatim — anything the
//! buffered parser would accept, the streaming parser also accepts.
//!
//! Nested-in-JSON Bool (inside arrays/objects/maps) takes a different
//! parser — [`super::json::bool::JsonBoolParser`] — that stays strict
//! (`true`/`false` only, lowercase) per RFC 8259 §3, since that's
//! what serde would have produced for the buffered nested path.

use typesayer_types::field::FieldValue;

use super::{super::event::ParseEvent, FieldParser};

pub(super) struct BoolTopLevelParser {
    path: String,
    accumulated: String,
}

impl BoolTopLevelParser {
    pub(super) const fn new(path: String) -> Self {
        Self {
            path,
            accumulated: String::new(),
        }
    }
}

/// True-set / false-set per buffered's tolerance.
fn parse_permissive_bool(trimmed: &str) -> Option<bool> {
    match trimmed.to_lowercase().as_str() {
        "true" | "yes" | "1" => Some(true),
        "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

impl FieldParser for BoolTopLevelParser {
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
        if let Some(b) = parse_permissive_bool(trimmed) {
            let value = FieldValue::Bool(b);
            (
                vec![ParseEvent::ValueSet {
                    path: self.path.clone(),
                    value: value.clone(),
                }],
                value,
            )
        } else {
            let msg = format!(
                "value {trimmed:?} is not a recognised boolean (accepted: true/yes/1, false/no/0)"
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

/// Top-level `Nullable<Bool>` parser. Same permissiveness as
/// [`BoolTopLevelParser`] but with the null sentinel rule layered on
/// top: trimmed empty OR case-insensitive `null` → `FieldValue::Null`.
pub(super) struct NullableBoolTopLevelParser {
    path: String,
    accumulated: String,
}

impl NullableBoolTopLevelParser {
    pub(super) const fn new(path: String) -> Self {
        Self {
            path,
            accumulated: String::new(),
        }
    }
}

impl FieldParser for NullableBoolTopLevelParser {
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
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
            return (
                vec![ParseEvent::ValueSet {
                    path: self.path.clone(),
                    value: FieldValue::Null,
                }],
                FieldValue::Null,
            );
        }
        if let Some(b) = parse_permissive_bool(trimmed) {
            let value = FieldValue::Bool(b);
            (
                vec![ParseEvent::ValueSet {
                    path: self.path.clone(),
                    value: value.clone(),
                }],
                value,
            )
        } else {
            let msg = format!(
                "value {trimmed:?} is not a recognised boolean (accepted: true/yes/1, false/no/0, null)"
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

    fn run(parser: impl FieldParser + 'static, input: &str) -> (Vec<ParseEvent>, FieldValue) {
        let mut p: Box<dyn FieldParser> = Box::new(parser);
        p.push(input);
        p.finish()
    }

    #[test]
    fn accepts_true() {
        let (_, v) = run(BoolTopLevelParser::new("/x".into()), "true");
        assert_eq!(v, FieldValue::Bool(true));
    }

    #[test]
    fn accepts_yes_case_insensitive() {
        let (_, v) = run(BoolTopLevelParser::new("/x".into()), "Yes");
        assert_eq!(v, FieldValue::Bool(true));
    }

    #[test]
    fn accepts_1_as_true() {
        let (_, v) = run(BoolTopLevelParser::new("/x".into()), "1");
        assert_eq!(v, FieldValue::Bool(true));
    }

    #[test]
    fn accepts_no_as_false() {
        let (_, v) = run(BoolTopLevelParser::new("/x".into()), "no");
        assert_eq!(v, FieldValue::Bool(false));
    }

    #[test]
    fn accepts_0_as_false() {
        let (_, v) = run(BoolTopLevelParser::new("/x".into()), "0");
        assert_eq!(v, FieldValue::Bool(false));
    }

    #[test]
    fn rejects_unrecognised() {
        let (events, v) = run(BoolTopLevelParser::new("/x".into()), "maybe");
        assert_eq!(v, FieldValue::Null);
        assert!(matches!(events[0], ParseEvent::StreamError { .. }));
    }

    #[test]
    fn nullable_accepts_null_word() {
        let (_, v) = run(NullableBoolTopLevelParser::new("/x".into()), "Null");
        assert_eq!(v, FieldValue::Null);
    }

    #[test]
    fn nullable_accepts_empty_as_null() {
        let (_, v) = run(NullableBoolTopLevelParser::new("/x".into()), "   ");
        assert_eq!(v, FieldValue::Null);
    }

    #[test]
    fn nullable_accepts_yes_as_true() {
        let (_, v) = run(NullableBoolTopLevelParser::new("/x".into()), "yes");
        assert_eq!(v, FieldValue::Bool(true));
    }

    #[test]
    fn nullable_rejects_unrecognised() {
        let (events, v) = run(NullableBoolTopLevelParser::new("/x".into()), "maybe");
        assert_eq!(v, FieldValue::Null);
        assert!(matches!(events[0], ParseEvent::StreamError { .. }));
    }
}
