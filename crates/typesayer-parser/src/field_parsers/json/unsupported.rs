// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Placeholder JSON value parser for FieldType variants not yet
//! implemented. Emits a clear [`ParseEvent::StreamError`] on first
//! push so the wrapping consumer returns a non-retryable failure
//! with a useful operator-facing message ("type X not yet supported
//! by streaming parser") rather than silently producing garbage or
//! panicking. Folded into [`super::json_value_parser_for`] as the
//! catch-all arm for FieldType variants without a concrete impl.

use typesayer_types::field::{FieldType, FieldValue};

use super::{super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser};

pub(super) struct JsonUnsupportedParser {
    path: String,
    field_type: FieldType,
    errored: bool,
}

impl JsonUnsupportedParser {
    pub(super) const fn new(path: String, field_type: FieldType) -> Self {
        Self {
            path,
            field_type,
            errored: false,
        }
    }
}

impl JsonValueParser for JsonUnsupportedParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        if self.errored {
            return JsonStep {
                consumed: chunk.len(),
                events: Vec::new(),
                completion: JsonCompletion::Errored("unsupported field type".into()),
            };
        }
        self.errored = true;
        let msg = format!(
            "JSON value parser for FieldType '{}' is not yet implemented in the \
             streaming parser",
            self.field_type
        );
        JsonStep {
            consumed: chunk.len(),
            events: vec![ParseEvent::StreamError {
                path: Some(self.path.clone()),
                message: msg.clone(),
            }],
            completion: JsonCompletion::Errored(msg),
        }
    }

    fn finish(self: Box<Self>, _path: &str) -> (Vec<ParseEvent>, FieldValue) {
        // If the parser was never pushed-to (empty field), still surface
        // the unsupported-type error so the caller knows why.
        if self.errored {
            (Vec::new(), FieldValue::Null)
        } else {
            (
                vec![ParseEvent::StreamError {
                    path: Some(self.path.clone()),
                    message: format!(
                        "FieldType '{}' is not yet supported by the streaming parser",
                        self.field_type
                    ),
                }],
                FieldValue::Null,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_error_on_first_push() {
        let mut p =
            JsonUnsupportedParser::new("/x".into(), FieldType::Map(Box::new(FieldType::String)));
        let step = p.push("{}");
        match step.completion {
            JsonCompletion::Errored(msg) => assert!(msg.contains("not yet implemented")),
            other => panic!("expected Errored, got {other:?}"),
        }
        assert_eq!(step.events.len(), 1);
        assert!(matches!(&step.events[0], ParseEvent::StreamError { .. }));
    }

    #[test]
    fn finish_without_push_emits_error_too() {
        let p = JsonUnsupportedParser::new("/x".into(), FieldType::Int);
        let (events, _) = Box::new(p).finish("/x");
        assert_eq!(events.len(), 1);
    }
}
