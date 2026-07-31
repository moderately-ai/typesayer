// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON enum parser. Wraps `JsonStringParser` and
//! validates the parsed string against the declared variant set at
//! `Done`. Mismatch becomes `Errored` with a message naming the bad
//! value and the allowed set — matches the buffered `format.rs:276-285`
//! validation behavior verbatim (case-sensitive exact match).
//!
//! Used by `json_value_parser_for(FieldType::Enum(variants), ...)`.
//! For nested JSON contexts (inside an array/object/map) the LM
//! emits the variant as a quoted JSON string (`"red"`), so we delegate
//! to the string parser and then validate.

use typesayer_types::field::FieldValue;

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser,
    string::JsonStringParser,
};

pub(super) struct JsonEnumParser {
    path: String,
    variants: Vec<String>,
    inner: Box<JsonStringParser>,
    state: State,
}

#[derive(Debug)]
enum State {
    Parsing,
    Done(FieldValue),
    Errored,
}

impl JsonEnumParser {
    pub(super) fn new(path: String, variants: Vec<String>) -> Self {
        Self {
            inner: Box::new(JsonStringParser::new(path.clone())),
            path,
            variants,
            state: State::Parsing,
        }
    }

    fn validate_and_complete(&mut self, parsed: String) -> JsonCompletion {
        if self.variants.iter().any(|v| v == &parsed) {
            let value = FieldValue::Str(parsed);
            self.state = State::Done(value.clone());
            JsonCompletion::Done(value)
        } else {
            let msg = format!(
                "value {parsed:?} is not one of the declared enum variants: {:?}",
                self.variants
            );
            self.state = State::Errored;
            JsonCompletion::Errored(msg)
        }
    }
}

impl JsonValueParser for JsonEnumParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        match &self.state {
            State::Done(v) => {
                return JsonStep {
                    consumed: chunk.len(),
                    events: Vec::new(),
                    completion: JsonCompletion::Done(v.clone()),
                };
            }
            State::Errored => {
                return JsonStep {
                    consumed: chunk.len(),
                    events: Vec::new(),
                    completion: JsonCompletion::Errored("enum parser already errored".into()),
                };
            }
            State::Parsing => {}
        }
        let step = self.inner.push(chunk);
        let JsonStep {
            consumed,
            mut events,
            completion,
        } = step;
        let completion = match completion {
            JsonCompletion::NeedMore => JsonCompletion::NeedMore,
            JsonCompletion::Done(FieldValue::Str(s)) => {
                let result = self.validate_and_complete(s);
                if let JsonCompletion::Errored(ref msg) = result {
                    events.push(ParseEvent::StreamError {
                        path: Some(self.path.clone()),
                        message: msg.clone(),
                    });
                }
                result
            }
            JsonCompletion::Done(other) => {
                let msg =
                    format!("enum parser expected a JSON string from inner parser, got {other:?}");
                self.state = State::Errored;
                events.push(ParseEvent::StreamError {
                    path: Some(self.path.clone()),
                    message: msg.clone(),
                });
                JsonCompletion::Errored(msg)
            }
            JsonCompletion::Errored(msg) => {
                self.state = State::Errored;
                JsonCompletion::Errored(msg)
            }
        };
        JsonStep {
            consumed,
            events,
            completion,
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        match self.state {
            State::Done(v) => (Vec::new(), v),
            State::Errored => (Vec::new(), FieldValue::Null),
            State::Parsing => self.inner.finish(path),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> JsonEnumParser {
        JsonEnumParser::new(
            "/color".into(),
            vec!["red".into(), "green".into(), "blue".into()],
        )
    }

    #[test]
    fn parses_valid_variant() {
        let mut p = parser();
        let step = p.push(r#""red","#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "red"),
            other => panic!("got {other:?}"),
        }
        assert!(step.events.is_empty());
    }

    #[test]
    fn rejects_unknown_variant() {
        let mut p = parser();
        let step = p.push(r#""yellow","#);
        match step.completion {
            JsonCompletion::Errored(msg) => {
                assert!(msg.contains("yellow"));
                assert!(msg.contains("red"));
            }
            other => panic!("got {other:?}"),
        }
        // Emits an error event so the wrapping FieldParser surfaces it
        // on the wire.
        assert_eq!(step.events.len(), 1);
        assert!(matches!(step.events[0], ParseEvent::StreamError { .. }));
    }

    #[test]
    fn case_sensitive() {
        let mut p = parser();
        let step = p.push(r#""Red","#);
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn chunk_boundary_mid_variant() {
        let mut p = parser();
        let s1 = p.push(r#""gre"#);
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push(r#"en","#);
        match s2.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "green"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn split_at_every_byte_boundary() {
        let input = r#""blue","#;
        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = parser();
            let s1 = p.push(&input[..split]);
            let final_c = if let JsonCompletion::Done(v) = s1.completion {
                JsonCompletion::Done(v)
            } else {
                p.push(&input[split..]).completion
            };
            match final_c {
                JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "blue"),
                other => panic!("split={split}: got {other:?}"),
            }
        }
    }
}
