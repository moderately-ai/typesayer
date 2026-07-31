// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental `Nullable<T>` JSON value parser.
//!
//! Disambiguates the `null` literal from `T`'s value form by peeking
//! the first non-whitespace byte and routing to either
//! [`super::null::JsonNullParser`] (when the byte is `n`) or lazily
//! constructing the inner type's parser via
//! [`super::json_value_parser_for`] for everything else. Recursive
//! nesting (`Nullable<List<Int>>`, `Nullable<Object<...>>`, etc.) flows
//! through the same dispatch.
//!
//! Buffered parity (`format.rs:288-294`): inside a JSON envelope, the
//! buffered path round-trips through `serde_json::Value`, which renders
//! `null` as `Value::Null` and any non-null as the inner shape. We
//! follow the same dispatch — `null` literal → `FieldValue::Null`;
//! otherwise delegate to the inner parser.

use typesayer_types::field::{FieldType, FieldValue};

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser,
    json_value_parser_for, null::JsonNullParser, skip_whitespace,
};

pub(super) struct JsonNullableParser {
    path: String,
    inner_type: FieldType,
    state: State,
}

enum State {
    /// Skipping leading whitespace; awaiting the first non-whitespace
    /// byte so we can choose null-path vs inner-type-path. The
    /// `held` buffer captures any whitespace we've already consumed
    /// in case the inner parser needs to see it too — but in practice
    /// every parser starts with its own whitespace-skip, so we drop
    /// it here.
    Choosing,
    /// Delegating to the null-literal parser.
    DelegatingNull(Box<JsonNullParser>),
    /// Delegating to the inner type's parser.
    DelegatingInner(Box<dyn JsonValueParser>),
    Done(FieldValue),
    Errored,
}

impl JsonNullableParser {
    pub(super) const fn new(path: String, inner_type: FieldType) -> Self {
        Self {
            path,
            inner_type,
            state: State::Choosing,
        }
    }
}

impl JsonValueParser for JsonNullableParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        let bytes = chunk.as_bytes();
        let mut cursor = 0;

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events: Vec::new(),
                    completion: match &self.state {
                        State::Done(v) => JsonCompletion::Done(v.clone()),
                        State::Errored => JsonCompletion::Errored("nullable parser errored".into()),
                        _ => JsonCompletion::NeedMore,
                    },
                };
            }

            match &mut self.state {
                State::Done(v) => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: JsonCompletion::Done(v.clone()),
                    };
                }
                State::Errored => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: JsonCompletion::Errored(
                            "nullable parser already errored".into(),
                        ),
                    };
                }

                State::Choosing => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    if bytes[cursor] == b'n' {
                        // Don't advance cursor — the null parser
                        // expects to see the `n` itself.
                        self.state =
                            State::DelegatingNull(Box::new(JsonNullParser::new(self.path.clone())));
                    } else {
                        // Construct the inner parser lazily so the
                        // `Nullable<expensive_to_build>` case doesn't
                        // pay the cost on a null value.
                        let inner = json_value_parser_for(&self.inner_type, self.path.clone());
                        self.state = State::DelegatingInner(inner);
                    }
                }

                State::DelegatingNull(parser) => {
                    let step = parser.push(&chunk[cursor..]);
                    cursor += step.consumed;
                    return finalize(&mut self.state, cursor, step);
                }

                State::DelegatingInner(parser) => {
                    let step = parser.push(&chunk[cursor..]);
                    cursor += step.consumed;
                    return finalize(&mut self.state, cursor, step);
                }
            }
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        let Self { state, .. } = *self;
        match state {
            State::Done(v) => (Vec::new(), v),
            State::Errored => (Vec::new(), FieldValue::Null),
            State::DelegatingNull(parser) => parser.finish(path),
            State::DelegatingInner(parser) => parser.finish(path),
            State::Choosing => (
                vec![ParseEvent::StreamError {
                    path: Some(path.to_owned()),
                    message: "nullable parser had no value at end-of-input".to_owned(),
                }],
                FieldValue::Null,
            ),
        }
    }
}

/// Convert a delegated parser's `JsonStep` into the wrapping
/// nullable parser's own `JsonStep`, updating local state to
/// `Done`/`Errored` as appropriate. Pulled out so both delegation
/// arms share the same finalisation logic without duplicating the
/// match.
fn finalize(state: &mut State, consumed: usize, step: JsonStep) -> JsonStep {
    let JsonStep {
        events, completion, ..
    } = step;
    let completion = match completion {
        JsonCompletion::NeedMore => JsonCompletion::NeedMore,
        JsonCompletion::Done(v) => {
            *state = State::Done(v.clone());
            JsonCompletion::Done(v)
        }
        JsonCompletion::Errored(msg) => {
            *state = State::Errored;
            JsonCompletion::Errored(msg)
        }
    };
    JsonStep {
        consumed,
        events,
        completion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nullable_string_parser() -> JsonNullableParser {
        JsonNullableParser::new("/p".into(), FieldType::String)
    }

    fn nullable_int_parser() -> JsonNullableParser {
        JsonNullableParser::new("/p".into(), FieldType::Int)
    }

    #[test]
    fn null_literal_yields_null() {
        let mut p = nullable_string_parser();
        let step = p.push("null,");
        assert!(matches!(
            step.completion,
            JsonCompletion::Done(FieldValue::Null)
        ));
    }

    #[test]
    fn non_null_delegates_to_string_parser() {
        let mut p = nullable_string_parser();
        let step = p.push(r#""hello","#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "hello"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn non_null_delegates_to_int_parser() {
        let mut p = nullable_int_parser();
        let step = p.push("42,");
        match step.completion {
            JsonCompletion::Done(FieldValue::Int(v)) => assert_eq!(v, 42),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn nested_nullable_of_list() {
        let mut p =
            JsonNullableParser::new("/p".into(), FieldType::List(Box::new(FieldType::String)));
        let step = p.push(r#"["a", "b"],"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::List(v)) => assert_eq!(v.len(), 2),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn leading_whitespace_before_null() {
        let mut p = nullable_string_parser();
        let step = p.push("   null,");
        assert!(matches!(
            step.completion,
            JsonCompletion::Done(FieldValue::Null)
        ));
    }

    #[test]
    fn chunk_boundary_mid_null() {
        let mut p = nullable_string_parser();
        let s1 = p.push("nu");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push("ll,");
        assert!(matches!(
            s2.completion,
            JsonCompletion::Done(FieldValue::Null)
        ));
    }

    #[test]
    fn chunk_boundary_mid_int() {
        let mut p = nullable_int_parser();
        let s1 = p.push("12");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push("345,");
        match s2.completion {
            JsonCompletion::Done(FieldValue::Int(v)) => assert_eq!(v, 12345),
            other => panic!("got {other:?}"),
        }
    }
}
