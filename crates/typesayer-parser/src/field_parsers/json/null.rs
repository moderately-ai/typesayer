// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON `null` literal parser.
//!
//! Trie-matches `null` byte-by-byte across chunk boundaries via the
//! `matched: usize` cursor. Used internally by `JsonNullableParser`
//! when it has peeked an `n` byte and routes to the null-literal path.
//! Not exposed via `json_value_parser_for` — there's no `FieldType` for
//! "always null" so a standalone null parser has no caller.

use typesayer_types::field::FieldValue;

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser, skip_whitespace,
};

const NULL_LITERAL: &str = "null";

pub(super) struct JsonNullParser {
    path: String,
    state: State,
}

#[derive(Debug)]
enum State {
    BeforeStart,
    InNull { matched: usize },
    Done,
    Errored,
}

impl JsonNullParser {
    pub(super) const fn new(path: String) -> Self {
        Self {
            path,
            state: State::BeforeStart,
        }
    }

    fn err(&mut self, consumed: usize, msg: impl Into<String>) -> JsonStep {
        let msg = msg.into();
        self.state = State::Errored;
        JsonStep {
            consumed,
            events: vec![ParseEvent::StreamError {
                path: Some(self.path.clone()),
                message: msg.clone(),
            }],
            completion: JsonCompletion::Errored(msg),
        }
    }
}

impl JsonValueParser for JsonNullParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        let bytes = chunk.as_bytes();
        let mut cursor = 0;

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events: Vec::new(),
                    completion: match &self.state {
                        State::Done => JsonCompletion::Done(FieldValue::Null),
                        State::Errored => JsonCompletion::Errored("null parser errored".into()),
                        _ => JsonCompletion::NeedMore,
                    },
                };
            }

            match &mut self.state {
                State::Done => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: JsonCompletion::Done(FieldValue::Null),
                    };
                }
                State::Errored => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: JsonCompletion::Errored("null parser already errored".into()),
                    };
                }

                State::BeforeStart => {
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
                        cursor += 1;
                        self.state = State::InNull { matched: 1 };
                    } else {
                        let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                        return self.err(cursor, format!("expected 'null', got {bad:?}"));
                    }
                }

                State::InNull { matched } => {
                    let expected = NULL_LITERAL.as_bytes();
                    while *matched < expected.len() && cursor < bytes.len() {
                        if bytes[cursor] != expected[*matched] {
                            let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                            return self
                                .err(cursor, format!("expected 'null' continuation, got {bad:?}"));
                        }
                        *matched += 1;
                        cursor += 1;
                    }
                    if *matched == expected.len() {
                        self.state = State::Done;
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::Done(FieldValue::Null),
                        };
                    }
                    return JsonStep {
                        consumed: cursor,
                        events: Vec::new(),
                        completion: JsonCompletion::NeedMore,
                    };
                }
            }
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        match self.state {
            State::Done | State::Errored => (Vec::new(), FieldValue::Null),
            _ => (
                vec![ParseEvent::StreamError {
                    path: Some(path.to_owned()),
                    message: format!("unterminated null literal (state: {:?})", self.state),
                }],
                FieldValue::Null,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_null_literal() {
        let mut p = JsonNullParser::new("/p".into());
        let step = p.push("null,");
        assert!(matches!(
            step.completion,
            JsonCompletion::Done(FieldValue::Null)
        ));
        assert_eq!(step.consumed, 4);
    }

    #[test]
    fn leading_whitespace_skipped() {
        let mut p = JsonNullParser::new("/p".into());
        let step = p.push("  null]");
        assert!(matches!(
            step.completion,
            JsonCompletion::Done(FieldValue::Null)
        ));
    }

    #[test]
    fn rejects_uppercase() {
        let mut p = JsonNullParser::new("/p".into());
        let step = p.push("NULL");
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn split_at_every_byte_boundary() {
        let input = "null,";
        for split in 1..input.len() {
            let mut p = JsonNullParser::new("/p".into());
            let s1 = p.push(&input[..split]);
            let final_c = if let JsonCompletion::Done(v) = s1.completion {
                JsonCompletion::Done(v)
            } else {
                p.push(&input[split..]).completion
            };
            match final_c {
                JsonCompletion::Done(FieldValue::Null) => {}
                other => panic!("split={split}: got {other:?}"),
            }
        }
    }
}
