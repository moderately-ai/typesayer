// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON boolean parser.
//!
//! Strict JSON per RFC 8259 §3: exactly `true` or `false`, lowercase.
//! Top-level Bool fields take a different parser
//! (`field_parsers::bool_top::BoolTopLevelParser`) that's permissive
//! about `yes`/`no`/`1`/`0` to match the buffered parser's tolerance.
//! Inside JSON envelopes (arrays, objects, maps), the parser stays
//! strict because that's what serde would have produced for the
//! buffered path.

use typesayer_types::field::FieldValue;

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser, skip_whitespace,
};

const TRUE_LITERAL: &str = "true";
const FALSE_LITERAL: &str = "false";

/// Strict JSON bool parser. Trie-matches `true`/`false` byte-by-byte
/// across chunk boundaries via the `matched: usize` cursor.
pub(super) struct JsonBoolParser {
    path: String,
    state: State,
}

#[derive(Debug)]
enum State {
    /// Skipping leading whitespace; awaiting `t` or `f`.
    BeforeStart,
    /// Matching against `"true"`; `matched` bytes done so far (1..=3).
    InTrue {
        matched: usize,
    },
    /// Matching against `"false"`; `matched` bytes done so far (1..=4).
    InFalse {
        matched: usize,
    },
    Done(bool),
    Errored,
}

impl JsonBoolParser {
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

impl JsonValueParser for JsonBoolParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        let bytes = chunk.as_bytes();
        let mut cursor = 0;

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events: Vec::new(),
                    completion: match &self.state {
                        State::Done(b) => JsonCompletion::Done(FieldValue::Bool(*b)),
                        State::Errored => JsonCompletion::Errored("bool parser errored".into()),
                        _ => JsonCompletion::NeedMore,
                    },
                };
            }

            match &mut self.state {
                State::Done(b) => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: JsonCompletion::Done(FieldValue::Bool(*b)),
                    };
                }
                State::Errored => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: JsonCompletion::Errored("bool parser already errored".into()),
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
                    match bytes[cursor] {
                        b't' => {
                            cursor += 1;
                            self.state = State::InTrue { matched: 1 };
                        }
                        b'f' => {
                            cursor += 1;
                            self.state = State::InFalse { matched: 1 };
                        }
                        _ => {
                            let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                            return self
                                .err(cursor, format!("expected 'true' or 'false', got {bad:?}"));
                        }
                    }
                }

                State::InTrue { matched } => {
                    let expected = TRUE_LITERAL.as_bytes();
                    while *matched < expected.len() && cursor < bytes.len() {
                        if bytes[cursor] != expected[*matched] {
                            let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                            return self.err(
                                cursor,
                                format!("expected '{TRUE_LITERAL}' continuation, got {bad:?}"),
                            );
                        }
                        *matched += 1;
                        cursor += 1;
                    }
                    if *matched == expected.len() {
                        self.state = State::Done(true);
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::Done(FieldValue::Bool(true)),
                        };
                    }
                    // Ran out of bytes mid-match.
                    return JsonStep {
                        consumed: cursor,
                        events: Vec::new(),
                        completion: JsonCompletion::NeedMore,
                    };
                }

                State::InFalse { matched } => {
                    let expected = FALSE_LITERAL.as_bytes();
                    while *matched < expected.len() && cursor < bytes.len() {
                        if bytes[cursor] != expected[*matched] {
                            let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                            return self.err(
                                cursor,
                                format!("expected '{FALSE_LITERAL}' continuation, got {bad:?}"),
                            );
                        }
                        *matched += 1;
                        cursor += 1;
                    }
                    if *matched == expected.len() {
                        self.state = State::Done(false);
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::Done(FieldValue::Bool(false)),
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
            State::Done(b) => (Vec::new(), FieldValue::Bool(b)),
            State::Errored => (Vec::new(), FieldValue::Null),
            _ => (
                vec![ParseEvent::StreamError {
                    path: Some(path.to_owned()),
                    message: format!("unterminated JSON bool (state: {:?})", self.state),
                }],
                FieldValue::Null,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> JsonStep {
        JsonBoolParser::new("/p".into()).push(input)
    }

    fn expect_done(step: JsonStep, expected: bool) -> usize {
        match step.completion {
            JsonCompletion::Done(FieldValue::Bool(v)) => assert_eq!(v, expected),
            other => panic!("expected Done(Bool({expected})), got {other:?}"),
        }
        assert!(step.events.is_empty());
        step.consumed
    }

    #[test]
    fn parses_true() {
        assert_eq!(expect_done(parse("true,"), true), 4);
    }

    #[test]
    fn parses_false() {
        assert_eq!(expect_done(parse("false]"), false), 5);
    }

    #[test]
    fn leading_whitespace_skipped() {
        assert_eq!(expect_done(parse("   true "), true), 7);
    }

    #[test]
    fn rejects_uppercase() {
        let step = parse("True");
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn rejects_yes() {
        let step = parse("yes");
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn chunk_boundary_mid_true() {
        let mut p = JsonBoolParser::new("/p".into());
        let s1 = p.push("tr");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push("ue,");
        match s2.completion {
            JsonCompletion::Done(FieldValue::Bool(true)) => {}
            other => panic!("got {other:?}"),
        }
        assert_eq!(s2.consumed, 2);
    }

    #[test]
    fn chunk_boundary_mid_false() {
        for split in 1..5 {
            let mut p = JsonBoolParser::new("/p".into());
            let s1 = p.push(&"false]"[..split]);
            let _ = s1;
            let s2 = p.push(&"false]"[split..]);
            match s2.completion {
                JsonCompletion::Done(FieldValue::Bool(false)) => {}
                other => panic!("split={split}: got {other:?}"),
            }
        }
    }

    #[test]
    fn split_at_every_byte_boundary_true() {
        let input = "true,";
        for split in 1..input.len() {
            let mut p = JsonBoolParser::new("/p".into());
            let s1 = p.push(&input[..split]);
            let final_c = match s1.completion {
                JsonCompletion::Done(v) => JsonCompletion::Done(v),
                _ => p.push(&input[split..]).completion,
            };
            match final_c {
                JsonCompletion::Done(FieldValue::Bool(true)) => {}
                other => panic!("split={split}: got {other:?}"),
            }
        }
    }
}
