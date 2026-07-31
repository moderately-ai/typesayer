// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consume exactly one JSON value's bytes and discard them.
//!
//! Used by [`super::object::JsonObjectParser`] when an LM-emitted key
//! isn't in the declared field list — the buffered parser silently
//! ignores undeclared keys (`format.rs:251-258` only iterates the
//! declared `object_fields`), so the streaming codec needs a way to
//! drain the value's syntactic span without typing it. Returns
//! `FieldValue::Null` as a placeholder; the caller is expected to
//! discard the value.
//!
//! Operates as a single byte-walker tracking three modes — string,
//! container, atomic — without recursing into any other parser, so
//! it has no circular dependency on the type-dispatched parsers.

use typesayer_types::field::FieldValue;

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser, skip_whitespace,
};

pub(super) struct JsonSkipParser {
    path: String,
    state: State,
}

enum State {
    /// Skipping leading whitespace; awaiting the first content byte
    /// to decide which mode to enter.
    BeforeStart,
    /// Inside a JSON string. `escaped` is true when the previous byte
    /// was an unconsumed backslash so the next byte is part of an
    /// escape sequence.
    InString {
        escaped: bool,
    },
    /// Inside a `{...}` or `[...]` container. `depth` is the current
    /// bracket-balance count (1 at entry, must reach 0 to exit). The
    /// `in_string` flags pause depth tracking so brackets inside a
    /// JSON string don't perturb the depth count.
    InContainer {
        depth: u32,
        in_string: bool,
        in_string_escaped: bool,
    },
    /// Inside a bare token (number, true, false, null). Walk until
    /// the first non-token byte; don't consume the terminator.
    InAtomic,
    Done,
    Errored,
}

impl JsonSkipParser {
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

/// True for bytes inside the bare-token charset (number / bool / null
/// literals share the same alphabetic + digit + sign + decimal-point
/// set; we accept the union to keep the byte walk simple).
const fn is_atomic_token_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'0'..=b'9'
            | b'a'..=b'z'
            | b'A'..=b'Z'
            | b'-'
            | b'+'
            | b'.'
            | b'_',
    )
}

impl JsonValueParser for JsonSkipParser {
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
                        State::Errored => JsonCompletion::Errored("skip parser errored".into()),
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
                        completion: JsonCompletion::Errored("skip parser already errored".into()),
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
                        b'"' => {
                            cursor += 1;
                            self.state = State::InString { escaped: false };
                        }
                        b'{' | b'[' => {
                            cursor += 1;
                            self.state = State::InContainer {
                                depth: 1,
                                in_string: false,
                                in_string_escaped: false,
                            };
                        }
                        b if is_atomic_token_byte(b) => {
                            self.state = State::InAtomic;
                        }
                        other => {
                            let bad = char::from(other);
                            return self
                                .err(cursor, format!("unexpected byte {bad:?} at start of value"));
                        }
                    }
                }

                State::InString { escaped } => {
                    while cursor < bytes.len() {
                        let b = bytes[cursor];
                        cursor += 1;
                        if *escaped {
                            *escaped = false;
                            continue;
                        }
                        match b {
                            b'\\' => *escaped = true,
                            b'"' => {
                                self.state = State::Done;
                                return JsonStep {
                                    consumed: cursor,
                                    events: Vec::new(),
                                    completion: JsonCompletion::Done(FieldValue::Null),
                                };
                            }
                            _ => {}
                        }
                    }
                    return JsonStep {
                        consumed: cursor,
                        events: Vec::new(),
                        completion: JsonCompletion::NeedMore,
                    };
                }

                State::InContainer {
                    depth,
                    in_string,
                    in_string_escaped,
                } => {
                    while cursor < bytes.len() {
                        let b = bytes[cursor];
                        cursor += 1;
                        if *in_string {
                            if *in_string_escaped {
                                *in_string_escaped = false;
                                continue;
                            }
                            match b {
                                b'\\' => *in_string_escaped = true,
                                b'"' => *in_string = false,
                                _ => {}
                            }
                            continue;
                        }
                        match b {
                            b'"' => *in_string = true,
                            b'{' | b'[' => *depth += 1,
                            b'}' | b']' => {
                                *depth -= 1;
                                if *depth == 0 {
                                    self.state = State::Done;
                                    return JsonStep {
                                        consumed: cursor,
                                        events: Vec::new(),
                                        completion: JsonCompletion::Done(FieldValue::Null),
                                    };
                                }
                            }
                            _ => {}
                        }
                    }
                    return JsonStep {
                        consumed: cursor,
                        events: Vec::new(),
                        completion: JsonCompletion::NeedMore,
                    };
                }

                State::InAtomic => {
                    while cursor < bytes.len() {
                        let b = bytes[cursor];
                        if is_atomic_token_byte(b) {
                            cursor += 1;
                            continue;
                        }
                        // Terminator (`,` `]` `}` whitespace, etc.) —
                        // value's bytes are exhausted; do not consume.
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
            // Atomic with no terminator (e.g. number ending right at
            // marker boundary) is legitimate — accept silently, as
            // are the explicit Done / Errored terminal states.
            State::Done | State::Errored | State::InAtomic => (Vec::new(), FieldValue::Null),
            other => (
                vec![ParseEvent::StreamError {
                    path: Some(path.to_owned()),
                    message: format!("skip parser unterminated (state: {other:?})"),
                }],
                FieldValue::Null,
            ),
        }
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforeStart => f.write_str("BeforeStart"),
            Self::InString { escaped } => write!(f, "InString(escaped={escaped})"),
            Self::InContainer { depth, .. } => write!(f, "InContainer(depth={depth})"),
            Self::InAtomic => f.write_str("InAtomic"),
            Self::Done => f.write_str("Done"),
            Self::Errored => f.write_str("Errored"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(input: &str) -> JsonStep {
        JsonSkipParser::new("/p".into()).push(input)
    }

    #[test]
    fn skips_string() {
        let step = run(r#""hello","#);
        assert!(matches!(
            step.completion,
            JsonCompletion::Done(FieldValue::Null)
        ));
        assert_eq!(step.consumed, 7);
    }

    #[test]
    fn skips_string_with_escapes() {
        let step = run(r#""hel\"lo","#);
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
    }

    #[test]
    fn skips_number() {
        let step = run("42,");
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
        assert_eq!(step.consumed, 2);
    }

    #[test]
    fn skips_true() {
        let step = run("true,");
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
    }

    #[test]
    fn skips_null() {
        let step = run("null,");
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
    }

    #[test]
    fn skips_array() {
        let step = run("[1, 2, 3],");
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
        assert_eq!(step.consumed, 9);
    }

    #[test]
    fn skips_nested_object() {
        let step = run(r#"{"a": {"b": [1, "two"]}, "c": 3},"#);
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
    }

    #[test]
    fn skips_array_with_bracket_in_string() {
        let step = run(r#"["[", "]"],"#);
        assert!(matches!(step.completion, JsonCompletion::Done(_)));
    }

    #[test]
    fn chunk_boundary_mid_container() {
        let mut p = JsonSkipParser::new("/p".into());
        let s1 = p.push(r"[1, 2");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push(", 3],");
        assert!(matches!(s2.completion, JsonCompletion::Done(_)));
    }

    #[test]
    fn chunk_boundary_mid_string() {
        let mut p = JsonSkipParser::new("/p".into());
        let s1 = p.push(r#""hel"#);
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push(r#"lo","#);
        assert!(matches!(s2.completion, JsonCompletion::Done(_)));
    }
}
