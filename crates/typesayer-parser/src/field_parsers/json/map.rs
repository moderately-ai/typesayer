// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON map parser for `FieldType::Map(value_type)`.
//!
//! Parses `{ "k1": v1, "k2": v2, ... }` where every value is typed
//! by the same declared `value_type`. State machine mirrors
//! [`super::object::JsonObjectParser`] minus the per-key field
//! lookup — every key contributes to the result, every value uses
//! the same parser dispatch.
//!
//! Buffered parity (`format.rs:262-273`): silently accepts any
//! string-keyed JSON object and types every value via the declared
//! value type. Emits `ObjectOpen` / `EntryAdded` / `ObjectClose`
//! event family (Maps and Objects share the same wire events on the
//! consumer's streaming event list).

use std::collections::BTreeMap;

use typesayer_types::field::{FieldType, FieldValue};

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser,
    json_value_parser_for, skip_whitespace, string::JsonStringParser,
};

pub(super) struct JsonMapParser {
    path: String,
    value_type: FieldType,
    state: State,
    entries: BTreeMap<String, FieldValue>,
    sent_open: bool,
}

enum State {
    BeforeOpen,
    AfterOpen,
    InKey {
        parser: Box<JsonStringParser>,
    },
    AfterKey {
        key: String,
    },
    BeforeValue {
        key: String,
    },
    InValue {
        parser: Box<dyn JsonValueParser>,
        key: String,
    },
    AfterValue,
    Done,
    Errored,
}

impl JsonMapParser {
    pub(super) const fn new(path: String, value_type: FieldType) -> Self {
        Self {
            path,
            value_type,
            state: State::BeforeOpen,
            entries: BTreeMap::new(),
            sent_open: false,
        }
    }

    fn entry_path(&self, key: &str) -> String {
        format!("{}/{key}", self.path)
    }

    fn build_close(&mut self) -> (ParseEvent, FieldValue) {
        let value = FieldValue::Object(std::mem::take(&mut self.entries));
        let close = ParseEvent::ObjectClose {
            path: self.path.clone(),
            value: value.clone(),
        };
        (close, value)
    }

    fn err(
        &mut self,
        consumed: usize,
        events: Vec<ParseEvent>,
        msg: impl Into<String>,
    ) -> JsonStep {
        let msg = msg.into();
        let mut events = events;
        let inner_already_errored = events
            .iter()
            .any(|e| matches!(e, ParseEvent::StreamError { .. }));
        self.state = State::Errored;
        if !inner_already_errored {
            events.push(ParseEvent::StreamError {
                path: Some(self.path.clone()),
                message: msg.clone(),
            });
        }
        JsonStep {
            consumed,
            events,
            completion: JsonCompletion::Errored(msg),
        }
    }
}

impl JsonValueParser for JsonMapParser {
    #[expect(
        clippy::too_many_lines,
        reason = "single-pass JSON map state machine; structurally identical to JsonObjectParser \
                  minus the field-lookup step"
    )]
    fn push(&mut self, chunk: &str) -> JsonStep {
        let bytes = chunk.as_bytes();
        let mut cursor = 0;
        let mut events: Vec<ParseEvent> = Vec::new();

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events,
                    completion: match &self.state {
                        State::Done => {
                            JsonCompletion::Done(FieldValue::Object(self.entries.clone()))
                        }
                        State::Errored => JsonCompletion::Errored("map parser errored".into()),
                        _ => JsonCompletion::NeedMore,
                    },
                };
            }

            match &mut self.state {
                State::Done => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events,
                        completion: JsonCompletion::Done(FieldValue::Object(self.entries.clone())),
                    };
                }
                State::Errored => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events,
                        completion: JsonCompletion::Errored("map parser already errored".into()),
                    };
                }

                State::BeforeOpen => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events,
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    if bytes[cursor] == b'{' {
                        cursor += 1;
                        self.state = State::AfterOpen;
                        if !self.sent_open {
                            events.push(ParseEvent::ObjectOpen {
                                path: self.path.clone(),
                            });
                            self.sent_open = true;
                        }
                    } else {
                        let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                        return self.err(
                            cursor,
                            events,
                            format!("expected '{{' to start map, got {bad:?}"),
                        );
                    }
                }

                State::AfterOpen => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events,
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    if bytes[cursor] == b'}' {
                        cursor += 1;
                        let (close, value) = self.build_close();
                        events.push(close);
                        self.state = State::Done;
                        return JsonStep {
                            consumed: cursor,
                            events,
                            completion: JsonCompletion::Done(value),
                        };
                    }
                    self.state = State::InKey {
                        parser: Box::new(JsonStringParser::new(format!("{}#key", self.path))),
                    };
                }

                State::InKey { parser } => {
                    let step = parser.push(&chunk[cursor..]);
                    cursor += step.consumed;
                    events.extend(step.events);
                    match step.completion {
                        JsonCompletion::NeedMore => {
                            return JsonStep {
                                consumed: cursor,
                                events,
                                completion: JsonCompletion::NeedMore,
                            };
                        }
                        JsonCompletion::Done(FieldValue::Str(key)) => {
                            self.state = State::AfterKey { key };
                        }
                        JsonCompletion::Done(other) => {
                            return self.err(
                                cursor,
                                events,
                                format!("map key parser yielded non-string {other:?}"),
                            );
                        }
                        JsonCompletion::Errored(msg) => {
                            return self.err(cursor, events, format!("map key failed: {msg}"));
                        }
                    }
                }

                State::AfterKey { key: _ } => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events,
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    if bytes[cursor] == b':' {
                        cursor += 1;
                        let State::AfterKey { key } =
                            std::mem::replace(&mut self.state, State::AfterValue)
                        else {
                            unreachable!("state was AfterKey before mem::replace")
                        };
                        self.state = State::BeforeValue { key };
                    } else {
                        let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                        return self.err(
                            cursor,
                            events,
                            format!("expected ':' after map key, got {bad:?}"),
                        );
                    }
                }

                State::BeforeValue { key: _ } => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events,
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    let State::BeforeValue { key } =
                        std::mem::replace(&mut self.state, State::AfterValue)
                    else {
                        unreachable!("state was BeforeValue before mem::replace")
                    };
                    let value_path = self.entry_path(&key);
                    let parser = json_value_parser_for(&self.value_type, value_path);
                    self.state = State::InValue { parser, key };
                }

                State::InValue { parser, key: _ } => {
                    let step = parser.push(&chunk[cursor..]);
                    cursor += step.consumed;
                    events.extend(step.events);
                    match step.completion {
                        JsonCompletion::NeedMore => {
                            return JsonStep {
                                consumed: cursor,
                                events,
                                completion: JsonCompletion::NeedMore,
                            };
                        }
                        JsonCompletion::Done(value) => {
                            let State::InValue { key, .. } =
                                std::mem::replace(&mut self.state, State::AfterValue)
                            else {
                                unreachable!("state was InValue before mem::replace")
                            };
                            events.push(ParseEvent::EntryAdded {
                                path: self.path.clone(),
                                key: key.clone(),
                                value: value.clone(),
                            });
                            self.entries.insert(key, value);
                        }
                        JsonCompletion::Errored(msg) => {
                            return self.err(cursor, events, format!("map value failed: {msg}"));
                        }
                    }
                }

                State::AfterValue => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events,
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    match bytes[cursor] {
                        b',' => {
                            cursor += 1;
                            let ws = skip_whitespace(&chunk[cursor..]);
                            cursor += ws;
                            if cursor >= bytes.len() {
                                self.state = State::InKey {
                                    parser: Box::new(JsonStringParser::new(format!(
                                        "{}#key",
                                        self.path
                                    ))),
                                };
                                return JsonStep {
                                    consumed: cursor,
                                    events,
                                    completion: JsonCompletion::NeedMore,
                                };
                            }
                            if bytes[cursor] == b'}' {
                                return self.err(
                                    cursor,
                                    events,
                                    "trailing comma in map (got ',' then '}')",
                                );
                            }
                            self.state = State::InKey {
                                parser: Box::new(JsonStringParser::new(format!(
                                    "{}#key",
                                    self.path
                                ))),
                            };
                        }
                        b'}' => {
                            cursor += 1;
                            let (close, value) = self.build_close();
                            events.push(close);
                            self.state = State::Done;
                            return JsonStep {
                                consumed: cursor,
                                events,
                                completion: JsonCompletion::Done(value),
                            };
                        }
                        other => {
                            let bad = char::from(other);
                            return self.err(
                                cursor,
                                events,
                                format!("expected ',' or '}}' after map value, got {bad:?}"),
                            );
                        }
                    }
                }
            }
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        let Self { entries, state, .. } = *self;
        let value = FieldValue::Object(entries);
        if matches!(state, State::Done) {
            return (Vec::new(), value);
        }
        let msg = format!("map parsing incomplete at end of input (state: {state:?})");
        (
            vec![ParseEvent::StreamError {
                path: Some(path.to_owned()),
                message: msg,
            }],
            value,
        )
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforeOpen => f.write_str("BeforeOpen"),
            Self::AfterOpen => f.write_str("AfterOpen"),
            Self::InKey { .. } => f.write_str("InKey"),
            Self::AfterKey { key } => write!(f, "AfterKey(key={key:?})"),
            Self::BeforeValue { key } => write!(f, "BeforeValue(key={key:?})"),
            Self::InValue { key, .. } => write!(f, "InValue(key={key:?})"),
            Self::AfterValue => f.write_str("AfterValue"),
            Self::Done => f.write_str("Done"),
            Self::Errored => f.write_str("Errored"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser(value_type: FieldType) -> JsonMapParser {
        JsonMapParser::new("/tags".into(), value_type)
    }

    #[test]
    fn parses_map_of_strings() {
        let mut p = parser(FieldType::String);
        let step = p.push(r#"{"a": "1", "b": "2"}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                assert_eq!(map.get("a"), Some(&FieldValue::Str("1".into())));
                assert_eq!(map.get("b"), Some(&FieldValue::Str("2".into())));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parses_map_of_ints() {
        let mut p = parser(FieldType::Int);
        let step = p.push(r#"{"a": 1, "b": 2, "c": 3}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                assert_eq!(map.get("a"), Some(&FieldValue::Int(1)));
                assert_eq!(map.get("b"), Some(&FieldValue::Int(2)));
                assert_eq!(map.get("c"), Some(&FieldValue::Int(3)));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn empty_map() {
        let mut p = parser(FieldType::String);
        let step = p.push("{}");
        assert!(matches!(
            step.completion,
            JsonCompletion::Done(FieldValue::Object(_))
        ));
    }

    #[test]
    fn chunk_boundary_at_every_split() {
        let input = r#"{"a": 1, "b": 2}"#;
        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = parser(FieldType::Int);
            let s1 = p.push(&input[..split]);
            let final_c = if let JsonCompletion::Done(v) = s1.completion {
                JsonCompletion::Done(v)
            } else {
                p.push(&input[split..]).completion
            };
            match final_c {
                JsonCompletion::Done(FieldValue::Object(map)) => {
                    assert_eq!(map.len(), 2);
                }
                other => panic!("split={split}: got {other:?}"),
            }
        }
    }
}
