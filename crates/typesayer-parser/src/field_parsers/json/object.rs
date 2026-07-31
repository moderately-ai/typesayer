// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON object parser for `FieldType::Object(fields)`.
//!
//! Parses `{ "k1": v1, "k2": v2, ... }` with per-key dispatch driven
//! by the declared `Vec<ObjectField>`. Buffered parity
//! (`format.rs:246-260`): declared keys present in the JSON are
//! deserialised against their declared types; undeclared keys are
//! silently discarded (parsed and dropped via [`super::skip::JsonSkipParser`]);
//! declared keys missing from the JSON are filled with `FieldValue::Null`
//! at finish IF their declared type is `Nullable<_>`, otherwise stay
//! absent from the result map.
//!
//! Emits `ObjectOpen` on the opening `{`, `EntryAdded` per declared
//! key as its value completes, and `ObjectClose` carrying the full
//! `FieldValue::Object` at the closing `}`.

use std::collections::BTreeMap;

use typesayer_types::field::{FieldType, FieldValue, ObjectField};

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser,
    json_value_parser_for, skip::JsonSkipParser, skip_whitespace, string::JsonStringParser,
};

pub(super) struct JsonObjectParser {
    path: String,
    fields: Vec<ObjectField>,
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
        declared: bool,
    },
    AfterValue,
    Done,
    Errored,
}

impl JsonObjectParser {
    pub(super) const fn new(path: String, fields: Vec<ObjectField>) -> Self {
        Self {
            path,
            fields,
            state: State::BeforeOpen,
            entries: BTreeMap::new(),
            sent_open: false,
        }
    }

    fn entry_path(&self, key: &str) -> String {
        // RFC 6901 §4: tokens are separated by `/`; keys with `/` or `~`
        // would need escaping. Most schemas avoid those — keep the
        // simple form for v1.
        format!("{}/{key}", self.path)
    }

    /// Look up the declared type for `key`. `None` for undeclared
    /// keys; the caller dispatches to [`JsonSkipParser`] to discard
    /// the value.
    fn declared_type(&self, key: &str) -> Option<FieldType> {
        self.fields
            .iter()
            .find(|f| f.name == key)
            .map(|f| f.field_type.clone())
    }

    /// Fill in `FieldValue::Null` for any declared `Nullable<_>` field
    /// that the JSON didn't supply. Matches buffered for `Nullable`
    /// fields specifically; non-nullable missing fields stay absent.
    fn fill_missing_nullable_fields(&mut self) {
        for f in &self.fields {
            if !self.entries.contains_key(&f.name) && matches!(f.field_type, FieldType::Nullable(_))
            {
                self.entries.insert(f.name.clone(), FieldValue::Null);
            }
        }
    }

    fn build_close(&mut self) -> (ParseEvent, FieldValue) {
        self.fill_missing_nullable_fields();
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
        // Suppress wrapper StreamError when an inner parser already
        // emitted one — one cause, one event. Mirrors the dedup the
        // array parser will get in Step 5.
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

impl JsonValueParser for JsonObjectParser {
    #[expect(
        clippy::too_many_lines,
        reason = "single-pass JSON object state machine; splitting it into per-state helpers \
                  would fragment the chunk-boundary handling without a readability win"
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
                        State::Errored => JsonCompletion::Errored("object parser errored".into()),
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
                        completion: JsonCompletion::Errored("object parser already errored".into()),
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
                            format!("expected '{{' to start object, got {bad:?}"),
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
                    // Not `}` — must be the start of a key. JsonStringParser
                    // owns the quote-handling; transition with a fresh
                    // parser. We do NOT advance cursor.
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
                                format!("object key parser yielded non-string {other:?}"),
                            );
                        }
                        JsonCompletion::Errored(msg) => {
                            return self.err(cursor, events, format!("object key failed: {msg}"));
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
                        // Move the key out — we're about to spawn the
                        // value parser and need it on State::BeforeValue.
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
                            format!("expected ':' after object key, got {bad:?}"),
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
                    let (parser, declared): (Box<dyn JsonValueParser>, bool) =
                        if let Some(field_type) = self.declared_type(&key) {
                            (json_value_parser_for(&field_type, value_path), true)
                        } else {
                            // Undeclared key — silently discard the
                            // value (matches buffered).
                            (Box::new(JsonSkipParser::new(value_path)), false)
                        };
                    self.state = State::InValue {
                        parser,
                        key,
                        declared,
                    };
                }

                State::InValue {
                    parser,
                    key: _,
                    declared: _,
                } => {
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
                            // Extract key + declared flag by replacing
                            // the variant — the inner parser is dropped.
                            let State::InValue { key, declared, .. } =
                                std::mem::replace(&mut self.state, State::AfterValue)
                            else {
                                unreachable!("state was InValue before mem::replace")
                            };
                            if declared {
                                events.push(ParseEvent::EntryAdded {
                                    path: self.path.clone(),
                                    key: key.clone(),
                                    value: value.clone(),
                                });
                                self.entries.insert(key, value);
                            }
                            // self.state already = AfterValue.
                        }
                        JsonCompletion::Errored(msg) => {
                            return self.err(cursor, events, format!("object value failed: {msg}"));
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
                            // Skip whitespace after `,` so we land on
                            // the next key's opening `"`.
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
                                    "trailing comma in object (got ',' then '}')",
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
                                format!("expected ',' or '}}' after object value, got {bad:?}"),
                            );
                        }
                    }
                }
            }
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        let Self {
            mut entries,
            fields,
            state,
            ..
        } = *self;
        if matches!(state, State::Done) {
            return (Vec::new(), FieldValue::Object(entries));
        }
        // Defensive partial-value return — fill missing nullable
        // fields and surface the incomplete state as an error.
        for f in &fields {
            if !entries.contains_key(&f.name) && matches!(f.field_type, FieldType::Nullable(_)) {
                entries.insert(f.name.clone(), FieldValue::Null);
            }
        }
        let value = FieldValue::Object(entries);
        let msg = format!("object parsing incomplete at end of input (state: {state:?})");
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
            Self::InValue { key, declared, .. } => {
                write!(f, "InValue(key={key:?}, declared={declared})")
            }
            Self::AfterValue => f.write_str("AfterValue"),
            Self::Done => f.write_str("Done"),
            Self::Errored => f.write_str("Errored"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person_fields() -> Vec<ObjectField> {
        vec![
            ObjectField {
                name: "name".into(),
                description: String::new(),
                field_type: FieldType::String,
            },
            ObjectField {
                name: "age".into(),
                description: String::new(),
                field_type: FieldType::Int,
            },
        ]
    }

    fn parser(fields: Vec<ObjectField>) -> JsonObjectParser {
        JsonObjectParser::new("/user".into(), fields)
    }

    #[test]
    fn parses_simple_object() {
        let mut p = parser(person_fields());
        let step = p.push(r#"{"name": "Alice", "age": 30}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                assert_eq!(map.get("name"), Some(&FieldValue::Str("Alice".into())));
                assert_eq!(map.get("age"), Some(&FieldValue::Int(30)));
            }
            other => panic!("got {other:?}"),
        }
        // Open + 2 entry_added + close.
        let counts = step
            .events
            .iter()
            .fold((0, 0, 0), |(o, e, c), ev| match ev {
                ParseEvent::ObjectOpen { .. } => (o + 1, e, c),
                ParseEvent::EntryAdded { .. } => (o, e + 1, c),
                ParseEvent::ObjectClose { .. } => (o, e, c + 1),
                _ => (o, e, c),
            });
        assert_eq!(counts, (1, 2, 1));
    }

    #[test]
    fn empty_object() {
        let mut p = parser(person_fields());
        let step = p.push("{}");
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => assert!(map.is_empty()),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn undeclared_keys_silently_discarded() {
        let mut p = parser(person_fields());
        let step = p.push(r#"{"name": "Alice", "extra": [1, 2, 3]}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                assert_eq!(map.get("name"), Some(&FieldValue::Str("Alice".into())));
                assert!(!map.contains_key("extra"));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn missing_nullable_field_defaulted_to_null() {
        let fields = vec![
            ObjectField {
                name: "name".into(),
                description: String::new(),
                field_type: FieldType::String,
            },
            ObjectField {
                name: "nickname".into(),
                description: String::new(),
                field_type: FieldType::Nullable(Box::new(FieldType::String)),
            },
        ];
        let mut p = JsonObjectParser::new("/u".into(), fields);
        let step = p.push(r#"{"name": "Alice"}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                assert_eq!(map.get("name"), Some(&FieldValue::Str("Alice".into())));
                // Nullable missing → Null.
                assert_eq!(map.get("nickname"), Some(&FieldValue::Null));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn missing_non_nullable_field_stays_absent() {
        let mut p = parser(person_fields());
        let step = p.push(r#"{"name": "Alice"}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                assert_eq!(map.get("name"), Some(&FieldValue::Str("Alice".into())));
                assert!(!map.contains_key("age"));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn chunk_boundary_mid_object() {
        let input = r#"{"name": "Alice", "age": 30}"#;
        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = parser(person_fields());
            let s1 = p.push(&input[..split]);
            let final_c = if let JsonCompletion::Done(v) = s1.completion {
                JsonCompletion::Done(v)
            } else {
                p.push(&input[split..]).completion
            };
            match final_c {
                JsonCompletion::Done(FieldValue::Object(map)) => {
                    assert_eq!(map.get("name"), Some(&FieldValue::Str("Alice".into())));
                    assert_eq!(map.get("age"), Some(&FieldValue::Int(30)));
                }
                other => panic!("split={split}: got {other:?}"),
            }
        }
    }

    #[test]
    fn nested_object() {
        let fields = vec![ObjectField {
            name: "user".into(),
            description: String::new(),
            field_type: FieldType::Object(person_fields()),
        }];
        let mut p = JsonObjectParser::new("/root".into(), fields);
        let step = p.push(r#"{"user": {"name": "Alice", "age": 30}}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Object(map)) => {
                let user = map.get("user").expect("user");
                let FieldValue::Object(inner) = user else {
                    panic!("expected Object")
                };
                assert_eq!(inner.get("name"), Some(&FieldValue::Str("Alice".into())));
            }
            other => panic!("got {other:?}"),
        }
    }
}
