// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON array parser, generic over inner element type.
//!
//! Implements RFC 8259 §5 array grammar: `[` `(elt (`,` elt)*)?` `]`,
//! with whitespace tolerated between any two tokens. The element
//! parser is dispatched via [`json_value_parser_for`] on the array's
//! inner [`FieldType`](typesayer_types::field::FieldType), so nested arrays
//! (`List<List<T>>`), arrays of objects, etc. all work recursively
//! through the same dispatch — this parser holds no element-type-
//! specific logic of its own.

use typesayer_types::field::{FieldType, FieldValue};

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser,
    json_value_parser_for, skip_whitespace,
};

pub(super) struct JsonArrayParser {
    path: String,
    inner_type: FieldType,
    state: State,
    elements: Vec<FieldValue>,
    sent_open: bool,
}

enum State {
    /// Skipping leading whitespace; awaiting `[`.
    BeforeOpen,
    /// Past `[`; awaiting first element or `]` (empty array).
    AfterOpen,
    /// Parsing one element via a child parser. `index` is the
    /// element's position; included for clean event paths.
    InElement {
        parser: Box<dyn JsonValueParser>,
        index: usize,
    },
    /// Past one element; awaiting `,` (more) or `]` (close).
    AfterElement,
    /// Past `]`; done.
    Done,
    /// Hit a terminal grammar error.
    Errored,
}

impl JsonArrayParser {
    pub(super) const fn new(path: String, inner_type: FieldType) -> Self {
        Self {
            path,
            inner_type,
            state: State::BeforeOpen,
            elements: Vec::new(),
            sent_open: false,
        }
    }

    /// Helper: synthesize a per-element path. Per RFC 6901 the
    /// element at index N of the array at path `/secondkey` is
    /// `/secondkey/N`.
    fn element_path(&self, index: usize) -> String {
        format!("{}/{}", self.path, index)
    }

    /// Build the close event AND the final value in one shot —
    /// returning both so the caller can emit the event AND attach the
    /// same value to the `JsonCompletion::Done` signal without
    /// accidentally taking ownership twice. (Original bug: taking
    /// elements into the event first then cloning self.elements for
    /// Done produced an empty Done value while the event had the
    /// real list.)
    fn build_close(&mut self) -> (ParseEvent, FieldValue) {
        let value = FieldValue::List(std::mem::take(&mut self.elements));
        let close = ParseEvent::ListClose {
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
        self.state = State::Errored;
        let mut all_events = events;
        // Dedup the wrapper StreamError when an inner element parser
        // already emitted one. One cause, one event — the granular
        // element-level event carries the per-position path; the array-
        // level wrapper would just add noise. Mirrors the dedup baked
        // into JsonObjectParser / JsonMapParser.
        let inner_already_errored = all_events
            .iter()
            .any(|e| matches!(e, ParseEvent::StreamError { .. }));
        if !inner_already_errored {
            all_events.push(ParseEvent::StreamError {
                path: Some(self.path.clone()),
                message: msg.clone(),
            });
        }
        JsonStep {
            consumed,
            events: all_events,
            completion: JsonCompletion::Errored(msg),
        }
    }
}

impl JsonValueParser for JsonArrayParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        let mut cursor = 0;
        let mut events = Vec::new();
        let bytes = chunk.as_bytes();

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events,
                    completion: JsonCompletion::NeedMore,
                };
            }

            match &mut self.state {
                State::Done => {
                    // Parent shouldn't push after Done; we no-op the
                    // remainder. Re-build the value from accumulated
                    // elements for the completion signal.
                    return JsonStep {
                        consumed: bytes.len(),
                        events,
                        completion: JsonCompletion::Done(FieldValue::List(self.elements.clone())),
                    };
                }
                State::Errored => {
                    return JsonStep {
                        consumed: bytes.len(),
                        events,
                        completion: JsonCompletion::Errored("array parser already errored".into()),
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
                    if bytes[cursor] == b'[' {
                        cursor += 1;
                        self.state = State::AfterOpen;
                        if !self.sent_open {
                            events.push(ParseEvent::ListOpen {
                                path: self.path.clone(),
                            });
                            self.sent_open = true;
                        }
                    } else {
                        let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                        return self.err(
                            cursor,
                            events,
                            format!("expected '[' to start array, got {bad:?}"),
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
                    if bytes[cursor] == b']' {
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
                    // Not `]` — must be the start of an element. Dispatch
                    // a child parser for the inner type and transition.
                    // Note: we do NOT advance cursor here — the child
                    // parser sees the same byte we just peeked at.
                    let index = self.elements.len();
                    let child = json_value_parser_for(&self.inner_type, self.element_path(index));
                    self.state = State::InElement {
                        parser: child,
                        index,
                    };
                }

                State::InElement { parser, index } => {
                    let index = *index;
                    let step = parser.push(&chunk[cursor..]);
                    let mut child_events = step.events;
                    events.append(&mut child_events);
                    cursor += step.consumed;
                    match step.completion {
                        JsonCompletion::NeedMore => {
                            return JsonStep {
                                consumed: cursor,
                                events,
                                completion: JsonCompletion::NeedMore,
                            };
                        }
                        JsonCompletion::Done(value) => {
                            // Emit element_added with the parsed value.
                            events.push(ParseEvent::ElementAdded {
                                path: self.path.clone(),
                                index,
                                value: value.clone(),
                            });
                            self.elements.push(value);
                            self.state = State::AfterElement;
                        }
                        JsonCompletion::Errored(msg) => {
                            return self.err(
                                cursor,
                                events,
                                format!("array element {index} failed: {msg}"),
                            );
                        }
                    }
                }

                State::AfterElement => {
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
                            // Transition to "awaiting next element". We
                            // can't use AfterOpen because that also
                            // accepts `]` (an empty array) — after a `,`
                            // a `]` would be a trailing-comma error. So
                            // we directly transition by spawning the
                            // child parser AFTER skipping whitespace.
                            let ws = skip_whitespace(&chunk[cursor..]);
                            cursor += ws;
                            if cursor >= bytes.len() {
                                // Whole chunk consumed; remember we're
                                // expecting an element (NOT a close).
                                // Re-enter AfterElement's loop variant
                                // would re-accept `]` — so instead we
                                // synthesize the child eagerly with
                                // empty input and let it return NeedMore.
                                let index = self.elements.len();
                                let child = json_value_parser_for(
                                    &self.inner_type,
                                    self.element_path(index),
                                );
                                self.state = State::InElement {
                                    parser: child,
                                    index,
                                };
                                return JsonStep {
                                    consumed: cursor,
                                    events,
                                    completion: JsonCompletion::NeedMore,
                                };
                            }
                            if bytes[cursor] == b']' {
                                return self.err(
                                    cursor,
                                    events,
                                    "trailing comma in array (got ',' then ']')",
                                );
                            }
                            // Dispatch next element parser.
                            let index = self.elements.len();
                            let child =
                                json_value_parser_for(&self.inner_type, self.element_path(index));
                            self.state = State::InElement {
                                parser: child,
                                index,
                            };
                        }
                        b']' => {
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
                                format!("expected ',' or ']' after array element, got {bad:?}"),
                            );
                        }
                    }
                }
            }
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        let value = FieldValue::List(self.elements);
        match self.state {
            State::Done => (Vec::new(), value),
            other => {
                let msg = format!("array parsing incomplete at end of input (state: {other:?})");
                (
                    vec![ParseEvent::StreamError {
                        path: Some(path.to_owned()),
                        message: msg,
                    }],
                    value,
                )
            }
        }
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforeOpen => f.write_str("BeforeOpen"),
            Self::AfterOpen => f.write_str("AfterOpen"),
            Self::InElement { index, .. } => write!(f, "InElement(index={index})"),
            Self::AfterElement => f.write_str("AfterElement"),
            Self::Done => f.write_str("Done"),
            Self::Errored => f.write_str("Errored"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list_of_strings() -> FieldType {
        FieldType::List(Box::new(FieldType::String))
    }

    fn parse_done(input: &str, expected: Vec<&str>) -> (usize, Vec<ParseEvent>) {
        let mut p = JsonArrayParser::new("/items".into(), FieldType::String);
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Done(FieldValue::List(elts)) => {
                let strs: Vec<&str> = elts
                    .iter()
                    .map(|v| match v {
                        FieldValue::Str(s) => s.as_str(),
                        other => panic!("expected Str, got {other:?}"),
                    })
                    .collect();
                assert_eq!(strs, expected, "input: {input:?}");
            }
            other => panic!("expected Done, got {other:?} for input {input:?}"),
        }
        (step.consumed, step.events)
    }

    fn parse_errored(input: &str) -> String {
        let mut p = JsonArrayParser::new("/items".into(), FieldType::String);
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Errored(msg) => msg,
            other => panic!("expected Errored, got {other:?} for input {input:?}"),
        }
    }

    #[test]
    fn empty_array() {
        let (consumed, events) = parse_done("[]", Vec::<&str>::new());
        assert_eq!(consumed, 2);
        // ListOpen + ListClose, no ElementAdded
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ParseEvent::ListOpen { .. }));
        assert!(matches!(events[1], ParseEvent::ListClose { .. }));
    }

    #[test]
    fn single_element() {
        let (consumed, events) = parse_done(r#"["a"]"#, vec!["a"]);
        assert_eq!(consumed, 5);
        // ListOpen, ElementAdded(0), ListClose
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ParseEvent::ListOpen { .. }));
        match &events[1] {
            ParseEvent::ElementAdded { path, index, value } => {
                assert_eq!(path, "/items");
                assert_eq!(*index, 0);
                assert_eq!(*value, FieldValue::Str("a".into()));
            }
            other => panic!("expected ElementAdded, got {other:?}"),
        }
        assert!(matches!(events[2], ParseEvent::ListClose { .. }));
    }

    #[test]
    fn multiple_elements_in_order() {
        let (_, events) = parse_done(r#"["a","b","c"]"#, vec!["a", "b", "c"]);
        let added: Vec<(usize, &str)> = events
            .iter()
            .filter_map(|e| match e {
                ParseEvent::ElementAdded {
                    index,
                    value: FieldValue::Str(s),
                    ..
                } => Some((*index, s.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(added, vec![(0, "a"), (1, "b"), (2, "c")]);
    }

    #[test]
    fn whitespace_between_tokens_tolerated() {
        parse_done(r#"  [ "a" , "b" , "c" ]  "#, vec!["a", "b", "c"]);
    }

    #[test]
    fn element_with_escapes() {
        parse_done(
            r#"["he said \"hi\"", "a\nb"]"#,
            vec!["he said \"hi\"", "a\nb"],
        );
    }

    #[test]
    fn trailing_comma_rejected() {
        let msg = parse_errored(r#"["a",]"#);
        assert!(msg.contains("trailing comma"), "got: {msg}");
    }

    #[test]
    fn unquoted_element_rejected() {
        let msg = parse_errored(r"[a]");
        assert!(msg.contains("expected '\"' to start string"), "got: {msg}");
    }

    #[test]
    fn missing_open_bracket_rejected() {
        let msg = parse_errored(r#""a"]"#);
        assert!(msg.contains("expected '['"), "got: {msg}");
    }

    #[test]
    fn unterminated_array_at_finish_errors() {
        let mut p = JsonArrayParser::new("/items".into(), FieldType::String);
        p.push(r#"["a", "b""#);
        let (events, value) = Box::new(p).finish("/items");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ParseEvent::StreamError { message, .. } => {
                assert!(message.contains("incomplete"), "got: {message}");
            }
            other => panic!("got {other:?}"),
        }
        // Best-effort partial value with what we collected
        assert_eq!(
            value,
            FieldValue::List(vec![
                FieldValue::Str("a".into()),
                FieldValue::Str("b".into())
            ])
        );
    }

    #[test]
    fn trailing_content_after_close_returned_via_consumed() {
        let input = r#"["a","b"] tail content"#;
        let mut p = JsonArrayParser::new("/items".into(), FieldType::String);
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Done(_) => {}
            other => panic!("got {other:?}"),
        }
        // Consumed only up through the closing `]`
        assert_eq!(&input[..step.consumed], r#"["a","b"]"#);
        assert_eq!(&input[step.consumed..], " tail content");
    }

    #[test]
    fn nested_list_of_strings() {
        let mut p = JsonArrayParser::new("/items".into(), list_of_strings());
        let step = p.push(r#"[["a","b"],["c"]]"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::List(outer)) => {
                assert_eq!(outer.len(), 2);
                let inner0 = match &outer[0] {
                    FieldValue::List(v) => v,
                    other => panic!("expected List, got {other:?}"),
                };
                let inner0_strs: Vec<&str> = inner0
                    .iter()
                    .map(|v| match v {
                        FieldValue::Str(s) => s.as_str(),
                        _ => panic!(),
                    })
                    .collect();
                assert_eq!(inner0_strs, vec!["a", "b"]);

                let inner1 = match &outer[1] {
                    FieldValue::List(v) => v,
                    other => panic!("expected List, got {other:?}"),
                };
                let inner1_strs: Vec<&str> = inner1
                    .iter()
                    .map(|v| match v {
                        FieldValue::Str(s) => s.as_str(),
                        _ => panic!(),
                    })
                    .collect();
                assert_eq!(inner1_strs, vec!["c"]);
            }
            other => panic!("got {other:?}"),
        }
        // Nested element paths should have been emitted properly. Outer
        // elements live at `/items/{0,1}`; inner elements at deeper
        // paths. Counting via `.filter(...).count()` rather than
        // collecting keeps the iteration single-pass.
        let outer_added_count = step
            .events
            .iter()
            .filter(|e| matches!(e, ParseEvent::ElementAdded { path, .. } if path == "/items"))
            .count();
        assert_eq!(outer_added_count, 2, "two outer elements added");
        let inner_added_count = step
            .events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    ParseEvent::ElementAdded { path, .. }
                        if path.starts_with("/items/0") || path.starts_with("/items/1")
                )
            })
            .count();
        assert!(
            inner_added_count >= 3,
            "expected inner element_added events; got {inner_added_count}"
        );
    }

    #[test]
    fn element_split_across_chunks() {
        let mut p = JsonArrayParser::new("/items".into(), FieldType::String);
        // Push the first chunk and discard its NeedMore events — the
        // completion-side assertion below verifies the full value.
        let _ = p.push(r#"["he"#);
        let step = p.push(r#"llo"]"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::List(v)) => {
                assert_eq!(v, vec![FieldValue::Str("hello".into())]);
            }
            other => panic!("got {other:?}"),
        }
    }

    use proptest::prelude::*;

    use crate::proptest_strategies::{
        arb_chunking_positions, arb_supported_list_type_and_value, chunk_at_positions,
    };

    proptest! {
        /// Round-trip: any supported `List<T>` value, JSON-encoded and
        /// arbitrarily chunked, parses back to the original `FieldValue`.
        /// Covers nested `List<List<...>>` via the recursive strategy.
        #[test]
        fn json_array_roundtrip_any_supported_list_any_chunking(
            (list_type, list_value) in arb_supported_list_type_and_value(),
            positions in arb_chunking_positions(),
        ) {
            let FieldType::List(inner_type) = list_type else {
                prop_assert!(false, "strategy must yield List<T>");
                return Ok(());
            };
            let encoded = serde_json::to_string(&list_value).expect("FieldValue is Serialize");
            let chunks = chunk_at_positions(&encoded, &positions);

            let mut parser = JsonArrayParser::new("/items".into(), *inner_type);
            let mut final_value: Option<FieldValue> = None;
            for chunk in &chunks {
                let step = parser.push(chunk);
                match step.completion {
                    JsonCompletion::NeedMore => {}
                    JsonCompletion::Done(v) => {
                        final_value = Some(v);
                        break;
                    }
                    JsonCompletion::Errored(msg) => {
                        prop_assert!(false, "unexpected error on safe input: {msg}");
                    }
                }
            }
            let got = final_value.expect("parser must reach Done on a complete JSON array");
            prop_assert_eq!(got, list_value);
        }
    }

    #[test]
    fn many_split_points_all_produce_same_result() {
        let input = r#"["a","b","c"]"#;
        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = JsonArrayParser::new("/items".into(), FieldType::String);
            let mut consumed_total = 0;
            let step1 = p.push(&input[..split]);
            consumed_total += step1.consumed;
            let final_completion = match step1.completion {
                JsonCompletion::NeedMore => {
                    let step2 = p.push(&input[consumed_total..]);
                    let _ = step2.consumed;
                    step2.completion
                }
                done @ JsonCompletion::Done(_) => done,
                err => err,
            };
            match final_completion {
                JsonCompletion::Done(FieldValue::List(v)) => {
                    let strs: Vec<&str> = v
                        .iter()
                        .map(|x| match x {
                            FieldValue::Str(s) => s.as_str(),
                            _ => panic!(),
                        })
                        .collect();
                    assert_eq!(strs, vec!["a", "b", "c"], "split={split}");
                }
                other => panic!("split={split}: {other:?}"),
            }
        }
    }
}
