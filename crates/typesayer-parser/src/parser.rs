// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level streaming parser for chat-adapter responses.
//!
//! Composes `MarkerDetector` (recognises
//! `[[ ## name ## ]]` boundaries) with per-field
//! `FieldParser` sub-parsers (handles the
//! per-FieldType content semantics). Emits a typed
//! [`super::event::ParseEvent`] stream that mirrors the schema tree
//! as values arrive.
//!
//! Mirrors the buffered `ChatAdapter::parse` (in `typesayer`)
//! contract:
//!
//! - Field order is what the LM emits, not declared order.
//! - First occurrence wins if a field marker repeats.
//! - `[[ ## completed ## ]]` is the stream-end sentinel; events after it are silently consumed.
//! - Single-string-output signature + no markers ever → the entire completion is the value (the
//!   buffered parser's `single output + no markers` fallback).

use std::collections::{BTreeMap, HashSet};

use typesayer_types::{
    field::{FieldType, FieldValue},
    signature::Signature,
};

use super::{
    event::ParseEvent,
    field_parsers::{FieldParser, parser_for},
    marker::{MarkerDetector, MarkerScan},
};

/// Push-driven streaming parser for chat-adapter completions.
///
/// Hold one per LM call; push provider `StreamDelta.content` chunks
/// via [`Self::push`]; call [`Self::finish`] when the provider
/// stream ends. Both return [`Vec<ParseEvent>`] for the caller (the
/// streaming consumer) to forward over SSE.
pub struct ChatStreamParser<'a> {
    signature: &'a Signature,
    output_names: HashSet<&'a str>,
    marker_detector: MarkerDetector,
    state: ParserState,
    completed_fields: BTreeMap<String, FieldValue>,
    /// Accumulator for the no-marker fallback case — content seen
    /// before any marker arrived, in case finish() needs to wrap it
    /// as a single string value.
    pre_marker_buffer: String,
    /// Sticky flag: set once we've seen any marker. Disables the
    /// no-marker fallback on finish().
    saw_any_marker: bool,
    /// Sticky flag: set once we've emitted `stream_start`. Mirrors
    /// the buffered parser's contract that the first event is always
    /// StreamStart.
    sent_stream_start: bool,
    /// Set once `[[ ## completed ## ]]` arrives so subsequent push()
    /// calls silently consume trailing tokens.
    seen_completed: bool,
}

enum ParserState {
    BeforeFirstField,
    InField {
        name: String,
        path: String,
        parser: Box<dyn FieldParser>,
    },
    AfterCompleted,
}

impl<'a> ChatStreamParser<'a> {
    /// Build a parser for `signature`. Field types are read at marker
    /// time via the signature's `output_fields()` lookup.
    #[must_use]
    pub fn new(signature: &'a Signature) -> Self {
        let output_names: HashSet<&'a str> =
            signature.output_fields().map(|f| f.name.as_str()).collect();
        Self {
            signature,
            output_names,
            marker_detector: MarkerDetector::new(),
            state: ParserState::BeforeFirstField,
            completed_fields: BTreeMap::new(),
            pre_marker_buffer: String::new(),
            saw_any_marker: false,
            sent_stream_start: false,
            seen_completed: false,
        }
    }

    /// Feed one chunk of provider text. Returns the events produced
    /// by this chunk in emission order.
    pub fn push(&mut self, chunk: &str) -> Vec<ParseEvent> {
        let mut events = Vec::new();
        if !self.sent_stream_start {
            events.push(ParseEvent::StreamStart {
                output_schema_hash: output_schema_hash(self.signature),
            });
            self.sent_stream_start = true;
        }
        if self.seen_completed || matches!(self.state, ParserState::AfterCompleted) {
            return events;
        }
        self.process(chunk, &mut events);
        events
    }

    /// Finalize at end of provider stream. Returns any closing
    /// events (FieldComplete for an unclosed final field, the
    /// no-marker fallback if applicable, StreamComplete).
    #[must_use]
    pub fn finish(mut self) -> Vec<ParseEvent> {
        let mut events = Vec::new();
        if !self.sent_stream_start {
            events.push(ParseEvent::StreamStart {
                output_schema_hash: output_schema_hash(self.signature),
            });
            self.sent_stream_start = true;
        }
        // Drain any trailing marker-tail content into the active path
        // before finalising — handles the case where the LM ended with
        // bytes the detector was holding against a possible marker.
        let trailing = self.marker_detector.drain_tail();
        if !trailing.is_empty() {
            self.consume_content(&trailing, &mut events);
        }

        // Take state out so &mut self method calls remain valid in
        // arm bodies; AfterCompleted is the natural terminal placeholder.
        let state = std::mem::replace(&mut self.state, ParserState::AfterCompleted);
        match state {
            ParserState::AfterCompleted => {
                // StreamComplete already emitted; nothing more.
            }
            ParserState::InField { name, path, parser } => {
                let (parser_events, value) = parser.finish();
                events.extend(parser_events);
                events.push(ParseEvent::FieldComplete {
                    path,
                    value: value.clone(),
                });
                self.completed_fields.insert(name, value);
                events.push(ParseEvent::StreamComplete {
                    output: FieldValue::Object(self.completed_fields),
                });
            }
            ParserState::BeforeFirstField => {
                // No marker ever appeared. If the signature has exactly
                // one String output field, fall back to "the whole
                // completion is that field's value" — mirrors the
                // buffered parser's single-output-no-marker case.
                if !self.saw_any_marker && self.try_synthesize_single_string_output(&mut events) {
                    let output = FieldValue::Object(std::mem::take(&mut self.completed_fields));
                    events.push(ParseEvent::StreamComplete { output });
                } else {
                    // No fallback applies — surface as an error.
                    events.push(ParseEvent::StreamError {
                        path: None,
                        message: "stream ended without any field markers".to_owned(),
                    });
                }
            }
        }
        events
    }

    fn process(&mut self, chunk: &str, events: &mut Vec<ParseEvent>) {
        // Fast path: most streaming pushes carry no marker. The first
        // detector call accepts the borrowed `chunk` directly, so the
        // hot no-marker case doesn't allocate a working copy. Only
        // when a marker is found AND its remainder is non-empty do we
        // need to loop, holding the remainder as an owned String.
        let mut remaining: String = match self.marker_detector.push(chunk) {
            MarkerScan::NoMarker { content_before } => {
                if !content_before.is_empty() {
                    self.consume_content(&content_before, events);
                }
                return;
            }
            MarkerScan::MarkerFound {
                content_before,
                name,
                remainder,
            } => {
                self.saw_any_marker = true;
                if !content_before.is_empty() {
                    self.consume_content(&content_before, events);
                }
                self.handle_marker(&name, events);
                if matches!(self.state, ParserState::AfterCompleted) || self.seen_completed {
                    return;
                }
                remainder
            }
        };

        // Slow path: the chunk contained one marker AND trailing
        // content (and possibly further markers). Drain it.
        loop {
            if remaining.is_empty() {
                return;
            }
            let scan = self.marker_detector.push(&remaining);
            match scan {
                MarkerScan::NoMarker { content_before } => {
                    if !content_before.is_empty() {
                        self.consume_content(&content_before, events);
                    }
                    return;
                }
                MarkerScan::MarkerFound {
                    content_before,
                    name,
                    remainder: next,
                } => {
                    self.saw_any_marker = true;
                    if !content_before.is_empty() {
                        self.consume_content(&content_before, events);
                    }
                    self.handle_marker(&name, events);
                    if matches!(self.state, ParserState::AfterCompleted) || self.seen_completed {
                        return;
                    }
                    remaining = next;
                }
            }
        }
    }

    /// Route a slice of plain content (no markers) to wherever it
    /// belongs — the active field's parser, or the pre-marker
    /// buffer if we haven't seen the first marker yet.
    fn consume_content(&mut self, content: &str, events: &mut Vec<ParseEvent>) {
        match &mut self.state {
            ParserState::InField { parser, .. } => {
                let sub_events = parser.push(content);
                events.extend(sub_events);
            }
            ParserState::BeforeFirstField => {
                self.pre_marker_buffer.push_str(content);
            }
            ParserState::AfterCompleted => {
                // Silently ignore trailing content after the sentinel.
            }
        }
    }

    /// Handle a marker arrival: close the current field (if any),
    /// then either transition to a new field or to AfterCompleted.
    fn handle_marker(&mut self, name: &str, events: &mut Vec<ParseEvent>) {
        // First, close any current field.
        let current_state = std::mem::replace(&mut self.state, ParserState::BeforeFirstField);
        let was_in_field = matches!(current_state, ParserState::InField { .. });
        if let ParserState::InField {
            name: prev_name,
            path: prev_path,
            parser,
        } = current_state
        {
            let (parser_events, value) = parser.finish();
            events.extend(parser_events);
            events.push(ParseEvent::FieldComplete {
                path: prev_path,
                value: value.clone(),
            });
            self.completed_fields.entry(prev_name).or_insert(value);
        }

        // Now dispatch on the new marker name.
        if name == "completed" {
            // No-real-marker case (sentinel arrived without any prior
            // field marker AND nothing previously completed). Two
            // sub-cases mirror the buffered parser's behaviour:
            //
            // 1. Single-String-output signature → rescue: promote `pre_marker_buffer` to that
            //    field's value (commit 1fb557e8 in chat.rs). Instruction-tuned models routinely
            //    emit the visible "I'm done" sentinel while dropping the opening field marker on
            //    single-output signatures as boilerplate.
            //
            // 2. Multi-output (or non-String single-output) signature → emit StreamError. The
            //    buffered `ChatAdapter::parse` returns `PredictError::NoFieldMarkers` for the same
            //    shape (chat.rs::parse_no_markers_multiple_outputs_fails); without this branch the
            //    streaming parser would silently emit `StreamComplete` with an empty `Object` and
            //    the consumer would return success-with-empty- output, masking what the buffered
            //    path treats as an error.
            if !was_in_field && self.completed_fields.is_empty() {
                let rescued = self.try_synthesize_single_string_output(events);
                if !rescued {
                    events.push(ParseEvent::StreamError {
                        path: None,
                        message: "stream ended without any field markers".to_owned(),
                    });
                }
            }
            let output = FieldValue::Object(std::mem::take(&mut self.completed_fields));
            events.push(ParseEvent::StreamComplete { output });
            self.state = ParserState::AfterCompleted;
            self.seen_completed = true;
            return;
        }

        // Is this a declared output field?
        if !self.output_names.contains(name) {
            // Unknown marker name — could be an input echo or a
            // hallucinated field name. The buffered parser silently
            // skips these (`if !output_names.contains(field_name) ||
            // field_name == "completed" { continue; }`). Mirror that.
            // We don't transition; we stay in BeforeFirstField.
            return;
        }

        // First occurrence wins — if we've already completed this
        // field, ignore subsequent markers for it (mirrors the
        // buffered parser's `if fields.contains_key(field_name) {
        // continue; }`).
        if self.completed_fields.contains_key(name) {
            return;
        }

        // Look up the declared FieldType for this name. The
        // contains check above guarantees the find will succeed.
        let field_type = self
            .signature
            .output_fields()
            .find(|f| f.name == name)
            .map(|f| f.field_type.clone());
        let Some(field_type) = field_type else {
            return; // unreachable per the contains check, defensively skip
        };

        let path = format!("/{name}");
        let parser = parser_for(&field_type, path.clone());
        events.push(ParseEvent::FieldStart {
            path: path.clone(),
            field_type,
        });
        self.state = ParserState::InField {
            name: name.to_owned(),
            path,
            parser,
        };
    }

    /// Emit a synthetic `FieldStart` + `ValueAppend` + `FieldComplete`
    /// triple for a single-String-output signature, sourcing the value
    /// from `pre_marker_buffer` and inserting it into
    /// `completed_fields`. Used by two rescue paths that mirror the
    /// buffered parser's single-output-no-real-marker fallback:
    /// `finish()` (no marker ever arrived) and `handle_marker()` (only
    /// the `completed` sentinel arrived). Returns true if the rescue
    /// applied; false if the signature isn't shaped as
    /// single-String-output.
    fn try_synthesize_single_string_output(&mut self, events: &mut Vec<ParseEvent>) -> bool {
        let single_string_field_name = {
            let mut iter = self.signature.output_fields();
            let first = iter.next();
            let rest_exists = iter.next().is_some();
            match (first, rest_exists) {
                (Some(f), false) if matches!(f.field_type, FieldType::String) => {
                    Some(f.name.clone())
                }
                _ => None,
            }
        };
        let Some(field_name) = single_string_field_name else {
            return false;
        };

        let path = format!("/{field_name}");
        let trimmed_value = FieldValue::Str(self.pre_marker_buffer.trim().to_owned());
        events.push(ParseEvent::FieldStart {
            path: path.clone(),
            field_type: FieldType::String,
        });
        if !self.pre_marker_buffer.is_empty() {
            events.push(ParseEvent::ValueAppend {
                path: path.clone(),
                delta: std::mem::take(&mut self.pre_marker_buffer),
            });
        }
        events.push(ParseEvent::FieldComplete {
            path,
            value: trimmed_value.clone(),
        });
        self.completed_fields.insert(field_name, trimmed_value);
        true
    }
}

/// Compute a stable identity hash of the signature's output spec.
/// Not a cryptographic hash — clients use this to verify they're
/// rendering against the schema they expected, not for any security
/// property. SipHash via `DefaultHasher` is fine for this purpose.
fn output_schema_hash(signature: &Signature) -> String {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };
    let mut h = DefaultHasher::new();
    for f in signature.output_fields() {
        f.name.hash(&mut h);
        f.field_type.type_label().hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use typesayer_types::field::FieldDef;

    use super::*;

    fn simple_string_signature() -> Signature {
        Signature::builder("test")
            .input(FieldDef::input("q", FieldType::String, "question"))
            .output(FieldDef::output("answer", FieldType::String, "the answer"))
            .build()
            .unwrap()
    }

    fn two_string_signature() -> Signature {
        // Two String outputs — drives the multi-output sentinel-only
        // rescue path that must fall back to StreamError parity with
        // the buffered parser's `NoFieldMarkers` behaviour.
        Signature::builder("test")
            .input(FieldDef::input("q", FieldType::String, "question"))
            .output(FieldDef::output("a", FieldType::String, "first"))
            .output(FieldDef::output("b", FieldType::String, "second"))
            .build()
            .unwrap()
    }

    fn list_of_strings_signature() -> Signature {
        Signature::builder("test")
            .input(FieldDef::input("q", FieldType::String, "question"))
            .output(FieldDef::output(
                "items",
                FieldType::List(Box::new(FieldType::String)),
                "items",
            ))
            .build()
            .unwrap()
    }

    fn collect_events(events: Vec<ParseEvent>) -> Vec<String> {
        events.iter().map(|e| e.event_type().to_owned()).collect()
    }

    #[test]
    fn simple_string_schema_emits_expected_sequence() {
        let sig = simple_string_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push("[[ ## answer ## ]]\nhello"));
        events.extend(parser.push(" world\n[[ ## completed ## ]]"));
        events.extend(parser.finish());

        let types = collect_events(events.clone());
        // Should be: stream_start, field_start, value_append*, field_complete, stream_complete
        assert_eq!(types[0], "stream_start");
        assert_eq!(types[1], "field_start");
        let value_append_count = types.iter().filter(|t| *t == "value_append").count();
        assert!(value_append_count >= 1, "expected at least 1 value_append");
        assert_eq!(types[types.len() - 2], "field_complete");
        assert_eq!(types[types.len() - 1], "stream_complete");

        // The final StreamComplete should carry the correct output.
        match events.last().unwrap() {
            ParseEvent::StreamComplete { output } => {
                let FieldValue::Object(map) = output else {
                    panic!("expected Object")
                };
                let answer = map.get("answer").unwrap();
                assert_eq!(answer, &FieldValue::Str("hello world".into()));
            }
            other => panic!("expected StreamComplete, got {other:?}"),
        }
    }

    #[test]
    fn no_marker_fallback_for_single_string_output() {
        let sig = simple_string_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push("Paris"));
        events.extend(parser.finish());

        let types = collect_events(events.clone());
        // Should be: stream_start, field_start, value_append, field_complete, stream_complete
        assert_eq!(
            types,
            vec![
                "stream_start",
                "field_start",
                "value_append",
                "field_complete",
                "stream_complete"
            ]
        );

        // Final value should be "Paris"
        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                assert_eq!(map.get("answer"), Some(&FieldValue::Str("Paris".into())));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn list_of_strings_schema_emits_open_added_close() {
        let sig = list_of_strings_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push(
            r#"[[ ## items ## ]]
["a", "b", "c"]
[[ ## completed ## ]]"#,
        ));
        events.extend(parser.finish());

        let types = collect_events(events.clone());
        // Expect: stream_start, field_start, list_open, element_added*3, list_close,
        // field_complete, stream_complete
        let element_added_count = types.iter().filter(|t| *t == "element_added").count();
        assert_eq!(element_added_count, 3);
        assert!(types.contains(&"list_open".to_owned()));
        assert!(types.contains(&"list_close".to_owned()));
        assert!(types.contains(&"stream_complete".to_owned()));

        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                let items = map.get("items").unwrap();
                let FieldValue::List(v) = items else {
                    panic!("expected List")
                };
                let strs: Vec<&str> = v
                    .iter()
                    .map(|x| match x {
                        FieldValue::Str(s) => s.as_str(),
                        _ => panic!(),
                    })
                    .collect();
                assert_eq!(strs, vec!["a", "b", "c"]);
            }
            other => panic!("got {other:?}"),
        }
    }

    fn string_and_list_signature() -> Signature {
        // Two-output signature: one String + one List<String>. Used
        // by the multi-field event-ordering test to verify the
        // top-level parser correctly composes the two FieldParsers
        // across a single LM response.
        Signature::builder("test")
            .input(FieldDef::input("q", FieldType::String, "question"))
            .output(FieldDef::output("a", FieldType::String, "first text"))
            .output(FieldDef::output(
                "b",
                FieldType::List(Box::new(FieldType::String)),
                "second list",
            ))
            .build()
            .unwrap()
    }

    #[test]
    fn two_field_schema_full_event_sequence_ordering() {
        // Schema {a: String, b: List<String>}. Verify the canonical
        // event sequence: stream_start, field_start(a), value_append(a),
        // field_complete(a), field_start(b), list_open(b),
        // element_added(b, 0), element_added(b, 1), list_close(b),
        // field_complete(b), stream_complete. This pins the multi-
        // field composition contract from the original plan.
        let sig = string_and_list_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push(
            r#"[[ ## a ## ]]
first
[[ ## b ## ]]
["x", "y"]
[[ ## completed ## ]]"#,
        ));
        events.extend(parser.finish());

        let types = collect_events(events.clone());
        assert_eq!(
            types,
            vec![
                "stream_start",
                "field_start",
                "value_append",
                "field_complete",
                "field_start",
                "list_open",
                "element_added",
                "element_added",
                "list_close",
                "field_complete",
                "stream_complete",
            ]
        );

        // Verify the final StreamComplete carries both fields
        // correctly mapped.
        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                assert_eq!(map.get("a"), Some(&FieldValue::Str("first".into())));
                assert_eq!(
                    map.get("b"),
                    Some(&FieldValue::List(vec![
                        FieldValue::Str("x".into()),
                        FieldValue::Str("y".into()),
                    ]))
                );
            }
            other => panic!("expected StreamComplete, got {other:?}"),
        }
    }

    #[test]
    fn out_of_declared_order_field_emission() {
        // Schema declares a-then-b, but the LM emits b before a.
        // The streaming parser must follow LM emission order on the
        // wire (FieldStart(/b) before FieldStart(/a)) — clients
        // render in arrival order, not schema order. The final
        // StreamComplete output is a BTreeMap which orders by key
        // (sorts a, b alphabetically), but the EVENT STREAM order
        // is what consumers see live.
        let sig = Signature::builder("test")
            .input(FieldDef::input("q", FieldType::String, "question"))
            .output(FieldDef::output("a", FieldType::String, "first"))
            .output(FieldDef::output("b", FieldType::String, "second"))
            .build()
            .unwrap();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(
            parser.push("[[ ## b ## ]]\nbcontent\n[[ ## a ## ]]\nacontent\n[[ ## completed ## ]]"),
        );
        events.extend(parser.finish());

        // Extract field_start events in order. The /b field must
        // appear FIRST on the wire even though the signature
        // declared a-then-b.
        let field_start_paths: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                ParseEvent::FieldStart { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(field_start_paths, vec!["/b", "/a"]);

        // StreamComplete output is by BTreeMap key order (a, b).
        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                assert_eq!(map.get("a"), Some(&FieldValue::Str("acontent".into())));
                assert_eq!(map.get("b"), Some(&FieldValue::Str("bcontent".into())));
            }
            other => panic!("expected StreamComplete, got {other:?}"),
        }
    }

    #[test]
    fn stream_error_mid_stream_halts_field_emission() {
        // A malformed JSON value inside a List<String> field should
        // emit StreamError and stop emitting per-element events for
        // that field — the JsonArrayParser transitions to the
        // errored state and silently consumes further input. The
        // wrapping ChatStreamParser must propagate this without
        // crashing, and the eventual StreamComplete still fires
        // (the sentinel is unconditional once seen).
        let sig = list_of_strings_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push(
            r#"[[ ## items ## ]]
["a", bad_token, "c"]
[[ ## completed ## ]]"#,
        ));
        events.extend(parser.finish());

        // Must have at least one StreamError.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. })),
            "malformed JSON in list must emit at least one StreamError"
        );

        // Must NOT have emitted element_added events for `c` (the
        // value after the error position) — the JsonArrayParser
        // entered errored state after `bad_token` and won't emit
        // further per-element events.
        let c_was_added = events.iter().any(
            |e| matches!(e, ParseEvent::ElementAdded { value: FieldValue::Str(s), .. } if s == "c"),
        );
        assert!(!c_was_added, "no element_added for `c` after StreamError");

        // Parser doesn't crash; StreamComplete still arrives.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamComplete { .. })),
            "StreamComplete must still fire after mid-stream StreamError"
        );
    }

    #[test]
    fn unknown_field_marker_silently_skipped() {
        // The LM hallucinates a field name not in the signature — the
        // buffered parser ignores it; we mirror that.
        let sig = simple_string_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push(
            "[[ ## hallucinated ## ]]\nignored\n[[ ## answer ## ]]\nactual\n[[ ## completed ## ]]",
        ));
        events.extend(parser.finish());

        let field_start_paths: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                ParseEvent::FieldStart { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(field_start_paths, vec!["/answer"]);

        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                assert_eq!(map.get("answer"), Some(&FieldValue::Str("actual".into())));
                assert!(!map.contains_key("hallucinated"));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn completed_sentinel_only_rescues_single_string_output() {
        // Mirrors chat.rs::parse_single_output_only_completed_marker_fallback
        // (commit 1fb557e8): instruction-tuned models routinely emit the
        // closing sentinel while dropping the opening field marker on
        // single-output signatures. Both parsers must produce the same
        // final value rather than silently dropping the buffered content.
        let sig = simple_string_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push("Paris is the capital of France.\n[[ ## completed ## ]]"));
        events.extend(parser.finish());

        // Expect the rescue path to synthesise field events before the
        // stream-end sentinel: stream_start, field_start, value_append,
        // field_complete, stream_complete.
        let types = collect_events(events.clone());
        assert_eq!(
            types,
            vec![
                "stream_start",
                "field_start",
                "value_append",
                "field_complete",
                "stream_complete",
            ]
        );

        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                assert_eq!(
                    map.get("answer"),
                    Some(&FieldValue::Str("Paris is the capital of France.".into()))
                );
            }
            other => panic!("expected StreamComplete with rescued value, got {other:?}"),
        }
    }

    #[test]
    fn completed_sentinel_only_multi_output_emits_stream_error() {
        // Mirrors chat.rs::parse_no_markers_multiple_outputs_fails:
        // the buffered parser returns `PredictError::NoFieldMarkers`
        // when the LM emits content + only the closing sentinel
        // (no real field markers) on a signature with multiple
        // output fields — there's no rescue heuristic for multi-
        // output, and silently emitting an empty `StreamComplete`
        // would mask what the buffered path treats as a parse error.
        // The streaming parser must emit a StreamError event before
        // the StreamComplete so the consumer returns
        // structured parse error instead of success-with-empty-output.
        let sig = two_string_signature();
        let mut parser = ChatStreamParser::new(&sig);
        let mut events = Vec::new();
        events.extend(parser.push("Some content\n[[ ## completed ## ]]"));
        events.extend(parser.finish());

        // The exact sequence: stream_start, stream_error, stream_complete.
        let types = collect_events(events.clone());
        assert_eq!(
            types,
            vec!["stream_start", "stream_error", "stream_complete"]
        );

        // StreamError carries a message referencing the missing markers.
        let error_msg = events
            .iter()
            .find_map(|e| match e {
                ParseEvent::StreamError { message, path } if path.is_none() => Some(message),
                _ => None,
            })
            .expect("multi-output sentinel-only must emit a path-less StreamError");
        assert!(
            error_msg.contains("without any field markers"),
            "StreamError message should describe the missing-marker condition; got {error_msg:?}"
        );

        // StreamComplete still fires (terminal sentinel acknowledged),
        // but the output is the empty object — no fields could be
        // synthesised. The accompanying StreamError is the signal
        // that the consumer should surface as structured parse error.
        match events.last().unwrap() {
            ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            } => {
                assert!(
                    map.is_empty(),
                    "multi-output sentinel-only yields empty output map"
                );
            }
            other => panic!("expected StreamComplete, got {other:?}"),
        }
    }

    #[test]
    fn schema_hash_stable_for_same_signature() {
        let sig1 = simple_string_signature();
        let sig2 = simple_string_signature();
        assert_eq!(output_schema_hash(&sig1), output_schema_hash(&sig2));
    }

    #[test]
    fn schema_hash_differs_for_different_signatures() {
        let s1 = simple_string_signature();
        let s2 = list_of_strings_signature();
        assert_ne!(output_schema_hash(&s1), output_schema_hash(&s2));
    }

    use proptest::prelude::*;

    use crate::proptest_strategies::{
        DriftPerturbation, arb_chunking_positions, arb_supported_type_and_value,
        arb_unsupported_field_type, chunk_at_positions, serialize_completion,
        serialize_completion_with_drift, single_output_signature,
    };

    #[test]
    fn parity_fence_json_chunked_at_various_points() {
        // Same shape as the proptest failure: List<List<Object<f1:
        // String>>> + FenceJson + various chunk boundaries.
        use typesayer_types::ObjectField;
        let field_type = FieldType::List(Box::new(FieldType::List(Box::new(FieldType::Object(
            vec![ObjectField {
                name: "f1".into(),
                description: String::new(),
                field_type: FieldType::String,
            }],
        )))));
        // Just two inner lists with two objects each.
        let value = FieldValue::List(vec![
            FieldValue::List(vec![
                FieldValue::Object(
                    std::iter::once(("f1".to_string(), FieldValue::Str("abc".to_string())))
                        .collect(),
                ),
                FieldValue::Object(
                    std::iter::once(("f1".to_string(), FieldValue::Str("def".to_string())))
                        .collect(),
                ),
            ]),
            FieldValue::List(vec![FieldValue::Object(
                std::iter::once(("f1".to_string(), FieldValue::Str("ghi".to_string()))).collect(),
            )]),
        ]);
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion_with_drift(
            "answer",
            &field_type,
            &value,
            DriftPerturbation::FenceJson,
        );
        // Try every split point.
        for split in 1..completion.len() {
            if !completion.is_char_boundary(split) {
                continue;
            }
            let mut parser = ChatStreamParser::new(&signature);
            let mut events = Vec::new();
            events.extend(parser.push(&completion[..split]));
            events.extend(parser.push(&completion[split..]));
            events.extend(parser.finish());
            let streamed_value = match events.last() {
                Some(ParseEvent::StreamComplete {
                    output: FieldValue::Object(map),
                }) => map.get("answer").cloned(),
                _ => None,
            };
            assert_eq!(
                streamed_value.as_ref(),
                Some(&value),
                "split={split} input={:?}",
                &completion[..split]
            );
        }
    }

    #[test]
    fn parity_fence_json_with_empty_string_first() {
        use typesayer_types::ObjectField;
        // The counterexample's first inner-list element had an empty
        // f1 — i.e. `[[{"f1": ""}, ...]]`. Test if empty strings inside
        // nested JSON throw off the parser.
        let field_type = FieldType::List(Box::new(FieldType::List(Box::new(FieldType::Object(
            vec![ObjectField {
                name: "f1".into(),
                description: String::new(),
                field_type: FieldType::String,
            }],
        )))));
        let value = FieldValue::List(vec![FieldValue::List(vec![
            FieldValue::Object(
                std::iter::once(("f1".to_string(), FieldValue::Str(String::new()))).collect(),
            ),
            FieldValue::Object(
                std::iter::once((
                    "f1".to_string(),
                    FieldValue::Str(
                        "CPcb.7tmqOtkv4WyIny0D9vfIcR!LpVu-aY6fmGOl HKNNVtt,S".to_string(),
                    ),
                ))
                .collect(),
            ),
        ])]);
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion_with_drift(
            "answer",
            &field_type,
            &value,
            DriftPerturbation::FenceJson,
        );
        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        events.extend(parser.push(&completion));
        events.extend(parser.finish());
        let streamed_value = match events.last() {
            Some(ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            }) => map.get("answer").cloned(),
            _ => None,
        };
        assert_eq!(streamed_value.as_ref(), Some(&value));
    }

    #[test]
    fn parity_fence_json_nested_list_of_list_of_object() {
        use typesayer_types::ObjectField;
        // Minimal reproducer for the proptest FenceJson failure mode.
        // List<List<Object<f1: String>>> with FenceJson perturbation.
        let field_type = FieldType::List(Box::new(FieldType::List(Box::new(FieldType::Object(
            vec![ObjectField {
                name: "f1".into(),
                description: String::new(),
                field_type: FieldType::String,
            }],
        )))));
        let value = FieldValue::List(vec![FieldValue::List(vec![FieldValue::Object(
            std::iter::once(("f1".to_string(), FieldValue::Str("hi".to_string()))).collect(),
        )])]);
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion_with_drift(
            "answer",
            &field_type,
            &value,
            DriftPerturbation::FenceJson,
        );
        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        events.extend(parser.push(&completion));
        events.extend(parser.finish());
        let streamed_value = match events.last() {
            Some(ParseEvent::StreamComplete {
                output: FieldValue::Object(map),
            }) => map.get("answer").cloned(),
            _ => None,
        };
        assert_eq!(streamed_value.as_ref(), Some(&value));
    }

    /// Run the top-level parser against `(field_type, value)` serialised
    /// to the LM wire format, split into chunks, then re-assemble. The
    /// final `StreamComplete` output must carry the input value back —
    /// modulo the well-known top-level-`String` trim (`StringParser`
    /// trims whitespace at finish; mirrored by `value.trim()` here).
    fn parse_value_through_chunks(
        field_type: &FieldType,
        value: &FieldValue,
        positions: &[u8],
    ) -> Vec<ParseEvent> {
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion("answer", field_type, value);
        let chunks = chunk_at_positions(&completion, positions);

        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());
        events
    }

    proptest! {
        /// Round-trip: any supported `(FieldType, FieldValue)` serialised
        /// to the LM wire format and arbitrarily chunked, parses back
        /// to the original value via `ChatStreamParser`.
        ///
        /// Exercises the full marker layer + per-FieldType dispatch +
        /// chunk-boundary handling end-to-end. The per-parser
        /// proptests (string, array) cover the lower layers; this one
        /// verifies that the top-level orchestration composes them
        /// correctly across the marker boundary.
        #[test]
        fn chat_stream_parser_roundtrips_any_supported_value(
            (field_type, value) in arb_supported_type_and_value(),
            positions in arb_chunking_positions(),
        ) {
            let events = parse_value_through_chunks(&field_type, &value, &positions);

            // Top-level String trims at finish (see
            // `surrounding_whitespace_trimmed_in_final_value`); match
            // that for the expected comparison.
            let expected = match (&field_type, &value) {
                (FieldType::String, FieldValue::Str(s)) => FieldValue::Str(s.trim().to_owned()),
                _ => value.clone(),
            };

            match events.last() {
                Some(ParseEvent::StreamComplete { output: FieldValue::Object(map) }) => {
                    prop_assert_eq!(map.get("answer"), Some(&expected));
                }
                Some(other) => {
                    prop_assert!(false, "expected StreamComplete, got {other:?}");
                }
                None => {
                    prop_assert!(false, "no events emitted");
                }
            }
        }

        /// Degrade-gracefully: any FieldType the streaming codec does
        /// NOT support, fed through the same wire format with a stub
        /// value, must emit a `StreamError` without panic and without
        /// hanging. The synthesised content is a single space (valid
        /// for any wire shape but won't parse as a structured value).
        #[test]
        fn unsupported_field_type_emits_stream_error_without_panic(
            field_type in arb_unsupported_field_type(),
            positions in arb_chunking_positions(),
        ) {
            let signature = single_output_signature("answer", field_type.clone());
            // For unsupported types we can't construct a matching
            // FieldValue cleanly, so synthesise the completion
            // directly with a stub content body. The space mid-marker
            // is intentional — it ensures the StringParser sees at
            // least one byte to push to its sub-parser.
            let completion = "[[ ## answer ## ]]\n \n[[ ## completed ## ]]";
            let chunks = chunk_at_positions(completion, &positions);

            let mut parser = ChatStreamParser::new(&signature);
            let mut events = Vec::new();
            for chunk in chunks {
                events.extend(parser.push(chunk));
            }
            events.extend(parser.finish());

            // The unsupported sub-parser is required to emit at least
            // one StreamError. Either the parser itself emits it (for
            // top-level FieldTypes routed to UnsupportedParser like
            // Media), or the JSON layer emits it (for FieldTypes
            // routed through JsonFieldParser → JsonUnsupportedParser
            // like Int/Map/Nullable). Both are acceptable; the
            // property is "never panics, always reports".
            let has_stream_error = events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }));
            prop_assert!(
                has_stream_error,
                "expected StreamError for unsupported type {field_type:?}, events: {events:?}"
            );
        }

    }
}
