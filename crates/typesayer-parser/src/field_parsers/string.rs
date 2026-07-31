// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sub-parser for [`FieldType::String`](typesayer_types::field::FieldType::String).
//!
//! Pass-through accumulator: every chunk becomes one `ValueAppend`
//! event for chat-UI typing-style rendering, and the final
//! `FieldComplete` carries the full accumulated string.
//!
//! Whitespace handling mirrors `ChatAdapter::parse` (in `typesayer`):
//! the buffered parser does `completion.trim()` per field. Doing the
//! same here would require holding all content until `finish()` to
//! know what to trim, which defeats progressive emission. Instead we
//! emit raw deltas mid-stream AND trim the accumulated buffer in
//! `finish()` so the final `FieldValue::Str` matches what the
//! buffered parser would produce — clients that reconstruct from
//! deltas see the raw stream; clients that consume `FieldComplete`
//! see the trimmed value.

use typesayer_types::field::FieldValue;

use super::{super::event::ParseEvent, FieldParser};

/// Pass-through accumulator for a single `String`-typed output field.
pub(super) struct StringParser {
    path: String,
    accumulated: String,
}

impl StringParser {
    pub(super) const fn new(path: String) -> Self {
        Self {
            path,
            accumulated: String::new(),
        }
    }
}

impl FieldParser for StringParser {
    fn push(&mut self, chunk: &str) -> Vec<ParseEvent> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.accumulated.push_str(chunk);
        vec![ParseEvent::ValueAppend {
            path: self.path.clone(),
            delta: chunk.to_owned(),
        }]
    }

    fn finish(self: Box<Self>) -> (Vec<ParseEvent>, FieldValue) {
        // Trim to match the buffered parser's per-field `completion.trim()`
        // behavior, so the reconstructed FieldValue is identical regardless
        // of which parser path produced it.
        let value = FieldValue::Str(self.accumulated.trim().to_owned());
        (Vec::new(), value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_chunk_emits_one_append_and_completes() {
        let mut p = StringParser::new("/answer".into());
        let events = p.push("hello world");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ParseEvent::ValueAppend { path, delta } => {
                assert_eq!(path, "/answer");
                assert_eq!(delta, "hello world");
            }
            other => panic!("expected ValueAppend, got {other:?}"),
        }
        let (trailing, value) = Box::new(p).finish();
        assert!(trailing.is_empty());
        assert_eq!(value, FieldValue::Str("hello world".into()));
    }

    #[test]
    fn multiple_chunks_emit_in_order_and_concatenate() {
        let mut p = StringParser::new("/answer".into());
        let mut events = Vec::new();
        events.extend(p.push("hel"));
        events.extend(p.push("lo "));
        events.extend(p.push("world"));
        assert_eq!(events.len(), 3);
        let deltas: Vec<&str> = events
            .iter()
            .map(|e| match e {
                ParseEvent::ValueAppend { delta, .. } => delta.as_str(),
                _ => panic!("non-append event"),
            })
            .collect();
        assert_eq!(deltas, vec!["hel", "lo ", "world"]);
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str("hello world".into()));
    }

    #[test]
    fn empty_chunk_emits_nothing() {
        let mut p = StringParser::new("/x".into());
        assert!(p.push("").is_empty());
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str(String::new()));
    }

    #[test]
    fn surrounding_whitespace_trimmed_in_final_value() {
        let mut p = StringParser::new("/x".into());
        p.push("\n  hello  \n");
        let (_, value) = Box::new(p).finish();
        // Deltas pass through verbatim; final value is trimmed to
        // match the buffered parser's behavior.
        assert_eq!(value, FieldValue::Str("hello".into()));
    }

    #[test]
    fn unicode_split_across_chunks_preserved() {
        let mut p = StringParser::new("/x".into());
        // Push a multi-byte character in pieces — note `push_str` is
        // byte-safe; UTF-8 split midway through a codepoint is a
        // caller mistake at the chunk level (StreamDelta.content is a
        // String so the LM provider has already validated UTF-8).
        p.push("café");
        p.push(" ☕");
        let (_, value) = Box::new(p).finish();
        assert_eq!(value, FieldValue::Str("café ☕".into()));
    }
}
