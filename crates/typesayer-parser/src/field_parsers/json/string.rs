// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON string parser.
//!
//! Implements RFC 8259 §7 string grammar: quoted, with `\"`, `\\`,
//! `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, and `\uXXXX` escapes. State
//! survives chunk boundaries so the parser can run against
//! arbitrarily-tokenised provider streams.
//!
//! Non-BMP surrogate-pair escapes (`😀` style for emoji
//! etc.) are NOT yet implemented — a lone surrogate half emits a
//! [`ParseEvent::StreamError`] and transitions to the errored state.
//! BMP literals encoded directly as UTF-8 in the input chunk (`"é"`)
//! work fine since the chunk is already a `&str`. Surrogate pair
//! support is a planned follow-up tracked in the streaming-foundation
//! plan's "future FieldType expansion" section.

use typesayer_types::field::FieldValue;

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser, skip_whitespace,
};

/// One-JSON-string parser.
pub(super) struct JsonStringParser {
    path: String,
    state: State,
    accumulated: String,
}

#[derive(Debug)]
enum State {
    /// Skipping leading whitespace; awaiting opening `"`.
    BeforeQuote,
    /// Past the opening `"`; accumulating chars; watching for `\` or `"`.
    InString,
    /// Past `\` inside a string; awaiting the escape character.
    InEscape,
    /// Past `\u`; collecting hex digits. `collected` accumulates
    /// digits MSB-first; `remaining` counts hex digits still needed.
    InUnicodeEscape { collected: u32, remaining: u8 },
    /// String fully parsed; subsequent pushes are an error
    /// (parent should stop pushing once we report Done).
    Done,
    /// Hit a grammar error; subsequent pushes silently consume.
    Errored,
}

impl JsonStringParser {
    pub(super) const fn new(path: String) -> Self {
        Self {
            path,
            state: State::BeforeQuote,
            accumulated: String::new(),
        }
    }

    /// Build a one-shot error step, advancing state to Errored.
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

impl JsonValueParser for JsonStringParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        let bytes = chunk.as_bytes();
        let mut cursor: usize = 0;

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events: Vec::new(),
                    completion: JsonCompletion::NeedMore,
                };
            }

            match &mut self.state {
                State::Done | State::Errored => {
                    // Contract: parent shouldn't push after Done/Errored,
                    // but if it does, no-op the rest of the chunk.
                    return JsonStep {
                        consumed: bytes.len(),
                        events: Vec::new(),
                        completion: match self.state {
                            State::Done => {
                                JsonCompletion::Done(FieldValue::Str(self.accumulated.clone()))
                            }
                            _ => JsonCompletion::Errored("parser already errored".into()),
                        },
                    };
                }

                State::BeforeQuote => {
                    let ws = skip_whitespace(&chunk[cursor..]);
                    cursor += ws;
                    if cursor >= bytes.len() {
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                    if bytes[cursor] == b'"' {
                        cursor += 1;
                        self.state = State::InString;
                    } else {
                        let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                        return self.err(
                            cursor,
                            format!("expected '\"' to start string, got {bad:?}"),
                        );
                    }
                }

                State::InString => {
                    // Walk chars (not bytes) so multi-byte UTF-8 literals
                    // inside the string land as one char each. The cursor
                    // tracks byte position; char_indices on the suffix
                    // gives us both.
                    let suffix = &chunk[cursor..];
                    let mut local_cursor = 0;
                    let mut closed = false;
                    let mut escape_started = false;
                    for (idx, ch) in suffix.char_indices() {
                        match ch {
                            '"' => {
                                local_cursor = idx + 1;
                                closed = true;
                                break;
                            }
                            '\\' => {
                                local_cursor = idx + 1;
                                escape_started = true;
                                break;
                            }
                            // Control chars must be \-escaped per RFC 8259
                            // §7. Reject literal control chars in the
                            // stream (the LM should be emitting them
                            // escaped if they appear at all).
                            c if (c as u32) < 0x20 => {
                                let absolute = cursor + idx;
                                return self.err(
                                    absolute,
                                    format!(
                                        "unescaped control character U+{:04X} inside string",
                                        c as u32
                                    ),
                                );
                            }
                            c => {
                                self.accumulated.push(c);
                            }
                        }
                    }
                    if closed {
                        let value = FieldValue::Str(std::mem::take(&mut self.accumulated));
                        self.state = State::Done;
                        return JsonStep {
                            consumed: cursor + local_cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::Done(value),
                        };
                    }
                    if escape_started {
                        cursor += local_cursor;
                        self.state = State::InEscape;
                    } else {
                        // Consumed everything; need more for either a
                        // closing quote or an escape.
                        return JsonStep {
                            consumed: bytes.len(),
                            events: Vec::new(),
                            completion: JsonCompletion::NeedMore,
                        };
                    }
                }

                State::InEscape => {
                    let suffix = &chunk[cursor..];
                    let Some(ch) = suffix.chars().next() else {
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::NeedMore,
                        };
                    };
                    let ch_len = ch.len_utf8();
                    match ch {
                        '"' => self.accumulated.push('"'),
                        '\\' => self.accumulated.push('\\'),
                        '/' => self.accumulated.push('/'),
                        'b' => self.accumulated.push('\u{0008}'),
                        'f' => self.accumulated.push('\u{000C}'),
                        'n' => self.accumulated.push('\n'),
                        'r' => self.accumulated.push('\r'),
                        't' => self.accumulated.push('\t'),
                        'u' => {
                            cursor += ch_len;
                            self.state = State::InUnicodeEscape {
                                collected: 0,
                                remaining: 4,
                            };
                            continue;
                        }
                        other => {
                            return self.err(
                                cursor + ch_len,
                                format!("invalid escape sequence: \\{other}"),
                            );
                        }
                    }
                    cursor += ch_len;
                    self.state = State::InString;
                }

                State::InUnicodeEscape {
                    collected,
                    remaining,
                } => {
                    let suffix = &chunk[cursor..];
                    let Some(ch) = suffix.chars().next() else {
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: JsonCompletion::NeedMore,
                        };
                    };
                    let Some(digit) = ch.to_digit(16) else {
                        return self.err(
                            cursor + ch.len_utf8(),
                            format!("invalid hex digit in \\u escape: {ch:?}"),
                        );
                    };
                    *collected = (*collected << 4) | digit;
                    *remaining -= 1;
                    cursor += ch.len_utf8();
                    if *remaining == 0 {
                        let cp = *collected;
                        if (0xD800..=0xDFFF).contains(&cp) {
                            return self.err(
                                cursor,
                                format!(
                                    "surrogate-pair escape (\\u{cp:04X}) not yet supported \
                                     in streaming parser"
                                ),
                            );
                        }
                        let Some(c) = char::from_u32(cp) else {
                            return self
                                .err(cursor, format!("invalid unicode code point: U+{cp:04X}"));
                        };
                        self.accumulated.push(c);
                        self.state = State::InString;
                    }
                }
            }
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        match self.state {
            State::Done => (Vec::new(), FieldValue::Str(self.accumulated)),
            // Any non-Done state at end-of-input means the LM cut off
            // mid-string. Surface the error and return what we have.
            other => {
                let msg = format!("unterminated JSON string (state: {other:?})");
                (
                    vec![ParseEvent::StreamError {
                        path: Some(path.to_owned()),
                        message: msg,
                    }],
                    FieldValue::Str(self.accumulated),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run one push and assert Done with the expected value.
    /// Returns the bytes_consumed for chunk-remainder assertions.
    fn parse_done(input: &str, expected: &str) -> usize {
        let mut p = JsonStringParser::new("/test".into());
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => {
                assert_eq!(s, expected, "input: {input:?}");
            }
            other => panic!("expected Done({expected:?}), got {other:?} for input {input:?}"),
        }
        assert!(
            step.events.is_empty(),
            "no events on success; got {:?}",
            step.events
        );
        step.consumed
    }

    fn parse_errored(input: &str) -> String {
        let mut p = JsonStringParser::new("/test".into());
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Errored(msg) => msg,
            other => panic!("expected Errored, got {other:?} for input {input:?}"),
        }
    }

    #[test]
    fn empty_string() {
        let consumed = parse_done(r#""""#, "");
        assert_eq!(consumed, 2);
    }

    #[test]
    fn simple_ascii() {
        let consumed = parse_done(r#""hello""#, "hello");
        assert_eq!(consumed, 7);
    }

    #[test]
    fn unicode_bmp_literal_in_input() {
        let consumed = parse_done(r#""héllo ☕""#, "héllo ☕");
        assert_eq!(consumed, r#""héllo ☕""#.len()); // byte length, not char count
    }

    #[test]
    fn standard_escapes_decode() {
        parse_done(r#""\"""#, "\"");
        parse_done(r#""\\""#, "\\");
        parse_done(r#""\/""#, "/");
        parse_done(r#""\b""#, "\u{0008}");
        parse_done(r#""\f""#, "\u{000C}");
        parse_done(r#""\n""#, "\n");
        parse_done(r#""\r""#, "\r");
        parse_done(r#""\t""#, "\t");
    }

    #[test]
    fn unicode_escape_bmp() {
        parse_done(r#""é""#, "é");
        parse_done(r#""é""#, "é"); // uppercase hex
        parse_done(r#""hello ☃ world""#, "hello ☃ world");
    }

    #[test]
    fn surrogate_half_rejected() {
        let msg = parse_errored(r#""\uD800""#);
        assert!(msg.contains("surrogate"), "got msg: {msg}");
    }

    #[test]
    fn invalid_escape_rejected() {
        let msg = parse_errored(r#""\x""#);
        assert!(msg.contains("invalid escape"), "got msg: {msg}");
    }

    #[test]
    fn invalid_hex_in_unicode_escape_rejected() {
        let msg = parse_errored(r#""\u00x""#);
        assert!(msg.contains("invalid hex"), "got msg: {msg}");
    }

    #[test]
    fn unterminated_string_at_finish_errors() {
        let mut p = JsonStringParser::new("/test".into());
        let step = p.push(r#""hello"#);
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = Box::new(p).finish("/test");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ParseEvent::StreamError { path, message } => {
                assert_eq!(path.as_deref(), Some("/test"));
                assert!(message.contains("unterminated"));
            }
            other => panic!("expected StreamError, got {other:?}"),
        }
        // We return the best-effort partial value
        assert_eq!(value, FieldValue::Str("hello".into()));
    }

    #[test]
    fn missing_opening_quote_errors() {
        let msg = parse_errored("hello");
        assert!(
            msg.contains("expected '\"' to start string"),
            "got msg: {msg}"
        );
    }

    #[test]
    fn unescaped_control_char_rejected() {
        let mut p = JsonStringParser::new("/test".into());
        // Literal newline inside string (not \n escape) is invalid per RFC 8259
        let input = "\"hello\nworld\"";
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Errored(msg) => {
                assert!(msg.contains("control character"), "got: {msg}");
            }
            other => panic!("expected Errored, got {other:?}"),
        }
    }

    #[test]
    fn leading_whitespace_skipped() {
        let consumed = parse_done(r#"   "hello""#, "hello");
        assert_eq!(consumed, 10); // 3 ws + 7 string
    }

    #[test]
    fn trailing_chars_after_close_not_consumed() {
        let input = r#""hello", "world"]"#;
        let mut p = JsonStringParser::new("/test".into());
        let step = p.push(input);
        match step.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "hello"),
            other => panic!("got {other:?}"),
        }
        // Consumed only the bytes up through the closing quote of "hello"
        assert_eq!(step.consumed, 7);
        // Caller would pass &input[7..] (`, "world"]`) to next parser
        assert_eq!(&input[step.consumed..], r#", "world"]"#);
    }

    #[test]
    fn mid_escape_chunk_boundary() {
        // Chunk ends right after the backslash; next chunk has the escape char
        let mut p = JsonStringParser::new("/test".into());
        let step1 = p.push(r#""hello \"#);
        assert!(matches!(step1.completion, JsonCompletion::NeedMore));
        let step2 = p.push(r#"n world""#);
        match step2.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "hello \n world"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn mid_unicode_escape_chunk_boundary() {
        // Chunk splits in the middle of a \u escape sequence
        let mut p = JsonStringParser::new("/test".into());
        let step1 = p.push(r#""\u00"#);
        assert!(matches!(step1.completion, JsonCompletion::NeedMore));
        let step2 = p.push(r#"e9""#);
        match step2.completion {
            JsonCompletion::Done(FieldValue::Str(s)) => assert_eq!(s, "é"),
            other => panic!("got {other:?}"),
        }
    }

    use proptest::prelude::*;

    use crate::proptest_strategies::{arb_chunking_positions, arb_safe_string, chunk_at_positions};

    proptest! {
        /// Round-trip: any safe string round-trips through any chunking
        /// of its JSON encoding back to the same string value, with the
        /// parser reporting `Done` exactly once.
        #[test]
        fn json_string_roundtrip_any_safe_string_any_chunking(
            raw in arb_safe_string(),
            positions in arb_chunking_positions(),
        ) {
            // serde_json gives us a quoted JSON-encoded form. Because
            // `arb_safe_string` excludes `"` / `\` / control chars, the
            // encoding is just `"<raw>"` with no escape sequences.
            let encoded = serde_json::to_string(&raw).expect("string is Serialize");
            let chunks = chunk_at_positions(&encoded, &positions);

            let mut parser = JsonStringParser::new("/p".into());
            let mut final_value: Option<String> = None;
            for chunk in &chunks {
                let step = parser.push(chunk);
                match step.completion {
                    JsonCompletion::NeedMore => {}
                    JsonCompletion::Done(FieldValue::Str(s)) => {
                        prop_assert!(final_value.is_none(), "Done emitted twice");
                        final_value = Some(s);
                        break;
                    }
                    JsonCompletion::Done(other) => {
                        prop_assert!(false, "Done with non-Str: {other:?}");
                    }
                    JsonCompletion::Errored(msg) => {
                        prop_assert!(false, "unexpected error on safe input: {msg}");
                    }
                }
            }

            let got = final_value.expect("complete JSON string must reach Done");
            prop_assert_eq!(got, raw);
        }
    }

    #[test]
    fn split_at_every_char_boundary() {
        // For every possible split point in this input, the two-push
        // sequence must produce the same parsed value as a single push.
        let input = r#""he\nllo é ☃""#;
        let expected = "he\nllo é ☃";

        for split in 1..input.len() {
            // Only split at valid char boundaries (UTF-8 safe).
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = JsonStringParser::new("/test".into());
            let mut consumed_total = 0;
            let step1 = p.push(&input[..split]);
            consumed_total += step1.consumed;
            let final_completion = match step1.completion {
                JsonCompletion::NeedMore => {
                    // Push the remainder starting from where step1 left off.
                    // Note: cursor advancing within step1 means consumed_total
                    // may equal `split` (all consumed) and we push the rest.
                    let step2 = p.push(&input[consumed_total..]);
                    let _ = step2.consumed;
                    step2.completion
                }
                done @ JsonCompletion::Done(_) => done,
                err => err,
            };
            match final_completion {
                JsonCompletion::Done(FieldValue::Str(s)) => {
                    assert_eq!(s, expected, "split={split}, input={input:?}");
                }
                other => panic!("split={split}: expected Done({expected:?}), got {other:?}"),
            }
        }
    }
}
