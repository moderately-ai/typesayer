// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental JSON number parser for `FieldType::Int` and
//! `FieldType::Float`.
//!
//! Accumulates number-token bytes (digits, sign, decimal point,
//! exponent) until a non-token byte arrives, then defers to
//! `i64::from_str` / `f64::from_str` for the final parse — matching the
//! buffered `format.rs::deserialize_value` behavior. Permissive in the
//! same places: leading zeros (`042`) are accepted because the
//! buffered path's `i64::from_str` accepts them; the streaming codec
//! mirrors that tolerance rather than enforcing strict-JSON.
//!
//! The terminator (`,` `]` `}` whitespace) is NOT consumed — the
//! parent state machine (array/object) sees it next. End-of-input
//! before a terminator (e.g. `[[ ## count ## ]]42[[ ## completed ## ]]`
//! where the next marker arrives directly after the number) falls
//! through `finish()`, which parses the accumulated buffer the same
//! way.

use std::str::FromStr;

use typesayer_types::field::FieldValue;

use super::{super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser};

/// Numeric kind — picks `i64` vs `f64` parsing AND constrains which
/// token chars are accepted. `Int` rejects `.`, `e`, `E`; `Float`
/// accepts the full grammar.
#[derive(Debug, Clone, Copy)]
pub(super) enum NumberKind {
    Int,
    Float,
}

/// One-JSON-number parser. State is just the accumulator + done/
/// errored flags; tokenisation is uniform-per-byte so the state machine
/// stays flat.
pub(super) struct JsonNumberParser {
    path: String,
    kind: NumberKind,
    accumulated: String,
    state: State,
}

#[derive(Debug)]
enum State {
    /// Skipping leading whitespace; awaiting the first token byte.
    BeforeStart,
    /// Past the first token byte; accumulating until terminator.
    InToken,
    /// Number fully parsed; subsequent pushes no-op.
    Done(FieldValue),
    /// Hit a terminal grammar error.
    Errored,
}

impl JsonNumberParser {
    pub(super) const fn int(path: String) -> Self {
        Self {
            path,
            kind: NumberKind::Int,
            accumulated: String::new(),
            state: State::BeforeStart,
        }
    }

    pub(super) const fn float(path: String) -> Self {
        Self {
            path,
            kind: NumberKind::Float,
            accumulated: String::new(),
            state: State::BeforeStart,
        }
    }

    /// Token chars are digits, leading `-`, decimal point, exponent
    /// marker, and exponent sign. Int rejects the float-only chars.
    /// `-` is accepted as a leading sign OR as an exponent sign right
    /// after `e`/`E` (Float only); `+` is exponent-sign only.
    fn is_token_byte(&self, byte: u8) -> bool {
        match byte {
            b'0'..=b'9' => true,
            b'-' => {
                self.accumulated.is_empty()
                    || (matches!(self.kind, NumberKind::Float)
                        && self.accumulated.ends_with(['e', 'E']))
            }
            b'+' => {
                matches!(self.kind, NumberKind::Float) && self.accumulated.ends_with(['e', 'E'])
            }
            b'.' | b'e' | b'E' => matches!(self.kind, NumberKind::Float),
            _ => false,
        }
    }

    fn try_finish(&mut self) -> JsonCompletion {
        if self.accumulated.is_empty() {
            self.state = State::Errored;
            return JsonCompletion::Errored(format!(
                "expected {} value, got empty token",
                kind_label(self.kind)
            ));
        }
        let parsed = match self.kind {
            NumberKind::Int => i64::from_str(&self.accumulated)
                .map(FieldValue::Int)
                .map_err(|e| e.to_string()),
            NumberKind::Float => f64::from_str(&self.accumulated)
                .map(FieldValue::Float)
                .map_err(|e| e.to_string()),
        };
        match parsed {
            Ok(value) => {
                self.state = State::Done(value.clone());
                JsonCompletion::Done(value)
            }
            Err(e) => {
                let msg = format!(
                    "failed to parse {} value {:?}: {e}",
                    kind_label(self.kind),
                    self.accumulated
                );
                self.state = State::Errored;
                JsonCompletion::Errored(msg)
            }
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

const fn kind_label(kind: NumberKind) -> &'static str {
    match kind {
        NumberKind::Int => "int",
        NumberKind::Float => "float",
    }
}

const fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

/// JSON syntactic terminator for a number value: whitespace or a
/// container-level delimiter. We never CONSUME terminators — the
/// parent state machine (array / object / field) picks them up.
const fn is_terminator(byte: u8) -> bool {
    is_whitespace(byte) || matches!(byte, b',' | b']' | b'}')
}

impl JsonValueParser for JsonNumberParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        let bytes = chunk.as_bytes();
        let mut cursor: usize = 0;

        loop {
            if cursor >= bytes.len() {
                return JsonStep {
                    consumed: cursor,
                    events: Vec::new(),
                    completion: match &self.state {
                        State::Done(v) => JsonCompletion::Done(v.clone()),
                        State::Errored => JsonCompletion::Errored("number parser errored".into()),
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
                        completion: JsonCompletion::Errored("number parser already errored".into()),
                    };
                }

                State::BeforeStart => {
                    if is_whitespace(bytes[cursor]) {
                        cursor += 1;
                        continue;
                    }
                    let byte = bytes[cursor];
                    if !self.is_token_byte(byte) {
                        let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                        return self.err(
                            cursor,
                            format!("expected {} value, got {bad:?}", kind_label(self.kind)),
                        );
                    }
                    self.accumulated.push(byte as char);
                    cursor += 1;
                    self.state = State::InToken;
                }

                State::InToken => {
                    let byte = bytes[cursor];
                    if self.is_token_byte(byte) {
                        self.accumulated.push(byte as char);
                        cursor += 1;
                        continue;
                    }
                    // Non-token byte. Two cases:
                    //  - Syntactic terminator (`,` `]` `}` whitespace): parse-and-complete without
                    //    consuming.
                    //  - Anything else (e.g. `.` in Int mode, `x` in Float mode): malformed value;
                    //    emit an error so the wrapping field reports rather than silently
                    //    truncating.
                    if is_terminator(byte) {
                        return JsonStep {
                            consumed: cursor,
                            events: Vec::new(),
                            completion: self.try_finish(),
                        };
                    }
                    let bad = chunk[cursor..].chars().next().unwrap_or(' ');
                    return self.err(
                        cursor,
                        format!(
                            "unexpected {bad:?} in {} value (accumulated: {:?})",
                            kind_label(self.kind),
                            self.accumulated
                        ),
                    );
                }
            }
        }
    }

    fn finish(mut self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        match &self.state {
            State::Done(v) => return (Vec::new(), v.clone()),
            State::Errored => return (Vec::new(), FieldValue::Null),
            _ => {}
        }
        // No terminator ever arrived — parse what we have. Matches the
        // case where the field's closing marker arrives immediately
        // after the number with no trailing whitespace.
        match self.try_finish() {
            JsonCompletion::Done(v) => (Vec::new(), v),
            JsonCompletion::Errored(msg) => (
                vec![ParseEvent::StreamError {
                    path: Some(path.to_owned()),
                    message: msg,
                }],
                FieldValue::Null,
            ),
            JsonCompletion::NeedMore => (
                vec![ParseEvent::StreamError {
                    path: Some(path.to_owned()),
                    message: format!(
                        "unterminated {} value (no token bytes accumulated)",
                        kind_label(self.kind)
                    ),
                }],
                FieldValue::Null,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_int(input: &str) -> JsonStep {
        JsonNumberParser::int("/p".into()).push(input)
    }

    fn parse_float(input: &str) -> JsonStep {
        JsonNumberParser::float("/p".into()).push(input)
    }

    fn expect_done_int(step: JsonStep, expected: i64) -> usize {
        match step.completion {
            JsonCompletion::Done(FieldValue::Int(v)) => assert_eq!(v, expected),
            other => panic!("expected Done(Int({expected})), got {other:?}"),
        }
        assert!(step.events.is_empty());
        step.consumed
    }

    fn expect_done_float(step: JsonStep, expected: f64) -> usize {
        match step.completion {
            JsonCompletion::Done(FieldValue::Float(v)) => {
                assert!((v - expected).abs() < 1e-12, "expected {expected}, got {v}");
            }
            other => panic!("expected Done(Float({expected})), got {other:?}"),
        }
        assert!(step.events.is_empty());
        step.consumed
    }

    fn expect_errored(step: JsonStep) -> String {
        match step.completion {
            JsonCompletion::Errored(msg) => msg,
            other => panic!("expected Errored, got {other:?}"),
        }
    }

    #[test]
    fn int_simple_with_terminator() {
        // `,` is a terminator; only the digits are consumed.
        let consumed = expect_done_int(parse_int("42,"), 42);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn int_negative() {
        expect_done_int(parse_int("-7]"), -7);
    }

    #[test]
    fn int_leading_zero_accepted_for_buffered_parity() {
        // `i64::from_str` accepts `042`; we match buffered.
        expect_done_int(parse_int("042 "), 42);
    }

    #[test]
    fn int_rejects_decimal_point() {
        let msg = expect_errored(parse_int("4.2,"));
        // `.` is rejected at the byte level for Int — the parser
        // surfaces a strict error rather than silently truncating to
        // `4` and leaving `.2,` to confuse the parent state machine.
        assert!(
            msg.contains("unexpected") && msg.contains("int value"),
            "got: {msg}"
        );
    }

    #[test]
    fn int_rejects_bare_sign() {
        // `-` then `,` terminator: try_finish parses "-" → fails.
        let msg = expect_errored(parse_int("-,"));
        assert!(msg.contains("failed to parse int"), "got: {msg}");
    }

    #[test]
    fn float_rejects_trailing_garbage() {
        let msg = expect_errored(parse_float("1.2x,"));
        assert!(
            msg.contains("unexpected") && msg.contains("float value"),
            "got: {msg}"
        );
    }

    #[test]
    fn float_simple() {
        // `3.25` rather than `3.14` to avoid clippy's PI-approx lint.
        expect_done_float(parse_float("3.25,"), 3.25);
    }

    #[test]
    fn float_scientific_notation() {
        expect_done_float(parse_float("1.5e10]"), 1.5e10);
    }

    #[test]
    fn float_negative_exponent() {
        expect_done_float(parse_float("2E-3 "), 2e-3);
    }

    #[test]
    fn float_integer_form() {
        expect_done_float(parse_float("42,"), 42.0);
    }

    #[test]
    fn finish_without_terminator_parses_accumulated() {
        let mut p = JsonNumberParser::int("/p".into());
        let step = p.push("42");
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = Box::new(p).finish("/p");
        assert!(events.is_empty(), "no error events on clean finish");
        assert_eq!(value, FieldValue::Int(42));
    }

    #[test]
    fn finish_without_token_bytes_emits_error() {
        let p = JsonNumberParser::int("/p".into());
        let (events, value) = Box::new(p).finish("/p");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ParseEvent::StreamError { .. }));
        assert_eq!(value, FieldValue::Null);
    }

    #[test]
    fn chunk_boundary_mid_number() {
        let mut p = JsonNumberParser::int("/p".into());
        let s1 = p.push("12");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push("345,");
        match s2.completion {
            JsonCompletion::Done(FieldValue::Int(v)) => assert_eq!(v, 12345),
            other => panic!("got {other:?}"),
        }
        assert_eq!(s2.consumed, 3);
    }

    #[test]
    fn chunk_boundary_mid_negative_sign() {
        let mut p = JsonNumberParser::int("/p".into());
        let s1 = p.push("-");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push("7]");
        match s2.completion {
            JsonCompletion::Done(FieldValue::Int(v)) => assert_eq!(v, -7),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn chunk_boundary_mid_exponent() {
        let mut p = JsonNumberParser::float("/p".into());
        let s1 = p.push("1.5e");
        assert!(matches!(s1.completion, JsonCompletion::NeedMore));
        let s2 = p.push("10,");
        match s2.completion {
            JsonCompletion::Done(FieldValue::Float(v)) => assert!((v - 1.5e10).abs() < 1e-3),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn leading_whitespace_skipped() {
        let consumed = expect_done_int(parse_int("   42,"), 42);
        assert_eq!(consumed, 5); // 3 ws + 2 digits
    }

    #[test]
    fn split_at_every_byte_boundary() {
        // For every possible split point in this input, the chunked
        // parse must reach the same final value as the single-push
        // parse.
        let input = "-12345.6789e-10,";
        let expected = -12345.6789e-10;
        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = JsonNumberParser::float("/p".into());
            let s1 = p.push(&input[..split]);
            let final_completion = if let JsonCompletion::Done(v) = s1.completion {
                JsonCompletion::Done(v)
            } else {
                p.push(&input[split..]).completion
            };
            match final_completion {
                JsonCompletion::Done(FieldValue::Float(v)) => {
                    assert!((v - expected).abs() < 1e-25, "split={split}: got {v}");
                }
                other => panic!("split={split}: expected Done(Float), got {other:?}"),
            }
        }
    }
}
