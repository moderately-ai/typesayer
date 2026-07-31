// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tail-buffered marker detector for the streaming chat-adapter parser.
//!
//! Scans incoming text chunks for `[[ ## <name> ## ]]` patterns. The
//! tricky case is markers split across token boundaries: a chunk might
//! end with `[[ ##` and the next chunk start with ` fname ## ]]`. The
//! detector holds an internal tail buffer of ambiguous trailing bytes
//! that could be the start of a marker until either the marker
//! completes or enough non-marker content arrives to confirm the bytes
//! are literal content (e.g. a string field happens to contain `[[`).
//!
//! Used by [`super::parser::ChatStreamParser`]; not part of the
//! public adapter API.

use std::sync::LazyLock;

use regex::Regex;

/// Full-marker regex. Captures the field name. Mirrors
/// `FIELD_MARKER` from `typesayer` — kept as a sibling rather than
/// re-exported because the streaming detector wraps it with tail-
/// buffering logic that doesn't belong on the buffered parser.
#[expect(
    clippy::expect_used,
    reason = "static LazyLock built from a literal regex: compile-time verifiable, \
              so a runtime parse failure would mean the binary was linked against \
              a broken `regex` crate and panicking at first-use is the only \
              meaningful recovery"
)]
static FULL_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[ ## (\w+) ## \]\]").expect("static regex is always valid"));

/// Result of one scan of `tail_buffer + chunk`.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum MarkerScan {
    /// No complete marker found. `content_before` is the prefix of the
    /// working slice that's safe to forward to the current field
    /// parser as plain content. The detector's tail buffer holds the
    /// remaining bytes that could be the start of a marker.
    NoMarker { content_before: String },
    /// A complete marker was found. `content_before` is content prior
    /// to the marker; `name` is the captured field name; `remainder`
    /// is everything after the marker — caller passes it back via
    /// another `push` to continue scanning (or holds it internally
    /// during a re-entrant scan).
    MarkerFound {
        content_before: String,
        name: String,
        remainder: String,
    },
}

/// Detects `[[ ## name ## ]]` markers across token-stream chunks with
/// tail-buffering to handle markers split mid-token.
#[derive(Debug, Default)]
pub(super) struct MarkerDetector {
    tail_buffer: String,
}

impl MarkerDetector {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Feed one chunk in; return the next scan result. May leave the
    /// detector with a non-empty tail buffer if the chunk ends with
    /// what looks like the start of a marker.
    pub(super) fn push(&mut self, chunk: &str) -> MarkerScan {
        // Fast path: tail buffer empty (true on the vast majority of
        // streaming pushes; `tail_buffer` is non-empty only briefly,
        // while a possible marker prefix is being assembled). Scan
        // `chunk` directly to avoid the up-front concatenation alloc.
        if self.tail_buffer.is_empty() {
            return Self::scan(chunk, &mut self.tail_buffer);
        }

        // Slow path: held bytes from a previous push need to be
        // combined with `chunk` before scanning so a marker split
        // across chunk boundaries is detected when its closing bytes
        // arrive.
        let mut working = std::mem::take(&mut self.tail_buffer);
        working.push_str(chunk);
        Self::scan(&working, &mut self.tail_buffer)
    }

    /// Scan one already-combined working slice for a marker and either
    /// return [`MarkerScan::MarkerFound`] or, on no match, refill
    /// `tail_buffer_out` with the longest trailing potential-prefix
    /// slice and return [`MarkerScan::NoMarker`]. Pulled out of `push`
    /// so the empty-tail fast path can call it on a borrowed chunk
    /// without first allocating an owned working copy.
    fn scan(working: &str, tail_buffer_out: &mut String) -> MarkerScan {
        if let Some(cap) = FULL_MARKER.captures(working) {
            let full_match = cap.get(0).unwrap_or_else(|| {
                unreachable!("regex group 0 always present on a match");
            });
            let name = cap.get(1).unwrap_or_else(|| {
                unreachable!("regex group 1 always present on this pattern");
            });
            return MarkerScan::MarkerFound {
                content_before: working[..full_match.start()].to_owned(),
                name: name.as_str().to_owned(),
                remainder: working[full_match.end()..].to_owned(),
            };
        }

        // No complete marker. Find the longest trailing slice of
        // `working` that could be a partial marker prefix. Everything
        // before that slice is safe to forward; the slice itself
        // refills the tail buffer.
        let split_point = longest_partial_marker_suffix_start(working);
        let content_before = working[..split_point].to_owned();
        working[split_point..].clone_into(tail_buffer_out);
        MarkerScan::NoMarker { content_before }
    }

    /// Drain any held tail bytes at end-of-stream. Called by the
    /// parser's `finish()` to recover content that was held against a
    /// possible-marker hypothesis the LM never completed (e.g. a
    /// string field that ended with literal `[[`).
    pub(super) fn drain_tail(&mut self) -> String {
        std::mem::take(&mut self.tail_buffer)
    }

    #[cfg(test)]
    pub(super) fn tail(&self) -> &str {
        &self.tail_buffer
    }
}

/// Find the byte index where a suffix of `s` starts that could be the
/// beginning of a marker. Returns `s.len()` when no suffix could be a
/// marker prefix (so the whole buffer is safe to forward).
///
/// Walks back at most `MAX_MARKER_PREFIX` bytes — beyond that the
/// suffix is too long to be a marker prefix (markers cap at the field
/// name's length, and we don't see fields longer than a few dozen
/// chars in practice). Bounded scan keeps the per-chunk cost O(1)
/// regardless of buffer size.
fn longest_partial_marker_suffix_start(s: &str) -> usize {
    let bytes = s.as_bytes();
    let scan_start = bytes.len().saturating_sub(MAX_MARKER_PREFIX);

    // Walk forward from scan_start; for each `[` we find, check if the
    // suffix from that point could be a marker prefix. Return the
    // earliest such position. That's the longest suffix that's
    // ambiguous; everything before it is safe to forward.
    for i in scan_start..bytes.len() {
        if bytes[i] == b'[' && is_potential_marker_prefix(&s[i..]) {
            return i;
        }
    }
    bytes.len()
}

/// Bound on how far back we scan for a potential marker prefix.
/// `"[[ ## " + 64 word chars + " ## ]"` = 12 + 64 = 76. Use 128 to be
/// safe against any reasonable field-name length without scanning
/// unbounded history.
const MAX_MARKER_PREFIX: usize = 128;

/// Returns `true` if `s` could be extended to form a complete
/// `[[ ## name ## ]]` marker. Walks the fixed grammar with a small
/// state machine; on the first character that violates the grammar,
/// returns `false`. Empty input returns `false` (nothing to buffer).
///
/// Crucially, returns `false` for a COMPLETE marker — the caller's
/// regex path handles those; this function is only for prefixes that
/// need to wait for more bytes.
fn is_potential_marker_prefix(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut st = State::Start;
    for ch in s.chars() {
        st = match (st, ch) {
            (State::Start, '[') => State::Bracket1,
            (State::Bracket1, '[') => State::Bracket2,
            (State::Bracket2, ' ') => State::Space1,
            (State::Space1, '#') => State::Hash1a,
            (State::Hash1a, '#') => State::Hash1b,
            (State::Hash1b, ' ') => State::Space2,
            (State::Space2, c) if is_name_char(c) => State::InsideName,
            (State::InsideName, c) if is_name_char(c) => State::InsideName,
            (State::InsideName, ' ') => State::Space3,
            (State::Space3, '#') => State::Hash2a,
            (State::Hash2a, '#') => State::Hash2b,
            (State::Hash2b, ' ') => State::Space4,
            (State::Space4, ']') => State::CloseBracket1,
            // A COMPLETE marker (CloseBracket1 + ']') would terminate the
            // partial-prefix scan, but the FULL_MARKER regex already
            // handled the complete-marker case before we entered the
            // partial-prefix path — so by the time we reach the
            // CloseBracket1 state here, the partial-prefix hypothesis is
            // refuted and we collapse it with the rest of the failures.
            _ => return false,
        }
    }
    true
}

const fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[derive(Debug, Clone, Copy)]
enum State {
    Start,
    Bracket1,
    Bracket2,
    Space1,
    Hash1a,
    Hash1b,
    Space2,
    InsideName,
    Space3,
    Hash2a,
    Hash2b,
    Space4,
    CloseBracket1,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect_one(chunk: &str) -> MarkerScan {
        let mut d = MarkerDetector::new();
        d.push(chunk)
    }

    #[test]
    fn complete_marker_in_one_chunk() {
        match detect_one("hello [[ ## answer ## ]] world") {
            MarkerScan::MarkerFound {
                content_before,
                name,
                remainder,
            } => {
                assert_eq!(content_before, "hello ");
                assert_eq!(name, "answer");
                assert_eq!(remainder, " world");
            }
            scan @ MarkerScan::NoMarker { .. } => {
                panic!("expected MarkerFound, got {scan:?}")
            }
        }
    }

    #[test]
    fn multiple_markers_in_one_chunk() {
        // The detector returns the FIRST marker; ChatStreamParser
        // re-pushes the `remainder` to find the next one. Verify
        // both calls cleanly hit MarkerFound — this is the loop
        // semantics ChatStreamParser::process relies on.
        let mut d = MarkerDetector::new();
        let scan_a = d.push("[[ ## a ## ]]content1[[ ## b ## ]]content2");
        let remainder_after_a = match scan_a {
            MarkerScan::MarkerFound {
                content_before,
                name,
                remainder,
            } => {
                assert_eq!(content_before, "");
                assert_eq!(name, "a");
                remainder
            }
            scan @ MarkerScan::NoMarker { .. } => {
                panic!("expected MarkerFound(a), got {scan:?}")
            }
        };
        // Re-push the remainder; the detector should find marker `b`.
        match d.push(&remainder_after_a) {
            MarkerScan::MarkerFound {
                content_before,
                name,
                remainder,
            } => {
                assert_eq!(content_before, "content1");
                assert_eq!(name, "b");
                assert_eq!(remainder, "content2");
            }
            scan @ MarkerScan::NoMarker { .. } => {
                panic!("expected MarkerFound(b), got {scan:?}")
            }
        }
        // Tail should be empty after the second marker found.
        assert_eq!(d.tail(), "");
    }

    #[test]
    fn empty_chunk_handled_safely() {
        // Empty input must not panic, must not leave the tail
        // buffer holding anything, and must report no content.
        let mut d = MarkerDetector::new();
        match d.push("") {
            MarkerScan::NoMarker { content_before } => assert_eq!(content_before, ""),
            scan @ MarkerScan::MarkerFound { .. } => {
                panic!("expected NoMarker for empty input, got {scan:?}")
            }
        }
        assert_eq!(d.tail(), "");
        // A second empty push should also be a no-op.
        match d.push("") {
            MarkerScan::NoMarker { content_before } => assert_eq!(content_before, ""),
            scan @ MarkerScan::MarkerFound { .. } => panic!("got {scan:?}"),
        }
        assert_eq!(d.tail(), "");
    }

    #[test]
    fn no_marker_clean_content() {
        let mut d = MarkerDetector::new();
        match d.push("just text here") {
            MarkerScan::NoMarker { content_before } => {
                assert_eq!(content_before, "just text here");
            }
            scan @ MarkerScan::MarkerFound { .. } => {
                panic!("expected NoMarker, got {scan:?}")
            }
        }
        assert_eq!(d.tail(), "");
    }

    #[test]
    fn marker_split_across_chunks_at_every_boundary() {
        let full = "before [[ ## fname ## ]] after";
        // Try splitting `full` at every character boundary; the union
        // of the two pushes' results should reconstruct the same
        // content-before / name / content-after as the single-chunk
        // case.
        for split in 1..full.len() {
            let mut d = MarkerDetector::new();
            let first = d.push(&full[..split]);
            let second = d.push(&full[split..]);

            // Exactly one of the two pushes must report MarkerFound.
            // NoMarker.content_before semantics depend on whether the
            // marker has been seen: BEFORE the marker, it's pre-marker
            // safe-to-forward content; AFTER the marker, it's
            // trailing content that belongs in `remainder`.
            let mut content_before = String::new();
            let mut name: Option<String> = None;
            let mut remainder = String::new();
            let mut marker_found = false;
            for scan in [first, second] {
                match scan {
                    MarkerScan::NoMarker { content_before: c } => {
                        if marker_found {
                            remainder.push_str(&c);
                        } else {
                            content_before.push_str(&c);
                        }
                    }
                    MarkerScan::MarkerFound {
                        content_before: c,
                        name: n,
                        remainder: r,
                    } => {
                        content_before.push_str(&c);
                        name = Some(n);
                        remainder.push_str(&r);
                        marker_found = true;
                    }
                }
            }
            assert_eq!(name.as_deref(), Some("fname"), "split={split}");
            assert_eq!(content_before, "before ", "split={split}");
            assert_eq!(remainder, " after", "split={split}");
            assert_eq!(
                d.tail(),
                "",
                "tail should be empty after marker found; split={split}"
            );
        }
    }

    #[test]
    fn tail_buffer_holds_partial_marker_prefixes() {
        let prefixes = [
            "[",
            "[[",
            "[[ ",
            "[[ #",
            "[[ ##",
            "[[ ## ",
            "[[ ## fname",
            "[[ ## fname ",
            "[[ ## fname #",
            "[[ ## fname ##",
            "[[ ## fname ## ",
            "[[ ## fname ## ]",
        ];
        for prefix in prefixes {
            let mut d = MarkerDetector::new();
            let chunk = format!("safe content {prefix}");
            match d.push(&chunk) {
                MarkerScan::NoMarker { content_before } => {
                    assert_eq!(content_before, "safe content ", "prefix={prefix:?}");
                    assert_eq!(d.tail(), prefix, "prefix={prefix:?}");
                }
                scan @ MarkerScan::MarkerFound { .. } => {
                    panic!("expected NoMarker for prefix {prefix:?}, got {scan:?}")
                }
            }
        }
    }

    #[test]
    fn literal_double_bracket_not_marker_flushes_after_disambiguation() {
        let mut d = MarkerDetector::new();
        // First push: `[[` buffered, content_before is "literal "
        match d.push("literal [[") {
            MarkerScan::NoMarker { content_before } => assert_eq!(content_before, "literal "),
            scan @ MarkerScan::MarkerFound { .. } => panic!("got {scan:?}"),
        }
        assert_eq!(d.tail(), "[[");
        // Second push extends the tail with characters that break the
        // marker grammar — `[[!` is not a marker prefix. The detector
        // should flush.
        match d.push("!not marker") {
            MarkerScan::NoMarker { content_before } => {
                assert_eq!(content_before, "[[!not marker");
            }
            scan @ MarkerScan::MarkerFound { .. } => panic!("got {scan:?}"),
        }
        assert_eq!(d.tail(), "");
    }

    #[test]
    fn completed_marker_split_across_chunks() {
        let mut d = MarkerDetector::new();
        let s1 = d.push("[[ ## comple");
        match s1 {
            MarkerScan::NoMarker { content_before } => assert_eq!(content_before, ""),
            scan @ MarkerScan::MarkerFound { .. } => {
                panic!("expected NoMarker, got {scan:?}")
            }
        }
        match d.push("ted ## ]]") {
            MarkerScan::MarkerFound {
                content_before,
                name,
                remainder,
            } => {
                assert_eq!(content_before, "");
                assert_eq!(name, "completed");
                assert_eq!(remainder, "");
            }
            scan @ MarkerScan::NoMarker { .. } => {
                panic!("expected MarkerFound, got {scan:?}")
            }
        }
    }

    #[test]
    fn drain_tail_returns_held_bytes() {
        let mut d = MarkerDetector::new();
        d.push("content [[ ##");
        assert_eq!(d.tail(), "[[ ##");
        assert_eq!(d.drain_tail(), "[[ ##");
        assert_eq!(d.tail(), "");
    }

    #[test]
    fn potential_prefix_rejects_complete_marker() {
        // A COMPLETE marker should NOT be a "potential prefix" — the
        // regex path handles those. This guarantees we don't double-
        // count a complete marker as also-a-prefix-of-itself.
        assert!(!is_potential_marker_prefix("[[ ## x ## ]]"));
    }

    #[test]
    fn potential_prefix_rejects_garbage() {
        for s in ["foo", "[x", "[[ #x", "[[ ## name## ]]", "[[ ## name ## ] ]"] {
            assert!(!is_potential_marker_prefix(s), "should reject {s:?}");
        }
    }
}
