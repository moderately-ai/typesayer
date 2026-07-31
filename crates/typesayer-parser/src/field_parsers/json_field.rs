// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level FieldParser that wraps a [`JsonValueParser`] with three
//! buffered-path drift-tolerance behaviors matching `format.rs::prepare_complex_json`:
//!
//! 1. **Code fence stripping** — `\`\`\`json\n...\n\`\`\`` and `\`\`\`\n...\n\`\`\`` are silently
//!    consumed from the prefix and (implicitly, via the "captured -> ignore trailing" branch) the
//!    suffix.
//! 2. **Surrounding-prose tolerance** — non-bracket bytes before the JSON value's opening `[` or
//!    `{` are skipped, so "Here's the data: [1,2,3]" parses cleanly.
//! 3. **Single-key wrapper unwrap** — `{"<field_name>": <value>}` is unwrapped to `<value>`.
//!    Matches buffered's behavior of dropping common LM `{"answer": [...]}` envelopes.
//!
//! Drift handling applies ONLY when the declared type is a container
//! (`List`, `Object`, `Map`, or `Nullable<container>`). Scalar fields
//! (`Int`, `Float`, `Bool`, `Enum`, `Nullable<scalar>`) skip drift
//! entirely — `format.rs:218-228` calls `i64::from_str`/`f64::from_str`
//! directly on the trimmed buffer without `prepare_complex_json`, and
//! the streaming codec matches that strictness.

use typesayer_types::field::{FieldType, FieldValue};

use super::{
    super::event::ParseEvent,
    FieldParser,
    json::{JsonCompletion, JsonValueParser, json_value_parser_for},
};

/// Soft cap on bytes the drift detector will buffer before giving up
/// and treating the content as un-wrapped. Field names in practice
/// are short (1-50 chars); 256 bytes gives a wide margin for the
/// `{"<field_name>": ` prefix even with whitespace and long names.
const DRIFT_DETECT_BUDGET: usize = 256;

pub(super) struct JsonFieldParser {
    path: String,
    field_type: FieldType,
    parser: Box<dyn JsonValueParser>,
    captured: Option<FieldValue>,
    errored: bool,
    drift: DriftState,
    /// Bytes pending drift-decision (fence detection + prose skip +
    /// single-key unwrap detection). Empty once `drift` transitions
    /// past `Detecting`. Stays empty for non-drift-eligible fields.
    drift_buffer: String,
    /// The field-name fragment of `self.path` (last `/`-delimited
    /// segment), cached so the wrapper detector's string-equality
    /// check doesn't re-parse on every push.
    field_name: String,
}

#[derive(Debug)]
enum DriftState {
    /// Drift handling not applicable for this field type — bytes pass
    /// straight to the inner parser without buffering.
    NotApplicable,
    /// Buffering bytes to detect fence prefix / prose-before / single-key
    /// wrapper. Transitions to `Passthrough` (after flushing the
    /// buffer minus any consumed prefix), `InsideWrapper` (after
    /// matching `{"<field_name>": ` and routing only the value), or
    /// stays here until the budget is exhausted.
    Detecting,
    /// Wrapper detected — the inner parser is being fed the value
    /// portion. After it returns `Done`, we transition to
    /// `AfterWrapperValue` to consume the closing `}`.
    InsideWrapper,
    /// Inner parser captured the wrapped value; consuming the trailing
    /// `}` (and any leading whitespace before it).
    AfterWrapperValue,
    /// No more drift work — every byte goes straight to the inner
    /// parser. Trailing fence close ` ``` ` is implicitly handled by
    /// JsonFieldParser's `captured.is_some() => Vec::new()` branch.
    Passthrough,
}

impl JsonFieldParser {
    pub(super) fn new(path: String, field_type: FieldType) -> Self {
        let parser = json_value_parser_for(&field_type, path.clone());
        let drift = if drift_applicable(&field_type) {
            DriftState::Detecting
        } else {
            DriftState::NotApplicable
        };
        let field_name = path.rsplit('/').next().unwrap_or(&path).to_owned();
        Self {
            path,
            field_type,
            parser,
            captured: None,
            errored: false,
            drift,
            drift_buffer: String::new(),
            field_name,
        }
    }

    fn maybe_value_set_event(&self, value: &FieldValue) -> Option<ParseEvent> {
        if Self::is_non_container_scalar(&self.field_type) {
            Some(ParseEvent::ValueSet {
                path: self.path.clone(),
                value: value.clone(),
            })
        } else {
            None
        }
    }

    fn is_non_container_scalar(ft: &FieldType) -> bool {
        match ft {
            FieldType::Int | FieldType::Float | FieldType::Bool | FieldType::Enum(_) => true,
            FieldType::Nullable(inner) => Self::is_non_container_scalar(inner),
            FieldType::String
            | FieldType::List(_)
            | FieldType::Object(_)
            | FieldType::Map(_)
            | FieldType::Media { .. }
            | FieldType::OneOf { .. }
            | FieldType::AnyOf { .. } => false,
        }
    }

    /// Feed bytes into the inner parser; capture Done/Errored.
    /// Returns the inner parser's events to bubble up.
    fn feed_inner(&mut self, chunk: &str) -> Vec<ParseEvent> {
        let step = self.parser.push(chunk);
        let mut events = step.events;
        match step.completion {
            JsonCompletion::NeedMore => {}
            JsonCompletion::Done(value) => {
                if let Some(ev) = self.maybe_value_set_event(&value) {
                    events.push(ev);
                }
                self.captured = Some(value);
            }
            JsonCompletion::Errored(_msg) => {
                self.errored = true;
            }
        }
        events
    }

    /// Attempt to resolve the drift detection now. Returns events from
    /// any inner-parser pushes triggered by the resolution. Mutates
    /// `self.drift` based on what was decided.
    fn try_resolve_drift(&mut self) -> Vec<ParseEvent> {
        // Step 1: strip leading whitespace + code fence prefix from the
        // buffer. Returns the fence-consumed prefix length OR None when
        // more bytes are needed to decide.
        let post_fence_offset = match detect_fence_prefix(&self.drift_buffer) {
            FenceDetect::NeedMore => return Vec::new(),
            FenceDetect::NoFence => skip_leading_whitespace(&self.drift_buffer),
            FenceDetect::Fence { skip } => skip,
        };

        // Step 2: from post_fence_offset, skip prose bytes (anything
        // that isn't a JSON container opener). Container types tolerate
        // prose-before per buffered behavior (`extract_balanced_json`
        // looks specifically for `[`/`{`).
        let after_prose = skip_prose_to_container_start(&self.drift_buffer[post_fence_offset..]);
        let value_start = post_fence_offset + after_prose;
        if value_start >= self.drift_buffer.len() {
            // Need more bytes to find the value start.
            if self.drift_buffer.len() < DRIFT_DETECT_BUDGET {
                return Vec::new();
            }
            // Budget exhausted with no JSON-value start — flush the
            // whole buffer to the inner parser and switch to passthrough.
            // The inner parser will error appropriately.
            return self.flush_buffer_and_passthrough(0);
        }

        let first_char = self.drift_buffer.as_bytes()[value_start];

        // Step 3: single-key wrapper detection. Only attempt when the
        // first content char is `{`. The wrapper looks like
        // `{"<field_name>": <value>}`. If the declared type is itself
        // Object/Map at top-level, the `{` is the value's own opener
        // and we should NOT try to unwrap. Detect by checking whether
        // the declared type's effective inner is Object/Map.
        if first_char == b'{' && !declared_is_object_or_map(&self.field_type) {
            match detect_wrapper_prefix(&self.drift_buffer[value_start..], &self.field_name) {
                WrapperDetect::NeedMore => {
                    if self.drift_buffer.len() < DRIFT_DETECT_BUDGET {
                        return Vec::new();
                    }
                    // Budget exhausted while waiting for `:` — flush
                    // everything as-is. Inner parser handles what it
                    // gets.
                    return self.flush_buffer_and_passthrough(0);
                }
                WrapperDetect::NotWrapper => {
                    // Not a wrapper — flush from value_start (the `{`
                    // is the actual value's opener for an Object/Map
                    // declared as something else, which is bizarre but
                    // we honour the LM's choice). Inner parser may
                    // error.
                    return self.flush_buffer_and_passthrough(value_start);
                }
                WrapperDetect::Wrapper { skip_past } => {
                    // Wrapper matched. Bytes [value_start + skip_past..]
                    // are the start of the wrapped value. Send those
                    // bytes to the inner parser; transition to
                    // InWrapper so the closing `}` is consumed later.
                    let inner_start = value_start + skip_past;
                    let remaining = self.drift_buffer.split_off(inner_start);
                    self.drift_buffer.clear();
                    self.drift = DriftState::InsideWrapper;
                    return self.feed_inner(&remaining);
                }
            }
        }

        // Step 4: no wrapper applicable — flush from value_start to
        // the inner parser and switch to passthrough.
        self.flush_buffer_and_passthrough(value_start)
    }

    /// Drain the buffer starting at `offset` to the inner parser and
    /// transition to `Passthrough`.
    fn flush_buffer_and_passthrough(&mut self, offset: usize) -> Vec<ParseEvent> {
        let drained = self.drift_buffer.split_off(offset);
        self.drift_buffer.clear();
        self.drift = DriftState::Passthrough;
        if drained.is_empty() {
            Vec::new()
        } else {
            self.feed_inner(&drained)
        }
    }
}

impl FieldParser for JsonFieldParser {
    fn push(&mut self, chunk: &str) -> Vec<ParseEvent> {
        if self.errored {
            return Vec::new();
        }
        if self.captured.is_some() {
            // Already done; subsequent content is trailing whitespace
            // or fence-close that we silently ignore.
            return Vec::new();
        }

        match self.drift {
            DriftState::NotApplicable | DriftState::Passthrough => self.feed_inner(chunk),
            DriftState::Detecting => {
                self.drift_buffer.push_str(chunk);
                self.try_resolve_drift()
            }
            DriftState::InsideWrapper => {
                let events = self.feed_inner(chunk);
                if self.captured.is_some() {
                    // Inner finished; the bytes after `consumed` (which
                    // we don't track here precisely) need to be checked
                    // for the closing `}`. To avoid complex tracking,
                    // we trust the inner parser's "already done -> noop"
                    // behavior: any leftover bytes from `chunk` past
                    // the JSON value were already passed through, and
                    // the JsonFieldParser's captured-guard at the top
                    // of push() will discard subsequent bytes. We just
                    // mark AfterWrapperValue so finish() doesn't
                    // require seeing `}` explicitly.
                    self.drift = DriftState::AfterWrapperValue;
                }
                events
            }
            DriftState::AfterWrapperValue => {
                // Captured; nothing to do. The captured-guard catches
                // this path before we'd reach here, but keep this arm
                // for defensive completeness.
                Vec::new()
            }
        }
    }

    fn finish(self: Box<Self>) -> (Vec<ParseEvent>, FieldValue) {
        if let Some(value) = self.captured {
            return (Vec::new(), value);
        }
        if self.errored {
            return (Vec::new(), FieldValue::Null);
        }
        // Not yet captured, not errored. Try to resolve any pending
        // drift detection FIRST so a buffered-only value (e.g. a
        // wrapper that arrived all at once but the inner parser didn't
        // see Done because the closing `}` was at the very end) still
        // gets through.
        let Self {
            path,
            field_type,
            mut parser,
            mut drift,
            drift_buffer,
            ..
        } = *self;
        let mut events = Vec::new();

        if matches!(drift, DriftState::Detecting) && !drift_buffer.is_empty() {
            // Flush whatever's in the buffer to the inner parser; it
            // may capture Done from the partial. We don't have a
            // mutable JsonFieldParser to call try_resolve_drift on, so
            // simulate by feeding the inner directly.
            let step = parser.push(&drift_buffer);
            events.extend(step.events);
            if let JsonCompletion::Done(v) = step.completion {
                if Self::is_non_container_scalar(&field_type) {
                    events.push(ParseEvent::ValueSet {
                        path,
                        value: v.clone(),
                    });
                }
                return (events, v);
            }
            drift = DriftState::Passthrough;
        }
        let _ = drift;

        let (inner_events, partial) = parser.finish(&path);
        events.extend(inner_events);
        if !matches!(partial, FieldValue::Null) && Self::is_non_container_scalar(&field_type) {
            events.push(ParseEvent::ValueSet {
                path,
                value: partial.clone(),
            });
        }
        (events, partial)
    }
}

/// Whether drift handling applies to the declared type. Matches
/// buffered's set: container types and Nullable<container>.
fn drift_applicable(ft: &FieldType) -> bool {
    match ft {
        FieldType::List(_)
        | FieldType::Object(_)
        | FieldType::Map(_)
        | FieldType::OneOf { .. }
        | FieldType::AnyOf { .. } => true,
        FieldType::Nullable(inner) => drift_applicable(inner),
        FieldType::String
        | FieldType::Int
        | FieldType::Float
        | FieldType::Bool
        | FieldType::Enum(_)
        | FieldType::Media { .. } => false,
    }
}

/// Whether the declared type's effective container is Object or Map.
/// Used by the wrapper detector: when the declared type IS Object/Map,
/// the leading `{` is the value's own opener, not a single-key
/// wrapper; we skip the unwrap detection.
fn declared_is_object_or_map(ft: &FieldType) -> bool {
    match ft {
        FieldType::Object(_) | FieldType::Map(_) => true,
        FieldType::Nullable(inner) => declared_is_object_or_map(inner),
        _ => false,
    }
}

/// Outcome of fence-prefix detection on the leading bytes of the
/// drift buffer.
enum FenceDetect {
    /// Not enough bytes to decide.
    NeedMore,
    /// No fence — caller treats the leading bytes as content.
    NoFence,
    /// Fence consumed `skip` bytes of the buffer's prefix.
    Fence { skip: usize },
}

/// Detect a code fence opener at the start of `buf` and return the
/// number of leading bytes to skip.
///
/// Buffered parity (`format.rs::strip_code_fence`): accepts both
/// `\`\`\`json\n` and `\`\`\`\n` and ALSO the no-newline forms where
/// the opening fence is immediately followed by the JSON container
/// (`\`\`\`[1,2,3]\`\`\`` or `\`\`\`json[1,2,3]\`\`\``). The
/// buffered code does `body.split_once('\n').map_or(body, |(_, rest)| rest)`
/// which uses the whole body when no newline is present.
///
/// Decision rules:
/// - No leading `\`\`\`` and we have at least 3 non-whitespace bytes → `NoFence`.
/// - Leading `\`\`\`` followed by `\n` → fence is `\`\`\`\n` (skip 4 + ws).
/// - Leading `\`\`\`json\n` → fence is `\`\`\`json\n` (skip 8 + ws).
/// - Leading `\`\`\`json` followed by `[` or `{` → fence is `\`\`\`json` (skip 7 + ws).
/// - Leading `\`\`\`` followed by `[` or `{` → fence is `\`\`\`` (skip 3 + ws).
/// - Insufficient bytes to disambiguate → `NeedMore`.
fn detect_fence_prefix(buf: &str) -> FenceDetect {
    let trimmed_start = skip_leading_whitespace(buf);
    let after_ws = &buf[trimmed_start..];

    // Not enough bytes to even see `\`\`\``. Decide based on what we have:
    // - Empty → wait.
    // - A `\`\`\`` prefix candidate (1-3 backticks) → wait.
    // - Anything else → definitely not a fence.
    if after_ws.is_empty() {
        return FenceDetect::NeedMore;
    }
    if after_ws.len() < 3 {
        if after_ws.bytes().all(|b| b == b'`') {
            return FenceDetect::NeedMore;
        }
        return FenceDetect::NoFence;
    }
    if !after_ws.starts_with("```") {
        return FenceDetect::NoFence;
    }
    let after_ticks = &after_ws[3..];

    // Try the five accepted opener shapes in order.
    if after_ticks.starts_with("json\n") {
        return FenceDetect::Fence {
            skip: trimmed_start + 3 + 5,
        };
    }
    if after_ticks.starts_with("json[") || after_ticks.starts_with("json{") {
        return FenceDetect::Fence {
            skip: trimmed_start + 3 + 4,
        };
    }
    if after_ticks.starts_with('\n') {
        return FenceDetect::Fence {
            skip: trimmed_start + 3 + 1,
        };
    }
    if after_ticks.starts_with('[') || after_ticks.starts_with('{') {
        return FenceDetect::Fence {
            skip: trimmed_start + 3,
        };
    }

    // Could we still match one of those with more bytes? `json\n`,
    // `json[`, `json{`, `\n`, `[`, `{` are the accepted forms.
    if "json[".starts_with(after_ticks) || "json{".starts_with(after_ticks) {
        return FenceDetect::NeedMore;
    }
    // After `\`\`\`` we have at least one definitive byte that isn't
    // the start of a known fence opener — treat as no-fence (the
    // `\`\`\`` is content, not a fence). This matches buffered's
    // `strip_code_fence` which simply returns the trimmed input
    // unchanged when the prefix-strip fails.
    FenceDetect::NoFence
}

fn skip_leading_whitespace(s: &str) -> usize {
    s.bytes()
        .take_while(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        .count()
}

/// Skip bytes until the first JSON container start (`[` or `{`).
/// Returns the offset of that byte (or `s.len()` if none found).
///
/// Matches buffered's `extract_balanced_json` (`format.rs:134-166`):
/// containers are the only types where surrounding prose is tolerated
/// (`prepare_complex_json` is only called from the `List` / `Object`
/// / `Map` arms in `deserialize_value`), so we only look for `[` and
/// `{`. A `"` or `0` byte mid-prose (e.g. `"Here's the data: ..."`)
/// is correctly treated as prose, not a value-start.
fn skip_prose_to_container_start(s: &str) -> usize {
    s.bytes().take_while(|b| !matches!(b, b'[' | b'{')).count()
}

enum WrapperDetect {
    /// Need more bytes to finish parsing the key + `:`.
    NeedMore,
    /// The first content was `{` but the first key doesn't match
    /// `field_name` — treat as bare value (no unwrap).
    NotWrapper,
    /// Wrapper matched; `skip_past` bytes from the start of the slice
    /// must be consumed before the inner value begins.
    Wrapper { skip_past: usize },
}

/// Detect a `{"<field_name>": <value>` prefix in `s` (which begins
/// with `{`). Returns the number of bytes from the start of `s` that
/// must be consumed to reach the wrapped value.
fn detect_wrapper_prefix(s: &str, field_name: &str) -> WrapperDetect {
    debug_assert!(s.as_bytes().first() == Some(&b'{'));
    let mut cursor = 1; // past `{`
    let bytes = s.as_bytes();
    cursor += skip_ws_at(bytes, cursor);
    if cursor >= bytes.len() {
        return WrapperDetect::NeedMore;
    }
    if bytes[cursor] != b'"' {
        return WrapperDetect::NotWrapper;
    }
    cursor += 1; // past opening "
    // Walk the field name bytes; assume no escapes in the key (field
    // names in practice are plain identifiers).
    let name_bytes = field_name.as_bytes();
    let mut name_cursor = 0;
    while name_cursor < name_bytes.len() && cursor < bytes.len() {
        if bytes[cursor] != name_bytes[name_cursor] {
            return WrapperDetect::NotWrapper;
        }
        name_cursor += 1;
        cursor += 1;
    }
    if name_cursor < name_bytes.len() {
        return WrapperDetect::NeedMore;
    }
    if cursor >= bytes.len() {
        return WrapperDetect::NeedMore;
    }
    if bytes[cursor] != b'"' {
        return WrapperDetect::NotWrapper;
    }
    cursor += 1; // past closing "
    cursor += skip_ws_at(bytes, cursor);
    if cursor >= bytes.len() {
        return WrapperDetect::NeedMore;
    }
    if bytes[cursor] != b':' {
        return WrapperDetect::NotWrapper;
    }
    cursor += 1; // past :
    cursor += skip_ws_at(bytes, cursor);
    WrapperDetect::Wrapper { skip_past: cursor }
}

fn skip_ws_at(bytes: &[u8], cursor: usize) -> usize {
    let mut k = 0;
    while cursor + k < bytes.len() && matches!(bytes[cursor + k], b' ' | b'\t' | b'\n' | b'\r') {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use typesayer_types::field::ObjectField;

    use super::*;

    fn make_parser(field_type: FieldType, name: &str) -> Box<dyn FieldParser> {
        Box::new(JsonFieldParser::new(format!("/{name}"), field_type))
    }

    #[test]
    fn list_of_int_plain_response() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("[1, 2, 3]");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn list_of_int_fence_stripped() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("```json\n[1, 2, 3]\n```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn list_of_int_bare_fence_stripped() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("```\n[1, 2, 3]\n```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn list_of_int_surrounding_prose_skipped() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("Here's the data: [1, 2, 3] hope it helps!");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn list_of_int_single_key_wrapper_unwrapped() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push(r#"{"items": [1, 2, 3]}"#);
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn list_of_int_fence_plus_wrapper() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push(
            r#"```json
{"items": [1, 2, 3]}
```"#,
        );
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn map_type_not_unwrapped() {
        let mut p = make_parser(FieldType::Map(Box::new(FieldType::Int)), "tags");
        // For a Map field, the `{` is the value's opener, not a
        // wrapper — even if the first key matches `tags`. Confirm we
        // don't unwrap and the map parses literally.
        p.push(r#"{"tags": 5, "other": 7}"#);
        let (_, v) = p.finish();
        match v {
            FieldValue::Object(map) => {
                assert_eq!(map.get("tags"), Some(&FieldValue::Int(5)));
                assert_eq!(map.get("other"), Some(&FieldValue::Int(7)));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn object_type_not_unwrapped() {
        let mut p = make_parser(
            FieldType::Object(vec![
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
            ]),
            "user",
        );
        p.push(r#"{"name": "Alice", "age": 30}"#);
        let (_, v) = p.finish();
        match v {
            FieldValue::Object(map) => {
                assert_eq!(map.get("name"), Some(&FieldValue::Str("Alice".into())));
                assert_eq!(map.get("age"), Some(&FieldValue::Int(30)));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn nullable_list_unwrapped() {
        let mut p = make_parser(
            FieldType::Nullable(Box::new(FieldType::List(Box::new(FieldType::Int)))),
            "items",
        );
        p.push(r#"{"items": [1, 2, 3]}"#);
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn wrong_key_not_unwrapped() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        // Field name "items" but LM emits {"other": [...]}. Not a
        // wrapper — but also not parseable as a List<Int>, so the
        // inner parser will error. We just confirm the unwrap doesn't
        // misfire.
        let mut events = p.push(r#"{"other": [1, 2, 3]}"#);
        let (finish_events, _) = p.finish();
        events.extend(finish_events);
        // Expect a StreamError because JsonArrayParser saw `{` not `[`.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn chunk_boundary_in_fence_prefix() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("```");
        p.push("json\n[1, 2, 3]\n```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn chunk_boundary_in_wrapper_key() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push(r#"{"ite"#);
        p.push(r#"ms": [1, 2, 3]}"#);
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn fence_no_newline() {
        // Buffered's `strip_code_fence` accepts ``` immediately
        // followed by the JSON value (no newline). Streaming matches.
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("```[1, 2, 3]```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn fence_json_no_newline() {
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("```json[1, 2, 3]```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn prose_with_quote_char_doesnt_misfire() {
        // The `'` in "Here's" shouldn't be confused for a JSON string
        // start. Our skip_prose stops only at `[` or `{`, matching
        // buffered's extract_balanced_json which scans for bracket
        // positions specifically.
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push(r#"Here's the "data" for you: [1, 2, 3]. Done."#);
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn chunk_boundary_after_three_backticks() {
        // Buffer ends right after ``` — detector must wait, NOT
        // assume it's a fence yet (could be ```anything).
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("```");
        p.push("\n[1, 2, 3]\n```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn chunk_boundary_mid_partial_fence() {
        // 1 backtick, then 2 more — the partial-fence buffer must
        // wait for all three to arrive before deciding fence vs prose.
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push("`");
        p.push("``json\n[1, 2, 3]\n```");
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn three_backticks_not_followed_by_known_opener_treated_as_no_fence() {
        // ```abc — not a fence (LM emitted ``` as literal content).
        // Inner parser will see `\`\`\`abc...` and fail, which is fine.
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        let mut events = p.push("```abc def");
        let (finish_events, _) = p.finish();
        events.extend(finish_events);
        // Without a JSON value-start, the inner parser receives
        // bogus content and emits StreamError. Confirm we don't
        // silently swallow it.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn one_byte_chunks_through_full_drift_sequence() {
        // Push the whole "```json\n{"items": [1, 2, 3]}\n```"
        // one byte at a time. Drift detection + wrapper unwrap must
        // both survive arbitrarily fine chunking.
        let input = r#"```json
{"items": [1, 2, 3]}
```"#;
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        for ch in input.chars() {
            let buf = ch.to_string();
            p.push(&buf);
        }
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn split_at_every_byte_through_fence_and_wrapper() {
        // Sweep every possible 2-chunk split point to ensure the
        // drift state machine recovers from any chunk boundary.
        let input = "```json\n{\"items\": [1, 2, 3]}\n```";
        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
            p.push(&input[..split]);
            p.push(&input[split..]);
            let (_, v) = p.finish();
            assert_eq!(
                v,
                FieldValue::List(vec![
                    FieldValue::Int(1),
                    FieldValue::Int(2),
                    FieldValue::Int(3)
                ]),
                "split={split}"
            );
        }
    }

    #[test]
    fn wrapper_with_whitespace_around_key() {
        // {"items"   :   [1, 2, 3]} — extra whitespace inside the
        // wrapper. Detector must consume the optional whitespace.
        let mut p = make_parser(FieldType::List(Box::new(FieldType::Int)), "items");
        p.push(r#"{  "items"   :   [1, 2, 3]  }"#);
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn deeply_nested_field_path_uses_last_segment_for_wrapper_check() {
        // The wrapper check uses the LAST `/`-segment of the path as
        // the field name. Confirm that a path like `/data/items` is
        // matched as field name `items`, not the full path.
        let mut p: Box<dyn FieldParser> = Box::new(JsonFieldParser::new(
            "/data/items".into(),
            FieldType::List(Box::new(FieldType::Int)),
        ));
        p.push(r#"{"items": [1, 2, 3]}"#);
        let (_, v) = p.finish();
        assert_eq!(
            v,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn int_field_no_drift_handling() {
        // Top-level Int does NOT route through JsonFieldParser
        // (BoolTopLevelParser etc. handle scalars at the marker layer),
        // but the dispatch for Nullable<Int> DOES — and drift
        // shouldn't apply since buffered doesn't either. Confirm
        // a wrapped Int response would NOT be unwrapped.
        let mut p = make_parser(FieldType::Nullable(Box::new(FieldType::Int)), "count");
        let mut events = p.push(r#"{"count": 42}"#);
        let (finish_events, _) = p.finish();
        events.extend(finish_events);
        // JsonNullableParser sees `{`, dispatches to JsonNumberParser,
        // which errors. The drift_applicable check returns false for
        // Nullable<Int>, so no unwrap is attempted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }
}
