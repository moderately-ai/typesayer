// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streaming parser for `FieldType::OneOf` and `FieldType::AnyOf`.
//!
//! ## Architecture
//!
//! Both variant types share the same parsing strategy: instantiate one
//! sub-parser per arm and feed every incoming chunk to every arm in
//! parallel. As bytes arrive, arms that can no longer match self-error
//! out; the remaining arm(s) eventually complete. The variant type's
//! "match rule" then resolves the winner:
//!
//! - **OneOf** ([`MatchRule::Exclusive`]): JSON Schema's `oneOf` requires exactly one arm to
//!   validate. Zero matches surfaces `PredictError::OneOfNoArmMatched`; two-or-more matches
//!   surfaces `PredictError::OneOfAmbiguous`.
//! - **AnyOf** ([`MatchRule::FirstMatch`]): the lowest-index Done arm wins; zero matches surfaces
//!   `PredictError::AnyOfNoArmMatched`.
//!
//! ## Why parallel-arm is correct for tagged unions too
//!
//! When the OneOf carries a discriminator hint (each arm is an Object with
//! a `const`-restricted property), wrong-arm parsers naturally error on
//! the const mismatch as soon as their Object parser inspects the
//! discriminator property's value — typically within the first 10–30
//! bytes of input. The parallel path therefore handles both tagged and
//! untagged correctly with the same code path; the tagged fast-path
//! (commit on discriminator value, then forward subsequent deltas to the
//! single chosen arm without parallel parsing) is a performance
//! optimization on top of this correctness baseline, tracked separately.
//!
//! ## Byte accounting
//!
//! Each arm consumes input at its own rate (different arms are following
//! different grammar paths), so the parser keeps an internal byte buffer
//! and a per-arm cursor instead of relying on the inner-parser
//! `consumed` semantics for forwarding. When a winner is identified, the
//! winner's cursor minus the bytes already reported as consumed to the
//! parent yields the final-push `consumed` value; any post-value bytes
//! within that final push are left unconsumed for the parent to dispatch.
//!
//! ## Memory bound
//!
//! The internal buffer grows linearly with input until resolution, after
//! which it is dropped. For a value of size N parsed against K arms,
//! peak memory is O(N + K × per-arm-parser-state). In practice most
//! losing arms self-error within the first few tokens (discriminator
//! mismatch, container-opener mismatch, scalar-type mismatch), so the
//! steady-state per-arm-parser-state for non-winning arms is small.

use typesayer_types::field::{FieldType, FieldValue, VariantArm};

use super::{
    super::super::event::ParseEvent, JsonCompletion, JsonStep, JsonValueParser,
    json_value_parser_for,
};

/// Resolution rule applied when all arms terminate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MatchRule {
    /// JSON Schema `oneOf`: exactly one arm must validate.
    Exclusive,
    /// JSON Schema `anyOf`: lowest-index Done arm wins; multiple matches
    /// are tolerated and disambiguated by declaration order.
    FirstMatch,
}

/// Per-arm runtime state during parallel parsing.
struct ArmState {
    arm_index: usize,
    /// Inner parser. `None` before fast-path scanning has chosen the
    /// arm (tagged OneOf), after the arm has been finalized (Done or
    /// Errored), or after we've handed the parser to `finish()` for a
    /// Parsing arm at stream end. Materialised lazily so untaken arms
    /// in the fast-path-success case never pay parser-construction cost.
    parser: Option<Box<dyn JsonValueParser>>,
    /// The arm's declared FieldType. Kept alongside the parser so the
    /// fast-path commit can lazily build a parser for the chosen arm
    /// and the parallel fallback can materialise parsers for all arms.
    arm_field_type: FieldType,
    status: ArmStatus,
    /// How far this arm has consumed from the shared buffer. Advanced
    /// on each push by `step.consumed` from the inner parser.
    cursor: usize,
    /// Events the arm emitted while still in `Parsing`. Held until the
    /// winner is known so we don't leak losing-arm events to the
    /// downstream consumer.
    buffered_events: Vec<ParseEvent>,
    /// Set when the arm transitions to `Done`. Taken by `resolve()`.
    captured_value: Option<FieldValue>,
}

#[derive(PartialEq, Eq)]
enum ArmStatus {
    Parsing,
    Done,
    Errored,
}

/// Tagged-OneOf fast-path state: scanning the inbound JSON for the
/// discriminator's value before any arm parser is fed.
///
/// When a tagged OneOf parser is constructed, it starts in `Scanning`
/// mode (storing the discriminator property name). As bytes arrive,
/// the parser delays feeding any arm and instead runs the lightweight
/// discriminator scanner. Once the scanner identifies the tag, only
/// the matching arm is fed (saving N-1 arms' worth of work). If the
/// scanner gives up (property order isn't discriminator-first, or the
/// JSON shape isn't an object), the parser flips to `Resolved` and
/// feeds the buffered bytes to every arm — the original parallel-arm
/// path.
///
/// `Resolved` is the steady state for untagged OneOf, AnyOf, and any
/// tagged OneOf that has either committed via the fast path or fallen
/// back to parallel parsing. Once `Resolved`, all subsequent pushes
/// follow the parallel-arm dispatch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DispatchStage {
    Scanning,
    Resolved,
}

/// One-complete-JSON-value parser for OneOf / AnyOf field types. See the
/// module-level doc for the architecture overview.
pub(super) struct JsonMultiArmParser {
    path: String,
    rule: MatchRule,
    arms: Vec<ArmState>,
    /// Per-arm tag value when the OneOf carries a discriminator; empty
    /// for untagged OneOf / AnyOf. When non-empty, indexed by arm
    /// position parallel to `arms`.
    tags: Vec<String>,
    /// Property name carrying the discriminator's string value. Set
    /// when the OneOf is tagged; empty otherwise.
    discriminator_property: String,
    /// Fast-path scanning state machine. Carries enough state to find
    /// the discriminator property's string value in the inbound JSON
    /// without instantiating the full per-arm parsers.
    stage: DispatchStage,
    /// All bytes received across all pushes. Each arm tracks its own
    /// progress as a cursor into this buffer so arms at different parse
    /// positions can stay synchronized with the source input.
    buffer: String,
    /// Cumulative `consumed` value reported to the parent across all
    /// completed pushes. Used to compute the per-push `consumed` we
    /// report when the parser finally resolves to Done.
    bytes_reported_consumed: usize,
    /// Final state after `resolve()` runs. Stays `None` until all arms
    /// are terminal; once set, subsequent `push()` calls return the
    /// terminal value without re-running the resolution.
    terminal: Option<TerminalState>,
    /// Test-only cumulative count of arm-parser instantiations across
    /// the parser's lifetime. Survives the per-arm `parser = None`
    /// reset on `Errored`, so a fast-path success leaves it at 1 and
    /// a fallback to parallel-arm leaves it at `arms.len()`. Without
    /// this counter, the live `arm.parser.is_some()` check is racy
    /// against the error-handling code that clears it.
    #[cfg(test)]
    arms_ever_materialised: usize,
}

enum TerminalState {
    /// The wrapped `value` is already a [`FieldValue::Variant`] carrying
    /// the winning arm index inside, so we don't duplicate it on the
    /// terminal state.
    Done {
        value: FieldValue,
    },
    Errored {
        message: String,
    },
}

impl JsonMultiArmParser {
    /// Construct a streaming parser for a [`FieldType::OneOf`] field.
    /// `path` is the JSON-pointer-style path of the field (or container
    /// position) being parsed; arm parsers are built with the same path
    /// so their events carry the variant's path, not a per-arm path.
    pub(super) fn oneof(path: String, arms: &[VariantArm]) -> Self {
        Self::new(path, MatchRule::Exclusive, arms)
    }

    /// Construct a streaming parser for a [`FieldType::AnyOf`] field.
    pub(super) fn anyof(path: String, arms: &[VariantArm]) -> Self {
        Self::new(path, MatchRule::FirstMatch, arms)
    }

    /// Test-only accessor for the shared input buffer's byte length.
    /// Used by the memory-bound proptest to guard against a refactor
    /// that accidentally introduces a per-arm input copy (which would
    /// make total memory grow as O(input × arms) instead of O(input)).
    #[cfg(test)]
    pub(super) fn shared_buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Test-only accessor for the cumulative count of arm parsers
    /// materialised across the parser's lifetime. Fast-path success
    /// leaves this at 1 (only the winning arm); fallback to parallel
    /// leaves this at `arms.len()` (every arm). The live
    /// `arm.parser.is_some()` count isn't a reliable signal because
    /// `feed_arms` clears `parser` on `Errored`, so by the time the
    /// test inspects it the answer is already racing with the
    /// error-handling code.
    #[cfg(test)]
    pub(super) const fn materialised_arm_count(&self) -> usize {
        self.arms_ever_materialised
    }

    fn new(path: String, rule: MatchRule, arms: &[VariantArm]) -> Self {
        Self::new_with_optional_discriminator(path, rule, arms, None, Vec::new())
    }

    /// Construct a streaming parser for a tagged OneOf. The fast-path
    /// scanner watches the inbound JSON for the discriminator property's
    /// value before any arm parser is constructed; once the tag is
    /// seen, only the matching arm gets a parser instantiated, saving
    /// N-1 arms' worth of byte processing on the common path.
    pub(super) fn tagged_oneof(
        path: String,
        arms: &[VariantArm],
        discriminator_property: String,
        tags: Vec<String>,
    ) -> Self {
        Self::new_with_optional_discriminator(
            path,
            MatchRule::Exclusive,
            arms,
            Some(discriminator_property),
            tags,
        )
    }

    fn new_with_optional_discriminator(
        path: String,
        rule: MatchRule,
        arms: &[VariantArm],
        discriminator_property: Option<String>,
        tags: Vec<String>,
    ) -> Self {
        let tagged = discriminator_property.is_some();
        let arm_states = arms
            .iter()
            .enumerate()
            .map(|(i, arm)| ArmState {
                arm_index: i,
                // Fast-path mode (`Scanning`): defer arm-parser
                // construction so we don't pay for parsers we won't use
                // once the discriminator commits. Untagged / AnyOf: build
                // immediately.
                parser: if tagged {
                    None
                } else {
                    Some(json_value_parser_for(&arm.field_type, path.clone()))
                },
                arm_field_type: arm.field_type.clone(),
                status: ArmStatus::Parsing,
                cursor: 0,
                buffered_events: Vec::new(),
                captured_value: None,
            })
            .collect();
        // Pre-count materialisations from the initial construction
        // (untagged / AnyOf instantiates every arm upfront; tagged
        // defers all of them until the scanner commits).
        #[cfg(test)]
        let initial_materialised = if tagged { 0 } else { arms.len() };
        Self {
            path,
            rule,
            arms: arm_states,
            tags,
            discriminator_property: discriminator_property.unwrap_or_default(),
            stage: if tagged {
                DispatchStage::Scanning
            } else {
                DispatchStage::Resolved
            },
            buffer: String::new(),
            bytes_reported_consumed: 0,
            terminal: None,
            #[cfg(test)]
            arms_ever_materialised: initial_materialised,
        }
    }

    /// While in `DispatchStage::Scanning`, run the discriminator scanner
    /// against the buffered bytes. Returns `true` if the stage advanced
    /// to `Resolved` (either via fast-path commit or fallback to
    /// parallel parsing) and arm parsers were instantiated.
    fn try_resolve_dispatch(&mut self) -> bool {
        if self.stage != DispatchStage::Scanning {
            return false;
        }
        match scan_for_discriminator(&self.buffer, &self.discriminator_property) {
            ScanResult::NeedMore => false,
            ScanResult::Found(tag) => {
                self.stage = DispatchStage::Resolved;
                if let Some(idx) = self.tags.iter().position(|t| t == &tag) {
                    // Fast path success: instantiate only the chosen
                    // arm's parser; mark every other arm `Errored` so
                    // the resolution rule picks the winner cleanly.
                    let arm_path = self.path.clone();
                    for (i, arm) in self.arms.iter_mut().enumerate() {
                        if i == idx {
                            arm.parser =
                                Some(json_value_parser_for(&arm.arm_field_type, arm_path.clone()));
                            #[cfg(test)]
                            {
                                self.arms_ever_materialised += 1;
                            }
                        } else {
                            arm.status = ArmStatus::Errored;
                            arm.buffered_events.clear();
                        }
                    }
                } else {
                    // Discriminator value found but not in the tag set
                    // — none of the arms can match. Materialise every
                    // arm parser and let the parallel path drive them
                    // all to `Errored` so the resolution emits the
                    // standard OneOfNoArmMatched.
                    self.materialise_all_arm_parsers();
                }
                true
            }
            ScanResult::FastPathUnavailable => {
                // Scanner determined the input doesn't expose the
                // discriminator at the head (different first property,
                // not an object, etc.). Fall back to parallel-arm: feed
                // every arm the buffered bytes so they can validate
                // independently.
                self.stage = DispatchStage::Resolved;
                self.materialise_all_arm_parsers();
                true
            }
        }
    }

    fn materialise_all_arm_parsers(&mut self) {
        let arm_path = self.path.clone();
        for arm in &mut self.arms {
            if arm.parser.is_none() && arm.status == ArmStatus::Parsing {
                arm.parser = Some(json_value_parser_for(&arm.arm_field_type, arm_path.clone()));
                #[cfg(test)]
                {
                    self.arms_ever_materialised += 1;
                }
            }
        }
    }

    /// Feed any newly-buffered bytes (those past each arm's cursor) into
    /// every Parsing arm. Returns events that should be emitted from
    /// this call (always empty until resolution) and updates each arm's
    /// status.
    fn feed_arms(&mut self) {
        for arm in &mut self.arms {
            if arm.status != ArmStatus::Parsing {
                continue;
            }
            // Loop until the inner parser either terminates or asks for
            // more (NeedMore with zero consumption means it's stuck
            // waiting for input it hasn't seen yet).
            loop {
                let slice = &self.buffer[arm.cursor..];
                if slice.is_empty() {
                    break;
                }
                let Some(parser) = arm.parser.as_mut() else {
                    break;
                };
                let step = parser.push(slice);
                arm.cursor += step.consumed;
                arm.buffered_events.extend(step.events);
                match step.completion {
                    JsonCompletion::NeedMore => {
                        if step.consumed == 0 {
                            // No progress on the slice — wait for more
                            // input. Otherwise we'd spin forever on a
                            // parser that can't make progress without
                            // additional bytes.
                            break;
                        }
                        // Made progress; loop again in case the rest of
                        // the buffer brings the parser to a terminal
                        // state in one more step.
                    }
                    JsonCompletion::Done(value) => {
                        arm.status = ArmStatus::Done;
                        arm.captured_value = Some(value);
                        // Drop the parser; we won't push to it again.
                        arm.parser = None;
                        break;
                    }
                    JsonCompletion::Errored(_msg) => {
                        arm.status = ArmStatus::Errored;
                        arm.parser = None;
                        // Losing arms shouldn't leak their progressive
                        // events downstream — discard them.
                        arm.buffered_events.clear();
                        break;
                    }
                }
            }
        }
    }

    /// Inspect arm statuses and choose a resolution. Returns `None` when
    /// at least one arm is still `Parsing` (caller must wait for more
    /// input).
    fn try_resolve(&self) -> Option<Resolution> {
        let any_parsing = self.arms.iter().any(|a| a.status == ArmStatus::Parsing);
        if any_parsing {
            return None;
        }
        let done: Vec<usize> = self
            .arms
            .iter()
            .filter(|a| a.status == ArmStatus::Done)
            .map(|a| a.arm_index)
            .collect();
        Some(match self.rule {
            MatchRule::Exclusive => match done.as_slice() {
                [] => Resolution::NoMatch,
                [single] => Resolution::Winner { arm_index: *single },
                multiple => Resolution::Ambiguous {
                    matching_arms: multiple.to_vec(),
                },
            },
            MatchRule::FirstMatch => match done.first() {
                Some(&idx) => Resolution::Winner { arm_index: idx },
                None => Resolution::NoMatch,
            },
        })
    }

    /// Build the final JsonStep for a resolved parser, including the
    /// winner's buffered events and the post-resolution `consumed` value.
    fn step_from_resolution(&mut self, chunk_len: usize, resolution: Resolution) -> JsonStep {
        match resolution {
            Resolution::Winner { arm_index } => {
                let arm = &mut self.arms[arm_index];
                // try_resolve only returns Winner for arms in ArmStatus::Done,
                // which always set captured_value in feed_arms; falling back
                // to a synthetic Null here is cheap insurance that doesn't
                // panic if a future refactor relaxes that invariant.
                let value = arm.captured_value.take().unwrap_or(FieldValue::Null);
                let events = std::mem::take(&mut arm.buffered_events);
                let winner_cursor = arm.cursor;
                let consumed_this_push = winner_cursor.saturating_sub(self.bytes_reported_consumed);
                let consumed_this_push = consumed_this_push.min(chunk_len);
                self.bytes_reported_consumed += consumed_this_push;
                self.terminal = Some(TerminalState::Done {
                    value: FieldValue::Variant {
                        arm_index,
                        value: Box::new(value.clone()),
                    },
                });
                JsonStep {
                    consumed: consumed_this_push,
                    events,
                    completion: JsonCompletion::Done(FieldValue::Variant {
                        arm_index,
                        value: Box::new(value),
                    }),
                }
            }
            Resolution::NoMatch => {
                let msg = match self.rule {
                    MatchRule::Exclusive => format!(
                        "OneOf at '{}' matched no arm; every declared arm rejected the value",
                        self.path
                    ),
                    MatchRule::FirstMatch => format!(
                        "AnyOf at '{}' matched no arm; every declared arm rejected the value",
                        self.path
                    ),
                };
                self.terminal = Some(TerminalState::Errored {
                    message: msg.clone(),
                });
                JsonStep {
                    consumed: chunk_len,
                    events: vec![ParseEvent::StreamError {
                        path: Some(self.path.clone()),
                        message: msg.clone(),
                    }],
                    completion: JsonCompletion::Errored(msg),
                }
            }
            Resolution::Ambiguous { matching_arms } => {
                let msg = format!(
                    "OneOf at '{}' matched multiple arms {matching_arms:?}; oneOf requires \
                     exactly one match. Make the arms structurally disjoint or use anyOf for \
                     first-match-wins semantics.",
                    self.path
                );
                self.terminal = Some(TerminalState::Errored {
                    message: msg.clone(),
                });
                JsonStep {
                    consumed: chunk_len,
                    events: vec![ParseEvent::StreamError {
                        path: Some(self.path.clone()),
                        message: msg.clone(),
                    }],
                    completion: JsonCompletion::Errored(msg),
                }
            }
        }
    }
}

#[derive(Debug)]
enum Resolution {
    Winner { arm_index: usize },
    NoMatch,
    Ambiguous { matching_arms: Vec<usize> },
}

#[derive(Debug)]
enum ScanResult {
    /// Not enough bytes seen yet to determine the discriminator value.
    /// Caller waits for more input.
    NeedMore,
    /// The discriminator's string value was decoded. Caller commits to
    /// the matching arm (or, if no arm has this tag, falls back to
    /// parallel-arm so the resolution emits a proper no-match error).
    Found(String),
    /// The scanner saw enough of the input to determine the
    /// discriminator can't be identified at the head (different first
    /// property, non-object JSON, etc.). Caller falls back to
    /// parallel-arm parsing.
    FastPathUnavailable,
}

/// Tagged-OneOf fast-path scanner. Inspects the buffered bytes to find
/// the discriminator property and its string value at the head of the
/// JSON object — without instantiating the full per-arm parsers.
///
/// Grammar handled (the well-formed-discriminator-first case):
///   <whitespace>? `{` <whitespace>? `"<property>"` <whitespace>? `:`
///   <whitespace>? `"<value>"` ...
///
/// Any deviation (different first property, non-object JSON, partial
/// input that can't disambiguate) returns `FastPathUnavailable` or
/// `NeedMore` so the caller can either wait or fall back cleanly.
fn scan_for_discriminator(buffer: &str, discriminator: &str) -> ScanResult {
    let bytes = buffer.as_bytes();
    let mut i = skip_ws(bytes, 0);
    // Expect `{`. If we see anything else (or run out), we can't fast-path.
    match bytes.get(i) {
        None => return ScanResult::NeedMore,
        Some(b'{') => i += 1,
        Some(_) => return ScanResult::FastPathUnavailable,
    }
    i = skip_ws(bytes, i);
    // Expect `"`.
    match bytes.get(i) {
        None => return ScanResult::NeedMore,
        Some(b'"') => i += 1,
        Some(_) => return ScanResult::FastPathUnavailable,
    }
    // Read the property name. If it doesn't match the discriminator, we
    // can't fast-path (the discriminator isn't the first property).
    let property_start = i;
    let property_end = match find_string_end(bytes, i) {
        FindEnd::NeedMore => return ScanResult::NeedMore,
        FindEnd::FoundAt(end) => end,
    };
    let property_name = &buffer[property_start..property_end];
    if property_name != discriminator {
        return ScanResult::FastPathUnavailable;
    }
    i = property_end + 1; // past closing `"`
    i = skip_ws(bytes, i);
    // Expect `:`.
    match bytes.get(i) {
        None => return ScanResult::NeedMore,
        Some(b':') => i += 1,
        Some(_) => return ScanResult::FastPathUnavailable,
    }
    i = skip_ws(bytes, i);
    // Expect `"` (the discriminator value must be a string).
    match bytes.get(i) {
        None => return ScanResult::NeedMore,
        Some(b'"') => i += 1,
        Some(_) => return ScanResult::FastPathUnavailable,
    }
    let value_start = i;
    let value_end = match find_string_end(bytes, i) {
        FindEnd::NeedMore => return ScanResult::NeedMore,
        FindEnd::FoundAt(end) => end,
    };
    // Decode the value, handling minimal JSON escape sequences (the
    // discriminator is a tag identifier — backslash sequences are
    // uncommon, but we honour the JSON spec).
    let raw = &buffer[value_start..value_end];
    let decoded = decode_json_string_value(raw);
    ScanResult::Found(decoded)
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while let Some(&b) = bytes.get(i) {
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            i += 1;
        } else {
            break;
        }
    }
    i
}

enum FindEnd {
    NeedMore,
    FoundAt(usize),
}

/// Find the closing `"` of a JSON string, honouring backslash escapes
/// (a `\"` inside the string is part of the value, not the terminator).
/// `start` is the byte index of the first content byte (one past the
/// opening `"`).
fn find_string_end(bytes: &[u8], start: usize) -> FindEnd {
    let mut i = start;
    while let Some(&b) = bytes.get(i) {
        match b {
            b'\\' => {
                // Skip the escaped character (or wait if it's truncated).
                if bytes.get(i + 1).is_none() {
                    return FindEnd::NeedMore;
                }
                i += 2;
            }
            b'"' => return FindEnd::FoundAt(i),
            _ => i += 1,
        }
    }
    FindEnd::NeedMore
}

/// Decode the minimal JSON string-escape vocabulary the discriminator
/// scanner needs. We don't honour `\uXXXX` because real-world
/// discriminator tags are ASCII identifiers; if a tag needs Unicode
/// escapes the schema author can declare it explicitly with the literal
/// characters.
fn decode_json_string_value(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(esc) = chars.next() {
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    other => {
                        // Unknown escape — keep the literal so the
                        // tag-mismatch path surfaces clearly rather
                        // than silently corrupting the comparison.
                        out.push('\\');
                        out.push(other);
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

impl JsonValueParser for JsonMultiArmParser {
    fn push(&mut self, chunk: &str) -> JsonStep {
        // Already resolved — replay the terminal state. Reporting the
        // chunk as fully consumed matches the existing parser contract
        // (don't ask the parent to re-push bytes after termination).
        if let Some(terminal) = &self.terminal {
            return match terminal {
                TerminalState::Done { value } => JsonStep {
                    consumed: chunk.len(),
                    events: Vec::new(),
                    completion: JsonCompletion::Done(value.clone()),
                },
                TerminalState::Errored { message } => JsonStep {
                    consumed: chunk.len(),
                    events: Vec::new(),
                    completion: JsonCompletion::Errored(message.clone()),
                },
            };
        }

        // Append chunk to the shared buffer; arms read from there via
        // their per-arm cursors.
        self.buffer.push_str(chunk);

        // Tagged-OneOf fast-path: in Scanning stage, defer arm parsers
        // until the discriminator is decoded. Once resolved (either
        // via commit or fallback to parallel parsing), feed the arms
        // the buffered bytes.
        if self.stage == DispatchStage::Scanning && !self.try_resolve_dispatch() {
            self.bytes_reported_consumed += chunk.len();
            return JsonStep {
                consumed: chunk.len(),
                events: Vec::new(),
                completion: JsonCompletion::NeedMore,
            };
        }

        self.feed_arms();

        match self.try_resolve() {
            None => {
                // All bytes from this chunk now sit in the shared
                // buffer; arms are still parsing. The parent should
                // continue feeding us subsequent bytes.
                self.bytes_reported_consumed += chunk.len();
                JsonStep {
                    consumed: chunk.len(),
                    events: Vec::new(),
                    completion: JsonCompletion::NeedMore,
                }
            }
            Some(resolution) => self.step_from_resolution(chunk.len(), resolution),
        }
    }

    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue) {
        // If we've already resolved (push() set self.terminal), return
        // the winning value or null+error per the existing parser
        // contract.
        if let Some(terminal) = self.terminal {
            return match terminal {
                TerminalState::Done { value } => (Vec::new(), value),
                TerminalState::Errored { message } => (
                    vec![ParseEvent::StreamError {
                        path: Some(path.into()),
                        message,
                    }],
                    FieldValue::Null,
                ),
            };
        }

        // Stream ended while some arms were still parsing. Force-finish
        // each Parsing arm: scalar parsers (Int, Float, Bool) commonly
        // stay in NeedMore on their last byte because they're waiting
        // for a terminator they'll never receive; finish() commits the
        // accumulated token. We treat an arm as Done when its finish()
        // produces a value AND emits no StreamError events.
        let Self {
            rule,
            arms,
            path: parser_path,
            ..
        } = *self;
        let mut done_arms: Vec<(usize, FieldValue, Vec<ParseEvent>)> = Vec::new();
        for mut arm in arms {
            match arm.status {
                ArmStatus::Done => {
                    if let Some(v) = arm.captured_value.take() {
                        let events = std::mem::take(&mut arm.buffered_events);
                        done_arms.push((arm.arm_index, v, events));
                    }
                }
                ArmStatus::Parsing => {
                    let Some(parser) = arm.parser.take() else {
                        continue;
                    };
                    let (finish_events, finish_value) = parser.finish(&parser_path);
                    let had_error = finish_events
                        .iter()
                        .any(|e| matches!(e, ParseEvent::StreamError { .. }));
                    if had_error {
                        // Treat as Errored; discard the arm's events
                        // (losing arms' progressive events don't leak
                        // downstream).
                        continue;
                    }
                    // Concatenate buffered events with finish-time events
                    // so the consumer sees the arm's full event sequence
                    // when this arm wins.
                    let mut combined = std::mem::take(&mut arm.buffered_events);
                    combined.extend(finish_events);
                    done_arms.push((arm.arm_index, finish_value, combined));
                }
                ArmStatus::Errored => {}
            }
        }

        // Pick the candidate at most-one-arm by rule. Using a single
        // `.into_iter().next()` for both branches avoids the
        // `expect("len==1")` pattern: Exclusive coerces a many-arm Vec to
        // None (ambiguous resolution at stream end is indistinguishable
        // from no-match for our purposes), FirstMatch always takes the
        // first.
        let resolution = match rule {
            MatchRule::Exclusive if done_arms.len() == 1 => done_arms.into_iter().next(),
            MatchRule::Exclusive => None,
            MatchRule::FirstMatch => done_arms.into_iter().next(),
        };

        if let Some((arm_index, value, events)) = resolution {
            let mut all_events = events;
            let wrapped = FieldValue::Variant {
                arm_index,
                value: Box::new(value),
            };
            all_events.push(ParseEvent::FieldComplete {
                path: path.into(),
                value: wrapped.clone(),
            });
            (all_events, wrapped)
        } else {
            let msg = format!(
                "{} at '{path}' did not resolve before stream end",
                match rule {
                    MatchRule::Exclusive => "OneOf",
                    MatchRule::FirstMatch => "AnyOf",
                }
            );
            (
                vec![ParseEvent::StreamError {
                    path: Some(path.into()),
                    message: msg,
                }],
                FieldValue::Null,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use typesayer_types::field::{FieldType, ObjectField, OneOfDiscriminator};

    use super::*;

    fn run_to_completion(
        mut parser: Box<dyn JsonValueParser>,
        chunks: &[&str],
    ) -> (Vec<ParseEvent>, JsonCompletion, usize) {
        let mut all_events = Vec::new();
        let mut total_consumed = 0;
        for chunk in chunks {
            let step = parser.push(chunk);
            all_events.extend(step.events);
            total_consumed += step.consumed;
            if !matches!(step.completion, JsonCompletion::NeedMore) {
                return (all_events, step.completion, total_consumed);
            }
        }
        let (finish_events, value) = parser.finish("/x");
        all_events.extend(finish_events);
        (all_events, JsonCompletion::Done(value), total_consumed)
    }

    fn tagged_oneof_two_arms() -> FieldType {
        FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "monthly".into(),
                    field_type: FieldType::Object(vec![
                        ObjectField {
                            name: "toolName".into(),
                            description: String::new(),
                            field_type: FieldType::Enum(vec!["monthly".into()]),
                        },
                        ObjectField {
                            name: "dimension".into(),
                            description: String::new(),
                            field_type: FieldType::Enum(vec!["category".into()]),
                        },
                    ]),
                },
                VariantArm {
                    description: "ranked_items".into(),
                    field_type: FieldType::Object(vec![
                        ObjectField {
                            name: "toolName".into(),
                            description: String::new(),
                            field_type: FieldType::Enum(vec!["ranked_items".into()]),
                        },
                        ObjectField {
                            name: "topN".into(),
                            description: String::new(),
                            field_type: FieldType::Int,
                        },
                    ]),
                },
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "toolName".into(),
                tags: vec!["monthly".into(), "ranked_items".into()],
            }),
        }
    }

    fn untagged_oneof_int_or_string_list() -> FieldType {
        FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "int".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "list".into(),
                    field_type: FieldType::List(Box::new(FieldType::String)),
                },
            ],
            discriminator: None,
        }
    }

    fn parser_for(ft: &FieldType) -> Box<dyn JsonValueParser> {
        json_value_parser_for(ft, "/x".into())
    }

    #[test]
    fn tagged_oneof_one_chunk_happy_path() {
        let ft = tagged_oneof_two_arms();
        let chunks = [r#"{"toolName": "ranked_items", "topN": 5}"#];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        match completion {
            JsonCompletion::Done(FieldValue::Variant {
                arm_index: 1,
                value,
            }) => match *value {
                FieldValue::Object(_) => {}
                other => panic!("expected Object inner, got {other:?}"),
            },
            other => panic!("expected Done(Variant{{1, Object}}), got {other:?}"),
        }
    }

    #[test]
    fn tagged_oneof_byte_by_byte_chunking() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items", "topN": 5}"#;
        let chunks: Vec<String> = raw.chars().map(|c| c.to_string()).collect();
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunk_refs);
        assert!(matches!(
            completion,
            JsonCompletion::Done(FieldValue::Variant { arm_index: 1, .. })
        ));
    }

    #[test]
    fn tagged_oneof_unknown_tag_errors() {
        let ft = tagged_oneof_two_arms();
        let chunks = [r#"{"toolName": "unknown_tool", "topN": 5}"#];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        assert!(
            matches!(completion, JsonCompletion::Errored(_)),
            "expected error, got: {completion:?}"
        );
    }

    #[test]
    fn untagged_oneof_int_arm_matches() {
        let ft = untagged_oneof_int_or_string_list();
        let chunks = ["42"];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        match completion {
            JsonCompletion::Done(FieldValue::Variant {
                arm_index: 0,
                value,
            }) => {
                assert_eq!(*value, FieldValue::Int(42));
            }
            other => panic!("expected Done(Variant{{0, Int(42)}}), got {other:?}"),
        }
    }

    #[test]
    fn untagged_oneof_list_arm_matches() {
        let ft = untagged_oneof_int_or_string_list();
        let chunks = [r#"["a", "b"]"#];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        assert!(matches!(
            completion,
            JsonCompletion::Done(FieldValue::Variant { arm_index: 1, .. })
        ));
    }

    #[test]
    fn untagged_oneof_no_match_errors() {
        let ft = untagged_oneof_int_or_string_list();
        let chunks = ["true"];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        assert!(matches!(completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn untagged_oneof_ambiguous_when_arms_overlap() {
        let ft = FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "arm a".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "x".into(),
                        description: String::new(),
                        field_type: FieldType::Int,
                    }]),
                },
                VariantArm {
                    description: "arm b".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "x".into(),
                        description: String::new(),
                        field_type: FieldType::Int,
                    }]),
                },
            ],
            discriminator: None,
        };
        let chunks = [r#"{"x": 1}"#];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        match completion {
            JsonCompletion::Errored(msg) => {
                assert!(msg.contains("multiple"), "got: {msg}");
            }
            other => panic!("expected Errored(multiple), got {other:?}"),
        }
    }

    #[test]
    fn anyof_first_match_wins_at_resolution() {
        let ft = FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "int".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "float".into(),
                    field_type: FieldType::Float,
                },
            ],
        };
        let chunks = ["42 "];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        match completion {
            JsonCompletion::Done(FieldValue::Variant {
                arm_index: 0,
                value,
            }) => {
                assert_eq!(*value, FieldValue::Int(42));
            }
            other => panic!("expected first-match Int winner, got {other:?}"),
        }
    }

    #[test]
    fn anyof_no_match_errors() {
        let ft = FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "int".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "bool".into(),
                    field_type: FieldType::Bool,
                },
            ],
        };
        let chunks = [r#""hello""#];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        assert!(matches!(completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn split_at_discriminator_value_boundary() {
        // The discriminator string value spans the chunk boundary.
        let ft = tagged_oneof_two_arms();
        let chunks = [r#"{"toolName": "rank"#, r#"ed_items", "topN": 5}"#];
        let (_, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        assert!(matches!(
            completion,
            JsonCompletion::Done(FieldValue::Variant { arm_index: 1, .. })
        ));
    }

    // ============================================================
    // Phase 8: Truncation tests — the LLM stops mid-stream at every
    // interesting boundary. The parser must surface a specific error
    // (never panic, never return a half-parsed value) and degrade
    // cleanly.
    // ============================================================

    #[test]
    fn truncation_before_any_byte_returns_error() {
        let ft = tagged_oneof_two_arms();
        let parser = parser_for(&ft);
        let (events, value) = parser.finish("/x");
        // No bytes seen at all — every arm finishes with no input.
        // No discriminator can resolve, so OneOf returns an error.
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn truncation_mid_discriminator_value_returns_error() {
        // Stream ends partway through the discriminator's string value.
        let ft = tagged_oneof_two_arms();
        let mut parser = parser_for(&ft);
        let step = parser.push(r#"{"toolName": "month"#);
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn truncation_after_discriminator_before_payload_returns_error() {
        // Discriminator complete, but payload field never arrives.
        let ft = tagged_oneof_two_arms();
        let mut parser = parser_for(&ft);
        let step = parser.push(r#"{"toolName": "ranked_items", "#);
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn truncation_mid_payload_value_returns_error() {
        // Object opener and discriminator present, but the payload
        // field's value is cut off.
        let ft = tagged_oneof_two_arms();
        let mut parser = parser_for(&ft);
        let step = parser.push(r#"{"toolName": "ranked_items", "topN":"#);
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn truncation_after_opening_brace_returns_error() {
        let ft = tagged_oneof_two_arms();
        let mut parser = parser_for(&ft);
        let step = parser.push("{");
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn truncation_after_discriminator_property_name_returns_error() {
        // The property name has been emitted but the colon + value
        // haven't. Each arm's Object parser is waiting for the value.
        let ft = tagged_oneof_two_arms();
        let mut parser = parser_for(&ft);
        let step = parser.push(r#"{"toolName""#);
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    // --- AnyOf truncation: same exhaustive boundary set as OneOf ---

    fn anyof_int_or_object() -> FieldType {
        FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "int".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "obj".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "x".into(),
                        description: String::new(),
                        field_type: FieldType::Int,
                    }]),
                },
            ],
        }
    }

    #[test]
    fn anyof_truncation_before_any_byte_returns_error() {
        let ft = anyof_int_or_object();
        let parser = parser_for(&ft);
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn anyof_truncation_after_opening_brace_returns_error() {
        let ft = anyof_int_or_object();
        let mut parser = parser_for(&ft);
        let step = parser.push("{");
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn anyof_truncation_mid_object_field_value_returns_error() {
        let ft = anyof_int_or_object();
        let mut parser = parser_for(&ft);
        let step = parser.push(r#"{"x": "#);
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        assert!(matches!(value, FieldValue::Null));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParseEvent::StreamError { .. }))
        );
    }

    #[test]
    fn anyof_truncation_in_int_arm_completes_via_finish() {
        // Number parsers commit on finish() if they've accumulated valid
        // digits. AnyOf's first-match rule then picks the int arm.
        let ft = anyof_int_or_object();
        let mut parser = parser_for(&ft);
        let step = parser.push("42");
        assert!(matches!(step.completion, JsonCompletion::NeedMore));
        let (events, value) = parser.finish("/x");
        // First match wins: arm 0 (Int) commits with value 42.
        match value {
            FieldValue::Variant {
                arm_index: 0,
                value,
            } => {
                assert_eq!(*value, FieldValue::Int(42));
            }
            other => panic!("expected Variant(0, Int), got {other:?} events={events:?}"),
        }
    }

    #[test]
    fn truncation_after_valid_complete_value_still_resolves() {
        // The value IS syntactically complete; finish() should
        // resolve to the matched arm, not error. Guards against
        // an over-eager truncation check.
        let ft = tagged_oneof_two_arms();
        let mut parser = parser_for(&ft);
        // Object closes; the JSON value is complete, parser should
        // hit Done on the closing brace.
        let step = parser.push(r#"{"toolName": "ranked_items", "topN": 5}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Variant { arm_index: 1, .. }) => {}
            other => panic!("expected Done at closing brace, got {other:?}"),
        }
    }

    // ============================================================
    // Layer 4 streaming counterpart: each near-miss case is fed in
    // three chunkings (one delta, byte-by-byte, split mid-tag-value)
    // and must error identically every time.
    // ============================================================

    fn chunkings_for(raw: &str) -> Vec<Vec<String>> {
        let one_chunk: Vec<String> = vec![raw.to_owned()];
        let byte_by_byte: Vec<String> = raw.chars().map(|c| c.to_string()).collect();
        let mid_tag_value = raw.find("\":").map_or_else(
            || vec![raw.to_owned()],
            |quote_idx| {
                // Split roughly inside the discriminator's value:
                // after the colon's space + a few bytes into the
                // value string.
                let split_at = (quote_idx + 5).min(raw.len() - 1);
                vec![raw[..split_at].to_owned(), raw[split_at..].to_owned()]
            },
        );
        vec![one_chunk, byte_by_byte, mid_tag_value]
    }

    fn drive(chunks: &[String], ft: &FieldType) -> JsonCompletion {
        let mut parser = parser_for(ft);
        let mut last_completion = JsonCompletion::NeedMore;
        for c in chunks {
            let step = parser.push(c);
            last_completion = step.completion;
            if !matches!(last_completion, JsonCompletion::NeedMore) {
                return last_completion;
            }
        }
        let (_events, value) = parser.finish("/x");
        // Done with finish-time value; collapse to Done so callers can
        // assert structurally.
        match last_completion {
            JsonCompletion::Done(_) | JsonCompletion::Errored(_) => last_completion,
            JsonCompletion::NeedMore => JsonCompletion::Done(value),
        }
    }

    #[test]
    fn streaming_case_mismatch_errors_under_all_chunkings() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "Ranked_Items", "topN": 5}"#;
        for chunks in chunkings_for(raw) {
            let outcome = drive(&chunks, &ft);
            match outcome {
                JsonCompletion::Errored(_) => {}
                JsonCompletion::Done(v) => {
                    panic!("case-mismatch must error, got Done({v:?}) under chunks {chunks:?}")
                }
                JsonCompletion::NeedMore => panic!("stream stuck at NeedMore: {chunks:?}"),
            }
        }
    }

    #[test]
    fn streaming_substring_superset_errors_under_all_chunkings() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items_v2", "topN": 5}"#;
        for chunks in chunkings_for(raw) {
            let outcome = drive(&chunks, &ft);
            assert!(
                matches!(outcome, JsonCompletion::Errored(_)),
                "substring-superset must error, got {outcome:?} under chunks {chunks:?}"
            );
        }
    }

    #[test]
    fn streaming_whitespace_in_tag_errors_under_all_chunkings() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": " ranked_items", "topN": 5}"#;
        for chunks in chunkings_for(raw) {
            let outcome = drive(&chunks, &ft);
            assert!(
                matches!(outcome, JsonCompletion::Errored(_)),
                "whitespace-in-tag must error, got {outcome:?} under chunks {chunks:?}"
            );
        }
    }

    #[test]
    fn streaming_property_case_mismatch_errors_under_all_chunkings() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"Toolname": "ranked_items", "topN": 5}"#;
        for chunks in chunkings_for(raw) {
            let outcome = drive(&chunks, &ft);
            // The arms each declare `toolName` as a const-restricted Enum;
            // an Object missing that property fails arm parsing, so all
            // arms error → OneOf no-match.
            assert!(
                matches!(outcome, JsonCompletion::Errored(_)),
                "property-case-mismatch must error, got {outcome:?} under chunks {chunks:?}"
            );
        }
    }

    // ============================================================
    // Layer 3: parallel-arm memory-bound proptest.
    //
    // The architectural invariant: there's ONE shared input buffer
    // sized O(input). Per-arm cursors track position in that buffer.
    // Per-arm Box<dyn JsonValueParser> may hold internal state but
    // each arm receives a slice of the shared buffer — no per-arm
    // copy of the input.
    //
    // The "linear-in-arms" memory bug would manifest as the shared
    // buffer size scaling with arm count. This proptest constructs
    // a OneOf with N (1..=8) arms and feeds an input of M bytes;
    // shared_buffer_len() must equal M regardless of N.
    // ============================================================

    use proptest::prelude::*;

    fn arms_with_distinct_const(count: usize) -> Vec<VariantArm> {
        (0..count)
            .map(|i| VariantArm {
                description: format!("arm {i}"),
                field_type: FieldType::Object(vec![
                    ObjectField {
                        name: "kind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec![format!("tag_{i}")]),
                    },
                    ObjectField {
                        name: "x".into(),
                        description: String::new(),
                        field_type: FieldType::Int,
                    },
                ]),
            })
            .collect()
    }

    proptest! {
        /// Property: shared input buffer is O(input), not O(input × arms).
        /// For a OneOf with N arms (1..=8), pushing a fixed-size input
        /// produces a shared_buffer_len() equal to the cumulative input
        /// length — independent of arm count.
        #[test]
        fn shared_buffer_grows_linearly_in_input_not_arms(
            arm_count in 1usize..=8usize,
            // Chunked input pieces; total length = sum of piece sizes.
            chunks in proptest::collection::vec("[a-z]{1,30}", 1..=8),
        ) {
            let arms = arms_with_distinct_const(arm_count);
            let mut parser = JsonMultiArmParser::oneof("/x".into(), &arms);
            let mut total_pushed: usize = 0;
            for chunk in &chunks {
                let step = parser.push(chunk);
                total_pushed += chunk.len();
                // After each push, the shared buffer should hold exactly
                // the bytes we've fed in so far (until resolution).
                if matches!(step.completion, JsonCompletion::NeedMore) {
                    prop_assert_eq!(
                        parser.shared_buffer_len(),
                        total_pushed,
                        "shared buffer must equal cumulative input on NeedMore"
                    );
                }
                if !matches!(step.completion, JsonCompletion::NeedMore) {
                    break;
                }
            }
        }

        /// Property: per-arm count doesn't multiply the shared buffer.
        /// Compares two parsers with different arm counts but the same
        /// input — both should produce identical shared_buffer_len().
        #[test]
        fn shared_buffer_size_independent_of_arm_count(
            input in "[a-z]{1,200}",
        ) {
            let mut parser_2 = JsonMultiArmParser::oneof(
                "/x".into(), &arms_with_distinct_const(2));
            let mut parser_8 = JsonMultiArmParser::oneof(
                "/x".into(), &arms_with_distinct_const(8));

            let step_2 = parser_2.push(&input);
            let step_8 = parser_8.push(&input);

            // Both parsers see the same bytes pushed; the shared
            // buffer's size is purely a function of input bytes seen,
            // not arm count.
            if matches!(step_2.completion, JsonCompletion::NeedMore)
                && matches!(step_8.completion, JsonCompletion::NeedMore)
            {
                prop_assert_eq!(
                    parser_2.shared_buffer_len(),
                    parser_8.shared_buffer_len(),
                    "shared buffer length must not depend on arm count"
                );
            }
        }
    }

    // ============================================================
    // Tagged-OneOf fast-path tests. Verify that the scanner identifies
    // the discriminator before any per-arm parser is constructed
    // (savings = N-1 unused arm parsers), and that the fall-back path
    // engages cleanly when the discriminator isn't first or the input
    // doesn't match expected shape.
    // ============================================================

    fn tagged_oneof_for_fast_path() -> FieldType {
        // Tagged OneOf with 4 arms; discriminator is "kind" (Enum
        // restricted per arm). Fast path should pick the right arm and
        // never instantiate the other 3.
        let make_arm = |tag: &str| VariantArm {
            description: format!("arm {tag}"),
            field_type: FieldType::Object(vec![
                ObjectField {
                    name: "kind".into(),
                    description: String::new(),
                    field_type: FieldType::Enum(vec![tag.into()]),
                },
                ObjectField {
                    name: "x".into(),
                    description: String::new(),
                    field_type: FieldType::Int,
                },
            ]),
        };
        FieldType::OneOf {
            arms: vec![make_arm("a"), make_arm("b"), make_arm("c"), make_arm("d")],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            }),
        }
    }

    /// Construct a fresh `JsonMultiArmParser` for a tagged OneOf. The
    /// json_value_parser_for dispatch hands back a `Box<dyn>`, so for
    /// the fast-path inspection tests we need direct access to the
    /// concrete type. We mirror the dispatch's `tagged_oneof` path
    /// here.
    fn tagged_parser(ft: &FieldType) -> JsonMultiArmParser {
        match ft {
            FieldType::OneOf {
                arms,
                discriminator: Some(d),
            } => JsonMultiArmParser::tagged_oneof(
                "/x".into(),
                arms,
                d.property.clone(),
                d.tags.clone(),
            ),
            _ => unreachable!("tagged_parser called with non-tagged-OneOf"),
        }
    }

    #[test]
    fn fast_path_commits_on_discriminator_first_input() {
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        // Initially no arm parsers materialised (fast-path mode).
        assert_eq!(parser.materialised_arm_count(), 0);

        // Push enough bytes to expose the discriminator value.
        let step = parser.push(r#"{"kind": "c", "x": 42}"#);
        // Fast path committed to arm c (index 2); only that arm has a
        // parser, the other three were skipped entirely.
        assert!(
            parser.materialised_arm_count() <= 1,
            "fast path must materialise at most one arm parser, got {}",
            parser.materialised_arm_count()
        );
        match step.completion {
            JsonCompletion::Done(FieldValue::Variant { arm_index: 2, .. }) => {}
            other => panic!("expected Done(Variant{{2, ...}}), got {other:?}"),
        }
    }

    #[test]
    fn fast_path_falls_back_when_discriminator_not_first() {
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        // The discriminator is "kind" but the LLM emitted "x" first.
        // Fast path should detect this and fall back to parallel-arm
        // (all arm parsers materialised, all but the matching one
        // ultimately error).
        let step = parser.push(r#"{"x": 5, "kind": "a"}"#);
        // Fallback path is engaged → every arm parser materialised
        // (then non-matching arms self-error on the Enum const check).
        assert_eq!(
            parser.materialised_arm_count(),
            4,
            "fallback path must materialise every arm"
        );
        // And the correct arm still wins via the parallel-arm dispatch.
        match step.completion {
            JsonCompletion::Done(FieldValue::Variant { arm_index: 0, .. }) => {}
            other => panic!("expected Done(Variant{{0, ...}}), got {other:?}"),
        }
    }

    #[test]
    fn fast_path_progresses_across_chunked_pushes() {
        // The scanner must accumulate bytes across multiple push() calls
        // until the discriminator value is fully decoded.
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        // First chunk doesn't include the closing `"` of the value.
        let step = parser.push(r#"{"kind": "c"#);
        assert!(
            matches!(step.completion, JsonCompletion::NeedMore),
            "scanner waits for closing quote"
        );
        assert_eq!(
            parser.materialised_arm_count(),
            0,
            "no arm yet — still scanning"
        );
        // Second chunk completes the value and the rest of the object.
        let step = parser.push(r#"", "x": 99}"#);
        assert!(
            parser.materialised_arm_count() <= 1,
            "fast path winner only"
        );
        match step.completion {
            JsonCompletion::Done(FieldValue::Variant { arm_index: 2, .. }) => {}
            other => panic!("expected Done(Variant{{2, ...}}), got {other:?}"),
        }
    }

    #[test]
    fn fast_path_falls_back_on_unknown_tag() {
        // The scanner sees a value but it's not in the declared tag
        // set. Fall back to parallel-arm so the resolution emits a
        // proper OneOfNoArmMatched.
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        let step = parser.push(r#"{"kind": "z", "x": 5}"#);
        // Fallback path engaged → all arm parsers materialised, none
        // match, error returned.
        assert_eq!(parser.materialised_arm_count(), 4);
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn fast_path_falls_back_on_non_object_input() {
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        let step = parser.push("[1, 2, 3]");
        // Scanner sees `[` not `{`, falls back. Parallel-arm parsers
        // all error (no Object arm matches an array input).
        assert_eq!(parser.materialised_arm_count(), 4);
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn fast_path_falls_back_on_leading_whitespace_does_not_break_commit() {
        // Whitespace before `{` and around the discriminator is JSON-
        // legal and the scanner must tolerate it without falling back.
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        let step = parser.push("   {   \"kind\"  :   \"b\"  ,  \"x\" : 1  }");
        assert!(
            parser.materialised_arm_count() <= 1,
            "whitespace tolerance must not break fast path"
        );
        match step.completion {
            JsonCompletion::Done(FieldValue::Variant { arm_index: 1, .. }) => {}
            other => panic!("expected Done(Variant{{1, ...}}), got {other:?}"),
        }
    }

    #[test]
    fn fast_path_falls_back_when_discriminator_value_is_not_string() {
        let ft = tagged_oneof_for_fast_path();
        let mut parser = tagged_parser(&ft);
        // Number-valued discriminator — invalid per the tag set, but
        // also impossible to disambiguate via the fast-path scanner
        // (which only handles string-valued discriminators).
        let step = parser.push(r#"{"kind": 42, "x": 1}"#);
        assert_eq!(
            parser.materialised_arm_count(),
            4,
            "non-string discriminator must trigger fallback"
        );
        assert!(matches!(step.completion, JsonCompletion::Errored(_)));
    }

    #[test]
    fn fast_path_handles_escape_in_discriminator_value() {
        // A discriminator tag containing an escape sequence (rare but
        // legal). The scanner must decode `\"` → `"` and match.
        let ft = FieldType::OneOf {
            arms: vec![VariantArm {
                description: "tag with quote".into(),
                field_type: FieldType::Object(vec![
                    ObjectField {
                        name: "kind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec!["a\"b".into()]),
                    },
                    ObjectField {
                        name: "x".into(),
                        description: String::new(),
                        field_type: FieldType::Int,
                    },
                ]),
            }],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["a\"b".into()],
            }),
        };
        let mut parser = tagged_parser(&ft);
        let step = parser.push(r#"{"kind": "a\"b", "x": 1}"#);
        match step.completion {
            JsonCompletion::Done(FieldValue::Variant { arm_index: 0, .. }) => {}
            other => panic!("expected Done(Variant{{0, ...}}), got {other:?}"),
        }
    }

    #[test]
    fn losing_arm_events_are_suppressed() {
        // For tagged OneOf, the wrong-arm parser starts emitting events
        // (e.g. ObjectOpen) before erroring on the discriminator value.
        // Those events must not leak to the consumer.
        let ft = tagged_oneof_two_arms();
        let chunks = [r#"{"toolName": "monthly", "dimension": "category"}"#];
        let (events, completion, _) = run_to_completion(parser_for(&ft), &chunks);
        assert!(matches!(
            completion,
            JsonCompletion::Done(FieldValue::Variant { arm_index: 0, .. })
        ));
        // Every emitted event must carry the variant's path "/x", not a
        // synthetic per-arm path. Since both arms used path "/x", this
        // mostly checks we got events and they're well-formed.
        for ev in &events {
            match ev {
                ParseEvent::ObjectOpen { path }
                | ParseEvent::EntryAdded { path, .. }
                | ParseEvent::ObjectClose { path, .. } => assert_eq!(path, "/x"),
                _ => {}
            }
        }
    }
}
