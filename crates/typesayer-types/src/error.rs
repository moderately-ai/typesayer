// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Predict engine error types.

/// Head budget for raw-completion excerpts attached to parse errors.
/// Calibrated to surface the first marker / opening JSON brace plus a
/// couple of fields' worth of content so the operator can see where the
/// model's output went off the rails.
const EXCERPT_HEAD_BYTES: usize = 400;

/// Tail budget for raw-completion excerpts attached to parse errors.
/// Smaller than the head because tail context is usually a trailing
/// `[[ ## completed ## ]]` (or a half-emitted final value) — enough to
/// confirm "did the model finish?" without dominating the error log.
const EXCERPT_TAIL_BYTES: usize = 200;

/// Build a head + tail excerpt of `raw` suitable for embedding in an error message.
///
/// Pure head when `raw` fits in the combined budget; otherwise emits
/// `<head>… (N bytes elided) …<tail>` with a real "bytes elided" count
/// so the operator knows how much was dropped.
///
/// Always slices on char boundaries — a raw completion can contain
/// arbitrary UTF-8 (model emitted prose, emoji in user-provided
/// inputs echoed back, etc.), and slicing through a multi-byte code
/// point would panic.
#[must_use]
pub fn build_excerpt(raw: &str) -> String {
    if raw.len() <= EXCERPT_HEAD_BYTES + EXCERPT_TAIL_BYTES {
        return raw.to_owned();
    }
    let head = safe_prefix(raw, EXCERPT_HEAD_BYTES);
    let tail = safe_suffix(raw, EXCERPT_TAIL_BYTES);
    let elided = raw.len().saturating_sub(head.len() + tail.len());
    format!("{head}\n… ({elided} bytes elided) …\n{tail}")
}

/// Largest char-boundary-safe prefix of `s` whose byte length is at
/// most `max_bytes`.
fn safe_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Largest char-boundary-safe suffix of `s` whose byte length is at
/// most `max_bytes`.
fn safe_suffix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len() - max_bytes;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// Errors that can occur in predict operations.
#[derive(Debug, thiserror::Error)]
pub enum PredictError {
    /// One or more required output fields are absent from the completion.
    ///
    /// Surfaced from an adapter's `parse` implementation when the
    /// parser found at least one field marker but not every required
    /// output field. The raw excerpt + total-bytes context lets an
    /// operator diagnose from the error alone instead of running the
    /// model again with extra logging.
    #[error(
        "missing output fields: {fields:?} (expected {} field(s); model emitted {raw_bytes_total} bytes)\n  raw excerpt:\n    {raw_excerpt}",
        expected.len(),
    )]
    MissingFields {
        /// The names of the missing fields.
        fields: Vec<String>,
        /// All output field names, in declaration order. Pairs with
        /// `fields` to surface "missing 2 of 3" context.
        expected: Vec<String>,
        /// Head + tail excerpt of the raw model completion at parse
        /// time. Empty when the error is constructed outside the
        /// adapter parse path (e.g. a `Prediction::get` lookup miss
        /// — though that uses [`FieldNotInPrediction`](Self::FieldNotInPrediction)).
        raw_excerpt: String,
        /// Total byte length of the raw completion. Reading the excerpt
        /// with `(0 of 50000)` context vs `(0 of 200)` context changes
        /// the operator's diagnosis (truncation vs malformed output).
        raw_bytes_total: usize,
    },

    /// Caller asked for a field via `Prediction::get`
    /// that is not present in the prediction. Distinct from
    /// [`MissingFields`](Self::MissingFields) — which is a parser-level
    /// "model failed to emit" — because this one is a "caller asked
    /// for the wrong key" and the raw-completion context is irrelevant.
    #[error("field '{field}' is not present in this prediction")]
    FieldNotInPrediction {
        /// The lookup key that was not in the prediction.
        field: String,
    },

    /// The language model stopped generating before producing a complete
    /// response — most often because the configured `max_tokens` budget
    /// was exhausted mid-output. The completion is therefore truncated
    /// and cannot be parsed for output fields. Surfaced ahead of
    /// [`MissingFields`](Self::MissingFields) so operators see the real
    /// root cause instead of a generic parse failure.
    #[error(
        "language model truncated output: stop_reason={stop_reason} \
         (output_tokens={output_tokens:?}); raise max_tokens or shrink the input \
         to give the model room to finish"
    )]
    Truncated {
        /// The normalized stop reason that triggered the early stop.
        /// Stringified at the boundary so this error type doesn't take
        /// a public dependency on `modelplease::StopReason`.
        stop_reason: String,
        /// Output tokens generated before truncation, when the provider
        /// reported usage. `None` for providers that omit usage on a
        /// truncated response.
        output_tokens: Option<u64>,
    },

    /// A field value could not be coerced to the declared [`FieldType`](crate::FieldType).
    #[error("field '{field}' type mismatch: expected {expected}, got {actual:?}")]
    FieldTypeMismatch {
        /// The field name.
        field: String,
        /// The expected type label.
        expected: String,
        /// The raw value that failed coercion.
        actual: String,
    },

    /// The completion contained no `[[ ## field ## ]]` markers at all.
    ///
    /// The raw excerpt names what the model emitted instead — usually
    /// either a code-fenced JSON object the prompt didn't ask for, a
    /// natural-language refusal, or a near-miss marker (`[[answer]]`
    /// without the surrounding `## ` decorations).
    #[error(
        "no field markers found in completion (expected {expected_markers:?}; \
         model emitted {raw_bytes_total} bytes)\n  raw excerpt:\n    {raw_excerpt}"
    )]
    NoFieldMarkers {
        /// Markers the parser hoped to find (one per output field).
        expected_markers: Vec<String>,
        /// Head + tail excerpt of the raw model completion.
        raw_excerpt: String,
        /// Total byte length of the raw completion.
        raw_bytes_total: usize,
    },

    /// Signature construction failed due to invalid configuration.
    #[error("invalid signature: {reason}")]
    InvalidSignature {
        /// Why the signature is invalid.
        reason: String,
    },

    /// JSON serialization or deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// A language model call failed.
    #[error("language model error: {0}")]
    LanguageModel(#[from] modelplease::LanguageModelError),

    /// An optimizer encountered a runtime error (e.g. exceeded max errors).
    #[error("optimizer error: {message}")]
    Optimizer {
        /// Description of the optimizer error.
        message: String,
    },

    /// The evaluation runner exceeded its error budget.
    #[error("evaluation error: {message}")]
    Evaluation {
        /// Description of the evaluation error.
        message: String,
    },

    /// File I/O error during state save or load.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A tagged [`OneOf`](crate::FieldType::OneOf) input is missing its
    /// discriminator property. Either the property is absent from the JSON
    /// object, or the value being parsed isn't a JSON object at all.
    #[error(
        "field '{field}' is a tagged variant but its discriminator property \
         `{discriminator}` is missing or the value is not a JSON object"
    )]
    OneOfTagMissing {
        /// The field path being parsed.
        field: String,
        /// The discriminator property name declared by the OneOf type.
        discriminator: String,
    },

    /// A tagged [`OneOf`](crate::FieldType::OneOf) input carried a discriminator
    /// value that doesn't match any declared arm tag. Lists the valid tags so
    /// the operator can compare against the offending value (case, whitespace,
    /// near-miss spellings).
    #[error(
        "field '{field}' discriminator value '{tag}' is not a valid arm tag; \
         expected one of: {valid_tags:?}"
    )]
    OneOfTagInvalid {
        /// The field path being parsed.
        field: String,
        /// The offending discriminator value.
        tag: String,
        /// The full set of valid arm tags declared by the OneOf type.
        valid_tags: Vec<String>,
    },

    /// An untagged [`OneOf`](crate::FieldType::OneOf) input matched zero arms.
    /// Each per-arm parse failure is preserved so the operator can see why
    /// every arm rejected the value rather than guessing.
    #[error(
        "field '{field}' did not match any OneOf arm ({} arm(s) tried)",
        arm_errors.len()
    )]
    OneOfNoArmMatched {
        /// The field path being parsed.
        field: String,
        /// Per-arm rejection reasons, parallel to the arms vector. The `usize`
        /// is the arm index.
        arm_errors: Vec<(usize, Box<Self>)>,
    },

    /// An untagged [`OneOf`](crate::FieldType::OneOf) input matched more than
    /// one arm. JSON Schema's `oneOf` semantics require exactly one match;
    /// ambiguity is a schema-design bug surfaced to the operator.
    #[error(
        "field '{field}' matched multiple OneOf arms ({matching_arms:?}); \
         oneOf requires exactly one match. Either make the arms structurally \
         disjoint, or use anyOf for first-match-wins semantics."
    )]
    OneOfAmbiguous {
        /// The field path being parsed.
        field: String,
        /// Indices of every arm that successfully parsed the value.
        matching_arms: Vec<usize>,
    },

    /// An [`AnyOf`](crate::FieldType::AnyOf) input matched zero arms. AnyOf is
    /// first-match-wins; "no arm matched" means every arm rejected the value.
    #[error(
        "field '{field}' did not match any AnyOf arm ({} arm(s) tried)",
        arm_errors.len()
    )]
    AnyOfNoArmMatched {
        /// The field path being parsed.
        field: String,
        /// Per-arm rejection reasons, parallel to the arms vector.
        arm_errors: Vec<(usize, Box<Self>)>,
    },
}

impl PredictError {
    /// Create a [`MissingFields`](PredictError::MissingFields) error
    /// from the chat-adapter parse path. The raw completion is in scope
    /// at the construction site; passing it through populates the
    /// excerpt + total-bytes context the operator needs to debug from
    /// the error alone.
    #[must_use]
    pub fn missing_fields_from_parse<F, E>(missing: F, expected: E, raw_completion: &str) -> Self
    where
        F: IntoIterator<Item = String>,
        E: IntoIterator<Item = String>,
    {
        Self::MissingFields {
            fields: missing.into_iter().collect(),
            expected: expected.into_iter().collect(),
            raw_excerpt: build_excerpt(raw_completion),
            raw_bytes_total: raw_completion.len(),
        }
    }

    /// Create a [`NoFieldMarkers`](PredictError::NoFieldMarkers) error
    /// from the chat-adapter parse path.
    #[must_use]
    pub fn no_field_markers_from_parse<E>(expected_markers: E, raw_completion: &str) -> Self
    where
        E: IntoIterator<Item = String>,
    {
        Self::NoFieldMarkers {
            expected_markers: expected_markers.into_iter().collect(),
            raw_excerpt: build_excerpt(raw_completion),
            raw_bytes_total: raw_completion.len(),
        }
    }

    /// Create an [`InvalidSignature`](PredictError::InvalidSignature) error.
    pub fn invalid_signature(reason: impl Into<String>) -> Self {
        Self::InvalidSignature {
            reason: reason.into(),
        }
    }

    /// Create an [`Optimizer`](PredictError::Optimizer) error.
    pub fn optimizer(message: impl Into<String>) -> Self {
        Self::Optimizer {
            message: message.into(),
        }
    }

    /// Create an [`Evaluation`](PredictError::Evaluation) error.
    pub fn evaluation(message: impl Into<String>) -> Self {
        Self::Evaluation {
            message: message.into(),
        }
    }

    /// Create a [`Truncated`](PredictError::Truncated) error from a
    /// language-model `StopReason` and optional output token count.
    #[must_use]
    pub fn truncated(stop_reason: &modelplease::StopReason, output_tokens: Option<u64>) -> Self {
        let label = match stop_reason {
            modelplease::StopReason::EndTurn => "end_turn".to_owned(),
            modelplease::StopReason::MaxTokens => "max_tokens".to_owned(),
            modelplease::StopReason::StopSequence => "stop_sequence".to_owned(),
            modelplease::StopReason::ToolUse => "tool_use".to_owned(),
            modelplease::StopReason::ContentFilter => "content_filter".to_owned(),
            modelplease::StopReason::Other(s) => s.clone(),
        };
        Self::Truncated {
            stop_reason: label,
            output_tokens,
        }
    }
}

/// A specialized `Result` type for predict operations.
pub type Result<T> = std::result::Result<T, PredictError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_display_carries_excerpt_and_total_bytes() {
        let completion = "[[ ## answer ## ]]\n42\n[[ ## completed ## ]]";
        let err = PredictError::missing_fields_from_parse(
            ["reasoning".to_owned()],
            ["reasoning".to_owned(), "answer".to_owned()],
            completion,
        );
        let msg = err.to_string();
        assert!(msg.contains("reasoning"), "names the missing field: {msg}");
        assert!(msg.contains("2 field"), "states expected count: {msg}");
        assert!(
            msg.contains(&format!("{} bytes", completion.len())),
            "names total bytes: {msg}"
        );
        assert!(
            msg.contains("[[ ## answer ## ]]"),
            "includes raw excerpt: {msg}"
        );
    }

    #[test]
    fn no_field_markers_display_carries_excerpt() {
        let completion = "Sorry, I cannot answer that question.";
        let err = PredictError::no_field_markers_from_parse(
            ["[[ ## answer ## ]]".to_owned()],
            completion,
        );
        let msg = err.to_string();
        assert!(
            msg.contains("[[ ## answer ## ]]"),
            "names expected marker: {msg}"
        );
        assert!(
            msg.contains(&format!("{} bytes", completion.len())),
            "names total bytes: {msg}"
        );
        assert!(msg.contains("Sorry"), "includes raw excerpt: {msg}");
    }

    #[test]
    fn field_not_in_prediction_display_names_field() {
        let err = PredictError::FieldNotInPrediction {
            field: "oops".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("oops"), "names the missing key: {msg}");
    }

    #[test]
    fn build_excerpt_inlines_short_input() {
        let short = "hello";
        assert_eq!(build_excerpt(short), short);
    }

    #[test]
    fn build_excerpt_truncates_with_elision_marker() {
        // 1000-byte ASCII string so head + tail = 600 bytes and 400 are
        // elided. The exact byte count appears in the elision separator
        // so an operator can size up "truncation vs malformed".
        let raw = "x".repeat(1000);
        let excerpt = build_excerpt(&raw);
        assert!(
            excerpt.contains("400 bytes elided"),
            "names elided count: {excerpt}"
        );
        assert!(excerpt.len() < raw.len(), "shorter than input");
    }

    #[test]
    fn build_excerpt_respects_char_boundaries() {
        // A multi-byte boundary right at EXCERPT_HEAD_BYTES (400) — slicing
        // naïvely would panic. Construct a payload where byte 400 falls
        // mid-codepoint by interleaving a 4-byte char near the cut.
        let mut raw = String::with_capacity(EXCERPT_HEAD_BYTES + EXCERPT_TAIL_BYTES + 200);
        raw.push_str(&"a".repeat(EXCERPT_HEAD_BYTES - 2));
        raw.push('🦀'); // 4 bytes — straddles the 400-byte cut
        raw.push_str(&"b".repeat(EXCERPT_TAIL_BYTES + 200));
        // Just calling it is the test — a char-boundary failure would panic.
        let excerpt = build_excerpt(&raw);
        assert!(!excerpt.is_empty());
    }

    #[test]
    fn field_type_mismatch_display() {
        let err = PredictError::FieldTypeMismatch {
            field: "age".to_string(),
            expected: "int".to_string(),
            actual: "not_a_number".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("age"));
        assert!(msg.contains("int"));
        assert!(msg.contains("not_a_number"));
    }

    #[test]
    fn invalid_signature_display() {
        let err = PredictError::invalid_signature("no output fields");
        assert!(err.to_string().contains("no output fields"));
    }

    #[test]
    fn serialization_error_from_serde() {
        let serde_err = serde_json::from_str::<String>("invalid").unwrap_err();
        let err = PredictError::from(serde_err);
        assert!(matches!(err, PredictError::Serialization(_)));
    }

    // ============================================================
    // Phase 8: Error-display ergonomics for the new variant errors.
    //
    // Each new variant's Display impl must surface the information a
    // production operator needs to debug from the error alone: the field
    // path, the offending value, the valid options (if any). Snapshot-
    // style assertions guard against drift under future refactors.
    // ============================================================

    #[test]
    fn one_of_tag_missing_display_names_field_and_discriminator() {
        let err = PredictError::OneOfTagMissing {
            field: "results[0].assignment".into(),
            discriminator: "toolName".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("results[0].assignment"), "field path: {msg}");
        assert!(msg.contains("toolName"), "discriminator name: {msg}");
    }

    #[test]
    fn one_of_tag_invalid_display_lists_valid_tags() {
        let err = PredictError::OneOfTagInvalid {
            field: "assignment".into(),
            tag: "Ranked_Items".into(),
            valid_tags: vec!["monthly_breakdown".into(), "ranked_items".into()],
        };
        let msg = err.to_string();
        assert!(msg.contains("assignment"), "field path: {msg}");
        assert!(msg.contains("Ranked_Items"), "offending tag: {msg}");
        assert!(msg.contains("monthly_breakdown"), "valid tag listed: {msg}");
        assert!(msg.contains("ranked_items"), "valid tag listed: {msg}");
    }

    #[test]
    fn one_of_no_arm_matched_display_includes_arm_count() {
        let arm_errors = vec![
            (
                0,
                Box::new(PredictError::FieldTypeMismatch {
                    field: "arm0".into(),
                    expected: "int".into(),
                    actual: "abc".into(),
                }),
            ),
            (
                1,
                Box::new(PredictError::FieldTypeMismatch {
                    field: "arm1".into(),
                    expected: "object".into(),
                    actual: "abc".into(),
                }),
            ),
        ];
        let err = PredictError::OneOfNoArmMatched {
            field: "value".into(),
            arm_errors,
        };
        let msg = err.to_string();
        assert!(msg.contains("value"), "field path: {msg}");
        assert!(msg.contains("2 arm"), "arm count: {msg}");
    }

    #[test]
    fn one_of_ambiguous_display_lists_matching_arms_and_remediation() {
        let err = PredictError::OneOfAmbiguous {
            field: "result".into(),
            matching_arms: vec![0, 2],
        };
        let msg = err.to_string();
        assert!(msg.contains("result"), "field path: {msg}");
        assert!(msg.contains("[0, 2]"), "matching arms: {msg}");
        // Remediation hint helps the schema author resolve the
        // ambiguity rather than just naming the symptom.
        assert!(
            msg.contains("disjoint") || msg.contains("anyOf"),
            "remediation hint: {msg}"
        );
    }

    #[test]
    fn any_of_no_arm_matched_display_includes_arm_count() {
        let arm_errors = vec![(
            0,
            Box::new(PredictError::FieldTypeMismatch {
                field: "arm0".into(),
                expected: "int".into(),
                actual: "abc".into(),
            }),
        )];
        let err = PredictError::AnyOfNoArmMatched {
            field: "data".into(),
            arm_errors,
        };
        let msg = err.to_string();
        assert!(msg.contains("data"), "field path: {msg}");
        assert!(msg.contains("AnyOf"), "construct name: {msg}");
    }
}
