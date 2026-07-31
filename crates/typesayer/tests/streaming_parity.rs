// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity integration tests between the buffered [`ChatAdapter::parse`] and the
//! streaming [`ChatStreamParser`].
//!
//! These proptest properties are the load-bearing correctness invariant of the
//! streaming codec: for any supported `(FieldType, FieldValue)`, any chunking of
//! the wire-format completion, and any drift perturbation, the streaming parser
//! must produce the same final `FieldValue` the buffered parser produces on the
//! same unchunked completion.
//!
//! The tests live here (rather than in `typesayer-parser`) because they
//! depend on `ChatAdapter::parse` from `typesayer` — the streaming crate
//! has no upward dependency on the adapter layer.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use proptest::prelude::*;
use typesayer::{Adapter, ChatAdapter};
use typesayer_parser::{
    ChatStreamParser, ParseEvent,
    proptest_strategies::{
        DriftPerturbation, arb_chunking_positions, arb_drift_perturbation,
        arb_supported_type_and_value, arb_tagged_oneof_type_and_value, arb_unsupported_field_type,
        arb_variant_type_and_value, chunk_at_positions, serialize_completion,
        serialize_completion_with_drift, single_output_signature,
    },
};
use typesayer_types::field::{FieldType, FieldValue};

/// Run the top-level parser against `(field_type, value)` serialised to the LM
/// wire format, split into chunks, then re-assemble. Returns the full event
/// sequence so callers can inspect both the final value and any intermediate
/// events.
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
    /// Round-trip: any supported `(FieldType, FieldValue)` serialised to the LM
    /// wire format and arbitrarily chunked, parses back to the original value via
    /// `ChatStreamParser`.
    ///
    /// Exercises the full marker layer + per-FieldType dispatch +
    /// chunk-boundary handling end-to-end.
    #[test]
    fn chat_stream_parser_roundtrips_any_supported_value(
        (field_type, value) in arb_supported_type_and_value(),
        positions in arb_chunking_positions(),
    ) {
        let events = parse_value_through_chunks(&field_type, &value, &positions);

        // Top-level String trims at finish; match that for the expected comparison.
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
}

proptest! {
    /// Degrade-gracefully: any FieldType the streaming codec does NOT support,
    /// fed through the same wire format with a stub value, must emit a
    /// `StreamError` without panic and without hanging.
    #[test]
    fn unsupported_field_type_emits_stream_error_without_panic(
        field_type in arb_unsupported_field_type(),
        positions in arb_chunking_positions(),
    ) {
        let signature = single_output_signature("answer", field_type.clone());
        // For unsupported types we can't construct a matching FieldValue
        // cleanly, so synthesise the completion directly with a stub content
        // body. The space mid-marker is intentional — it ensures the
        // StringParser sees at least one byte to push to its sub-parser.
        let completion = "[[ ## answer ## ]]\n \n[[ ## completed ## ]]";
        let chunks = chunk_at_positions(completion, &positions);

        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());

        // The unsupported sub-parser is required to emit at least one
        // StreamError. Either the parser itself emits it (for top-level
        // FieldTypes routed to UnsupportedParser like Media), or the JSON layer
        // emits it (for FieldTypes routed through JsonFieldParser →
        // JsonUnsupportedParser like Int/Map/Nullable). Both are acceptable;
        // the property is "never panics, always reports".
        let has_stream_error = events.iter().any(|e| matches!(e, ParseEvent::StreamError { .. }));
        prop_assert!(
            has_stream_error,
            "expected StreamError for unsupported type {field_type:?}, events: {events:?}"
        );
    }
}

proptest! {
    /// **Parity property** — the load-bearing correctness invariant of the
    /// streaming codec. For any supported `(FieldType, FieldValue)`, any chunking
    /// of the wire-format completion, and any drift perturbation (fence / prose /
    /// wrapper / combo), the streaming parser produces the SAME `FieldValue` the
    /// buffered `ChatAdapter::parse` would produce on the same unchunked
    /// completion. This is what guarantees customers don't see different outcomes
    /// between the two parsing modes for the same LM response.
    #[test]
    fn buffered_and_streaming_parsers_agree(
        (field_type, value) in arb_supported_type_and_value(),
        positions in arb_chunking_positions(),
        perturb in arb_drift_perturbation(),
    ) {
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion_with_drift(
            "answer", &field_type, &value, perturb,
        );

        // Buffered parse — sync via a per-case tokio runtime.
        // ChatAdapter::parse is async but does no actual IO, so the runtime
        // build is the only real cost (a few µs).
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let buffered_result: Result<BTreeMap<String, typesayer_types::FieldValue>, _> =
            runtime.block_on(async {
                ChatAdapter::default().parse(&signature, &completion).await
            });
        let Ok(buffered_map) = buffered_result else {
            // If buffered fails, we don't require streaming to succeed either
            // — the wire format may be pathological under the chosen
            // perturbation. Skip the case.
            return Ok(());
        };
        let Some(buffered_value) = buffered_map.get("answer") else {
            return Ok(());
        };

        // Streaming parse — chunked.
        let chunks = chunk_at_positions(&completion, &positions);
        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());

        let streamed_value = match events.last() {
            Some(ParseEvent::StreamComplete { output: FieldValue::Object(map) }) => {
                map.get("answer").cloned()
            }
            _ => None,
        };

        // The exception is top-level String trimming: the streaming
        // StringParser trims at finish to match buffered.
        let expected = match (&field_type, &value) {
            (FieldType::String, FieldValue::Str(s)) => FieldValue::Str(s.trim().to_owned()),
            _ => buffered_value.clone(),
        };
        prop_assert_eq!(
            streamed_value.as_ref(),
            Some(&expected),
            "buffered/streaming divergence on {:?} with perturbation {:?}",
            field_type,
            perturb
        );

        // Even without a perturbation, the streaming parser should never emit a
        // StreamError on a value the buffered parser accepted.
        if perturb == DriftPerturbation::None {
            let has_stream_error =
                events.iter().any(|e| matches!(e, ParseEvent::StreamError { .. }));
            prop_assert!(
                !has_stream_error,
                "streaming emitted StreamError on a buffered-accepted value: {events:?}"
            );
        }
    }
}

proptest! {
    /// **Variant parity property** — companion to the main parity proptest,
    /// restricted to [`FieldType::OneOf`] (untagged) and [`FieldType::AnyOf`]
    /// with disjoint arm shapes (so the generated values unambiguously match
    /// exactly one arm). Asserts buffered and streaming agree on the final
    /// [`FieldValue::Variant`], preserving the same `arm_index` and inner-value
    /// structural equality.
    ///
    /// Drift perturbations (code fence, prose, fence+wrapper where safe) are
    /// applied to the wire format the same way the main parity test does, so the
    /// variant parsers must tolerate the same LLM-output drift as the existing
    /// FieldTypes.
    #[test]
    fn variant_buffered_and_streaming_agree(
        (field_type, value) in arb_variant_type_and_value(),
        positions in arb_chunking_positions(),
        perturb in arb_drift_perturbation(),
    ) {
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion_with_drift(
            "answer", &field_type, &value, perturb,
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let buffered_result: Result<BTreeMap<String, typesayer_types::FieldValue>, _> =
            runtime.block_on(async {
                ChatAdapter::default().parse(&signature, &completion).await
            });
        let Ok(buffered_map) = buffered_result else {
            return Ok(());
        };
        let Some(buffered_value) = buffered_map.get("answer") else {
            return Ok(());
        };

        let chunks = chunk_at_positions(&completion, &positions);
        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());

        let streamed_value = match events.last() {
            Some(ParseEvent::StreamComplete { output: FieldValue::Object(map) }) => {
                map.get("answer").cloned()
            }
            _ => None,
        };

        prop_assert_eq!(
            streamed_value.as_ref(),
            Some(buffered_value),
            "variant buffered/streaming divergence on {:?}",
            field_type
        );
    }
}

proptest! {
    /// Parity property restricted to tagged [`FieldType::OneOf`] (with
    /// discriminator inferred from `const`-restricted arm properties). Object
    /// arms inevitably overlap on shape (every arm is `{type: object,
    /// properties: ...}`), so untagged disambiguation depends on the
    /// discriminator alone. Asserts the streaming dispatcher routes via the
    /// discriminator the same way buffered does — including under drift
    /// perturbations that exercise the wire-format tolerance the LLM output may
    /// trigger.
    #[test]
    fn tagged_variant_buffered_and_streaming_agree(
        (field_type, value) in arb_tagged_oneof_type_and_value(),
        positions in arb_chunking_positions(),
        perturb in arb_drift_perturbation(),
    ) {
        let signature = single_output_signature("answer", field_type.clone());
        let completion = serialize_completion_with_drift(
            "answer", &field_type, &value, perturb,
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let buffered_result: Result<BTreeMap<String, typesayer_types::FieldValue>, _> =
            runtime.block_on(async {
                ChatAdapter::default().parse(&signature, &completion).await
            });
        let Ok(buffered_map) = buffered_result else {
            return Ok(());
        };
        let Some(buffered_value) = buffered_map.get("answer") else {
            return Ok(());
        };

        let chunks = chunk_at_positions(&completion, &positions);
        let mut parser = ChatStreamParser::new(&signature);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());

        let streamed_value = match events.last() {
            Some(ParseEvent::StreamComplete { output: FieldValue::Object(map) }) => {
                map.get("answer").cloned()
            }
            _ => None,
        };

        prop_assert_eq!(
            streamed_value.as_ref(),
            Some(buffered_value),
            "tagged-variant buffered/streaming divergence on {:?}",
            field_type
        );
    }
}
