// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execution trace capture for per-predictor analysis.
//!
//! An [`ExecutionTrace`] records every [`Predict`](crate::Predict) invocation
//! during a module's `forward()` call. Traces feed optimizer reflection loops
//! (GEPA) and per-predictor demo attribution (`MIPROv2`).

use std::collections::BTreeMap;

use typesayer_types::field::FieldValue;

use crate::prediction::Prediction;

/// A single predictor invocation record.
///
/// Captures the predictor's name, the inputs it received, and the prediction
/// it produced. Collected by [`Predict::call_traced()`](crate::Predict::call_traced).
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// The dotted-path name of the predictor (e.g. `"classify"`, `"chain.qa"`).
    pub predictor_name: String,
    /// The input field values passed to this predictor.
    pub inputs: BTreeMap<String, FieldValue>,
    /// The prediction produced by this predictor.
    pub prediction: Prediction,
}

/// An ordered collection of trace entries from a module's forward pass.
///
/// Records every predictor invocation in execution order. For multi-step
/// modules, this captures the full pipeline.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    entries: Vec<TraceEntry>,
}

impl ExecutionTrace {
    /// Create an empty execution trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a trace entry.
    pub fn push(&mut self, entry: TraceEntry) {
        self.entries.push(entry);
    }

    /// All trace entries in execution order.
    #[must_use]
    pub fn entries(&self) -> &[TraceEntry] {
        &self.entries
    }

    /// Consume the trace and return the underlying entries.
    #[must_use]
    pub fn into_entries(self) -> Vec<TraceEntry> {
        self.entries
    }

    /// Number of recorded predictor invocations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the trace is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append all entries from another trace.
    pub fn extend(&mut self, other: Self) {
        self.entries.extend(other.entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(name: &str) -> TraceEntry {
        TraceEntry {
            predictor_name: name.to_owned(),
            inputs: BTreeMap::from([("q".into(), FieldValue::Str("test".into()))]),
            prediction: Prediction::new(
                BTreeMap::from([("a".into(), FieldValue::Str("answer".into()))]),
                None,
            ),
        }
    }

    #[test]
    fn empty_trace() {
        let trace = ExecutionTrace::new();
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
        assert!(trace.entries().is_empty());
    }

    #[test]
    fn push_and_entries() {
        let mut trace = ExecutionTrace::new();
        trace.push(sample_entry("first"));
        trace.push(sample_entry("second"));

        assert_eq!(trace.len(), 2);
        assert!(!trace.is_empty());
        assert_eq!(trace.entries()[0].predictor_name, "first");
        assert_eq!(trace.entries()[1].predictor_name, "second");
    }

    #[test]
    fn into_entries_consumes() {
        let mut trace = ExecutionTrace::new();
        trace.push(sample_entry("qa"));

        let entries = trace.into_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].predictor_name, "qa");
    }

    #[test]
    fn extend_merges_traces() {
        let mut trace1 = ExecutionTrace::new();
        trace1.push(sample_entry("first"));

        let mut trace2 = ExecutionTrace::new();
        trace2.push(sample_entry("second"));
        trace2.push(sample_entry("third"));

        trace1.extend(trace2);
        assert_eq!(trace1.len(), 3);
        assert_eq!(trace1.entries()[0].predictor_name, "first");
        assert_eq!(trace1.entries()[1].predictor_name, "second");
        assert_eq!(trace1.entries()[2].predictor_name, "third");
    }

    #[test]
    fn trace_entry_fields_accessible() {
        let entry = sample_entry("qa");
        assert_eq!(entry.predictor_name, "qa");
        assert_eq!(entry.inputs["q"], FieldValue::Str("test".into()));
        assert_eq!(entry.prediction.get::<String>("a").unwrap(), "answer");
    }
}
