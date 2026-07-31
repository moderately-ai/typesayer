// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Labeled data points with declared input/output separation.
//!
//! [`Example`] stores all fields in a flat map and declares which keys are inputs.
//! This matches `DSPy`'s `Example` representation and is the natural format for
//! datasets and optimizer training data.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize, ser::SerializeMap};
use typesayer_types::field::FieldValue;

use crate::adapter::Demo;

/// A labeled data point with a declared set of input keys.
///
/// Unlike [`Demo`], which splits inputs and outputs at construction time,
/// `Example` stores all fields in a flat map and uses `input_keys` to
/// separate them. This is the natural type for datasets and DSPy-compatible
/// serialization.
///
/// # Conversions
///
/// ```rust
/// use std::collections::{BTreeMap, HashSet};
///
/// use typesayer::{Demo, Example, FieldValue};
///
/// let demo = Demo {
///     inputs: BTreeMap::from([("question".into(), FieldValue::Str("2+2?".into()))]),
///     outputs: BTreeMap::from([("answer".into(), FieldValue::Str("4".into()))]),
/// };
///
/// let example = Example::from(demo.clone());
/// assert_eq!(example.inputs().len(), 1);
/// assert_eq!(example.labels().len(), 1);
///
/// let roundtripped: Demo = example.into();
/// assert_eq!(roundtripped.inputs["question"], demo.inputs["question"]);
/// ```
#[derive(Debug, Clone)]
pub struct Example {
    fields: BTreeMap<String, FieldValue>,
    input_keys: HashSet<String>,
}

impl Example {
    /// Create a new example from a flat field map and declared input keys.
    #[must_use]
    pub const fn new(fields: BTreeMap<String, FieldValue>, input_keys: HashSet<String>) -> Self {
        Self { fields, input_keys }
    }

    /// Get a field value by name.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&FieldValue> {
        self.fields.get(key)
    }

    /// All input fields (those whose names are in `input_keys`).
    #[must_use]
    pub fn inputs(&self) -> BTreeMap<&str, &FieldValue> {
        self.fields
            .iter()
            .filter(|(k, _)| self.input_keys.contains(k.as_str()))
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// All output/label fields (those whose names are NOT in `input_keys`).
    #[must_use]
    pub fn labels(&self) -> BTreeMap<&str, &FieldValue> {
        self.fields
            .iter()
            .filter(|(k, _)| !self.input_keys.contains(k.as_str()))
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// The declared input key names.
    #[must_use]
    pub const fn input_keys(&self) -> &HashSet<String> {
        &self.input_keys
    }

    /// All fields in this example.
    #[must_use]
    pub const fn fields(&self) -> &BTreeMap<String, FieldValue> {
        &self.fields
    }

    /// Consume this example and return the underlying field map.
    #[must_use]
    pub fn into_fields(self) -> BTreeMap<String, FieldValue> {
        self.fields
    }
}

/// Serializes as a flat JSON dict — only the fields, no `input_keys`.
/// This matches `DSPy`'s on-disk demo format.
impl Serialize for Example {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for (k, v) in &self.fields {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// Deserializes from a flat JSON dict. `input_keys` will be empty — the caller
/// must supply them separately or infer from a signature.
impl<'de> Deserialize<'de> for Example {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let fields = BTreeMap::<String, FieldValue>::deserialize(deserializer)?;
        Ok(Self {
            fields,
            input_keys: HashSet::new(),
        })
    }
}

impl From<Demo> for Example {
    fn from(demo: Demo) -> Self {
        let input_keys: HashSet<String> = demo.inputs.keys().cloned().collect();
        let mut fields = demo.inputs;
        fields.extend(demo.outputs);
        Self { fields, input_keys }
    }
}

impl From<Example> for Demo {
    fn from(ex: Example) -> Self {
        let mut inputs = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        for (key, value) in ex.fields {
            if ex.input_keys.contains(&key) {
                inputs.insert(key, value);
            } else {
                outputs.insert(key, value);
            }
        }
        Self { inputs, outputs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_example() -> Example {
        Example::new(
            BTreeMap::from([
                ("question".into(), FieldValue::Str("2+2?".into())),
                ("answer".into(), FieldValue::Str("4".into())),
            ]),
            HashSet::from(["question".into()]),
        )
    }

    #[test]
    fn inputs_returns_input_keyed_fields() {
        let ex = sample_example();
        let inputs = ex.inputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs["question"], &FieldValue::Str("2+2?".into()));
    }

    #[test]
    fn labels_returns_complement() {
        let ex = sample_example();
        let labels = ex.labels();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels["answer"], &FieldValue::Str("4".into()));
    }

    #[test]
    fn from_demo_preserves_fields_and_keys() {
        let demo = Demo {
            inputs: BTreeMap::from([("q".into(), FieldValue::Str("hi".into()))]),
            outputs: BTreeMap::from([("a".into(), FieldValue::Str("hello".into()))]),
        };
        let ex = Example::from(demo);
        assert_eq!(ex.fields().len(), 2);
        assert!(ex.input_keys().contains("q"));
        assert!(!ex.input_keys().contains("a"));
    }

    #[test]
    fn into_demo_splits_correctly() {
        let ex = sample_example();
        let demo: Demo = ex.into();
        assert_eq!(demo.inputs.len(), 1);
        assert_eq!(demo.outputs.len(), 1);
        assert!(demo.inputs.contains_key("question"));
        assert!(demo.outputs.contains_key("answer"));
    }

    #[test]
    fn serializes_as_flat_dict() {
        let ex = sample_example();
        let json = serde_json::to_value(&ex).unwrap();
        assert!(json.is_object());
        let obj = json.as_object().unwrap();
        // Flat dict — no "input_keys", no "fields" wrapper
        assert!(obj.contains_key("question"));
        assert!(obj.contains_key("answer"));
        assert!(!obj.contains_key("input_keys"));
    }

    #[test]
    fn deserializes_from_flat_dict() {
        let json = serde_json::json!({"question": "2+2?", "answer": "4"});
        let ex: Example = serde_json::from_value(json).unwrap();
        assert_eq!(ex.fields().len(), 2);
        assert!(ex.input_keys().is_empty()); // caller must supply
    }

    #[test]
    fn round_trip_via_demo() {
        let original = sample_example();
        let demo: Demo = original.clone().into();
        let restored = Example::from(demo);
        assert_eq!(original.fields(), restored.fields());
        assert_eq!(original.input_keys(), restored.input_keys());
    }
}
