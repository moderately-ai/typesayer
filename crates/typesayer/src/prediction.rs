// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prediction result with typed field access.

use std::collections::BTreeMap;

use modelplease::Usage;
use typesayer_types::{
    error::{PredictError, Result},
    field::FieldValue,
};

/// The result of a [`Predict::call()`](crate::Predict::call) invocation.
///
/// Wraps the parsed output fields with typed accessors and optional
/// usage metadata from the language model response.
#[derive(Debug, Clone)]
pub struct Prediction {
    fields: BTreeMap<String, FieldValue>,
    usage: Option<Usage>,
    model_id: Option<String>,
}

impl Prediction {
    /// Create a new prediction from parsed fields and optional usage.
    ///
    /// Prefer [`Prediction::with_model_id`] when the caller knows which
    /// model produced the output — surfacing the model identifier lets
    /// downstream consumers (e.g. audit records) persist it without
    /// having to thread it alongside the prediction.
    #[must_use]
    pub const fn new(fields: BTreeMap<String, FieldValue>, usage: Option<Usage>) -> Self {
        Self {
            fields,
            usage,
            model_id: None,
        }
    }

    /// Create a new prediction, recording the model identifier that
    /// produced it.
    #[must_use]
    pub const fn with_model_id(
        fields: BTreeMap<String, FieldValue>,
        usage: Option<Usage>,
        model_id: String,
    ) -> Self {
        Self {
            fields,
            usage,
            model_id: Some(model_id),
        }
    }

    /// The identifier of the model that produced this prediction, when
    /// recorded. `None` for predictions constructed by consumers that
    /// don't carry the id — notably test fixtures.
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Attach a model identifier to an existing prediction. Used by the
    /// application to record which model resolved the request after
    /// `Predict::call` returns and the prediction is otherwise complete.
    pub fn set_model_id(&mut self, model_id: String) {
        self.model_id = Some(model_id);
    }

    /// Get a typed value for the given field name.
    ///
    /// # Errors
    ///
    /// - [`PredictError::FieldNotInPrediction`] if the field is not present
    /// - [`PredictError::FieldTypeMismatch`] if the value cannot be converted to `T`
    pub fn get<T: TryFromFieldValue>(&self, field: &str) -> Result<T> {
        let value = self
            .fields
            .get(field)
            .ok_or_else(|| PredictError::FieldNotInPrediction {
                field: field.to_owned(),
            })?;
        T::try_from_field_value(value, field)
    }

    /// Get an optional typed value — returns `Ok(None)` for missing fields
    /// or [`FieldValue::Null`] values.
    ///
    /// Use this instead of [`get`](Self::get) for nullable or optional fields.
    ///
    /// # Errors
    ///
    /// Returns a [`PredictError::FieldTypeMismatch`] when the stored value
    /// exists and is not null but cannot be converted into `T`.
    pub fn get_optional<T: TryFromFieldValue>(&self, field: &str) -> Result<Option<T>> {
        match self.fields.get(field) {
            None | Some(FieldValue::Null) => Ok(None),
            Some(value) => T::try_from_field_value(value, field).map(Some),
        }
    }

    /// Get a reference to the raw [`FieldValue`] for the given field name.
    #[must_use]
    pub fn get_value(&self, field: &str) -> Option<&FieldValue> {
        self.fields.get(field)
    }

    /// Get a reference to all parsed fields.
    #[must_use]
    pub const fn fields(&self) -> &BTreeMap<String, FieldValue> {
        &self.fields
    }

    /// Consume the prediction and return the underlying field map.
    #[must_use]
    pub fn into_fields(self) -> BTreeMap<String, FieldValue> {
        self.fields
    }

    /// Token usage statistics from the language model response, if available.
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
}

/// Convert a [`FieldValue`] reference into a concrete Rust type.
///
/// Implemented for common types: `String`, `i64`, `f64`, `bool`,
/// `Vec<FieldValue>`, `BTreeMap<String, FieldValue>`, and `FieldValue` (identity).
pub trait TryFromFieldValue: Sized {
    /// Attempt to convert the given field value into this type.
    ///
    /// # Errors
    ///
    /// Returns a [`PredictError::FieldTypeMismatch`] when the stored value
    /// cannot be represented as `Self`.
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self>;
}

impl TryFromFieldValue for String {
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self> {
        match value {
            FieldValue::Str(s) => Ok(s.clone()),
            other => Err(PredictError::FieldTypeMismatch {
                field: field_name.to_string(),
                expected: "str".to_string(),
                actual: format!("{other:?}"),
            }),
        }
    }
}

impl TryFromFieldValue for i64 {
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self> {
        match value {
            FieldValue::Int(n) => Ok(*n),
            other => Err(PredictError::FieldTypeMismatch {
                field: field_name.to_string(),
                expected: "int".to_string(),
                actual: format!("{other:?}"),
            }),
        }
    }
}

impl TryFromFieldValue for f64 {
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self> {
        match value {
            FieldValue::Float(f) => Ok(*f),
            other => Err(PredictError::FieldTypeMismatch {
                field: field_name.to_string(),
                expected: "float".to_string(),
                actual: format!("{other:?}"),
            }),
        }
    }
}

impl TryFromFieldValue for bool {
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self> {
        match value {
            FieldValue::Bool(b) => Ok(*b),
            other => Err(PredictError::FieldTypeMismatch {
                field: field_name.to_string(),
                expected: "bool".to_string(),
                actual: format!("{other:?}"),
            }),
        }
    }
}

impl TryFromFieldValue for Vec<FieldValue> {
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self> {
        match value {
            FieldValue::List(list) => Ok(list.clone()),
            other => Err(PredictError::FieldTypeMismatch {
                field: field_name.to_string(),
                expected: "list".to_string(),
                actual: format!("{other:?}"),
            }),
        }
    }
}

impl TryFromFieldValue for BTreeMap<String, FieldValue> {
    fn try_from_field_value(value: &FieldValue, field_name: &str) -> Result<Self> {
        match value {
            FieldValue::Object(map) => Ok(map.clone()),
            other => Err(PredictError::FieldTypeMismatch {
                field: field_name.to_string(),
                expected: "object".to_string(),
                actual: format!("{other:?}"),
            }),
        }
    }
}

impl TryFromFieldValue for FieldValue {
    fn try_from_field_value(value: &FieldValue, _field_name: &str) -> Result<Self> {
        Ok(value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prediction() -> Prediction {
        Prediction::new(
            BTreeMap::from([
                ("answer".into(), FieldValue::Str("Paris".into())),
                ("count".into(), FieldValue::Int(42)),
                ("score".into(), FieldValue::Float(0.95)),
                ("valid".into(), FieldValue::Bool(true)),
                (
                    "items".into(),
                    FieldValue::List(vec![FieldValue::Str("a".into())]),
                ),
                (
                    "meta".into(),
                    FieldValue::Object(BTreeMap::from([(
                        "key".into(),
                        FieldValue::Str("val".into()),
                    )])),
                ),
            ]),
            Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            }),
        )
    }

    #[test]
    fn get_string() {
        let pred = sample_prediction();
        assert_eq!(pred.get::<String>("answer").unwrap(), "Paris");
    }

    #[test]
    fn get_int() {
        let pred = sample_prediction();
        assert_eq!(pred.get::<i64>("count").unwrap(), 42);
    }

    #[test]
    fn get_float() {
        let pred = sample_prediction();
        assert!((pred.get::<f64>("score").unwrap() - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn get_bool() {
        let pred = sample_prediction();
        assert!(pred.get::<bool>("valid").unwrap());
    }

    #[test]
    fn get_list() {
        let pred = sample_prediction();
        let items = pred.get::<Vec<FieldValue>>("items").unwrap();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn get_object() {
        let pred = sample_prediction();
        let meta = pred.get::<BTreeMap<String, FieldValue>>("meta").unwrap();
        assert_eq!(meta["key"], FieldValue::Str("val".into()));
    }

    #[test]
    fn get_field_value_identity() {
        let pred = sample_prediction();
        let val = pred.get::<FieldValue>("answer").unwrap();
        assert_eq!(val, FieldValue::Str("Paris".into()));
    }

    #[test]
    fn get_type_mismatch() {
        let pred = sample_prediction();
        let err = pred.get::<i64>("answer").unwrap_err();
        assert!(matches!(err, PredictError::FieldTypeMismatch { .. }));
    }

    #[test]
    fn get_missing_field() {
        let pred = sample_prediction();
        let err = pred.get::<String>("nonexistent").unwrap_err();
        assert!(matches!(err, PredictError::FieldNotInPrediction { .. }));
    }

    #[test]
    fn usage_present() {
        let pred = sample_prediction();
        let usage = pred.usage().unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn usage_absent() {
        let pred = Prediction::new(BTreeMap::new(), None);
        assert!(pred.usage().is_none());
    }

    #[test]
    fn into_fields_consumes() {
        let pred = sample_prediction();
        let fields = pred.into_fields();
        assert_eq!(fields["answer"], FieldValue::Str("Paris".into()));
    }

    #[test]
    fn get_optional_present() {
        let pred = sample_prediction();
        let val = pred.get_optional::<String>("answer").unwrap();
        assert_eq!(val, Some("Paris".to_owned()));
    }

    #[test]
    fn get_optional_null() {
        let pred = Prediction::new(BTreeMap::from([("name".into(), FieldValue::Null)]), None);
        let val = pred.get_optional::<String>("name").unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn get_optional_missing() {
        let pred = sample_prediction();
        let val = pred.get_optional::<String>("nonexistent").unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn get_optional_type_mismatch() {
        let pred = sample_prediction();
        let err = pred.get_optional::<i64>("answer").unwrap_err();
        assert!(matches!(err, PredictError::FieldTypeMismatch { .. }));
    }
}
