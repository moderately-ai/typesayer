// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Field value serialization and deserialization traits.
//!
//! [`FieldSerializer`] and [`FieldDeserializer`] are independent concerns from
//! message structure ([`Adapter`](crate::adapter::Adapter)). This separation
//! allows mixing formats — e.g. serialize inputs as one format but deserialize
//! outputs as another.

use std::collections::BTreeMap;

use typesayer_types::{
    error::{PredictError, Result},
    field::{FieldType, FieldValue},
};

/// Serializes a [`FieldValue`] into text for inclusion in a prompt.
///
/// This trait is intentionally synchronous — value serialization is CPU-bound
/// string/JSON work with no I/O.
pub trait FieldSerializer: Send + Sync {
    /// Convert a typed value into its text representation.
    ///
    /// # Errors
    ///
    /// Returns a [`PredictError`] when the value cannot be serialized under
    /// the declared `field_type` (implementations define their own criteria).
    fn serialize(&self, value: &FieldValue, field_type: &FieldType) -> Result<String>;
}

/// Deserializes raw text from a completion into a [`FieldValue`].
///
/// This trait is intentionally synchronous — value deserialization is CPU-bound
/// string/JSON work with no I/O.
pub trait FieldDeserializer: Send + Sync {
    /// Parse raw text into a typed value according to the declared field type.
    ///
    /// # Errors
    ///
    /// Returns a [`PredictError`] when `raw` cannot be parsed as `field_type`
    /// (malformed JSON, type mismatch, out-of-range value, etc.).
    fn deserialize(
        &self,
        raw: &str,
        field_name: &str,
        field_type: &FieldType,
    ) -> Result<FieldValue>;
}

/// JSON-based field serializer.
///
/// Primitives are serialized as their natural text representation (no wrapping
/// quotes for strings). Complex types (List, Object, Map) are serialized as JSON.
pub struct JsonFieldSerializer;

impl FieldSerializer for JsonFieldSerializer {
    fn serialize(&self, value: &FieldValue, _field_type: &FieldType) -> Result<String> {
        Ok(serialize_value(value))
    }
}

/// JSON-based field deserializer.
///
/// Primitives are parsed from their text representation. Complex types (List,
/// Object, Map) are parsed as JSON.
pub struct JsonFieldDeserializer;

impl FieldDeserializer for JsonFieldDeserializer {
    fn deserialize(
        &self,
        raw: &str,
        field_name: &str,
        field_type: &FieldType,
    ) -> Result<FieldValue> {
        deserialize_value(raw, field_name, field_type)
    }
}

/// Serialize a [`FieldValue`] to its text representation.
///
/// `FieldValue::Media` returns a placeholder string — media values are
/// emitted as non-text `ContentPart`s by the adapter and never reach a
/// text serializer. Returning a marker rather than panicking keeps any
/// accidental call observable in logs without breaking the call site.
pub fn serialize_value(value: &FieldValue) -> String {
    match value {
        FieldValue::Str(s) => s.clone(),
        FieldValue::Int(n) => n.to_string(),
        FieldValue::Float(f) => f.to_string(),
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Null => "null".to_owned(),
        // Complex types get JSON serialization
        FieldValue::List(_) | FieldValue::Object(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
        }
        FieldValue::Media(_) => "<media:out-of-band>".to_owned(),
        // The inner value already carries any discriminator key (the
        // OneOf/AnyOf parser injects it into the matched arm's Object
        // before constructing the FieldValue::Variant), so emitting the
        // inner value's JSON is lossless on the wire.
        FieldValue::Variant { value, .. } => serialize_value(value),
    }
}

/// Convert a `serde_json::Value` to a raw string suitable for re-deserialization.
///
/// For JSON strings, extracts the inner string (no wrapping quotes).
/// For other types, uses `serde_json::to_string` (which produces valid JSON).
fn json_value_to_raw_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// Strip a surrounding Markdown code fence (```` ``` ```` or ```` ```json … ``` ````)
/// and return the inner content. LLMs frequently fence JSON output even when
/// told not to; the fence is never part of the value. Returns the trimmed
/// input unchanged when there is no fence.
fn strip_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // Drop an optional `json` language tag right after the opening fence, then
    // the remainder of that opening line.
    let body = after_open.strip_prefix("json").unwrap_or(after_open);
    let body = body.split_once('\n').map_or(body, |(_, rest)| rest);
    body.trim()
        .strip_suffix("```")
        .map_or_else(|| body.trim(), str::trim)
}

/// Return the balanced `{…}`/`[…]` span starting at the first opening bracket,
/// ignoring brackets that appear inside JSON string literals. Used to discard
/// prose a model appends after (or before) a complete JSON value. Returns
/// `None` when there is no opening bracket or the span never closes (truncated
/// output) — callers then fall through to a serde error that names the cause.
///
/// Depth counts any bracket type rather than matching `{`↔`}` strictly; serde
/// performs the real structural validation afterward, so a count-balanced but
/// malformed span (e.g. `{1,2]`) simply surfaces as a serde error.
fn extract_balanced_json(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, &b) in bytes[start..].iter().enumerate() {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    // `start` and `start + offset` index ASCII brackets, so the
                    // slice lands on UTF-8 char boundaries even with multibyte
                    // content between them.
                    return Some(&s[start..=start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Normalize a complex-type completion before strict JSON parsing: strip a code
/// fence, extract the balanced JSON span (discarding any prose the model wrote
/// before or after the value), then unwrap a single-key object whose key is the
/// field name (e.g. `{"queries": [...]}` for the `queries` field). Models
/// commonly wrap the answer in an object keyed by the field; unwrapping lets the
/// inner value parse against its declared type. Only a key matching `field_name`
/// is unwrapped — a different single key is left intact so genuine data is never
/// silently misread.
fn prepare_complex_json(raw: &str, field_name: &str) -> String {
    let stripped = strip_code_fence(raw);
    let candidate = extract_balanced_json(stripped).unwrap_or(stripped);
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(candidate)
        && map.len() == 1
        && let Some(inner) = map.get(field_name)
    {
        return inner.to_string();
    }
    candidate.to_owned()
}

/// Deserialize raw text into a [`FieldValue`] according to the declared type.
fn deserialize_value(raw: &str, field_name: &str, field_type: &FieldType) -> Result<FieldValue> {
    let trimmed = raw.trim();

    let mismatch = |expected: &str| PredictError::FieldTypeMismatch {
        field: field_name.to_string(),
        expected: expected.to_string(),
        actual: trimmed.to_string(),
    };

    // Complex types parse via serde; keep the concrete serde cause so operators
    // can tell trailing prose from truncation from a stray control char at first
    // sight, instead of auditing the raw value by hand.
    let mismatch_parse =
        |expected: &str, err: &serde_json::Error| PredictError::FieldTypeMismatch {
            field: field_name.to_string(),
            expected: expected.to_string(),
            actual: format!("{trimmed} (serde error: {err})"),
        };

    match field_type {
        FieldType::Media { .. } => Err(PredictError::FieldTypeMismatch {
            field: field_name.to_string(),
            expected: "media (out-of-band content part)".to_string(),
            actual: trimmed.to_string(),
        }),
        FieldType::String => Ok(FieldValue::Str(trimmed.to_owned())),

        FieldType::Int => trimmed
            .parse::<i64>()
            .map(FieldValue::Int)
            .map_err(|_| mismatch("int")),

        FieldType::Float => trimmed
            .parse::<f64>()
            .map(FieldValue::Float)
            .map_err(|_| mismatch("float")),

        FieldType::Bool => match trimmed.to_lowercase().as_str() {
            "true" | "yes" | "1" => Ok(FieldValue::Bool(true)),
            "false" | "no" | "0" => Ok(FieldValue::Bool(false)),
            _ => Err(mismatch("bool")),
        },

        FieldType::List(inner_type) => {
            let prepared = prepare_complex_json(trimmed, field_name);
            let parsed: Vec<serde_json::Value> =
                serde_json::from_str(&prepared).map_err(|e| mismatch_parse("list", &e))?;
            let values = parsed
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    let element_name = format!("{field_name}[{i}]");
                    let raw_element = json_value_to_raw_string(&v);
                    deserialize_value(&raw_element, &element_name, inner_type)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(FieldValue::List(values))
        }

        FieldType::Object(object_fields) => {
            let prepared = prepare_complex_json(trimmed, field_name);
            let parsed: BTreeMap<String, serde_json::Value> =
                serde_json::from_str(&prepared).map_err(|e| mismatch_parse("object", &e))?;
            let mut result = BTreeMap::new();
            for obj_field in object_fields {
                if let Some(v) = parsed.get(&obj_field.name) {
                    let raw_field = json_value_to_raw_string(v);
                    let nested_name = format!("{field_name}.{}", obj_field.name);
                    let value = deserialize_value(&raw_field, &nested_name, &obj_field.field_type)?;
                    result.insert(obj_field.name.clone(), value);
                } else if matches!(obj_field.field_type, FieldType::Nullable(_)) {
                    // Missing nullable field is allowed; default to Null.
                    // Matches the chat adapter's top-level "skip-nullable"
                    // semantics for declared output fields.
                    result.insert(obj_field.name.clone(), FieldValue::Null);
                } else {
                    // Missing required field — surface a type-mismatch so
                    // the caller can distinguish "wrong shape" from "right
                    // shape, wrong values". Critical for untagged OneOf:
                    // a subset arm matching its required fields and a
                    // superset arm with missing required fields must
                    // disambiguate on presence, not silently accept the
                    // subset for both.
                    return Err(PredictError::FieldTypeMismatch {
                        field: format!("{field_name}.{}", obj_field.name),
                        expected: obj_field.field_type.type_label(),
                        actual: "(missing required field)".into(),
                    });
                }
            }
            Ok(FieldValue::Object(result))
        }

        FieldType::Map(value_type) => {
            let prepared = prepare_complex_json(trimmed, field_name);
            let parsed: BTreeMap<String, serde_json::Value> =
                serde_json::from_str(&prepared).map_err(|e| mismatch_parse("map", &e))?;
            let mut result = BTreeMap::new();
            for (key, v) in parsed {
                let raw_val = json_value_to_raw_string(&v);
                let nested_name = format!("{field_name}.{key}");
                let value = deserialize_value(&raw_val, &nested_name, value_type)?;
                result.insert(key, value);
            }
            Ok(FieldValue::Object(result))
        }

        FieldType::Enum(variants) => {
            if variants.iter().any(|v| v == trimmed) {
                Ok(FieldValue::Str(trimmed.to_owned()))
            } else {
                Err(PredictError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: format!("enum[{}]", variants.join(", ")),
                    actual: trimmed.to_string(),
                })
            }
        }

        FieldType::Nullable(inner_type) => {
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
                Ok(FieldValue::Null)
            } else {
                deserialize_value(raw, field_name, inner_type)
            }
        }

        FieldType::OneOf {
            arms,
            discriminator,
        } => discriminator.as_ref().map_or_else(
            || deserialize_oneof_untagged(trimmed, field_name, arms),
            |disc| deserialize_oneof_tagged(trimmed, field_name, arms, disc),
        ),

        FieldType::AnyOf { arms } => deserialize_anyof(trimmed, field_name, arms),
    }
}

/// Tagged-OneOf parse: look up the discriminator value, dispatch to the
/// matching arm. Per the field-type validation the arm's structure is an
/// Object containing the discriminator as a const-restricted field, but the
/// inner deserializer treats the arm's `field_type` generically — it could
/// in principle be anything an Object can hold.
fn deserialize_oneof_tagged(
    raw: &str,
    field_name: &str,
    arms: &[typesayer_types::field::VariantArm],
    discriminator: &typesayer_types::field::OneOfDiscriminator,
) -> Result<FieldValue> {
    let prepared = prepare_complex_json(raw, field_name);
    let map: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&prepared).map_err(|_| PredictError::OneOfTagMissing {
            field: field_name.to_string(),
            discriminator: discriminator.property.clone(),
        })?;

    let tag_value =
        map.get(&discriminator.property)
            .ok_or_else(|| PredictError::OneOfTagMissing {
                field: field_name.to_string(),
                discriminator: discriminator.property.clone(),
            })?;

    let tag_str = tag_value
        .as_str()
        .ok_or_else(|| PredictError::FieldTypeMismatch {
            field: format!("{field_name}.{}", discriminator.property),
            expected: "string discriminator value".to_string(),
            actual: tag_value.to_string(),
        })?;

    let arm_index = discriminator
        .tags
        .iter()
        .position(|t| t == tag_str)
        .ok_or_else(|| PredictError::OneOfTagInvalid {
            field: field_name.to_string(),
            tag: tag_str.to_string(),
            valid_tags: discriminator.tags.clone(),
        })?;

    let arm = &arms[arm_index];
    let arm_raw = serde_json::to_string(&map).unwrap_or_else(|_| prepared.clone());
    let arm_path = format!("{field_name}#{tag_str}");
    let value = deserialize_value(&arm_raw, &arm_path, &arm.field_type)?;
    Ok(FieldValue::Variant {
        arm_index,
        value: Box::new(value),
    })
}

/// Untagged-OneOf parse: try every arm, demand exactly one success. Zero
/// matches surfaces every per-arm error so the operator can see why each
/// arm rejected the value. Multiple matches surfaces all matching indices
/// so the schema author can disambiguate.
fn deserialize_oneof_untagged(
    raw: &str,
    field_name: &str,
    arms: &[typesayer_types::field::VariantArm],
) -> Result<FieldValue> {
    let mut matches: Vec<(usize, FieldValue)> = Vec::new();
    let mut errors: Vec<(usize, Box<PredictError>)> = Vec::new();
    for (i, arm) in arms.iter().enumerate() {
        let arm_path = format!("{field_name}#arm{i}");
        match deserialize_value(raw, &arm_path, &arm.field_type) {
            Ok(v) => matches.push((i, v)),
            Err(e) => errors.push((i, Box::new(e))),
        }
    }
    // Convert match-count to a result without resorting to .expect():
    // a 1-element Vec collapses cleanly via destructuring; the 0 / >1
    // cases construct their respective error variants directly.
    if matches.len() > 1 {
        return Err(PredictError::OneOfAmbiguous {
            field: field_name.to_string(),
            matching_arms: matches.into_iter().map(|(i, _)| i).collect(),
        });
    }
    if let Some((arm_index, value)) = matches.into_iter().next() {
        return Ok(FieldValue::Variant {
            arm_index,
            value: Box::new(value),
        });
    }
    Err(PredictError::OneOfNoArmMatched {
        field: field_name.to_string(),
        arm_errors: errors,
    })
}

/// AnyOf parse: try arms in declared order; first match wins. All-fail
/// surfaces every per-arm error so the operator can see the rejection
/// reasons.
fn deserialize_anyof(
    raw: &str,
    field_name: &str,
    arms: &[typesayer_types::field::VariantArm],
) -> Result<FieldValue> {
    let mut errors: Vec<(usize, Box<PredictError>)> = Vec::new();
    for (i, arm) in arms.iter().enumerate() {
        let arm_path = format!("{field_name}#arm{i}");
        match deserialize_value(raw, &arm_path, &arm.field_type) {
            Ok(value) => {
                return Ok(FieldValue::Variant {
                    arm_index: i,
                    value: Box::new(value),
                });
            }
            Err(e) => errors.push((i, Box::new(e))),
        }
    }
    Err(PredictError::AnyOfNoArmMatched {
        field: field_name.to_string(),
        arm_errors: errors,
    })
}

#[cfg(test)]
mod tests {
    use typesayer_types::field::ObjectField;

    use super::*;

    fn json_serialize(value: &FieldValue, field_type: &FieldType) -> String {
        JsonFieldSerializer.serialize(value, field_type).unwrap()
    }

    fn json_deserialize(raw: &str, field_name: &str, field_type: &FieldType) -> Result<FieldValue> {
        JsonFieldDeserializer.deserialize(raw, field_name, field_type)
    }

    #[test]
    fn serialize_str() {
        let result = json_serialize(&FieldValue::Str("hello".into()), &FieldType::String);
        assert_eq!(result, "hello");
    }

    #[test]
    fn serialize_int() {
        let result = json_serialize(&FieldValue::Int(42), &FieldType::Int);
        assert_eq!(result, "42");
    }

    #[test]
    fn serialize_float() {
        let result = json_serialize(&FieldValue::Float(2.5), &FieldType::Float);
        assert_eq!(result, "2.5");
    }

    #[test]
    fn serialize_bool() {
        let result = json_serialize(&FieldValue::Bool(true), &FieldType::Bool);
        assert_eq!(result, "true");
    }

    #[test]
    fn serialize_null() {
        let result = json_serialize(
            &FieldValue::Null,
            &FieldType::Nullable(Box::new(FieldType::String)),
        );
        assert_eq!(result, "null");
    }

    #[test]
    fn serialize_list() {
        let value = FieldValue::List(vec![FieldValue::Int(1), FieldValue::Int(2)]);
        let result = json_serialize(&value, &FieldType::List(Box::new(FieldType::Int)));
        assert_eq!(result, "[1,2]");
    }

    #[test]
    fn serialize_object() {
        let value = FieldValue::Object(BTreeMap::from([(
            "name".into(),
            FieldValue::Str("Alice".into()),
        )]));
        let result = json_serialize(
            &value,
            &FieldType::Object(vec![ObjectField {
                name: "name".into(),
                description: String::new(),
                field_type: FieldType::String,
            }]),
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["name"], "Alice");
    }

    #[test]
    fn deserialize_str() {
        let result = json_deserialize("hello", "field", &FieldType::String).unwrap();
        assert_eq!(result, FieldValue::Str("hello".into()));
    }

    #[test]
    fn deserialize_int() {
        let result = json_deserialize("42", "field", &FieldType::Int).unwrap();
        assert_eq!(result, FieldValue::Int(42));
    }

    #[test]
    fn deserialize_int_invalid() {
        let result = json_deserialize("abc", "age", &FieldType::Int);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("age"));
        assert!(err.contains("int"));
    }

    #[test]
    fn deserialize_float() {
        let result = json_deserialize("2.5", "field", &FieldType::Float).unwrap();
        assert_eq!(result, FieldValue::Float(2.5));
    }

    #[test]
    fn deserialize_bool_variants() {
        for (input, expected) in [
            ("true", true),
            ("True", true),
            ("yes", true),
            ("1", true),
            ("false", false),
            ("False", false),
            ("no", false),
            ("0", false),
        ] {
            let result = json_deserialize(input, "field", &FieldType::Bool).unwrap();
            assert_eq!(
                result,
                FieldValue::Bool(expected),
                "failed for input: {input}"
            );
        }
    }

    #[test]
    fn deserialize_bool_invalid() {
        let result = json_deserialize("maybe", "field", &FieldType::Bool);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_enum_valid() {
        let ft = FieldType::Enum(vec!["positive".into(), "negative".into(), "neutral".into()]);
        let result = json_deserialize("positive", "sentiment", &ft).unwrap();
        assert_eq!(result, FieldValue::Str("positive".into()));
    }

    #[test]
    fn deserialize_enum_invalid() {
        let ft = FieldType::Enum(vec!["positive".into(), "negative".into()]);
        let result = json_deserialize("maybe", "sentiment", &ft);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("enum"));
    }

    #[test]
    fn deserialize_nullable_null() {
        let ft = FieldType::Nullable(Box::new(FieldType::String));
        assert_eq!(
            json_deserialize("null", "f", &ft).unwrap(),
            FieldValue::Null
        );
        assert_eq!(json_deserialize("", "f", &ft).unwrap(), FieldValue::Null);
        assert_eq!(
            json_deserialize("  NULL  ", "f", &ft).unwrap(),
            FieldValue::Null
        );
    }

    #[test]
    fn deserialize_nullable_present() {
        let ft = FieldType::Nullable(Box::new(FieldType::Int));
        let result = json_deserialize("42", "f", &ft).unwrap();
        assert_eq!(result, FieldValue::Int(42));
    }

    #[test]
    fn deserialize_list() {
        let ft = FieldType::List(Box::new(FieldType::Int));
        let result = json_deserialize("[1, 2, 3]", "nums", &ft).unwrap();
        assert_eq!(
            result,
            FieldValue::List(vec![
                FieldValue::Int(1),
                FieldValue::Int(2),
                FieldValue::Int(3)
            ])
        );
    }

    #[test]
    fn deserialize_object() {
        let ft = FieldType::Object(vec![
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
        ]);
        let result = json_deserialize(r#"{"name": "Alice", "age": 30}"#, "person", &ft).unwrap();
        match result {
            FieldValue::Object(map) => {
                assert_eq!(map["name"], FieldValue::Str("Alice".into()));
                assert_eq!(map["age"], FieldValue::Int(30));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_map() {
        let ft = FieldType::Map(Box::new(FieldType::Int));
        let result = json_deserialize(r#"{"a": 1, "b": 2}"#, "scores", &ft).unwrap();
        match result {
            FieldValue::Object(map) => {
                assert_eq!(map["a"], FieldValue::Int(1));
                assert_eq!(map["b"], FieldValue::Int(2));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_primitive() {
        let value = FieldValue::Int(42);
        let ft = FieldType::Int;
        let serialized = json_serialize(&value, &ft);
        let deserialized = json_deserialize(&serialized, "f", &ft).unwrap();
        assert_eq!(value, deserialized);
    }

    #[test]
    fn round_trip_list() {
        let value = FieldValue::List(vec![
            FieldValue::Str("hello".into()),
            FieldValue::Str("world".into()),
        ]);
        let ft = FieldType::List(Box::new(FieldType::String));
        let serialized = json_serialize(&value, &ft);
        let deserialized = json_deserialize(&serialized, "f", &ft).unwrap();
        assert_eq!(value, deserialized);
    }

    #[test]
    fn deserialize_trims_whitespace() {
        let result = json_deserialize("  42  ", "f", &FieldType::Int).unwrap();
        assert_eq!(result, FieldValue::Int(42));
    }

    // --- Lenient deserialization of common LLM drift (code fences + single-key
    // wrappers). These encode the documented production failures; before the
    // leniency fix they error with "expected list/object".

    fn str_list(items: &[&str]) -> FieldValue {
        FieldValue::List(items.iter().map(|s| FieldValue::Str((*s).into())).collect())
    }

    #[test]
    fn deserialize_list_from_json_code_fence() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize("```json\n[\"a\", \"b\"]\n```", "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_list_from_bare_code_fence() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize("```\n[\"a\", \"b\"]\n```", "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_list_from_single_key_wrapper() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"{"queries": ["a", "b"]}"#, "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_list_bare_array_still_works() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"["a", "b"]"#, "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_list_wrapper_wrong_key_not_unwrapped() {
        // A single-key object whose key != field name must NOT be unwrapped —
        // that would misread genuine data. It stays a mismatch for a list field.
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"{"other": ["a", "b"]}"#, "queries", &ft);
        assert!(result.is_err(), "wrong-key wrapper must not be unwrapped");
    }

    #[test]
    fn deserialize_object_from_code_fence() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "name".into(),
            description: String::new(),
            field_type: FieldType::String,
        }]);
        let result =
            json_deserialize("```json\n{\"name\": \"Alice\"}\n```", "person", &ft).unwrap();
        match result {
            FieldValue::Object(map) => assert_eq!(map["name"], FieldValue::Str("Alice".into())),
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_object_from_single_key_wrapper() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "name".into(),
            description: String::new(),
            field_type: FieldType::String,
        }]);
        let result = json_deserialize(r#"{"person": {"name": "Alice"}}"#, "person", &ft).unwrap();
        match result {
            FieldValue::Object(map) => assert_eq!(map["name"], FieldValue::Str("Alice".into())),
            other => panic!("expected Object, got {other:?}"),
        }
    }

    // --- Balanced-prefix extraction: a complete JSON value followed by prose
    // (common LLM drift) parses by discarding the trailing tail. Truncated
    // output still errors, now with the serde cause attached.

    #[test]
    fn deserialize_list_with_trailing_prose() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result =
            json_deserialize("[\"a\", \"b\"]\nThis finds the items.", "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_object_with_trailing_prose() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "name".into(),
            description: String::new(),
            field_type: FieldType::String,
        }]);
        let result =
            json_deserialize(r#"{"name": "Alice"} hope this helps"#, "person", &ft).unwrap();
        match result {
            FieldValue::Object(map) => assert_eq!(map["name"], FieldValue::Str("Alice".into())),
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_single_key_wrapper_with_trailing_prose() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"{"queries": ["a", "b"]} done"#, "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_list_brace_in_string_not_truncated() {
        // A `}` inside a string literal must not be mistaken for a closing
        // bracket; the scanner has to track string state to stay correct.
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"["a }", "b"]"#, "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a }", "b"]));
    }

    #[test]
    fn deserialize_list_truncated_still_errors() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let err = json_deserialize(r#"["a", "b""#, "queries", &ft)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected list"), "got: {err}");
        assert!(err.contains("serde error"), "got: {err}");
    }

    #[test]
    fn deserialize_list_invalid_surfaces_serde_error() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let err = json_deserialize("not json at all", "items", &ft)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected list"), "got: {err}");
        assert!(err.contains("serde error"), "got: {err}");
    }

    #[test]
    fn deserialize_object_invalid_surfaces_serde_error() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "a".into(),
            description: String::new(),
            field_type: FieldType::Int,
        }]);
        let err = json_deserialize("not an object", "obj", &ft)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected object"), "got: {err}");
        assert!(err.contains("serde error"), "got: {err}");
    }

    #[test]
    fn deserialize_map_invalid_surfaces_serde_error() {
        let ft = FieldType::Map(Box::new(FieldType::Int));
        let err = json_deserialize("[1, 2, 3]", "m", &ft)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected map"), "got: {err}");
        assert!(err.contains("serde error"), "got: {err}");
    }

    #[test]
    fn deserialize_mismatch_preserves_raw_value() {
        // The raw model output stays in `actual` alongside the serde cause, so
        // operators see both what was produced and why it failed.
        let ft = FieldType::List(Box::new(FieldType::String));
        let err = json_deserialize("definitely not json", "items", &ft)
            .unwrap_err()
            .to_string();
        assert!(err.contains("definitely not json"), "got: {err}");
        assert!(err.contains("serde error"), "got: {err}");
    }

    #[test]
    fn deserialize_scalar_mismatch_has_no_serde_error_suffix() {
        // Int/float/bool failures use the plain mismatch path; their wording must
        // not gain a serde-error suffix — guards the strict-superset invariant.
        let int_err = json_deserialize("abc", "age", &FieldType::Int)
            .unwrap_err()
            .to_string();
        assert!(int_err.contains("expected int"), "got: {int_err}");
        assert!(!int_err.contains("serde error"), "got: {int_err}");

        let bool_err = json_deserialize("maybe", "flag", &FieldType::Bool)
            .unwrap_err()
            .to_string();
        assert!(!bool_err.contains("serde error"), "got: {bool_err}");
    }

    // --- Direct coverage of the balanced-bracket scanner. Precise span checks
    // for the cases the end-to-end tests exercise indirectly.

    #[test]
    fn extract_balanced_bare_containers_are_whole() {
        assert_eq!(extract_balanced_json("[1, 2, 3]"), Some("[1, 2, 3]"));
        assert_eq!(extract_balanced_json(r#"{"a": 1}"#), Some(r#"{"a": 1}"#));
        assert_eq!(extract_balanced_json("[]"), Some("[]"));
        assert_eq!(extract_balanced_json("{}"), Some("{}"));
    }

    #[test]
    fn extract_balanced_drops_surrounding_prose() {
        assert_eq!(extract_balanced_json("[1, 2] and the rest"), Some("[1, 2]"));
        assert_eq!(extract_balanced_json("here: [1, 2]"), Some("[1, 2]"));
        assert_eq!(
            extract_balanced_json(r#"answer: {"a": 1} done"#),
            Some(r#"{"a": 1}"#)
        );
    }

    #[test]
    fn extract_balanced_handles_nested_mixed_brackets() {
        let s = r#"{"a": [1, {"b": 2}], "c": 3} trailing"#;
        assert_eq!(
            extract_balanced_json(s),
            Some(r#"{"a": [1, {"b": 2}], "c": 3}"#)
        );
    }

    #[test]
    fn extract_balanced_ignores_brackets_inside_strings() {
        // A `}`/`]` inside a string literal must not be counted as a close.
        assert_eq!(
            extract_balanced_json(r#"["a } ] b"]"#),
            Some(r#"["a } ] b"]"#)
        );
    }

    #[test]
    fn extract_balanced_ignores_escaped_quote_in_string() {
        // The escaped quote does not close the string, so the `]` inside stays
        // part of the value and the span closes only at the final bracket.
        let s = r#"["a\"]b"]"#;
        assert_eq!(extract_balanced_json(s), Some(s));
    }

    #[test]
    fn extract_balanced_truncated_returns_none() {
        assert_eq!(extract_balanced_json(r#"["a", "b""#), None);
        assert_eq!(extract_balanced_json(r#"{"a": 1"#), None);
    }

    #[test]
    fn extract_balanced_no_bracket_returns_none() {
        assert_eq!(extract_balanced_json("just prose"), None);
        assert_eq!(extract_balanced_json("42"), None);
        assert_eq!(extract_balanced_json(""), None);
    }

    #[test]
    fn extract_balanced_takes_first_of_multiple_values() {
        // Documented permissive trade: concatenated values keep only the first.
        assert_eq!(
            extract_balanced_json(r#"{"a":1}{"b":2}"#),
            Some(r#"{"a":1}"#)
        );
    }

    // --- End-to-end balanced-prefix extraction across types and edge shapes.

    #[test]
    fn deserialize_list_from_leading_prose() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result =
            json_deserialize(r#"Here are the queries: ["a", "b"]"#, "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_object_value_with_bracket_in_string() {
        // A `}` inside a string value must not truncate the object.
        let ft = FieldType::Object(vec![ObjectField {
            name: "note".into(),
            description: String::new(),
            field_type: FieldType::String,
        }]);
        let result = json_deserialize(r#"{"note": "use } carefully"} thanks"#, "obj", &ft).unwrap();
        match result {
            FieldValue::Object(map) => {
                assert_eq!(map["note"], FieldValue::Str("use } carefully".into()));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_map_with_trailing_prose() {
        let ft = FieldType::Map(Box::new(FieldType::Int));
        let result = json_deserialize(r#"{"x": 1, "y": 2} and so on"#, "counts", &ft).unwrap();
        match result {
            FieldValue::Object(map) => {
                assert_eq!(map["x"], FieldValue::Int(1));
                assert_eq!(map["y"], FieldValue::Int(2));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_object_single_key_wrapper_with_trailing_prose() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "name".into(),
            description: String::new(),
            field_type: FieldType::String,
        }]);
        let result =
            json_deserialize(r#"{"person": {"name": "Bob"}} (done)"#, "person", &ft).unwrap();
        match result {
            FieldValue::Object(map) => assert_eq!(map["name"], FieldValue::Str("Bob".into())),
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_nested_list_with_trailing_prose() {
        let ft = FieldType::List(Box::new(FieldType::List(Box::new(FieldType::Int))));
        let result = json_deserialize("[[1, 2], [3]] trailing text", "grid", &ft).unwrap();
        assert_eq!(
            result,
            FieldValue::List(vec![
                FieldValue::List(vec![FieldValue::Int(1), FieldValue::Int(2)]),
                FieldValue::List(vec![FieldValue::Int(3)]),
            ])
        );
    }

    #[test]
    fn deserialize_list_takes_first_of_multiple_values() {
        // The permissive trade we opted into: trailing extra values are discarded
        // rather than erroring. Locked so a future change is a conscious one.
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"["a"] ["b"]"#, "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a"]));
    }

    #[test]
    fn deserialize_list_from_fence_with_trailing_prose() {
        // strip_code_fence leaves the trailing text when the closing fence isn't
        // the final token; balanced extraction then recovers the array.
        let ft = FieldType::List(Box::new(FieldType::String));
        let result =
            json_deserialize("```json\n[\"a\", \"b\"]\n```\nmore text", "queries", &ft).unwrap();
        assert_eq!(result, str_list(&["a", "b"]));
    }

    #[test]
    fn deserialize_list_escaped_quote_in_element() {
        let ft = FieldType::List(Box::new(FieldType::String));
        let result = json_deserialize(r#"["a\"]b"] extra"#, "queries", &ft).unwrap();
        assert_eq!(
            result,
            FieldValue::List(vec![FieldValue::Str("a\"]b".into())])
        );
    }

    #[test]
    fn deserialize_object_truncated_surfaces_serde_error() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "a".into(),
            description: String::new(),
            field_type: FieldType::Int,
        }]);
        let err = json_deserialize(r#"{"a": 1"#, "obj", &ft)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected object"), "got: {err}");
        assert!(err.contains("serde error"), "got: {err}");
    }

    // ============================================================
    // Phase 3: OneOf / AnyOf buffered deserialization
    // ============================================================

    use typesayer_types::field::{OneOfDiscriminator, VariantArm};

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
                            field_type: FieldType::Enum(vec!["category".into(), "merchant".into()]),
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

    fn untagged_oneof_two_arms() -> FieldType {
        FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "int arm".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "list arm".into(),
                    field_type: FieldType::List(Box::new(FieldType::String)),
                },
            ],
            discriminator: None,
        }
    }

    fn anyof_two_arms() -> FieldType {
        FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "str arm".into(),
                    field_type: FieldType::String,
                },
                VariantArm {
                    description: "int arm".into(),
                    field_type: FieldType::Int,
                },
            ],
        }
    }

    // --- OneOf tagged ---

    #[test]
    fn deserialize_oneof_tagged_happy_path() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items", "topN": 5}"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 1,
                value,
            } => match *value {
                FieldValue::Object(map) => {
                    assert_eq!(map["topN"], FieldValue::Int(5));
                    assert_eq!(map["toolName"], FieldValue::Str("ranked_items".into()));
                }
                other => panic!("expected Object, got {other:?}"),
            },
            other => panic!("expected Variant(1, ...), got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_tagged_missing_discriminator() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::OneOfTagMissing { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn deserialize_oneof_tagged_unknown_tag_lists_valid_tags() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "unknown", "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        match err {
            PredictError::OneOfTagInvalid {
                tag, valid_tags, ..
            } => {
                assert_eq!(tag, "unknown");
                assert!(valid_tags.contains(&"monthly".into()));
                assert!(valid_tags.contains(&"ranked_items".into()));
            }
            other => panic!("expected OneOfTagInvalid, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_tagged_case_mismatch_rejected() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "Ranked_Items", "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::OneOfTagInvalid { .. }),
            "got: {err:?}"
        );
    }

    // --- Discriminator near-miss catalog (Layer 4) ---

    #[test]
    fn deserialize_oneof_tagged_whitespace_in_tag_rejected() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": " ranked_items", "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        match err {
            PredictError::OneOfTagInvalid { tag, .. } => {
                assert_eq!(tag, " ranked_items");
            }
            other => panic!("expected OneOfTagInvalid, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_tagged_substring_superset_rejected() {
        // A tag like "monthly_v2" is structurally similar to a valid tag
        // but distinct; the parser must reject loudly with the full
        // valid-tag list rather than guess-match.
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items_v2", "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        match err {
            PredictError::OneOfTagInvalid {
                tag, valid_tags, ..
            } => {
                assert_eq!(tag, "ranked_items_v2");
                assert!(valid_tags.iter().any(|t| t == "ranked_items"));
            }
            other => panic!("expected OneOfTagInvalid, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_tagged_property_case_mismatch_rejected() {
        // The DISCRIMINATOR PROPERTY name itself is case-sensitive; an
        // LLM emitting `Toolname` instead of `toolName` should trip the
        // missing-discriminator path rather than silently use whichever
        // key happens to be present.
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"Toolname": "ranked_items", "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::OneOfTagMissing { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn deserialize_oneof_tagged_extra_whitespace_around_value_accepted() {
        // Inside the JSON, whitespace around a string value is JSON-grammar
        // whitespace (between `:` and `"`), not part of the tag string.
        // Serde handles this; the discriminator value is the de-quoted
        // string. This case must succeed.
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName":    "ranked_items"  ,  "topN":  5  }"#;
        let value = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(value, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn deserialize_oneof_tagged_discriminator_as_object_rejected() {
        // Non-string discriminator (object) → FieldTypeMismatch on the
        // discriminator's path.
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": {"nested": "ranked_items"}, "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::FieldTypeMismatch { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn deserialize_oneof_tagged_non_string_discriminator() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": 42, "topN": 5}"#;
        let err = json_deserialize(raw, "assignment", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::FieldTypeMismatch { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn deserialize_oneof_tagged_non_object_input() {
        let ft = tagged_oneof_two_arms();
        let err = json_deserialize("42", "assignment", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::OneOfTagMissing { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn deserialize_oneof_tagged_through_code_fence() {
        let ft = tagged_oneof_two_arms();
        let raw = "```json\n{\"toolName\": \"ranked_items\", \"topN\": 5}\n```";
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn deserialize_oneof_tagged_with_trailing_prose() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items", "topN": 5} that's my answer"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn deserialize_oneof_tagged_inside_list() {
        let ft = FieldType::List(Box::new(tagged_oneof_two_arms()));
        let raw = r#"[
            {"toolName": "monthly", "dimension": "category"},
            {"toolName": "ranked_items", "topN": 5}
        ]"#;
        let result = json_deserialize(raw, "items", &ft).unwrap();
        match result {
            FieldValue::List(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(items[0], FieldValue::Variant { arm_index: 0, .. }));
                assert!(matches!(items[1], FieldValue::Variant { arm_index: 1, .. }));
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    // --- OneOf untagged ---

    #[test]
    fn deserialize_oneof_untagged_single_match() {
        let ft = untagged_oneof_two_arms();
        let raw = r#"["a", "b", "c"]"#;
        let result = json_deserialize(raw, "value", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 1,
                value,
            } => {
                assert!(matches!(*value, FieldValue::List(_)));
            }
            other => panic!("expected Variant(1, List), got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_untagged_int_arm_match() {
        let ft = untagged_oneof_two_arms();
        let result = json_deserialize("42", "value", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 0,
                value,
            } => {
                assert_eq!(*value, FieldValue::Int(42));
            }
            other => panic!("expected Variant(0, Int(42)), got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_untagged_no_match_surfaces_arm_errors() {
        let ft = untagged_oneof_two_arms();
        let raw = "not a json value at all";
        let err = json_deserialize(raw, "value", &ft).unwrap_err();
        match err {
            PredictError::OneOfNoArmMatched { arm_errors, .. } => {
                assert_eq!(arm_errors.len(), 2);
            }
            other => panic!("expected OneOfNoArmMatched, got {other:?}"),
        }
    }

    // --- Layer 5: subset/superset arm-ambiguity ---
    //
    // Plan line 290: Arm A = {x: int}, Arm B = {x: int, y: int}, input
    // {x: 1} should match A only because B requires y. Verifies that the
    // Object deserializer surfaces missing required fields rather than
    // silently accepting partial objects — without this guard, the
    // subset/superset case would always be ambiguous.

    #[test]
    fn deserialize_oneof_subset_superset_strict_match() {
        let arm_a = VariantArm {
            description: "x only".into(),
            field_type: FieldType::Object(vec![ObjectField {
                name: "x".into(),
                description: String::new(),
                field_type: FieldType::Int,
            }]),
        };
        let arm_b = VariantArm {
            description: "x + required y".into(),
            field_type: FieldType::Object(vec![
                ObjectField {
                    name: "x".into(),
                    description: String::new(),
                    field_type: FieldType::Int,
                },
                ObjectField {
                    name: "y".into(),
                    description: String::new(),
                    field_type: FieldType::Int,
                },
            ]),
        };
        let ft = FieldType::OneOf {
            arms: vec![arm_a, arm_b],
            discriminator: None,
        };
        // {x: 1} should match A only — B requires both x and y.
        let result = json_deserialize(r#"{"x": 1}"#, "value", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 0,
                value,
            } => {
                if let FieldValue::Object(map) = *value {
                    assert_eq!(map["x"], FieldValue::Int(1));
                } else {
                    panic!("expected Object inner");
                }
            }
            other => panic!("expected Variant(0, ...), got {other:?}"),
        }
    }

    #[test]
    fn deserialize_oneof_subset_superset_both_match_is_ambiguous() {
        // Sanity check: when the input HAS y, both arms validate
        // (subset A matches because A doesn't require absence; superset
        // B matches because B's required fields are present). The buffered
        // semantics treat this as ambiguous, which matches JSON Schema
        // oneOf's "exactly one" rule.
        let arm_a = VariantArm {
            description: "x only".into(),
            field_type: FieldType::Object(vec![ObjectField {
                name: "x".into(),
                description: String::new(),
                field_type: FieldType::Int,
            }]),
        };
        let arm_b = VariantArm {
            description: "x + y".into(),
            field_type: FieldType::Object(vec![
                ObjectField {
                    name: "x".into(),
                    description: String::new(),
                    field_type: FieldType::Int,
                },
                ObjectField {
                    name: "y".into(),
                    description: String::new(),
                    field_type: FieldType::Int,
                },
            ]),
        };
        let ft = FieldType::OneOf {
            arms: vec![arm_a, arm_b],
            discriminator: None,
        };
        let err = json_deserialize(r#"{"x": 1, "y": 2}"#, "value", &ft).unwrap_err();
        assert!(
            matches!(err, PredictError::OneOfAmbiguous { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn deserialize_oneof_untagged_ambiguous_match() {
        // Two object arms with the same shape — JSON Schema authors are
        // expected to make arms structurally disjoint; if they don't, we
        // surface the ambiguity rather than silently picking one.
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
        let err = json_deserialize(r#"{"x": 1}"#, "value", &ft).unwrap_err();
        match err {
            PredictError::OneOfAmbiguous { matching_arms, .. } => {
                assert_eq!(matching_arms, vec![0, 1]);
            }
            other => panic!("expected OneOfAmbiguous, got {other:?}"),
        }
    }

    // --- AnyOf ---

    #[test]
    fn deserialize_anyof_first_match_wins() {
        let ft = anyof_two_arms();
        // "42" matches BOTH String (as the literal string "42") and Int.
        // First arm (String) wins.
        let result = json_deserialize("42", "value", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 0,
                value,
            } => {
                assert_eq!(*value, FieldValue::Str("42".into()));
            }
            other => panic!("expected Variant(0, Str), got {other:?}"),
        }
    }

    #[test]
    fn deserialize_anyof_falls_through_to_later_arm() {
        // For a value that only matches a later arm, that's the winner.
        let ft = FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "int arm".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "str arm".into(),
                    field_type: FieldType::String,
                },
            ],
        };
        let result = json_deserialize("hello world", "value", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 1,
                value,
            } => {
                assert_eq!(*value, FieldValue::Str("hello world".into()));
            }
            other => panic!("expected Variant(1, Str), got {other:?}"),
        }
    }

    #[test]
    fn deserialize_anyof_no_match_collects_arm_errors() {
        let ft = FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "int arm".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "bool arm".into(),
                    field_type: FieldType::Bool,
                },
            ],
        };
        let err = json_deserialize("hello", "value", &ft).unwrap_err();
        match err {
            PredictError::AnyOfNoArmMatched { arm_errors, .. } => {
                assert_eq!(arm_errors.len(), 2);
            }
            other => panic!("expected AnyOfNoArmMatched, got {other:?}"),
        }
    }

    // --- Layer 2: Buffered drift catalog for variants ---
    //
    // Hand-crafted "weird LLM output" cases that exercise the buffered
    // parser's drift tolerance on variant fields. Code fences, trailing
    // prose, leading prose, and other documented recovery paths must
    // succeed for variants the same way they do for List/Object/Map.

    #[test]
    fn drift_catalog_tagged_code_fence_json() {
        let ft = tagged_oneof_two_arms();
        let raw = "```json\n{\"toolName\": \"ranked_items\", \"topN\": 5}\n```";
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_tagged_code_fence_bare() {
        let ft = tagged_oneof_two_arms();
        let raw = "```\n{\"toolName\": \"ranked_items\", \"topN\": 5}\n```";
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_tagged_leading_prose() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"Here is my choice: {"toolName": "ranked_items", "topN": 5}"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_tagged_trailing_prose() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items", "topN": 5} that's the answer"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_tagged_single_key_wrapper() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"assignment": {"toolName": "ranked_items", "topN": 5}}"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_tagged_fence_plus_wrapper() {
        let ft = tagged_oneof_two_arms();
        let raw = "```json\n{\"assignment\": {\"toolName\": \"ranked_items\", \"topN\": 5}}\n```";
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_anyof_code_fence() {
        let ft = anyof_two_arms();
        let raw = "```json\n\"hello world\"\n```";
        let result = json_deserialize(raw, "data", &ft).unwrap();
        match result {
            FieldValue::Variant {
                arm_index: 0,
                value,
            } => {
                assert!(matches!(*value, FieldValue::Str(_)));
            }
            other => panic!("expected Variant(String arm), got {other:?}"),
        }
    }

    #[test]
    fn drift_catalog_untagged_oneof_trailing_prose() {
        let ft = untagged_oneof_two_arms();
        let raw = "[1, 2, 3] and the rest is prose";
        // List arm matches (Int rejects the bracket).
        let result = json_deserialize(raw, "value", &untagged_oneof_int_or_int_list()).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
        // Reference unused param so clippy doesn't complain.
        let _ = ft;
    }

    fn untagged_oneof_int_or_int_list() -> FieldType {
        FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "int".into(),
                    field_type: FieldType::Int,
                },
                VariantArm {
                    description: "list".into(),
                    field_type: FieldType::List(Box::new(FieldType::Int)),
                },
            ],
            discriminator: None,
        }
    }

    #[test]
    fn drift_catalog_tagged_internal_whitespace_accepted() {
        let ft = tagged_oneof_two_arms();
        let raw = r#"{
            "toolName":   "ranked_items",
            "topN":   5
        }"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    #[test]
    fn drift_catalog_tagged_brace_inside_string_not_truncated() {
        // A `}` inside the discriminator value mustn't be mistaken for
        // a closing brace by the balanced-extractor.
        let ft = tagged_oneof_two_arms();
        let raw = r#"{"toolName": "ranked_items", "topN": 5} extra }"#;
        let result = json_deserialize(raw, "assignment", &ft).unwrap();
        assert!(matches!(result, FieldValue::Variant { arm_index: 1, .. }));
    }

    // --- Serialization (round-trip) ---

    #[test]
    fn serialize_variant_emits_inner_value() {
        let value = FieldValue::Variant {
            arm_index: 0,
            value: Box::new(FieldValue::Object(BTreeMap::from([
                ("toolName".into(), FieldValue::Str("monthly".into())),
                ("dimension".into(), FieldValue::Str("category".into())),
            ]))),
        };
        let serialized = json_serialize(&value, &tagged_oneof_two_arms());
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed["toolName"], "monthly");
        assert_eq!(parsed["dimension"], "category");
    }

    #[test]
    fn oneof_tagged_round_trip() {
        let ft = tagged_oneof_two_arms();
        let original = FieldValue::Variant {
            arm_index: 0,
            value: Box::new(FieldValue::Object(BTreeMap::from([
                ("toolName".into(), FieldValue::Str("monthly".into())),
                ("dimension".into(), FieldValue::Str("category".into())),
            ]))),
        };
        let serialized = json_serialize(&original, &ft);
        let restored = json_deserialize(&serialized, "f", &ft).unwrap();
        assert_eq!(original, restored);
    }
}
