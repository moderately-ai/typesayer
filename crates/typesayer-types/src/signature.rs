// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Signature definitions for structured LLM prediction.
//!
//! A [`Signature`] defines the input/output contract for a prediction: what fields
//! the caller provides, what fields the language model produces, and the task
//! instructions that guide the model.

use serde::{Deserialize, Serialize};

use crate::{
    error::{PredictError, Result},
    field::{FieldDef, FieldKind, FieldType, OneOfDiscriminator, VariantArm},
};

/// A signature defines the input/output contract for a prediction.
///
/// Build with [`Signature::builder`]. Field ordering is preserved and deterministic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    instructions: String,
    fields: Vec<FieldDef>,
}

impl Signature {
    /// Create a builder for constructing a signature.
    pub fn builder(instructions: impl Into<String>) -> SignatureBuilder {
        SignatureBuilder {
            instructions: instructions.into(),
            fields: Vec::new(),
        }
    }

    /// The task instructions that guide the language model.
    #[must_use]
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// Iterate over input fields in declaration order.
    pub fn input_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| f.kind == FieldKind::Input)
    }

    /// Iterate over output fields in declaration order.
    pub fn output_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| f.kind == FieldKind::Output)
    }

    /// All fields in declaration order.
    #[must_use]
    pub fn fields(&self) -> &[FieldDef] {
        &self.fields
    }

    /// Serialize this signature to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `PredictError` wrapping a `serde_json` error if serialization
    /// fails (not expected for valid Signature values).
    pub fn dump_state(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    /// Update the task instructions.
    pub fn set_instructions(&mut self, instructions: impl Into<String>) {
        self.instructions = instructions.into();
    }

    /// Return a new signature with an output field prepended before existing outputs.
    ///
    /// Used by `Predict::chain_of_thought` to
    /// inject a reasoning field ahead of the declared outputs.
    #[must_use]
    pub fn with_prepended_output(&self, field: FieldDef) -> Self {
        let mut new_fields = Vec::with_capacity(self.fields.len() + 1);
        let mut inserted = false;
        for f in &self.fields {
            if f.kind == FieldKind::Output && !inserted {
                new_fields.push(FieldDef {
                    kind: FieldKind::Output,
                    ..field.clone()
                });
                inserted = true;
            }
            new_fields.push(f.clone());
        }
        if !inserted {
            new_fields.push(FieldDef {
                kind: FieldKind::Output,
                ..field
            });
        }
        Self {
            instructions: self.instructions.clone(),
            fields: new_fields,
        }
    }

    /// Deserialize a signature from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `PredictError` when the JSON is malformed or describes a
    /// signature with no input or output fields.
    pub fn load_state(json: &str) -> Result<Self> {
        let sig: Self = serde_json::from_str(json)?;
        // Validate the deserialized signature has inputs and outputs
        if sig.input_fields().next().is_none() {
            return Err(PredictError::invalid_signature(
                "signature must have at least one input field",
            ));
        }
        if sig.output_fields().next().is_none() {
            return Err(PredictError::invalid_signature(
                "signature must have at least one output field",
            ));
        }
        Ok(sig)
    }
}

/// Builder for constructing a [`Signature`].
///
/// Enforces that the signature has at least one input and one output field at
/// build time.
pub struct SignatureBuilder {
    instructions: String,
    fields: Vec<FieldDef>,
}

impl SignatureBuilder {
    /// Add an input field. The field's `kind` is set to [`FieldKind::Input`]
    /// regardless of what was passed.
    #[must_use]
    pub fn input(mut self, mut def: FieldDef) -> Self {
        def.kind = FieldKind::Input;
        self.fields.push(def);
        self
    }

    /// Add an output field. The field's `kind` is set to [`FieldKind::Output`]
    /// regardless of what was passed.
    #[must_use]
    pub fn output(mut self, mut def: FieldDef) -> Self {
        def.kind = FieldKind::Output;
        self.fields.push(def);
        self
    }

    /// Validate and build the signature.
    ///
    /// Validation covers: at-least-one input + at-least-one output
    /// (existing); structural validation of every transitively-reachable
    /// [`FieldType::OneOf`] and
    /// [`FieldType::AnyOf`] field (new) — non-empty
    /// arms, discriminator parallelism with arms, unique tag values, and
    /// arm-shape consistency with the named discriminator property.
    ///
    /// Schema-conversion-driven construction (via `field_type_from_schema` in
    /// `typesayer`) is validated at conversion time and would not
    /// produce malformed variants here. The validation in this builder is the
    /// safety net for callers that construct `FieldType::OneOf` / `AnyOf`
    /// programmatically.
    ///
    /// # Errors
    ///
    /// Returns [`PredictError::InvalidSignature`] if there are no input or
    /// output fields, or if any variant field violates the structural
    /// constraints above.
    pub fn build(self) -> Result<Signature> {
        let has_input = self.fields.iter().any(|f| f.kind == FieldKind::Input);
        let has_output = self.fields.iter().any(|f| f.kind == FieldKind::Output);

        if !has_input {
            return Err(PredictError::invalid_signature(
                "signature must have at least one input field",
            ));
        }
        if !has_output {
            return Err(PredictError::invalid_signature(
                "signature must have at least one output field",
            ));
        }

        for field in &self.fields {
            validate_variant_shapes(&field.name, &field.field_type)?;
        }

        Ok(Signature {
            instructions: self.instructions,
            fields: self.fields,
        })
    }
}

/// Recursively validate every variant FieldType reachable from `ft` rooted
/// at `path`. Path is used for the error message so callers can locate the
/// offending field within nested structures.
fn validate_variant_shapes(path: &str, ft: &FieldType) -> Result<()> {
    match ft {
        FieldType::OneOf {
            arms,
            discriminator,
        } => {
            validate_variant_arms(path, arms, "OneOf")?;
            if let Some(disc) = discriminator {
                validate_one_of_discriminator(path, arms, disc)?;
            }
            for (i, arm) in arms.iter().enumerate() {
                let arm_path = format!("{path}#arm{i}");
                validate_variant_shapes(&arm_path, &arm.field_type)?;
            }
            Ok(())
        }
        FieldType::AnyOf { arms } => {
            validate_variant_arms(path, arms, "AnyOf")?;
            for (i, arm) in arms.iter().enumerate() {
                let arm_path = format!("{path}#arm{i}");
                validate_variant_shapes(&arm_path, &arm.field_type)?;
            }
            Ok(())
        }
        FieldType::List(inner) | FieldType::Nullable(inner) | FieldType::Map(inner) => {
            validate_variant_shapes(path, inner)
        }
        FieldType::Object(fields) => {
            for f in fields {
                let nested = format!("{path}.{}", f.name);
                validate_variant_shapes(&nested, &f.field_type)?;
            }
            Ok(())
        }
        FieldType::String
        | FieldType::Int
        | FieldType::Float
        | FieldType::Bool
        | FieldType::Enum(_)
        | FieldType::Media { .. } => Ok(()),
    }
}

fn validate_variant_arms(path: &str, arms: &[VariantArm], kind: &str) -> Result<()> {
    if arms.is_empty() {
        return Err(PredictError::invalid_signature(format!(
            "{kind} field at `{path}` must declare at least one arm"
        )));
    }
    Ok(())
}

/// For a discriminator-tagged OneOf, enforce: tags.len() == arms.len(),
/// unique tag values, and every arm is an Object that includes the named
/// property carrying the arm's tag as a const-restricted `Enum` (or
/// `Enum(vec![tag])` produced by the schema converter's `const` fold).
fn validate_one_of_discriminator(
    path: &str,
    arms: &[VariantArm],
    discriminator: &OneOfDiscriminator,
) -> Result<()> {
    if discriminator.tags.len() != arms.len() {
        return Err(PredictError::invalid_signature(format!(
            "OneOf at `{path}` discriminator has {} tags but {} arms; the \
             vectors must be parallel",
            discriminator.tags.len(),
            arms.len()
        )));
    }

    let mut seen = std::collections::HashSet::with_capacity(discriminator.tags.len());
    for tag in &discriminator.tags {
        if !seen.insert(tag.as_str()) {
            return Err(PredictError::invalid_signature(format!(
                "OneOf at `{path}` discriminator tag `{tag}` appears more than \
                 once; tags must be unique"
            )));
        }
    }

    for (i, arm) in arms.iter().enumerate() {
        let arm_path = format!("{path}#{tag}", tag = discriminator.tags[i]);
        validate_arm_carries_discriminator(
            &arm_path,
            &arm.field_type,
            &discriminator.property,
            &discriminator.tags[i],
        )?;
    }
    Ok(())
}

/// Walk through a single Nullable layer if present (mirrors the schema
/// converter's inference), then assert the arm is an `Object` declaring the
/// discriminator property with an `Enum` whose only variant is the arm's tag.
fn validate_arm_carries_discriminator(
    arm_path: &str,
    arm_type: &FieldType,
    property: &str,
    tag: &str,
) -> Result<()> {
    let inner = match arm_type {
        FieldType::Nullable(inner) => inner.as_ref(),
        other => other,
    };
    let FieldType::Object(fields) = inner else {
        return Err(PredictError::invalid_signature(format!(
            "tagged OneOf arm at `{arm_path}` must be an Object (or \
             Nullable<Object>); got `{}`",
            arm_type.type_label()
        )));
    };
    let prop = fields.iter().find(|f| f.name == property).ok_or_else(|| {
        PredictError::invalid_signature(format!(
            "tagged OneOf arm at `{arm_path}` is missing the discriminator \
             property `{property}`"
        ))
    })?;
    match &prop.field_type {
        FieldType::Enum(variants) if variants.iter().any(|v| v == tag) => Ok(()),
        other => Err(PredictError::invalid_signature(format!(
            "tagged OneOf arm at `{arm_path}` declares discriminator \
             property `{property}` as `{}`, but the discriminator's tag \
             `{tag}` requires an Enum variant containing it (typical: \
             single-element Enum from a const-restricted schema)",
            other.type_label()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::FieldType;

    fn simple_signature() -> Signature {
        Signature::builder("Answer the question.")
            .input(FieldDef::input(
                "question",
                FieldType::String,
                "The user question",
            ))
            .output(FieldDef::output("answer", FieldType::String, "The answer"))
            .build()
            .unwrap()
    }

    #[test]
    fn builder_produces_valid_signature() {
        let sig = simple_signature();
        assert_eq!(sig.instructions(), "Answer the question.");
        assert_eq!(sig.input_fields().count(), 1);
        assert_eq!(sig.output_fields().count(), 1);
    }

    #[test]
    fn builder_no_inputs_fails() {
        let result = Signature::builder("test")
            .output(FieldDef::output("answer", FieldType::String, "answer"))
            .build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("input"));
    }

    #[test]
    fn builder_no_outputs_fails() {
        let result = Signature::builder("test")
            .input(FieldDef::input("question", FieldType::String, "question"))
            .build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("output"));
    }

    #[test]
    fn field_ordering_preserved() {
        let sig = Signature::builder("test")
            .input(FieldDef::input("a", FieldType::String, "first"))
            .input(FieldDef::input("b", FieldType::Int, "second"))
            .output(FieldDef::output("x", FieldType::String, "third"))
            .output(FieldDef::output("y", FieldType::Bool, "fourth"))
            .build()
            .unwrap();

        let input_names: Vec<_> = sig.input_fields().map(|f| f.name.as_str()).collect();
        assert_eq!(input_names, vec!["a", "b"]);

        let output_names: Vec<_> = sig.output_fields().map(|f| f.name.as_str()).collect();
        assert_eq!(output_names, vec!["x", "y"]);
    }

    #[test]
    fn dump_load_round_trip() {
        let sig = Signature::builder("Classify sentiment.")
            .input(FieldDef::input("text", FieldType::String, "Input text"))
            .output(FieldDef::output(
                "sentiment",
                FieldType::Enum(vec!["positive".into(), "negative".into(), "neutral".into()]),
                "The sentiment",
            ))
            .build()
            .unwrap();

        let json = sig.dump_state().unwrap();
        let restored = Signature::load_state(&json).unwrap();

        assert_eq!(sig.instructions(), restored.instructions());
        assert_eq!(sig.fields().len(), restored.fields().len());
        for (a, b) in sig.fields().iter().zip(restored.fields().iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn dump_load_with_nested_types() {
        let sig = Signature::builder("Extract info.")
            .input(FieldDef::input("doc", FieldType::String, "The document"))
            .output(FieldDef::output(
                "entities",
                FieldType::List(Box::new(FieldType::Object(vec![
                    crate::field::ObjectField {
                        name: "name".into(),
                        description: "Entity name".into(),
                        field_type: FieldType::String,
                    },
                    crate::field::ObjectField {
                        name: "type".into(),
                        description: "Entity type".into(),
                        field_type: FieldType::String,
                    },
                ]))),
                "Extracted entities",
            ))
            .build()
            .unwrap();

        let json = sig.dump_state().unwrap();
        let restored = Signature::load_state(&json).unwrap();
        assert_eq!(sig.fields(), restored.fields());
    }

    // ============================================================
    // OneOf / AnyOf structural validation at Signature::build.
    //
    // Schema-conversion-driven construction already rejects malformed
    // input. These tests guard the programmatic-construction path:
    // a caller who builds a FieldType::OneOf by hand with mismatched
    // tag/arm parallelism, duplicate tags, or arms that lack the named
    // discriminator property must get a build-time error rather than
    // undefined behavior at parse time.
    // ============================================================

    use crate::field::{ObjectField, OneOfDiscriminator, VariantArm};

    fn tagged_oneof_object_arm(tag: &str, extra_field: (&str, FieldType)) -> VariantArm {
        VariantArm {
            description: tag.into(),
            field_type: FieldType::Object(vec![
                ObjectField {
                    name: "kind".into(),
                    description: String::new(),
                    field_type: FieldType::Enum(vec![tag.into()]),
                },
                ObjectField {
                    name: extra_field.0.into(),
                    description: String::new(),
                    field_type: extra_field.1,
                },
            ]),
        }
    }

    #[test]
    fn build_rejects_oneof_with_empty_arms() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![],
                    discriminator: None,
                },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least one arm"), "got: {err}");
        assert!(err.contains("OneOf"), "got: {err}");
    }

    #[test]
    fn build_rejects_anyof_with_empty_arms() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::AnyOf { arms: vec![] },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least one arm"), "got: {err}");
        assert!(err.contains("AnyOf"), "got: {err}");
    }

    #[test]
    fn build_rejects_discriminator_with_mismatched_tag_count() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![
                        tagged_oneof_object_arm("a", ("x", FieldType::Int)),
                        tagged_oneof_object_arm("b", ("y", FieldType::String)),
                    ],
                    discriminator: Some(OneOfDiscriminator {
                        property: "kind".into(),
                        tags: vec!["a".into()],
                    }),
                },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("1 tags but 2 arms"), "got: {err}");
        assert!(err.contains("parallel"), "got: {err}");
    }

    #[test]
    fn build_rejects_duplicate_discriminator_tags() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![
                        tagged_oneof_object_arm("dup", ("x", FieldType::Int)),
                        tagged_oneof_object_arm("dup", ("y", FieldType::String)),
                    ],
                    discriminator: Some(OneOfDiscriminator {
                        property: "kind".into(),
                        tags: vec!["dup".into(), "dup".into()],
                    }),
                },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("dup"), "got: {err}");
        assert!(err.contains("unique"), "got: {err}");
    }

    #[test]
    fn build_rejects_tagged_arm_that_is_not_an_object() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![VariantArm {
                        description: "scalar arm".into(),
                        field_type: FieldType::Int,
                    }],
                    discriminator: Some(OneOfDiscriminator {
                        property: "kind".into(),
                        tags: vec!["a".into()],
                    }),
                },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("must be an Object"), "got: {err}");
    }

    #[test]
    fn build_rejects_tagged_arm_missing_discriminator_property() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![VariantArm {
                        description: "no discriminator field".into(),
                        field_type: FieldType::Object(vec![ObjectField {
                            name: "x".into(),
                            description: String::new(),
                            field_type: FieldType::Int,
                        }]),
                    }],
                    discriminator: Some(OneOfDiscriminator {
                        property: "kind".into(),
                        tags: vec!["a".into()],
                    }),
                },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("missing the discriminator property"),
            "got: {err}"
        );
        assert!(err.contains("kind"), "got: {err}");
    }

    #[test]
    fn build_rejects_tagged_arm_with_wrong_enum_for_discriminator() {
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![VariantArm {
                        description: "wrong tag enum".into(),
                        field_type: FieldType::Object(vec![
                            ObjectField {
                                name: "kind".into(),
                                description: String::new(),
                                field_type: FieldType::Enum(vec!["other".into()]),
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
                        tags: vec!["a".into()],
                    }),
                },
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Enum variant containing"), "got: {err}");
    }

    #[test]
    fn build_accepts_well_formed_tagged_oneof() {
        Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::OneOf {
                    arms: vec![
                        tagged_oneof_object_arm("a", ("x", FieldType::Int)),
                        tagged_oneof_object_arm("b", ("y", FieldType::String)),
                    ],
                    discriminator: Some(OneOfDiscriminator {
                        property: "kind".into(),
                        tags: vec!["a".into(), "b".into()],
                    }),
                },
                "",
            ))
            .build()
            .expect("well-formed tagged OneOf must build");
    }

    #[test]
    fn build_validates_variant_nested_inside_list_and_nullable() {
        // Nested empty-arm OneOf reachable through List<Nullable<OneOf>>:
        // the recursive walker should catch it.
        let result = Signature::builder("inst")
            .input(FieldDef::input("q", FieldType::String, ""))
            .output(FieldDef::output(
                "out",
                FieldType::List(Box::new(FieldType::Nullable(Box::new(FieldType::OneOf {
                    arms: vec![],
                    discriminator: None,
                })))),
                "",
            ))
            .build();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least one arm"), "got: {err}");
    }

    #[test]
    fn load_state_validates_fields() {
        // JSON with no input fields
        let json = serde_json::json!({
            "instructions": "test",
            "fields": [{
                "name": "answer",
                "description": "answer",
                "field_type": {"kind": "string"},
                "kind": "output"
            }]
        });
        let result = Signature::load_state(&json.to_string());
        assert!(result.is_err());
    }

    #[test]
    fn builder_forces_kind() {
        // Even if you pass a FieldDef with the wrong kind, builder overrides it
        let wrong_kind = FieldDef::output("question", FieldType::String, "A question");
        let sig = Signature::builder("test")
            .input(wrong_kind) // builder forces Input
            .output(FieldDef::output("answer", FieldType::String, "answer"))
            .build()
            .unwrap();

        let input = sig.input_fields().next().unwrap();
        assert_eq!(input.kind, FieldKind::Input);
    }
}
