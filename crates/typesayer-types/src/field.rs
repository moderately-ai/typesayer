// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Field types, definitions, and values for structured LLM prediction.

use std::{collections::BTreeMap, fmt};

use enumset::EnumSet;
use modelplease::{MediaKind, MediaSource, SourceKind};
use serde::{Deserialize, Serialize};

/// The type of a field in a signature.
///
/// Exhaustive by design — adding a new variant causes compiler errors at all
/// unhandled match arms. There is no `Any` or `Custom` escape hatch.
///
/// Sum-type variants ([`OneOf`](Self::OneOf), [`AnyOf`](Self::AnyOf)) model
/// JSON Schema's `oneOf` / `anyOf` composition primitives. JSON Schema's other
/// composition keywords (`allOf`, `const`, `$ref`, `dependentSchemas`,
/// `if`/`then`/`else`) are folded or transformed at schema-conversion time into
/// the existing variants — they have no runtime representation here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "inner", rename_all = "snake_case")]
pub enum FieldType {
    /// A UTF-8 string.
    String,
    /// A 64-bit signed integer.
    Int,
    /// A 64-bit floating-point number.
    Float,
    /// A boolean value.
    Bool,
    /// A homogeneous list of values.
    List(Box<Self>),
    /// A structured object with named fields.
    Object(Vec<ObjectField>),
    /// A string-keyed map with homogeneous values.
    Map(Box<Self>),
    /// One of a fixed set of string variants.
    Enum(Vec<std::string::String>),
    /// A nullable wrapper around another type.
    Nullable(Box<Self>),
    /// Exactly one arm matches. Models JSON Schema's `oneOf`.
    ///
    /// When `discriminator` is `Some`, parsing is deterministic via tag
    /// dispatch — the inbound JSON's `property` field selects the arm by its
    /// `tags[i]` value. When `discriminator` is `None`, parsing tries every
    /// arm and demands exactly one succeed; zero matches and multiple matches
    /// both surface specific errors.
    OneOf {
        /// Alternatives, in declaration order.
        arms: Vec<VariantArm>,
        /// Discriminator hint for deterministic dispatch when the arms share
        /// a `const`-valued property. Inferred at schema-conversion time.
        discriminator: Option<OneOfDiscriminator>,
    },
    /// At least one arm matches; first match wins. Models JSON Schema's
    /// `anyOf`. Parsing tries arms in declared order; the first arm whose
    /// deserializer succeeds wins, and the rest are not attempted. Suitable
    /// when arms may legitimately overlap and the consumer is content with
    /// declared-order priority.
    AnyOf {
        /// Alternatives, in declared order.
        arms: Vec<VariantArm>,
    },
    /// A media reference (image / document / audio / video).
    ///
    /// `accepted_sources` constrains which [`SourceKind`]s a caller may
    /// pass at value-construction time — the application preflight uses
    /// this to reject application configurations whose declared sources
    /// the model doesn't support.
    Media {
        /// Which modality this field carries (`Image`, `Document`, ...).
        kind: MediaKind,
        /// Source kinds the slot accepts. Default is "all" when the
        /// schema doesn't specify; preflight narrows it against the
        /// model's capability table.
        accepted_sources: EnumSet<SourceKind>,
    },
}

/// One arm of a [`OneOf`](FieldType::OneOf) or [`AnyOf`](FieldType::AnyOf).
///
/// Carries both the structural type and a human-readable description so the
/// chat adapter can render per-arm guidance in the prompt without losing the
/// `description` keyword from the source JSON Schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantArm {
    /// Description of the arm — typically the `description` keyword from the
    /// source JSON Schema arm, falling back to the discriminator tag or an
    /// auto-generated name when absent.
    pub description: std::string::String,
    /// The arm's structural type.
    pub field_type: FieldType,
}

/// Discriminator hint for a tagged [`OneOf`](FieldType::OneOf).
///
/// Present when every arm is an Object that names a single property whose
/// value is a unique `const` string. The schema converter infers this from
/// the source `oneOf` structure and optionally cross-checks against an
/// explicit OpenAPI `discriminator: {propertyName: P}` hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneOfDiscriminator {
    /// The property name whose value selects the arm (e.g. `"toolName"`).
    pub property: std::string::String,
    /// `tags[i]` is the const value of `property` in `arms[i]`. Parallel to
    /// the parent's `arms` vector; all values are unique strings.
    pub tags: Vec<std::string::String>,
}

impl FieldType {
    /// Returns a human-readable type label for use in system prompts.
    #[must_use]
    pub fn type_label(&self) -> std::string::String {
        match self {
            Self::String => "str".to_owned(),
            Self::Int => "int".to_owned(),
            Self::Float => "float".to_owned(),
            Self::Bool => "bool".to_owned(),
            Self::List(inner) => format!("list[{}]", inner.type_label()),
            Self::Object(fields) => {
                let parts: Vec<_> = fields
                    .iter()
                    .map(|f| format!("{}: {}", f.name, f.field_type.type_label()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            Self::Map(value_type) => format!("map[str, {}]", value_type.type_label()),
            Self::Enum(variants) => {
                format!("enum[{}]", variants.join(", "))
            }
            Self::Nullable(inner) => format!("optional[{}]", inner.type_label()),
            Self::OneOf {
                arms,
                discriminator,
            } => discriminator.as_ref().map_or_else(
                || {
                    let parts: Vec<_> = arms.iter().map(|a| a.field_type.type_label()).collect();
                    format!("oneof[{}]", parts.join(" | "))
                },
                |disc| format!("oneof[{}: {}]", disc.property, disc.tags.join(" | ")),
            ),
            Self::AnyOf { arms } => {
                let parts: Vec<_> = arms.iter().map(|a| a.field_type.type_label()).collect();
                format!("anyof[{}]", parts.join(" | "))
            }
            Self::Media { kind, .. } => format!("media[{}]", kind.label()),
        }
    }

    /// Concrete serialization guidance for an *output* field of this type,
    /// for the system prompt and the live output-format reminder. `None`
    /// means no note is needed — a bare string, or a media slot that's
    /// emitted as an out-of-band content part rather than text.
    ///
    /// The type label alone (`list[str]`) doesn't tell a model the wire
    /// format, so well-behaved prompts still emit code fences, single-key
    /// wrappers, or markdown lists. Stating the exact shape makes formatting
    /// the adapter's responsibility, not the caller's. Mirrors DSPy's
    /// `translate_field_type`.
    #[must_use]
    pub fn output_format_hint(&self) -> Option<std::string::String> {
        match self {
            Self::String | Self::Media { .. } => None,
            Self::Int => Some("a single integer".to_owned()),
            Self::Float => Some("a single number".to_owned()),
            Self::Bool => Some("`true` or `false`".to_owned()),
            Self::Enum(variants) => Some(format!("exactly one of: {}", variants.join(", "))),
            Self::List(inner) => Some(format!(
                "a JSON array of {}, e.g. [\"...\", \"...\"] — not an object, not a code fence",
                inner.type_label()
            )),
            Self::Object(fields) => {
                let example = object_payload_example(fields);
                Some(format!(
                    "a JSON object matching {} — e.g. {example} — not a code fence",
                    self.type_label()
                ))
            }
            Self::Map(value_type) => Some(format!(
                "a JSON object with string keys and {} values — \
                 e.g. {{\"key1\": ..., \"key2\": ...}} — not a code fence",
                value_type.type_label()
            )),
            Self::Nullable(inner) => {
                // Nullable's hint says both halves: what a present value
                // looks like AND when null is the correct emission. The
                // "when not applicable" framing converts the user's
                // most-common LLM-emits-null failure mode (model wants
                // to express "doesn't apply" but the schema author
                // didn't tell it to drop the marker) into the right
                // mental model — drop the marker for not-applicable,
                // emit null only when the value is genuinely null-valued.
                Some(inner.output_format_hint().map_or_else(
                    || "a value, or null when the value is not applicable".to_owned(),
                    |hint| format!("{hint}, or null when the value is not applicable"),
                ))
            }
            Self::OneOf { discriminator, .. } => Some(discriminator.as_ref().map_or_else(
                || {
                    "a JSON value matching exactly one of the shapes listed under \"Variant \
                     shapes\" above"
                        .to_owned()
                },
                |d| {
                    // Parenthesise the "(see ...)" cross-reference so a wrapping
                    // `Nullable` can append its ", or null" clause without
                    // creating "see X above, or null" ambiguity. Including a
                    // representative tag value as part of the example makes
                    // the shape concrete without forcing the model to scroll
                    // back to the variant-shapes block to pick one.
                    let example_tag = d.tags.first().map_or("...", String::as_str);
                    format!(
                        "a JSON object whose `{property}` field selects the variant — \
                         e.g. {{\"{property}\": \"{example_tag}\", ...}} \
                         (see \"Variant shapes\" above for each arm's fields)",
                        property = d.property,
                    )
                },
            )),
            Self::AnyOf { .. } => Some(
                "a JSON value matching any of the shapes listed under \"Variant shapes\" \
                 above (first match wins)"
                    .to_owned(),
            ),
        }
    }
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.type_label())
    }
}

/// Build a tiny example payload for an [`Object`](FieldType::Object) field
/// to embed in the output-format hint. The type label alone says *what*;
/// the example says *how* — both pieces together collapse a class of
/// "model emitted a code-fenced thing in the wrong shape" errors.
///
/// Caps at the first three fields and uses a per-type placeholder so the
/// example stays one line regardless of how wide the declared object is.
fn object_payload_example(fields: &[ObjectField]) -> String {
    let parts: Vec<String> = fields
        .iter()
        .take(3)
        .map(|f| format!("\"{}\": {}", f.name, type_placeholder(&f.field_type)))
        .collect();
    let suffix = if fields.len() > 3 { ", ..." } else { "" };
    format!("{{{}{suffix}}}", parts.join(", "))
}

/// A one-token JSON placeholder for the given type, used in synthetic
/// example payloads. Concrete enough that the model can pattern-match
/// (`123` for int, `"..."` for string) without claiming a specific
/// value the schema does not commit to.
const fn type_placeholder(field_type: &FieldType) -> &'static str {
    match field_type {
        FieldType::String | FieldType::Enum(_) | FieldType::Media { .. } => "\"...\"",
        FieldType::Int => "123",
        FieldType::Float => "1.5",
        FieldType::Bool => "true",
        FieldType::List(_) => "[...]",
        FieldType::Object(_) | FieldType::Map(_) => "{...}",
        FieldType::Nullable(_) => "null",
        FieldType::OneOf { .. } | FieldType::AnyOf { .. } => "...",
    }
}

/// A field within an [`Object`](FieldType::Object) type.
///
/// Unlike [`FieldDef`], this has no input/output discriminant — nested object
/// fields are always part of their parent's structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectField {
    /// The field name.
    pub name: std::string::String,
    /// A human-readable description of the field.
    pub description: std::string::String,
    /// The type of this field.
    pub field_type: FieldType,
}

/// Whether a field is an input or output of a signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    /// An input field provided by the caller.
    Input,
    /// An output field produced by the language model.
    Output,
}

/// A top-level field definition within a [`Signature`](crate::Signature).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    /// The field name.
    pub name: std::string::String,
    /// A human-readable description of the field.
    pub description: std::string::String,
    /// The type of this field.
    pub field_type: FieldType,
    /// Whether this is an input or output field.
    pub kind: FieldKind,
    /// Whether this input field's value is stable across calls within a
    /// conversation and therefore eligible to sit inside the cacheable
    /// prefix of the user message. Default `false` (backwards-compatible):
    /// only fields a application explicitly declares stable are cached. The
    /// `ChatAdapter` places a cache breakpoint after
    /// the last cacheable input field; meaningless on output fields.
    #[serde(default)]
    pub cacheable: bool,
    /// Concrete example values for this field, sourced from the
    /// JSON Schema `examples` keyword on the originating schema. Rendered
    /// in the system prompt under the field's description so the model
    /// sees "what good output looks like" without bloating the
    /// instructions string. Defaults to empty; the schema converter
    /// caps at the first three examples to bound prompt size.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<serde_json::Value>,
}

impl FieldDef {
    /// Create an input field definition.
    ///
    /// `field_type` sits between the two `String` args by design: it's the
    /// only non-string positional arg, so callers cannot silently
    /// transpose `name` and `description` without a compile error from
    /// the misplaced `FieldType` enum value.
    pub fn input(
        name: impl Into<std::string::String>,
        field_type: FieldType,
        description: impl Into<std::string::String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            field_type,
            kind: FieldKind::Input,
            cacheable: false,
            examples: Vec::new(),
        }
    }

    /// Create an output field definition. See [`Self::input`] for the
    /// rationale behind the arg order (`field_type` between the two
    /// `String` args).
    pub fn output(
        name: impl Into<std::string::String>,
        field_type: FieldType,
        description: impl Into<std::string::String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            field_type,
            kind: FieldKind::Output,
            cacheable: false,
            examples: Vec::new(),
        }
    }

    /// Attach concrete example values to this field. Builder-style for
    /// test ergonomics — production paths populate [`Self::examples`]
    /// directly via the JSON Schema converter in `schema.rs`.
    #[must_use]
    pub fn with_examples(mut self, examples: Vec<serde_json::Value>) -> Self {
        self.examples = examples;
        self
    }
}

/// A typed value extracted from an LM completion or supplied as input.
///
/// Mirrors [`FieldType`] but holds actual data. Produced by adapter parsing.
///
/// # Serde behavior
///
/// Uses `#[serde(untagged)]` — serde tries variants in declaration order during
/// deserialization. This means JSON `42` deserializes as `Int(42)` (not `Float`),
/// `3.14` as `Float(3.14)`, `true` as `Bool(true)`, etc. This ordering is
/// intentional and deterministic: whole numbers become `Int`, decimal numbers
/// become `Float`. Do not reorder variants without considering the serde impact.
///
/// The `Media` variant is tagged via its inner [`MediaValue`] struct
/// (which carries an explicit `kind` discriminator) rather than the
/// untagged numeric/string variants. Serde tries each untagged variant
/// in order, and `MediaValue`'s struct shape is unambiguous against
/// the scalar variants.
///
/// The `Variant` arm is placed **before** `Object` so its specific
/// `{arm_index, value}` shape matches first. The narrow consequence: a real
/// Object whose declared fields are exactly `{arm_index: <non-negative int>,
/// value: <any>}` (and nothing else) cannot round-trip through serde-untagged
/// — it would be parsed as `Variant`. No production schema uses these field
/// names, so the ambiguity is theoretical, but consumers persisting custom
/// shapes should avoid this exact field pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldValue {
    /// A string value.
    Str(std::string::String),
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A list of values.
    List(Vec<Self>),
    /// A matched arm of a [`OneOf`](FieldType::OneOf) or [`AnyOf`](FieldType::AnyOf)
    /// field. `arm_index` is the zero-based position of the matched arm in the
    /// owning FieldType's `arms` vector; consumers can switch on it without
    /// re-running arm validation.
    Variant {
        /// Zero-based index of the matched arm in the owning FieldType's
        /// `arms` vector.
        arm_index: usize,
        /// The inner value produced by the matched arm's deserializer.
        value: Box<Self>,
    },
    /// A structured object with named fields.
    Object(BTreeMap<std::string::String, Self>),
    /// A media reference (image / document / audio / video).
    Media(MediaValue),
    /// A null value (from a nullable field).
    Null,
}

/// A media field value. Carries the [`MediaSource`] (where the bytes
/// come from) and a duplicate `kind` discriminator so consumers can
/// switch on the modality without inspecting the source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaValue {
    /// Modality this value represents.
    pub kind: MediaKind,
    /// Where the bytes come from.
    pub source: MediaSource,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_label_primitives() {
        assert_eq!(FieldType::String.type_label(), "str");
        assert_eq!(FieldType::Int.type_label(), "int");
        assert_eq!(FieldType::Float.type_label(), "float");
        assert_eq!(FieldType::Bool.type_label(), "bool");
    }

    #[test]
    fn type_label_nested() {
        let list_int = FieldType::List(Box::new(FieldType::Int));
        assert_eq!(list_int.type_label(), "list[int]");

        let nullable_str = FieldType::Nullable(Box::new(FieldType::String));
        assert_eq!(nullable_str.type_label(), "optional[str]");

        let map_float = FieldType::Map(Box::new(FieldType::Float));
        assert_eq!(map_float.type_label(), "map[str, float]");
    }

    #[test]
    fn type_label_enum() {
        let e = FieldType::Enum(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(e.type_label(), "enum[a, b, c]");
    }

    #[test]
    fn type_label_object() {
        let obj = FieldType::Object(vec![
            ObjectField {
                name: "x".into(),
                description: "x coord".into(),
                field_type: FieldType::Int,
            },
            ObjectField {
                name: "y".into(),
                description: "y coord".into(),
                field_type: FieldType::Int,
            },
        ]);
        assert_eq!(obj.type_label(), "{x: int, y: int}");
    }

    #[test]
    fn field_def_input_constructor() {
        let f = FieldDef::input("question", FieldType::String, "The user question");
        assert_eq!(f.name, "question");
        assert_eq!(f.kind, FieldKind::Input);
    }

    #[test]
    fn field_def_output_constructor() {
        let f = FieldDef::output("answer", FieldType::String, "The answer");
        assert_eq!(f.name, "answer");
        assert_eq!(f.kind, FieldKind::Output);
    }

    #[test]
    fn field_type_serde_round_trip_primitive() {
        let ft = FieldType::String;
        let json = serde_json::to_string(&ft).unwrap();
        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(ft, deserialized);
    }

    #[test]
    fn field_type_serde_round_trip_nested() {
        let ft = FieldType::List(Box::new(FieldType::Nullable(Box::new(FieldType::Int))));
        let json = serde_json::to_string(&ft).unwrap();
        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(ft, deserialized);
    }

    #[test]
    fn field_type_serde_round_trip_object() {
        let ft = FieldType::Object(vec![ObjectField {
            name: "name".into(),
            description: "A name".into(),
            field_type: FieldType::String,
        }]);
        let json = serde_json::to_string(&ft).unwrap();
        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(ft, deserialized);
    }

    #[test]
    fn field_value_serde_round_trip() {
        let val = FieldValue::Object(BTreeMap::from([
            ("name".into(), FieldValue::Str("Alice".into())),
            ("age".into(), FieldValue::Int(30)),
            ("active".into(), FieldValue::Bool(true)),
        ]));
        let json = serde_json::to_string(&val).unwrap();
        let deserialized: FieldValue = serde_json::from_str(&json).unwrap();
        assert_eq!(val, deserialized);
    }

    #[test]
    fn field_def_serde_round_trip() {
        let fd = FieldDef::input("text", FieldType::String, "Input text");
        let json = serde_json::to_string(&fd).unwrap();
        let deserialized: FieldDef = serde_json::from_str(&json).unwrap();
        assert_eq!(fd, deserialized);
    }

    #[test]
    fn output_format_hint_none_for_string_and_media() {
        assert!(FieldType::String.output_format_hint().is_none());
        let media = FieldType::Media {
            kind: MediaKind::Image,
            accepted_sources: EnumSet::all(),
        };
        assert!(media.output_format_hint().is_none());
    }

    #[test]
    fn output_format_hint_list_says_json_array() {
        let hint = FieldType::List(Box::new(FieldType::String))
            .output_format_hint()
            .expect("list has a hint");
        assert!(hint.contains("JSON array"), "got: {hint}");
        assert!(hint.contains('['), "should show array brackets: {hint}");
    }

    #[test]
    fn output_format_hint_bool_says_true_false() {
        let hint = FieldType::Bool
            .output_format_hint()
            .expect("bool has a hint");
        assert!(
            hint.contains("true") && hint.contains("false"),
            "got: {hint}"
        );
    }

    #[test]
    fn output_format_hint_enum_lists_variants() {
        let hint = FieldType::Enum(vec!["yes".into(), "no".into()])
            .output_format_hint()
            .expect("enum has a hint");
        assert!(hint.contains("yes") && hint.contains("no"), "got: {hint}");
    }

    #[test]
    fn output_format_hint_nullable_mentions_null() {
        let hint = FieldType::Nullable(Box::new(FieldType::List(Box::new(FieldType::String))))
            .output_format_hint()
            .expect("nullable has a hint");
        assert!(hint.contains("null"), "got: {hint}");
    }

    #[test]
    fn output_format_hint_object_and_map_say_json_object() {
        let obj = FieldType::Object(vec![ObjectField {
            name: "x".into(),
            description: String::new(),
            field_type: FieldType::Int,
        }]);
        assert!(
            obj.output_format_hint()
                .expect("object hint")
                .contains("JSON object")
        );
        let map = FieldType::Map(Box::new(FieldType::Int));
        assert!(
            map.output_format_hint()
                .expect("map hint")
                .contains("JSON object")
        );
    }

    // --- OneOf / AnyOf type-lattice tests ---

    fn variant_arm_obj(name: &str, field_type: FieldType) -> VariantArm {
        VariantArm {
            description: format!("arm: {name}"),
            field_type,
        }
    }

    #[test]
    fn type_label_oneof_tagged_names_discriminator_and_tags() {
        let ft = FieldType::OneOf {
            arms: vec![
                variant_arm_obj("a", FieldType::Object(vec![])),
                variant_arm_obj("b", FieldType::Object(vec![])),
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["a".into(), "b".into()],
            }),
        };
        let label = ft.type_label();
        assert!(label.contains("oneof"), "got: {label}");
        assert!(label.contains("kind"), "got: {label}");
        assert!(label.contains('a'), "got: {label}");
        assert!(label.contains('b'), "got: {label}");
    }

    #[test]
    fn type_label_oneof_untagged_lists_arm_labels() {
        let ft = FieldType::OneOf {
            arms: vec![
                variant_arm_obj("int", FieldType::Int),
                variant_arm_obj("str", FieldType::String),
            ],
            discriminator: None,
        };
        let label = ft.type_label();
        assert!(label.starts_with("oneof["), "got: {label}");
        assert!(label.contains("int"), "got: {label}");
        assert!(label.contains("str"), "got: {label}");
    }

    #[test]
    fn type_label_anyof_lists_arm_labels() {
        let ft = FieldType::AnyOf {
            arms: vec![
                variant_arm_obj("int", FieldType::Int),
                variant_arm_obj("str", FieldType::String),
            ],
        };
        let label = ft.type_label();
        assert!(label.starts_with("anyof["), "got: {label}");
        assert!(label.contains("int"), "got: {label}");
        assert!(label.contains("str"), "got: {label}");
    }

    #[test]
    fn output_format_hint_oneof_tagged_points_to_variant_shapes() {
        let ft = FieldType::OneOf {
            arms: vec![variant_arm_obj("a", FieldType::Object(vec![]))],
            discriminator: Some(OneOfDiscriminator {
                property: "toolName".into(),
                tags: vec!["a".into()],
            }),
        };
        let hint = ft.output_format_hint().expect("hint");
        assert!(hint.contains("toolName"), "got: {hint}");
        assert!(hint.contains("Variant shapes"), "got: {hint}");
    }

    #[test]
    fn output_format_hint_oneof_untagged_points_to_variant_shapes() {
        let ft = FieldType::OneOf {
            arms: vec![variant_arm_obj("int", FieldType::Int)],
            discriminator: None,
        };
        let hint = ft.output_format_hint().expect("hint");
        assert!(hint.contains("exactly one"), "got: {hint}");
        assert!(hint.contains("Variant shapes"), "got: {hint}");
    }

    #[test]
    fn output_format_hint_anyof_mentions_first_match() {
        let ft = FieldType::AnyOf {
            arms: vec![variant_arm_obj("int", FieldType::Int)],
        };
        let hint = ft.output_format_hint().expect("hint");
        assert!(hint.contains("first match"), "got: {hint}");
    }

    #[test]
    fn type_label_composes_oneof_inside_list() {
        let ft = FieldType::List(Box::new(FieldType::OneOf {
            arms: vec![
                variant_arm_obj("a", FieldType::Object(vec![])),
                variant_arm_obj("b", FieldType::Object(vec![])),
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["a".into(), "b".into()],
            }),
        }));
        let label = ft.type_label();
        assert!(label.starts_with("list[oneof["), "got: {label}");
        assert!(label.contains("kind"), "got: {label}");
    }

    #[test]
    fn field_type_oneof_serde_round_trip() {
        let ft = FieldType::OneOf {
            arms: vec![
                variant_arm_obj("a", FieldType::Object(vec![])),
                variant_arm_obj("b", FieldType::Object(vec![])),
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["a".into(), "b".into()],
            }),
        };
        let json = serde_json::to_string(&ft).unwrap();
        let restored: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(ft, restored);
    }

    #[test]
    fn field_type_anyof_serde_round_trip() {
        let ft = FieldType::AnyOf {
            arms: vec![
                variant_arm_obj("int", FieldType::Int),
                variant_arm_obj("str", FieldType::String),
            ],
        };
        let json = serde_json::to_string(&ft).unwrap();
        let restored: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(ft, restored);
    }

    #[test]
    fn field_value_variant_serde_round_trip() {
        let value = FieldValue::Variant {
            arm_index: 1,
            value: Box::new(FieldValue::Object(BTreeMap::from([(
                "toolName".into(),
                FieldValue::Str("ranked_items".into()),
            )]))),
        };
        let json = serde_json::to_string(&value).unwrap();
        let restored: FieldValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value, restored);
    }

    #[test]
    fn field_value_object_still_round_trips_with_variant_in_lattice() {
        // Guards the documented serde-untagged-ordering constraint —
        // a real Object whose declared fields aren't exactly
        // `{arm_index, value}` must still parse as Object, not Variant.
        let value = FieldValue::Object(BTreeMap::from([
            ("name".into(), FieldValue::Str("Alice".into())),
            ("age".into(), FieldValue::Int(30)),
            ("active".into(), FieldValue::Bool(true)),
        ]));
        let json = serde_json::to_string(&value).unwrap();
        let restored: FieldValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value, restored);
    }
}
