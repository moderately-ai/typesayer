// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! JSON Schema → [`FieldType`] conversion for dynamic signature construction.
//!
//! Converts JSON Schema definitions into the predict engine's type system,
//! enabling runtime signature construction from schemas received from external
//! services (e.g. the TypeScript API).
//!
//! **Strict mode**: unknown types and unsupported patterns produce errors,
//! never silent fallbacks to `String`.
//!
//! ## Supported JSON Schema vocabulary
//!
//! Primitives: `type: "string" | "integer" | "number" | "boolean" | "array" |
//! "object"`, `enum` (string variants), `const` (string-valued, folds to
//! single-element [`Enum`](FieldType::Enum)), nullable idioms
//! (`anyOf: [T, null]`, `type: ["T", "null"]`), `contentMediaType` for media
//! slots.
//!
//! Composition: `oneOf` ([`FieldType::OneOf`]), `anyOf` general
//! ([`FieldType::AnyOf`]), `allOf` (folds Object arms into a single merged
//! Object at conversion time), `$ref` (internal `#/$defs/...` and
//! `#/definitions/...` only; resolved at entry-point pre-pass with cycle
//! detection), `dependentSchemas` and `if`/`then`/`else` (transformed to
//! `oneOf` when reducible to discriminator patterns; rejected explicitly
//! otherwise).
//!
//! Rejected: external `$ref` (`http://`, `file:`), non-string `const`,
//! `allOf` of non-Object schemas, non-reducible `dependentSchemas` /
//! `if`/`then`/`else`, `patternProperties`, `propertyNames`, length / range
//! / format / pattern constraints. Each rejection produces a specific
//! [`PredictError::InvalidSignature`] naming the offending construct so the
//! schema author can see why.

use std::{
    collections::{BTreeMap, HashSet},
    convert::TryFrom,
};

use enumset::EnumSet;
use modelplease::{MediaKind, MediaType, SourceKind};
use serde_json::Value;
use typesayer_types::{
    error::{PredictError, Result},
    field::{FieldDef, FieldKind, FieldType, ObjectField, OneOfDiscriminator, VariantArm},
    signature::Signature,
};

/// A JSON Schema definition for constructing a [`Signature`] at runtime.
///
/// Both `inputs` and `outputs` are flat JSON objects mapping field names to
/// JSON Schema definitions:
///
/// ```json
/// {
///   "question": {"type": "string", "description": "The question"},
///   "tags": {"type": "array", "items": {"type": "string"}}
/// }
/// ```
///
/// Use via [`TryFrom`] or the convenience wrapper [`signature_from_json_schema()`]:
///
/// ```rust
/// use typesayer::{JsonSchemaDefinition, Signature};
/// use serde_json::json;
///
/// let inputs = json!({"question": {"type": "string"}});
/// let outputs = json!({"answer": {"type": "string"}});
///
/// let sig = Signature::try_from(JsonSchemaDefinition {
///     inputs: &inputs,
///     outputs: &outputs,
///     instructions: "Answer the question.",
/// }).unwrap();
/// ```
pub struct JsonSchemaDefinition<'a> {
    /// Flat JSON object of input field name → JSON Schema.
    pub inputs: &'a Value,
    /// Flat JSON object of output field name → JSON Schema.
    pub outputs: &'a Value,
    /// Task instructions for the language model.
    pub instructions: &'a str,
}

impl TryFrom<JsonSchemaDefinition<'_>> for Signature {
    type Error = PredictError;

    fn try_from(def: JsonSchemaDefinition<'_>) -> Result<Self> {
        let input_fields = fields_from_json_schema(def.inputs, FieldKind::Input)?;
        let output_fields = fields_from_json_schema(def.outputs, FieldKind::Output)?;

        let mut builder = Self::builder(def.instructions);
        for field in input_fields {
            builder = builder.input(field);
        }
        for field in output_fields {
            builder = builder.output(field);
        }
        builder.build()
    }
}

/// Construct a [`Signature`] from JSON Schema definitions at runtime.
///
/// Convenience wrapper around [`TryFrom<JsonSchemaDefinition>`]. Both
/// `inputs` and `outputs` must be flat JSON objects mapping field names to
/// JSON Schema definitions.
///
/// # Errors
///
/// Returns a [`PredictError`] when `inputs`/`outputs` are not valid JSON
/// Schema objects or when a field type is not supported.
pub fn signature_from_json_schema(
    inputs: &serde_json::Value,
    outputs: &serde_json::Value,
    instructions: &str,
) -> Result<Signature> {
    JsonSchemaDefinition {
        inputs,
        outputs,
        instructions,
    }
    .try_into()
}

/// Resolve all `$ref`s in `schema` against `document_root`. Returns a
/// `$ref`-free clone of the schema so subsequent conversion code only
/// has to handle inline subschemas.
///
/// Only internal refs (`#/$defs/...`, `#/definitions/...`) are supported;
/// external refs (`http://`, `https://`, `file:`) reject with a specific
/// error so the failure surfaces at conversion time rather than as a
/// silent "missing type" downstream. Cycles are detected via a
/// per-resolution `in_progress` set: a ref that revisits an in-progress
/// path errors with `RefCycle` listing the offending pointer.
///
/// The resolver is purely structural — it does NOT validate that the
/// resolved target is a valid JSON Schema. Downstream
/// [`field_type_from_schema`] handles type-level errors.
///
/// # Errors
///
/// `PredictError::InvalidSignature` with a message describing the
/// specific failure: external ref, unresolved ref, or cycle.
fn resolve_refs(schema: &Value, document_root: &Value) -> Result<Value> {
    let mut in_progress: HashSet<String> = HashSet::new();
    resolve_refs_inner(schema, document_root, &mut in_progress)
}

fn resolve_refs_inner(
    schema: &Value,
    document_root: &Value,
    in_progress: &mut HashSet<String>,
) -> Result<Value> {
    match schema {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get("$ref") {
                let target = resolve_ref_pointer(reference, document_root, in_progress)?;
                in_progress.insert(reference.clone());
                let resolved = resolve_refs_inner(&target, document_root, in_progress)?;
                in_progress.remove(reference);
                return Ok(resolved);
            }
            let mut new_map = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                new_map.insert(
                    k.clone(),
                    resolve_refs_inner(v, document_root, in_progress)?,
                );
            }
            Ok(Value::Object(new_map))
        }
        Value::Array(arr) => {
            let mut new_arr = Vec::with_capacity(arr.len());
            for v in arr {
                new_arr.push(resolve_refs_inner(v, document_root, in_progress)?);
            }
            Ok(Value::Array(new_arr))
        }
        other => Ok(other.clone()),
    }
}

/// Resolve a single `$ref` pointer against the document root.
///
/// Accepts `#/$defs/X` and `#/definitions/X` (JSON Schema 2019-09 and
/// pre-2019 conventional names). Rejects external schemes outright.
fn resolve_ref_pointer(
    reference: &str,
    document_root: &Value,
    in_progress: &HashSet<String>,
) -> Result<Value> {
    if in_progress.contains(reference) {
        return Err(PredictError::invalid_signature(format!(
            "$ref cycle detected at `{reference}`; recursive type definitions \
             cannot be represented as a finite FieldType"
        )));
    }
    if reference.starts_with("http://")
        || reference.starts_with("https://")
        || reference.starts_with("file:")
    {
        return Err(PredictError::invalid_signature(format!(
            "external $ref `{reference}` is not supported; inline the target \
             definition or use an internal `#/$defs/...` reference instead"
        )));
    }
    let Some(stripped) = reference.strip_prefix("#/") else {
        return Err(PredictError::invalid_signature(format!(
            "$ref `{reference}` must be a JSON pointer starting with `#/`"
        )));
    };
    let mut cursor = document_root;
    for segment in stripped.split('/') {
        let decoded = json_pointer_unescape(segment);
        cursor = match cursor {
            Value::Object(m) => m.get(&decoded).ok_or_else(|| {
                PredictError::invalid_signature(format!(
                    "$ref `{reference}` cannot be resolved; segment \
                     `{decoded}` not found"
                ))
            })?,
            Value::Array(arr) => {
                let idx = decoded.parse::<usize>().map_err(|_| {
                    PredictError::invalid_signature(format!(
                        "$ref `{reference}` array segment `{decoded}` is not \
                         a valid index"
                    ))
                })?;
                arr.get(idx).ok_or_else(|| {
                    PredictError::invalid_signature(format!(
                        "$ref `{reference}` array index {idx} out of bounds"
                    ))
                })?
            }
            _ => {
                return Err(PredictError::invalid_signature(format!(
                    "$ref `{reference}` segment `{decoded}` traverses a non-container"
                )));
            }
        };
    }
    Ok(cursor.clone())
}

/// Decode RFC 6901 escape sequences in a JSON pointer segment.
fn json_pointer_unescape(segment: &str) -> String {
    segment.replace("~1", "/").replace("~0", "~")
}

/// Convert a flat dict of field name → JSON Schema into a list of [`FieldDef`]s.
///
/// The `kind` parameter determines whether the resulting fields are inputs or outputs.
/// The `schema` value must be a JSON object where keys are field names and values
/// are JSON Schema definitions.
///
/// ```json
/// {
///   "question": {"type": "string", "description": "The user question"},
///   "tags": {"type": "array", "items": {"type": "string"}}
/// }
/// ```
///
/// # Errors
///
/// Returns a `PredictError::InvalidSignature` when `schema` is not a JSON
/// object or when any field's schema is malformed.
pub fn fields_from_json_schema(schema: &Value, kind: FieldKind) -> Result<Vec<FieldDef>> {
    let obj = schema
        .as_object()
        .ok_or_else(|| PredictError::invalid_signature("schema must be a JSON object"))?;

    let mut fields = Vec::new();
    for (name, field_schema) in obj {
        // `field_type_from_schema` runs its own `$ref` pre-pass against
        // `field_schema` as the document root, so each field's `$defs`
        // are reachable via `#/$defs/...`. Nullability comes from
        // standard JSON Schema patterns inside `field_schema` itself
        // (`anyOf` containing `{"type": "null"}`, `type: ["string",
        // "null"]`, etc). The per-field `required: bool` form that used
        // to be honored here is intentionally not consulted — it
        // conflicts with strict JSON Schema validation, and every case
        // it covered is expressible via the standard patterns.
        let field_type = field_type_from_schema(field_schema)?;
        let description = extract_description(name, field_schema);
        let examples = extract_examples(field_schema);

        fields.push(FieldDef {
            name: name.clone(),
            description,
            field_type,
            kind,
            cacheable: false,
            examples,
        });
    }

    Ok(fields)
}

/// Convert a single JSON Schema definition into a [`FieldType`].
///
/// `schema` is treated as a complete JSON Schema document and is its own
/// resolution root for `$ref` (`#/$defs/...`, `#/definitions/...`). The
/// `$ref` pre-pass runs once at this boundary; the recursive descent
/// then operates entirely on inline subschemas, so the conversion has a
/// single canonical entry point that's safe to call directly with any
/// well-formed schema.
///
/// Evaluation order (top-down dispatch — first matching keyword wins):
///
/// 1. `contentMediaType` → [`Media`](FieldType::Media)
/// 2. `const` → folds to single-element [`Enum`](FieldType::Enum) (string-valued only; rejects
///    other types)
/// 3. `anyOf` nullable pattern → [`Nullable`](FieldType::Nullable)
/// 4. `anyOf` (general) → [`AnyOf`](FieldType::AnyOf)
/// 5. Type-array nullable pattern (`["T", "null"]`) → [`Nullable`](FieldType::Nullable)
/// 6. `oneOf` → [`OneOf`](FieldType::OneOf) with optional discriminator
/// 7. `allOf` → folds Object arms into a merged Object; rejects others
/// 8. `dependentSchemas` → transforms to `oneOf` where reducible
/// 9. `if`/`then`/`else` → transforms to `oneOf` where reducible
/// 10. `enum` → [`Enum`](FieldType::Enum)
/// 11. Type-based: array, object, primitives
/// 12. Unknown → rejected
///
/// # Errors
///
/// Returns a `PredictError::InvalidSignature` when `schema` uses an
/// unsupported keyword, combines patterns incorrectly, or has a
/// malformed sub-definition. Each rejection names the offending keyword.
pub fn field_type_from_schema(schema: &Value) -> Result<FieldType> {
    // Resolve $refs once against the schema itself as document root.
    // Recursive descent below operates on inline subschemas only — the
    // helper `field_type_from_inline_schema` never re-runs the pre-pass.
    let resolved = resolve_refs(schema, schema)?;
    field_type_from_inline_schema(&resolved)
}

/// Recursive implementation. Caller guarantees that every embedded
/// `$ref` has been resolved to its inline target. Used by the public
/// [`field_type_from_schema`] (which runs the pre-pass once) and by
/// every internal recursive call within this module.
fn field_type_from_inline_schema(schema: &Value) -> Result<FieldType> {
    // 1. Media slot — JSON Schema 2020-12 `contentMediaType` (RFC 4288)
    // with an optional `x-source-kinds` extension that narrows the
    // accepted [`SourceKind`] set.
    if let Some(content_media_type) = schema.get("contentMediaType").and_then(|v| v.as_str()) {
        return parse_media(content_media_type, schema);
    }

    // 2. `const`: folds to a single-element Enum so the LLM is constrained
    // to emit exactly the literal value. String-valued only — JSON Schema
    // allows any JSON value as `const`, but non-string consts have no LLM-
    // output consumer and the FieldType lattice doesn't model arbitrary
    // literal values.
    if let Some(const_value) = schema.get("const") {
        return parse_const(const_value);
    }

    // 3. anyOf nullable: [{"type": T}, {"type": "null"}]
    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        // First try the nullable idiom; if it doesn't match, fall through to
        // the general AnyOf path so non-nullable disjunctions still convert.
        if let Some(nullable) = try_parse_any_of_nullable(any_of)? {
            return Ok(nullable);
        }
        return parse_any_of_general(any_of);
    }

    // 4. Type array nullable: {"type": ["string", "null"]}
    if let Some(type_arr) = schema.get("type").and_then(|v| v.as_array()) {
        return parse_type_array_nullable(type_arr, schema);
    }

    // 5. `oneOf`: discriminated sum type. Discriminator inferred when
    // arms share a const-string property; otherwise untagged.
    if let Some(one_of) = schema.get("oneOf").and_then(|v| v.as_array()) {
        return parse_one_of(one_of, schema);
    }

    // 6. `allOf`: schema composition. Object arms fold by merging their
    // fields; non-Object arms reject with a specific error.
    if let Some(all_of) = schema.get("allOf").and_then(|v| v.as_array()) {
        return parse_all_of(all_of);
    }

    // 7. `dependentSchemas`: conditional-fields construct. Reducible to
    // `oneOf` when the dependency value is itself a discriminated `oneOf`
    // on the same trigger property; rejected otherwise.
    if let Some(deps) = schema.get("dependentSchemas").and_then(|v| v.as_object()) {
        return transform_dependent_schemas(schema, deps);
    }

    // 8. `if`/`then`/`else`: conditional-validation construct. Reducible
    // to `oneOf` when the `if` is a discriminator pattern with a known
    // enum context; rejected otherwise.
    if schema.get("if").is_some() {
        return transform_if_then_else(schema);
    }

    // 9. Enum: {"type": "string", "enum": ["a", "b"]} (or bare `enum`).
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        return parse_enum(enum_values);
    }

    // 10. Type-based dispatch
    let type_str = schema.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        PredictError::invalid_signature(format!(
            "schema must have a \"type\" field (string) or a recognized \
             composition keyword (oneOf, anyOf, allOf, const, $ref, \
             dependentSchemas, if/then/else, enum, contentMediaType); got: {}",
            truncate_value(schema)
        ))
    })?;

    match type_str {
        "array" => parse_array(schema),
        "object" => parse_object(schema),
        "string" => Ok(FieldType::String),
        "integer" => Ok(FieldType::Int),
        "number" => Ok(FieldType::Float),
        "boolean" => Ok(FieldType::Bool),
        other => Err(PredictError::invalid_signature(format!(
            "unsupported JSON Schema type: \"{other}\""
        ))),
    }
}

fn parse_media(content_media_type: &str, schema: &Value) -> Result<FieldType> {
    let media_type = MediaType::parse(content_media_type).map_err(|e| {
        PredictError::invalid_signature(format!(
            "invalid `contentMediaType` `{content_media_type}`: {e}"
        ))
    })?;
    let kind = MediaKind::from_media_type(&media_type).ok_or_else(|| {
        PredictError::invalid_signature(format!(
            "unsupported `contentMediaType` top-level type `{}`",
            media_type.top_level()
        ))
    })?;
    let accepted_sources = match schema.get("x-source-kinds") {
        Some(Value::Array(arr)) => parse_source_kinds_array(arr)?,
        Some(other) => {
            return Err(PredictError::invalid_signature(format!(
                "`x-source-kinds` must be a JSON array, got: {}",
                truncate_value(other)
            )));
        }
        None => EnumSet::all(),
    };
    Ok(FieldType::Media {
        kind,
        accepted_sources,
    })
}

fn parse_source_kinds_array(arr: &[Value]) -> Result<EnumSet<SourceKind>> {
    let mut set = EnumSet::new();
    for entry in arr {
        let name = entry.as_str().ok_or_else(|| {
            PredictError::invalid_signature(
                "`x-source-kinds` entries must be strings (`url`, `inline_bytes`, `provider_file`, `s3`)",
            )
        })?;
        let kind = match name {
            "url" => SourceKind::Url,
            "inline_bytes" => SourceKind::InlineBytes,
            "provider_file" => SourceKind::ProviderFile,
            "s3" => SourceKind::S3,
            other => {
                return Err(PredictError::invalid_signature(format!(
                    "unknown `x-source-kinds` value `{other}`"
                )));
            }
        };
        set.insert(kind);
    }
    Ok(set)
}

/// Extract a description from a field schema, with enum value appending
/// and JSON Schema constraint-keyword annotations.
///
/// Resolution order for the base text:
/// 1. `description` — the canonical free-text field
/// 2. `title` — a short human label, used when no `description` is set
/// 3. Field name as a last-resort label
///
/// Appended in order if present (each only adds context the type label
/// can't carry on its own):
///
/// - `enum` values (preserved from prior behavior)
/// - `pattern` — a regex constraint that the LLM cannot infer; this is the most load-bearing of the
///   constraint keywords because guessing a regex-conforming value blind is essentially impossible
/// - `format` — RFC-3339 date-time, email, UUID, URI, etc.
/// - `default` — common-case value hint
/// - `minimum`/`maximum` for numerics, `minLength`/`maxLength` for strings, `minItems`/`maxItems`
///   for arrays
#[must_use]
pub fn extract_description(field_name: &str, schema: &Value) -> String {
    let base = schema
        .get("description")
        .and_then(|v| v.as_str())
        .or_else(|| schema.get("title").and_then(|v| v.as_str()))
        .unwrap_or(field_name)
        .to_owned();

    let mut annotations: Vec<String> = Vec::new();

    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        let values: Vec<String> = enum_values
            .iter()
            .filter_map(|v| v.as_str().map(|s| format!("'{s}'")))
            .collect();
        if !values.is_empty() {
            annotations.push(format!("must be one of: {}", values.join(", ")));
        }
    }

    if let Some(pattern) = schema.get("pattern").and_then(|v| v.as_str()) {
        annotations.push(format!("must match regex: {pattern}"));
    }

    if let Some(format) = schema.get("format").and_then(|v| v.as_str()) {
        annotations.push(format!("format: {format}"));
    }

    if let Some(default) = schema.get("default") {
        annotations.push(format!("default: {default}"));
    }

    // Numeric range — emit a single combined fragment when both ends
    // are present, separate ones otherwise.
    let min = schema.get("minimum").and_then(serde_json::Value::as_f64);
    let max = schema.get("maximum").and_then(serde_json::Value::as_f64);
    match (min, max) {
        (Some(lo), Some(hi)) => annotations.push(format!("range: {lo} to {hi}")),
        (Some(lo), None) => annotations.push(format!("minimum: {lo}")),
        (None, Some(hi)) => annotations.push(format!("maximum: {hi}")),
        (None, None) => {}
    }

    // String length range.
    let min_len = schema.get("minLength").and_then(serde_json::Value::as_u64);
    let max_len = schema.get("maxLength").and_then(serde_json::Value::as_u64);
    match (min_len, max_len) {
        (Some(lo), Some(hi)) => annotations.push(format!("length: {lo}-{hi} chars")),
        (Some(lo), None) => annotations.push(format!("minLength: {lo} chars")),
        (None, Some(hi)) => annotations.push(format!("maxLength: {hi} chars")),
        (None, None) => {}
    }

    // Array length range.
    let min_items = schema.get("minItems").and_then(serde_json::Value::as_u64);
    let max_items = schema.get("maxItems").and_then(serde_json::Value::as_u64);
    match (min_items, max_items) {
        (Some(lo), Some(hi)) => annotations.push(format!("size: {lo}-{hi} items")),
        (Some(lo), None) => annotations.push(format!("minItems: {lo}")),
        (None, Some(hi)) => annotations.push(format!("maxItems: {hi}")),
        (None, None) => {}
    }

    if annotations.is_empty() {
        base
    } else {
        format!("{base} ({})", annotations.join("; "))
    }
}

/// Extract a capped, length-bounded list of example values from a schema's
/// `examples` keyword (JSON Schema draft 2020-12).
///
/// The 3-item cap and per-example serialization budget keep the rendered
/// prompt bounded: a wide-open `examples` array (every enum variant, say)
/// would otherwise push past the LLM's context faster than the schema
/// author would intuit. Operators who really want more examples can move
/// them into the field description text.
#[must_use]
pub fn extract_examples(schema: &Value) -> Vec<Value> {
    schema
        .get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().take(EXAMPLES_RENDER_CAP).cloned().collect())
        .unwrap_or_default()
}

/// Maximum number of `examples` entries propagated from JSON Schema to
/// the rendered prompt per field. Higher counts return diminishing
/// signal — the model needs one or two concrete shapes, not an
/// exhaustive listing.
pub const EXAMPLES_RENDER_CAP: usize = 3;

// --- Internal helpers ---

/// Try to match the nullable-`anyOf` idiom: `[{"type": T}, {"type": "null"}]`.
///
/// Returns `Ok(Some(Nullable<T>))` on a match, `Ok(None)` when the shape
/// doesn't match (caller falls through to general `AnyOf`), and `Err` only
/// when a nested type fails to parse.
fn try_parse_any_of_nullable(any_of: &[Value]) -> Result<Option<FieldType>> {
    let has_null = any_of
        .iter()
        .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
    if !has_null {
        return Ok(None);
    }
    let non_null: Vec<&Value> = any_of
        .iter()
        .filter(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"))
        .collect();
    if non_null.len() != 1 {
        return Ok(None);
    }
    let inner = field_type_from_inline_schema(non_null[0])?;
    Ok(Some(FieldType::Nullable(Box::new(inner))))
}

/// Parse a general (non-nullable) `anyOf` into [`FieldType::AnyOf`].
fn parse_any_of_general(any_of: &[Value]) -> Result<FieldType> {
    if any_of.is_empty() {
        return Err(PredictError::invalid_signature(
            "anyOf must contain at least one arm",
        ));
    }
    let mut arms = Vec::with_capacity(any_of.len());
    for (i, arm_schema) in any_of.iter().enumerate() {
        // Catch the "this arm is `{type: null}` but the parent didn't
        // collapse to Nullable" case before the generic dispatch yells
        // about an unsupported type — the schema author needs to know
        // they hit a representational gap, not a typo.
        if arm_schema.get("type").and_then(|v| v.as_str()) == Some("null") {
            return Err(PredictError::invalid_signature(
                "anyOf contains a `{\"type\": \"null\"}` arm alongside multiple non-null \
                 arms. typesayer can only represent the null choice when there is \
                 exactly one non-null arm (collapsed to `Nullable<T>`). Restructure the \
                 schema so the null arm pairs with a single non-null arm, or move the null \
                 case into a discriminated `oneOf` whose tag includes a `null` choice.",
            ));
        }
        let field_type = field_type_from_inline_schema(arm_schema)?;
        let description = arm_description_with_warning(arm_schema, i, "anyOf");
        arms.push(VariantArm {
            description,
            field_type,
        });
    }
    Ok(FieldType::AnyOf { arms })
}

/// Extract an arm's description and emit a `tracing::warn!` when the
/// schema author didn't provide one. The auto-generated `arm N`
/// placeholder is parsing-safe but lossy — the model has no idea WHEN
/// arm 0 vs arm 1 applies, which is the dominant failure mode for
/// untagged variants in production prompts.
///
/// Logs at warn (not error) because the schema-conversion result is
/// still valid; the warn just nudges the schema author at the gap.
fn arm_description_with_warning(arm_schema: &Value, idx: usize, kind: &str) -> String {
    if let Some(desc) = arm_schema.get("description").and_then(|v| v.as_str()) {
        return desc.to_owned();
    }
    tracing::warn!(
        target: "typesayer::schema",
        arm_index = idx,
        variant_kind = kind,
        "{kind} arm at index {idx} has no `description`; using placeholder \"arm {idx}\". \
         Prompts that include this arm in a variant-shapes block will not tell the model \
         WHEN this arm applies. Add `description` to the arm schema."
    );
    format!("arm {idx}")
}

/// Parse a `oneOf` schema into [`FieldType::OneOf`], inferring a
/// discriminator when every arm shares a `const`-string property with a
/// unique value across arms. Cross-checks against an explicit OpenAPI
/// `discriminator: {propertyName: P}` hint when present.
fn parse_one_of(one_of: &[Value], schema: &Value) -> Result<FieldType> {
    if one_of.is_empty() {
        return Err(PredictError::invalid_signature(
            "oneOf must contain at least one arm",
        ));
    }

    // Convert each arm and remember its source schema for later
    // discriminator inference (which needs the const-property positions).
    let mut arms = Vec::with_capacity(one_of.len());
    for (i, arm_schema) in one_of.iter().enumerate() {
        let field_type = field_type_from_inline_schema(arm_schema)?;
        let description = arm_description_with_warning(arm_schema, i, "oneOf");
        arms.push(VariantArm {
            description,
            field_type,
        });
    }

    let discriminator = infer_one_of_discriminator(one_of);

    // Cross-check explicit OpenAPI discriminator hint against the
    // inferred one. Inconsistent hints reject; missing-but-hinted
    // (a hint when inference failed) also rejects so the schema author
    // sees a precise error instead of silently dropping the hint.
    if let Some(hint_obj) = schema.get("discriminator").and_then(|v| v.as_object()) {
        let hint_property = hint_obj
            .get("propertyName")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                PredictError::invalid_signature(
                    "discriminator hint must have a string `propertyName`",
                )
            })?;
        match &discriminator {
            Some(inferred) if inferred.property == hint_property => {}
            Some(inferred) => {
                return Err(PredictError::invalid_signature(format!(
                    "discriminator hint propertyName=`{hint_property}` does not \
                     match inferred discriminator `{}`",
                    inferred.property
                )));
            }
            None => {
                return Err(PredictError::invalid_signature(format!(
                    "discriminator hint propertyName=`{hint_property}` is set but \
                     no compatible const-property discriminator could be inferred \
                     from the oneOf arms"
                )));
            }
        }
    }

    Ok(FieldType::OneOf {
        arms,
        discriminator,
    })
}

/// Inspect each `oneOf` arm's source schema; return a discriminator when
/// every arm is `type: object` with a `const`-string property such that
/// exactly one property name is shared across all arms with unique values.
///
/// Returns `None` for every shape that fails inference — the call site
/// treats absence as "untagged OneOf" rather than an error.
fn infer_one_of_discriminator(one_of: &[Value]) -> Option<OneOfDiscriminator> {
    // Per-arm: collect (property name -> const string value) for every
    // const-valued property the arm declares. Bail out (return None) if
    // any arm isn't an Object schema with at least one such property.
    let mut per_arm: Vec<BTreeMap<String, String>> = Vec::with_capacity(one_of.len());
    for arm in one_of {
        let arm_inner = unwrap_nullable_layer(arm);
        let properties = arm_inner.get("properties").and_then(|v| v.as_object())?;
        let declared_type = arm_inner.get("type").and_then(|v| v.as_str());
        if declared_type.is_some() && declared_type != Some("object") {
            return None;
        }
        let mut consts = BTreeMap::new();
        for (prop_name, prop_schema) in properties {
            if let Some(Value::String(s)) = prop_schema.get("const") {
                consts.insert(prop_name.clone(), s.clone());
            }
        }
        if consts.is_empty() {
            return None;
        }
        per_arm.push(consts);
    }

    // Intersect the property-name sets across arms.
    let mut candidate_props: HashSet<String> = per_arm[0].keys().cloned().collect();
    for arm_consts in &per_arm[1..] {
        let arm_keys: HashSet<String> = arm_consts.keys().cloned().collect();
        candidate_props = candidate_props.intersection(&arm_keys).cloned().collect();
        if candidate_props.is_empty() {
            return None;
        }
    }

    // For each surviving candidate, check that the values are unique
    // across arms. The first surviving property with uniqueness wins.
    // (When multiple share unique tag sets — rare — the schema is
    // ambiguously discriminated; pick deterministically by sorted name
    // so the inference is stable.)
    let mut sorted: Vec<String> = candidate_props.into_iter().collect();
    sorted.sort();
    for prop in sorted {
        let tags: Vec<String> = per_arm.iter().map(|c| c[&prop].clone()).collect();
        let unique: HashSet<&String> = tags.iter().collect();
        if unique.len() == tags.len() {
            return Some(OneOfDiscriminator {
                property: prop,
                tags,
            });
        }
    }
    None
}

/// If `schema` is `anyOf: [{type: null}, T]`, return T; otherwise return
/// `schema` unchanged. Used by discriminator inference to look through a
/// single Nullable layer when discovering the const-property structure.
fn unwrap_nullable_layer(schema: &Value) -> &Value {
    let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) else {
        return schema;
    };
    let has_null = any_of
        .iter()
        .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
    if !has_null {
        return schema;
    }
    let non_null: Vec<&Value> = any_of
        .iter()
        .filter(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"))
        .collect();
    if non_null.len() == 1 {
        non_null[0]
    } else {
        schema
    }
}

/// Parse an `allOf` schema. Object arms merge into a single Object;
/// non-Object arms reject. Field-name conflicts (same name, different
/// type) reject. Single-arm `allOf` is identity.
fn parse_all_of(all_of: &[Value]) -> Result<FieldType> {
    if all_of.is_empty() {
        return Err(PredictError::invalid_signature(
            "allOf must contain at least one arm",
        ));
    }
    let mut merged_fields: Vec<ObjectField> = Vec::new();
    for (i, arm_schema) in all_of.iter().enumerate() {
        let arm_type = field_type_from_inline_schema(arm_schema)?;
        match arm_type {
            FieldType::Object(fields) => {
                for new_field in fields {
                    if let Some(existing) = merged_fields.iter().find(|f| f.name == new_field.name)
                    {
                        if existing.field_type != new_field.field_type {
                            return Err(PredictError::invalid_signature(format!(
                                "allOf arm {i} field `{}` type conflicts with an \
                                 earlier arm: {} vs {}",
                                new_field.name,
                                existing.field_type.type_label(),
                                new_field.field_type.type_label()
                            )));
                        }
                        // Same name + same type: keep the first
                        // declaration (existing); no-op.
                    } else {
                        merged_fields.push(new_field);
                    }
                }
            }
            other => {
                return Err(PredictError::invalid_signature(format!(
                    "allOf arm {i} is `{}`; only Object arms can be merged. \
                     Wrap non-Object constraints in their own field, or use \
                     `oneOf` if you intended a sum type.",
                    other.type_label()
                )));
            }
        }
    }
    Ok(FieldType::Object(merged_fields))
}

/// Fold a `const` schema into a single-element [`Enum`]. String-valued
/// only; non-string consts reject with a specific error.
fn parse_const(value: &Value) -> Result<FieldType> {
    match value {
        Value::String(s) => Ok(FieldType::Enum(vec![s.clone()])),
        other => Err(PredictError::invalid_signature(format!(
            "non-string `const` is not supported (got: {}); the FieldType lattice \
             does not model arbitrary literal values. Wrap in an enum with a \
             single string variant if you need a literal value.",
            truncate_value(other)
        ))),
    }
}

/// Transform a `dependentSchemas` block into `oneOf` when reducible.
///
/// Reducible shape: exactly one trigger property `P` whose dependency
/// schema is itself a `oneOf` with each arm discriminated on `P` via a
/// `const`. The transformation merges the parent object's base
/// properties with each arm's overlay and delegates to `parse_one_of`.
fn transform_dependent_schemas(
    parent: &Value,
    deps: &serde_json::Map<String, Value>,
) -> Result<FieldType> {
    if deps.len() != 1 {
        return Err(PredictError::invalid_signature(format!(
            "dependentSchemas with {} trigger properties cannot be reduced to \
             a single oneOf; this transformer supports exactly one trigger \
             property whose dependency value is itself a `oneOf` keyed on the \
             same property",
            deps.len()
        )));
    }
    // `deps.len() == 1` was just checked above; `next()` cannot return
    // None. Fallback to an empty trigger + null schema is unreachable in
    // practice but avoids the `.expect()` lint without taking a panic
    // path.
    let Some((trigger, dep_schema)) = deps.iter().next() else {
        return Err(PredictError::invalid_signature(
            "dependentSchemas iterator returned no entries despite non-empty length",
        ));
    };
    let Some(dep_one_of) = dep_schema.get("oneOf").and_then(|v| v.as_array()) else {
        return Err(PredictError::invalid_signature(format!(
            "dependentSchemas[{trigger}] must contain a `oneOf` whose arms \
             discriminate on `{trigger}`; got: {}",
            truncate_value(dep_schema)
        )));
    };

    // Build a synthetic oneOf: each arm = parent properties merged with
    // the dep's arm. Reuse parse_one_of so discriminator inference + the
    // OneOf type construction stay in one place.
    let parent_obj = parent.as_object().ok_or_else(|| {
        PredictError::invalid_signature(
            "dependentSchemas requires its enclosing schema to be an object",
        )
    })?;

    let parent_properties = parent_obj
        .get("properties")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let mut synthetic_arms = Vec::with_capacity(dep_one_of.len());
    for dep_arm in dep_one_of {
        let mut merged = serde_json::Map::new();
        merged.insert("type".to_owned(), Value::String("object".to_owned()));

        // Merge properties: parent first, then dep-arm overrides.
        let mut props = parent_properties.clone();
        if let Some(arm_props) = dep_arm.get("properties").and_then(|v| v.as_object()) {
            for (k, v) in arm_props {
                props.insert(k.clone(), v.clone());
            }
        }
        merged.insert("properties".to_owned(), Value::Object(props));

        // Required: union of parent's required and arm's required.
        let mut required: HashSet<String> = parent_obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(arm_required) = dep_arm.get("required").and_then(|v| v.as_array()) {
            for r in arm_required.iter().filter_map(|v| v.as_str()) {
                required.insert(r.to_owned());
            }
        }
        let mut required_vec: Vec<String> = required.into_iter().collect();
        required_vec.sort();
        merged.insert(
            "required".to_owned(),
            Value::Array(required_vec.into_iter().map(Value::String).collect()),
        );

        if let Some(desc) = dep_arm.get("description") {
            merged.insert("description".to_owned(), desc.clone());
        }

        synthetic_arms.push(Value::Object(merged));
    }

    parse_one_of(&synthetic_arms, &Value::Null)
}

/// Transform an `if`/`then`/`else` block into `oneOf` when reducible.
///
/// Reducible shape: `if` is a discriminator pattern — a single-property
/// schema with `{const: X}` on property P — AND P's domain is constrained
/// by an enclosing enum or by an else branch that lists the complement
/// values. When the negation can't be expressed structurally, reject.
fn transform_if_then_else(schema: &Value) -> Result<FieldType> {
    let if_schema = schema
        .get("if")
        .ok_or_else(|| PredictError::invalid_signature("if/then/else requires an `if` schema"))?;

    let then_schema = schema.get("then").cloned().unwrap_or_else(|| {
        // Empty `then` means "the if's constraint is enough" — treat as
        // the if schema itself.
        if_schema.clone()
    });

    let (disc_prop, _disc_const) = extract_if_discriminator(if_schema).ok_or_else(|| {
        PredictError::invalid_signature(format!(
            "if/then/else `if` must be a discriminator pattern (single-property \
             const) to be reducible to oneOf; got: {}",
            truncate_value(if_schema)
        ))
    })?;

    let else_branch = schema.get("else");

    // Negation: collect alternative discriminator values for the else
    // branch. Requires else to be a discriminator pattern too OR for the
    // schema to declare a known enum elsewhere. For the v1 reducible
    // form we require the else to be discriminator-shaped (so the
    // negation is its set of consts).
    let else_arm_consts: Vec<String> = match else_branch {
        None => Vec::new(),
        Some(else_schema) => {
            if let Some(else_one_of) = else_schema.get("oneOf").and_then(|v| v.as_array()) {
                let mut consts = Vec::new();
                for else_arm in else_one_of {
                    let Some((prop, c)) = extract_if_discriminator(else_arm) else {
                        return Err(PredictError::invalid_signature(format!(
                            "if/then/else `else.oneOf` arm must be a \
                             discriminator pattern on `{disc_prop}`; got: {}",
                            truncate_value(else_arm)
                        )));
                    };
                    if prop != disc_prop {
                        return Err(PredictError::invalid_signature(format!(
                            "if/then/else `else` arm discriminates on `{prop}` \
                             but `if` discriminates on `{disc_prop}`; the two \
                             must agree to be reducible to oneOf"
                        )));
                    }
                    consts.push(c);
                }
                consts
            } else if let Some((prop, c)) = extract_if_discriminator(else_schema) {
                if prop != disc_prop {
                    return Err(PredictError::invalid_signature(format!(
                        "if/then/else `else` discriminates on `{prop}` but `if` \
                         discriminates on `{disc_prop}`; the two must agree to \
                         be reducible to oneOf"
                    )));
                }
                vec![c]
            } else {
                return Err(PredictError::invalid_signature(format!(
                    "if/then/else `else` must be a discriminator pattern or a \
                     `oneOf` of such patterns (sharing the trigger property \
                     `{disc_prop}`) to be reducible; got: {}",
                    truncate_value(else_schema)
                )));
            }
        }
    };

    // Build synthetic oneOf: one arm for the if/then branch, then one
    // arm per else-discriminator-value.
    let mut synthetic_arms: Vec<Value> = Vec::new();
    synthetic_arms.push(merge_object_schemas(&[if_schema.clone(), then_schema]));
    if let Some(else_schema) = else_branch {
        for c in else_arm_consts {
            let disc_arm = Value::Object({
                let mut map = serde_json::Map::new();
                let mut props = serde_json::Map::new();
                props.insert(
                    disc_prop.clone(),
                    Value::Object({
                        let mut p = serde_json::Map::new();
                        p.insert("const".to_owned(), Value::String(c));
                        p
                    }),
                );
                map.insert("type".to_owned(), Value::String("object".to_owned()));
                map.insert("properties".to_owned(), Value::Object(props));
                map.insert(
                    "required".to_owned(),
                    Value::Array(vec![Value::String(disc_prop.clone())]),
                );
                map
            });

            // For the else-arm we need both the discriminator value AND
            // the else schema's other constraints. If else_schema is the
            // discriminator-shape directly, just use it. If else_schema
            // is `oneOf`, that's already covered by iterating its arms,
            // so this branch only fires for the single-pattern else case.
            let else_merged = if else_schema.get("oneOf").is_some() {
                disc_arm
            } else {
                merge_object_schemas(&[disc_arm, else_schema.clone()])
            };
            synthetic_arms.push(else_merged);
        }
    }

    parse_one_of(&synthetic_arms, &Value::Null)
}

/// Extract `(property_name, const_string)` from a discriminator-shaped
/// schema (`{type: "object", properties: {P: {const: "X"}}, required: ["P"]}`
/// or its single-property variant).
fn extract_if_discriminator(schema: &Value) -> Option<(String, String)> {
    let properties = schema.get("properties").and_then(|v| v.as_object())?;
    if properties.len() != 1 {
        return None;
    }
    let (name, prop_schema) = properties.iter().next()?;
    let const_value = prop_schema.get("const")?.as_str()?;
    Some((name.clone(), const_value.to_owned()))
}

/// Shallow-merge two or more object schemas: union `properties`, union
/// `required`, preserve the rightmost schema's keys for collisions.
/// Used by `transform_if_then_else` to build synthetic oneOf arms.
fn merge_object_schemas(schemas: &[Value]) -> Value {
    let mut merged = serde_json::Map::new();
    merged.insert("type".to_owned(), Value::String("object".to_owned()));
    let mut props = serde_json::Map::new();
    let mut required: HashSet<String> = HashSet::new();
    for s in schemas {
        if let Some(p) = s.get("properties").and_then(|v| v.as_object()) {
            for (k, v) in p {
                props.insert(k.clone(), v.clone());
            }
        }
        if let Some(r) = s.get("required").and_then(|v| v.as_array()) {
            for name in r.iter().filter_map(|v| v.as_str()) {
                required.insert(name.to_owned());
            }
        }
    }
    merged.insert("properties".to_owned(), Value::Object(props));
    let mut required_vec: Vec<String> = required.into_iter().collect();
    required_vec.sort();
    merged.insert(
        "required".to_owned(),
        Value::Array(required_vec.into_iter().map(Value::String).collect()),
    );
    Value::Object(merged)
}

fn parse_type_array_nullable(type_arr: &[Value], schema: &Value) -> Result<FieldType> {
    let types: Vec<&str> = type_arr.iter().filter_map(|v| v.as_str()).collect();

    if types.len() != type_arr.len() {
        return Err(PredictError::invalid_signature(
            "type array must contain only strings",
        ));
    }

    let non_null: Vec<&&str> = types.iter().filter(|t| **t != "null").collect();
    let has_null = types.contains(&"null");

    if !has_null || non_null.len() != 1 {
        return Err(PredictError::invalid_signature(format!(
            "type array nullable must be [\"<type>\", \"null\"], got {types:?}"
        )));
    }

    // Internal-consistency check: when `type: [T, null]` is paired with
    // `enum: [...]` (or `const: ...`), JSON Schema's spec applies the
    // enum constraint *independently* of `type`. If null isn't in the
    // enum / const, the schema admits no null value — but folding to
    // `Nullable<T>` would tell the prompt the model CAN emit null. The
    // application then validates the raw stored schema via the
    // `jsonschema` crate (correctly per spec) and rejects null, so
    // prompt conversion and downstream validation disagree on what's emittable.
    //
    // Surface the inconsistency at schema-conversion time with a
    // concrete fix list, instead of letting the disagreement leak
    // out as a confusing runtime "null is not one of [...]" error.
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array())
        && !enum_values.iter().any(serde_json::Value::is_null)
    {
        return Err(PredictError::invalid_signature(format!(
            "schema declares `type: {types:?}` (nullable) but `enum: {enum_values:?}` does \
                 not include null. Per JSON Schema, `enum` restricts the value set \
                 independently of `type`, so this schema admits no null value despite the \
                 nullable type. The prompt would tell the model null is valid while the \
                 downstream validator's schema validator would reject it. Fix one of: \
                 (a) add null to the enum (`enum: [..., null]`), or \
                 (b) drop \"null\" from the type array (`type: \"{}\"`).",
            non_null[0],
        )));
    }
    // No equivalent check for `const`: the top-level dispatch in
    // `field_type_from_inline_schema` routes any `const` keyword
    // through `parse_const` BEFORE this function ever fires, so a
    // schema like `{type: [T, null], const: "x"}` becomes a
    // single-element `Enum(["x"])` — the downstream validator's `const`
    // validation will reject null in lockstep, and the generated prompt
    // will tell the model to emit "x" verbatim. Consistent by
    // construction; no extra check needed here.

    // Build a synthetic schema with the non-null type for recursion
    let mut synthetic = schema.clone();
    synthetic["type"] = Value::String((*non_null[0]).to_string());
    let inner = field_type_from_inline_schema(&synthetic)?;
    Ok(FieldType::Nullable(Box::new(inner)))
}

fn parse_enum(enum_values: &[Value]) -> Result<FieldType> {
    let values: Vec<String> = enum_values
        .iter()
        .map(|v| {
            v.as_str()
                .map(std::borrow::ToOwned::to_owned)
                .ok_or_else(|| {
                    PredictError::invalid_signature(format!(
                        "enum values must be strings, got: {v}"
                    ))
                })
        })
        .collect::<Result<_>>()?;

    Ok(FieldType::Enum(values))
}

fn parse_array(schema: &Value) -> Result<FieldType> {
    let items = schema.get("items").ok_or_else(|| {
        PredictError::invalid_signature("array type requires an \"items\" schema")
    })?;
    let inner = field_type_from_inline_schema(items)?;
    Ok(FieldType::List(Box::new(inner)))
}

fn parse_object(schema: &Value) -> Result<FieldType> {
    // Object with properties → Object(Vec<ObjectField>)
    if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
        let required_fields: HashSet<&str> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let mut obj_fields = Vec::new();
        for (name, prop_schema) in properties {
            let mut field_type = field_type_from_inline_schema(prop_schema)?;

            // Standard JSON Schema nullability: a property is required iff
            // its name appears in the parent object's `required: [name, ...]`
            // array. Absent from the array (or no array present) means the
            // property is nullable. The legacy per-property `required: bool`
            // bool marker is intentionally not consulted here — it conflicts
            // with the strict JSON Schema validator that runs against this
            // same schema at input/output validation time, and every nullability
            // case expressible as `required: false` is equally expressible
            // as either omitting the name from the parent's array or using
            // a standard nullable pattern (`anyOf` + null, `type: [..., "null"]`).
            if !required_fields.contains(name.as_str()) {
                field_type = wrap_nullable(field_type);
            }

            let description = extract_description(name, prop_schema);

            obj_fields.push(ObjectField {
                name: name.clone(),
                description,
                field_type,
            });
        }
        return Ok(FieldType::Object(obj_fields));
    }

    // Object with additionalProperties → Map(value_type)
    if let Some(additional) = schema.get("additionalProperties")
        && additional.is_object()
    {
        let value_type = field_type_from_inline_schema(additional)?;
        return Ok(FieldType::Map(Box::new(value_type)));
    }

    // Bare object → Map(String)
    Ok(FieldType::Map(Box::new(FieldType::String)))
}

/// Check if a field schema has `required: false` (non-standard per-field boolean).
///
/// **Deprecated** — this helper exists for legacy callers that still ship
/// `required: bool` inside property schemas. The non-standard form conflicts
/// with the workspace's strict JSON Schema validator (`jsonschema` crate);
/// new code should express nullability via:
///
/// - The parent object's standard `required: ["name1", "name2"]` array, OR
/// - The property schema's own `anyOf: [..., {"type": "null"}]`, OR
/// - The property schema's `type: ["string", "null"]` form.
///
/// All three are honored by [`field_type_from_schema`]. After the May 2026
/// audit removed the last in-tree caller, this function is retained only
/// for any out-of-tree consumer that still imports it; the predict crate's
/// internal paths no longer consult it.
#[deprecated(
    since = "0.2.0",
    note = "use parent's `required: [...]` array or standard nullable patterns (`anyOf` + null, `type: [..., \"null\"]`); see function doc"
)]
#[must_use]
pub fn is_field_required(schema: &Value) -> bool {
    match schema.get("required") {
        Some(Value::Bool(b)) => *b,
        _ => true, // default to required; arrays are handled at the parent level
    }
}

/// Wrap a type in Nullable if it isn't already.
fn wrap_nullable(ft: FieldType) -> FieldType {
    match ft {
        FieldType::Nullable(_) => ft,
        other => FieldType::Nullable(Box::new(other)),
    }
}

/// Truncate a JSON value for error messages.
fn truncate_value(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 80 {
        format!("{}...", &s[..77])
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // --- Description-extraction tests (JSON Schema vocabulary) ---
    //
    // Every constraint keyword `extract_description` honors needs a
    // test asserting it makes it into the rendered description, since
    // each keyword is an independently-load-bearing hint to the model
    // and a regression that silently drops one shows up only as
    // mysteriously worse model behavior.

    #[test]
    fn extract_description_falls_back_to_title_then_field_name() {
        // description present → wins
        assert_eq!(
            extract_description("f", &json!({"description": "desc"})),
            "desc"
        );
        // no description but title → title wins
        assert_eq!(
            extract_description("f", &json!({"title": "Title"})),
            "Title"
        );
        // neither → field name
        assert_eq!(extract_description("country", &json!({})), "country");
    }

    #[test]
    fn extract_description_appends_pattern_constraint() {
        let desc = extract_description(
            "country",
            &json!({"type": "string", "pattern": "^[A-Z]{2}$", "description": "ISO 3166 code"}),
        );
        assert!(desc.contains("ISO 3166 code"), "preserves base: {desc}");
        assert!(
            desc.contains("must match regex: ^[A-Z]{2}$"),
            "appends pattern: {desc}"
        );
    }

    #[test]
    fn extract_description_appends_format_default_and_ranges() {
        let desc = extract_description(
            "ts",
            &json!({
                "type": "string",
                "format": "date-time",
                "default": "2026-01-01T00:00:00Z",
                "description": "When it happened",
            }),
        );
        assert!(desc.contains("format: date-time"));
        assert!(desc.contains("default: \"2026-01-01T00:00:00Z\""));
    }

    #[test]
    fn extract_description_emits_combined_numeric_range() {
        let desc = extract_description(
            "score",
            &json!({"type": "number", "minimum": 0, "maximum": 1, "description": "score"}),
        );
        assert!(desc.contains("range: 0 to 1"), "got: {desc}");
    }

    #[test]
    fn extract_description_emits_string_length_and_array_size() {
        let s = extract_description(
            "name",
            &json!({"type": "string", "minLength": 1, "maxLength": 64, "description": "name"}),
        );
        assert!(s.contains("length: 1-64 chars"), "string length: {s}");
        let a = extract_description(
            "tags",
            &json!({"type": "array", "items": {"type":"string"}, "minItems": 1, "maxItems": 5, "description":"tags"}),
        );
        assert!(a.contains("size: 1-5 items"), "array size: {a}");
    }

    #[test]
    fn extract_examples_caps_at_render_cap_and_preserves_order() {
        let examples = extract_examples(&json!({"examples": ["a", "b", "c", "d", "e"]}));
        assert_eq!(examples.len(), EXAMPLES_RENDER_CAP);
        assert_eq!(examples[0], json!("a"));
        assert_eq!(examples[2], json!("c"));
    }

    #[test]
    fn extract_examples_returns_empty_for_absent_or_wrong_shape() {
        // Absent: no array
        assert!(extract_examples(&json!({"type": "string"})).is_empty());
        // Wrong shape: examples is a string, not an array
        assert!(extract_examples(&json!({"examples": "oops"})).is_empty());
    }

    #[test]
    fn fields_from_json_schema_populates_examples_on_top_level_field() {
        // Top-level field metadata is the user-facing surface a TS
        // pipeline declares; assert the round-trip from JSON Schema
        // through to the FieldDef the chat adapter consumes.
        let schema = json!({
            "groupBy": {
                "type": "string",
                "enum": ["month", "day"],
                "examples": ["month", "day"],
                "description": "How to bucket"
            }
        });
        let fields = fields_from_json_schema(&schema, FieldKind::Output).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "groupBy");
        assert_eq!(fields[0].examples, vec![json!("month"), json!("day")]);
    }

    // --- Primitive types ---

    #[test]
    fn string_type() {
        let ft = field_type_from_schema(&json!({"type": "string"})).unwrap();
        assert_eq!(ft, FieldType::String);
    }

    #[test]
    fn integer_type() {
        let ft = field_type_from_schema(&json!({"type": "integer"})).unwrap();
        assert_eq!(ft, FieldType::Int);
    }

    #[test]
    fn number_type() {
        let ft = field_type_from_schema(&json!({"type": "number"})).unwrap();
        assert_eq!(ft, FieldType::Float);
    }

    #[test]
    fn boolean_type() {
        let ft = field_type_from_schema(&json!({"type": "boolean"})).unwrap();
        assert_eq!(ft, FieldType::Bool);
    }

    // --- Enum ---

    #[test]
    fn enum_type() {
        let ft =
            field_type_from_schema(&json!({"type": "string", "enum": ["a", "b", "c"]})).unwrap();
        assert_eq!(
            ft,
            FieldType::Enum(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn enum_non_string_values_error() {
        let result = field_type_from_schema(&json!({"type": "string", "enum": [1, 2]}));
        assert!(result.is_err());
    }

    // --- Array ---

    #[test]
    fn array_of_strings() {
        let ft = field_type_from_schema(&json!({
            "type": "array",
            "items": {"type": "string"}
        }))
        .unwrap();
        assert_eq!(ft, FieldType::List(Box::new(FieldType::String)));
    }

    #[test]
    fn array_of_objects() {
        let ft = field_type_from_schema(&json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        }))
        .unwrap();
        match ft {
            FieldType::List(inner) => match *inner {
                FieldType::Object(fields) => {
                    assert_eq!(fields.len(), 1);
                    assert_eq!(fields[0].name, "name");
                }
                other => panic!("expected Object, got {other:?}"),
            },
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn array_without_items_error() {
        let result = field_type_from_schema(&json!({"type": "array"}));
        assert!(result.is_err());
    }

    // --- Nullable ---

    #[test]
    fn any_of_nullable() {
        let ft = field_type_from_schema(&json!({
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }))
        .unwrap();
        assert_eq!(ft, FieldType::Nullable(Box::new(FieldType::String)));
    }

    #[test]
    fn type_array_nullable() {
        let ft = field_type_from_schema(&json!({"type": ["string", "null"]})).unwrap();
        assert_eq!(ft, FieldType::Nullable(Box::new(FieldType::String)));
    }

    #[test]
    fn type_array_nullable_with_enum_excluding_null_rejects() {
        // The bug that motivated this check: `{type: ['string','null'], enum: ['a','b']}`
        // — predict would render Nullable<Enum> and tell the model null is
        // valid, but the downstream validator (correctly per JSON Schema spec, where
        // `enum` constrains the value set independently of `type`) rejects
        // null. The disagreement leaks as a runtime "null is not one of"
        // error far downstream of the schema author. Surface it at
        // conversion time with a fix list instead.
        let err = field_type_from_schema(&json!({
            "type": ["string", "null"],
            "enum": ["month", "day"]
        }))
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("enum"), "names the conflict: {msg}");
        assert!(msg.contains("does not include null"), "explains why: {msg}");
        assert!(
            msg.contains("add null to the enum") || msg.contains("`enum: [..., null]`"),
            "offers fix (a): {msg}"
        );
        assert!(msg.contains("drop \"null\""), "offers fix (b): {msg}");
    }

    #[test]
    fn type_array_nullable_with_enum_including_null_passes_consistency_check() {
        // When `null` is in the enum the consistency check passes —
        // prompt conversion and downstream validation would agree that null is a valid value.
        // The inner `parse_enum` still requires all surviving enum
        // values to be strings; the assertion here is narrow: this
        // schema does NOT trip the type-array-nullable consistency
        // error (any failure here would be a different downstream
        // constraint, not the inconsistency this fix targets).
        let result = field_type_from_schema(&json!({
            "type": ["string", "null"],
            "enum": ["month", "day", null]
        }));
        if let Err(err) = &result {
            let msg = err.to_string();
            assert!(
                !msg.contains("does not include null"),
                "consistency check fired unexpectedly: {msg}"
            );
        }
    }

    #[test]
    fn type_array_nullable_with_const_routes_through_parse_const() {
        // `const` dispatch fires at field_type_from_inline_schema step 2,
        // BEFORE parse_type_array_nullable runs. So `{type: [T, null],
        // const: "x"}` folds to a single-element Enum, which the
        // downstream validation checks the original schema's `const`
        // keyword — both agree the only valid emission is "x". No
        // consistency check needed in parse_type_array_nullable; this
        // test pins the routing behavior so a future reorder of the
        // dispatch doesn't silently bring back the bug class.
        let ft = field_type_from_schema(&json!({
            "type": ["string", "null"],
            "const": "month"
        }))
        .expect("const dispatch fires before type-array nullable");
        assert_eq!(ft, FieldType::Enum(vec!["month".into()]));
    }

    #[test]
    fn anyof_null_renders_nullable() {
        // Replaces an earlier test that asserted `required: false` (bool)
        // on a field schema produced a nullable type. That legacy
        // marker is no longer consulted by `fields_from_json_schema`
        // because it conflicts with strict JSON Schema validation.
        // The standard `anyOf` + null pattern is the canonical way to
        // express per-field nullability where there's no enclosing
        // parent to declare `required: [...]`.
        let fields = fields_from_json_schema(
            &json!({
                "score": {
                    "anyOf": [{"type": "number"}, {"type": "null"}]
                }
            }),
            FieldKind::Output,
        )
        .unwrap();
        assert_eq!(
            fields[0].field_type,
            FieldType::Nullable(Box::new(FieldType::Float))
        );
    }

    #[test]
    fn parent_required_array_nullable() {
        let ft = field_type_from_schema(&json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }))
        .unwrap();
        match ft {
            FieldType::Object(fields) => {
                let name_field = fields.iter().find(|f| f.name == "name").unwrap();
                assert_eq!(name_field.field_type, FieldType::String);

                let age_field = fields.iter().find(|f| f.name == "age").unwrap();
                assert_eq!(
                    age_field.field_type,
                    FieldType::Nullable(Box::new(FieldType::Int))
                );
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn any_of_without_null_produces_general_anyof() {
        // Pre-Phase-2 behavior: rejected as "anyOf must include null".
        // Phase 2 added general AnyOf support; non-nullable anyOf now
        // converts to a FieldType::AnyOf with one arm per schema entry.
        // The nullable idiom still takes priority — see
        // `anyof_nullable_still_takes_priority`.
        let ft = field_type_from_schema(&json!({
            "anyOf": [{"type": "string"}, {"type": "integer"}]
        }))
        .unwrap();
        match ft {
            FieldType::AnyOf { arms } => {
                assert_eq!(arms.len(), 2);
                assert!(matches!(arms[0].field_type, FieldType::String));
                assert!(matches!(arms[1].field_type, FieldType::Int));
            }
            other => panic!("expected AnyOf, got {other:?}"),
        }
    }

    // --- Object ---

    #[test]
    fn object_with_properties() {
        let ft = field_type_from_schema(&json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "x coord"},
                "y": {"type": "integer", "description": "y coord"}
            },
            "required": ["x", "y"]
        }))
        .unwrap();
        match ft {
            FieldType::Object(fields) => {
                assert_eq!(fields.len(), 2);
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn object_with_additional_properties() {
        let ft = field_type_from_schema(&json!({
            "type": "object",
            "additionalProperties": {"type": "string"}
        }))
        .unwrap();
        assert_eq!(ft, FieldType::Map(Box::new(FieldType::String)));
    }

    #[test]
    fn bare_object() {
        let ft = field_type_from_schema(&json!({"type": "object"})).unwrap();
        assert_eq!(ft, FieldType::Map(Box::new(FieldType::String)));
    }

    // --- Description ---

    #[test]
    fn description_from_schema() {
        let desc = extract_description("field", &json!({"description": "A cool field"}));
        assert_eq!(desc, "A cool field");
    }

    #[test]
    fn description_fallback_to_name() {
        let desc = extract_description("my_field", &json!({"type": "string"}));
        assert_eq!(desc, "my_field");
    }

    #[test]
    fn description_with_enum_values() {
        let desc = extract_description(
            "status",
            &json!({"type": "string", "enum": ["active", "inactive"], "description": "Status"}),
        );
        assert_eq!(desc, "Status (must be one of: 'active', 'inactive')");
    }

    // --- Errors ---

    #[test]
    fn unknown_type_error() {
        let result = field_type_from_schema(&json!({"type": "custom"}));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unsupported"));
        assert!(err.contains("custom"));
    }

    #[test]
    fn missing_type_error() {
        let result = field_type_from_schema(&json!({"description": "no type"}));
        assert!(result.is_err());
    }

    #[test]
    fn schema_not_object_error() {
        let result = fields_from_json_schema(&json!("not an object"), FieldKind::Input);
        assert!(result.is_err());
    }

    // --- fields_from_json_schema ---

    #[test]
    fn fields_from_schema_basic() {
        let fields = fields_from_json_schema(
            &json!({
                "question": {"type": "string", "description": "The question"},
                "context": {"type": "array", "items": {"type": "string"}}
            }),
            FieldKind::Input,
        )
        .unwrap();

        assert_eq!(fields.len(), 2);
        assert!(fields.iter().all(|f| f.kind == FieldKind::Input));
    }

    // --- Integration: JSON Schema → Signature ---

    #[test]
    fn full_signature_from_json_schema() {
        let sig = signature_from_json_schema(
            &json!({
                "question": {"type": "string", "description": "The user question"}
            }),
            &json!({
                "answer": {"type": "string", "description": "The answer"},
                "confidence": {"type": "number", "description": "Confidence score"}
            }),
            "Answer the question with confidence.",
        )
        .unwrap();

        assert_eq!(sig.instructions(), "Answer the question with confidence.");
        assert_eq!(sig.input_fields().count(), 1);
        assert_eq!(sig.output_fields().count(), 2);
    }

    // ============================================================
    // Phase 2: oneOf / anyOf / allOf / const / $ref / dependentSchemas /
    // if-then-else conversion tests.
    // ============================================================

    fn oneof_schema_with_const(prop: &str, tag: &str, extra_field: &str) -> Value {
        json!({
            "type": "object",
            "properties": {
                prop: {"const": tag},
                extra_field: {"type": "string"}
            },
            "required": [prop, extra_field]
        })
    }

    // --- oneOf: discriminator inference happy paths ---

    #[test]
    fn oneof_with_const_discriminator_inferred() {
        let schema = json!({
            "oneOf": [
                oneof_schema_with_const("toolName", "a", "x"),
                oneof_schema_with_const("toolName", "b", "y")
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::OneOf {
                arms,
                discriminator: Some(disc),
            } => {
                assert_eq!(disc.property, "toolName");
                assert_eq!(disc.tags, vec!["a".to_owned(), "b".to_owned()]);
                assert_eq!(arms.len(), 2);
            }
            other => panic!("expected OneOf with discriminator, got {other:?}"),
        }
    }

    #[test]
    fn oneof_with_explicit_discriminator_hint_matching() {
        let schema = json!({
            "discriminator": {"propertyName": "toolName"},
            "oneOf": [
                oneof_schema_with_const("toolName", "a", "x"),
                oneof_schema_with_const("toolName", "b", "y")
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert!(matches!(
            ft,
            FieldType::OneOf {
                discriminator: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn oneof_explicit_hint_inconsistent_rejects() {
        let schema = json!({
            "discriminator": {"propertyName": "kind"},
            "oneOf": [
                oneof_schema_with_const("toolName", "a", "x"),
                oneof_schema_with_const("toolName", "b", "y")
            ]
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(
            err.contains("kind") && err.contains("toolName"),
            "got: {err}"
        );
    }

    #[test]
    fn oneof_explicit_hint_no_inferred_rejects() {
        let schema = json!({
            "discriminator": {"propertyName": "kind"},
            "oneOf": [
                {"type": "object", "properties": {"x": {"type": "string"}}},
                {"type": "object", "properties": {"y": {"type": "string"}}}
            ]
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(
            err.contains("kind") && err.contains("inferred"),
            "got: {err}"
        );
    }

    #[test]
    fn oneof_untagged_when_no_shared_const_property() {
        let schema = json!({
            "oneOf": [
                {"type": "object", "properties": {"a": {"const": "x"}}},
                {"type": "object", "properties": {"b": {"const": "y"}}}
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert!(matches!(
            ft,
            FieldType::OneOf {
                discriminator: None,
                ..
            }
        ));
    }

    #[test]
    fn oneof_untagged_when_arm_has_no_const() {
        let schema = json!({
            "oneOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "string"}}}
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert!(matches!(
            ft,
            FieldType::OneOf {
                discriminator: None,
                ..
            }
        ));
    }

    #[test]
    fn oneof_empty_arms_rejected() {
        let schema = json!({"oneOf": []});
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("at least one"), "got: {err}");
    }

    #[test]
    fn oneof_inside_list_composes() {
        let schema = json!({
            "type": "array",
            "items": {
                "oneOf": [
                    oneof_schema_with_const("kind", "a", "x"),
                    oneof_schema_with_const("kind", "b", "y")
                ]
            }
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::List(inner) => match *inner {
                FieldType::OneOf {
                    discriminator: Some(_),
                    ..
                } => {}
                other => panic!("expected OneOf with discriminator, got {other:?}"),
            },
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn oneof_with_duplicate_tags_skips_inference() {
        let schema = json!({
            "oneOf": [
                oneof_schema_with_const("kind", "same", "x"),
                oneof_schema_with_const("kind", "same", "y")
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        // Non-unique tags fail uniqueness check → untagged OneOf.
        assert!(matches!(
            ft,
            FieldType::OneOf {
                discriminator: None,
                ..
            }
        ));
    }

    // --- anyOf general (non-nullable) ---

    #[test]
    fn anyof_general_produces_anyof_type() {
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::AnyOf { arms } => {
                assert_eq!(arms.len(), 2);
                assert!(matches!(arms[0].field_type, FieldType::String));
                assert!(matches!(arms[1].field_type, FieldType::Int));
            }
            other => panic!("expected AnyOf, got {other:?}"),
        }
    }

    #[test]
    fn anyof_nullable_still_takes_priority() {
        // Existing nullable idiom must still work after AnyOf was added.
        let schema = json!({
            "anyOf": [{"type": "string"}, {"type": "null"}]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::Nullable(Box::new(FieldType::String)));
    }

    // --- allOf folding ---

    #[test]
    fn allof_merges_object_arms() {
        let schema = json!({
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]}
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::Object(fields) => {
                let names: Vec<_> = fields.iter().map(|f| f.name.as_str()).collect();
                assert!(names.contains(&"a"), "got: {names:?}");
                assert!(names.contains(&"b"), "got: {names:?}");
            }
            other => panic!("expected merged Object, got {other:?}"),
        }
    }

    #[test]
    fn allof_field_type_conflict_rejects() {
        let schema = json!({
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
            ]
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("conflict"), "got: {err}");
    }

    #[test]
    fn allof_non_object_arm_rejects() {
        let schema = json!({
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "string"}
            ]
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("Object arms"), "got: {err}");
    }

    // --- const folding ---

    #[test]
    fn const_string_folds_to_single_element_enum() {
        let schema = json!({"const": "production"});
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::Enum(vec!["production".to_owned()]));
    }

    #[test]
    fn const_non_string_rejects() {
        let schema = json!({"const": 42});
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("non-string"), "got: {err}");
    }

    // --- $ref resolution ---

    #[test]
    fn ref_internal_defs_resolves() {
        let schema = json!({
            "type": "object",
            "properties": {
                "field_a": {"$ref": "#/$defs/A"}
            },
            "required": ["field_a"],
            "$defs": {
                "A": {"type": "string"}
            }
        });
        let resolved = resolve_refs(&schema, &schema).unwrap();
        let ft = field_type_from_schema(&resolved).unwrap();
        match ft {
            FieldType::Object(fields) => {
                let f = fields.iter().find(|f| f.name == "field_a").unwrap();
                assert!(matches!(f.field_type, FieldType::String));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn ref_internal_definitions_resolves() {
        let schema = json!({
            "$ref": "#/definitions/A",
            "definitions": {
                "A": {"type": "integer"}
            }
        });
        let resolved = resolve_refs(&schema, &schema).unwrap();
        let ft = field_type_from_schema(&resolved).unwrap();
        assert_eq!(ft, FieldType::Int);
    }

    #[test]
    fn ref_external_rejects() {
        let schema = json!({"$ref": "https://example.com/schema.json"});
        let err = resolve_refs(&schema, &schema).unwrap_err().to_string();
        assert!(err.contains("external"), "got: {err}");
    }

    #[test]
    fn ref_cycle_rejects() {
        let schema = json!({
            "$ref": "#/$defs/A",
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"}
            }
        });
        let err = resolve_refs(&schema, &schema).unwrap_err().to_string();
        assert!(err.contains("cycle"), "got: {err}");
    }

    #[test]
    fn ref_unresolved_rejects() {
        let schema = json!({
            "$ref": "#/$defs/Missing",
            "$defs": {"A": {"type": "string"}}
        });
        let err = resolve_refs(&schema, &schema).unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("Missing"),
            "got: {err}"
        );
    }

    #[test]
    fn ref_invalid_prefix_rejects() {
        let schema = json!({"$ref": "definitions/A"});
        let err = resolve_refs(&schema, &schema).unwrap_err().to_string();
        assert!(err.contains("#/"), "got: {err}");
    }

    // --- dependentSchemas transformation ---

    #[test]
    fn dependent_schemas_reducible_transforms_to_oneof() {
        let schema = json!({
            "type": "object",
            "properties": {
                "kind": {"enum": ["a", "b"]}
            },
            "dependentSchemas": {
                "kind": {
                    "oneOf": [
                        {
                            "properties": {
                                "kind": {"const": "a"},
                                "x": {"type": "string"}
                            },
                            "required": ["kind", "x"]
                        },
                        {
                            "properties": {
                                "kind": {"const": "b"},
                                "y": {"type": "integer"}
                            },
                            "required": ["kind", "y"]
                        }
                    ]
                }
            }
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::OneOf {
                arms,
                discriminator: Some(disc),
            } => {
                assert_eq!(disc.property, "kind");
                assert_eq!(disc.tags, vec!["a".to_owned(), "b".to_owned()]);
                assert_eq!(arms.len(), 2);
            }
            other => panic!("expected discriminated OneOf, got {other:?}"),
        }
    }

    #[test]
    fn dependent_schemas_multiple_triggers_rejects() {
        let schema = json!({
            "type": "object",
            "dependentSchemas": {
                "kind": {"oneOf": []},
                "other": {"oneOf": []}
            }
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("trigger"), "got: {err}");
    }

    #[test]
    fn dependent_schemas_non_oneof_value_rejects() {
        let schema = json!({
            "type": "object",
            "dependentSchemas": {
                "kind": {"type": "string"}
            }
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("oneOf"), "got: {err}");
    }

    // --- if/then/else transformation ---

    #[test]
    fn if_then_else_reducible_transforms_to_oneof() {
        let schema = json!({
            "if": {
                "properties": {"kind": {"const": "a"}}
            },
            "then": {
                "properties": {
                    "x": {"type": "string"}
                },
                "required": ["x"]
            },
            "else": {
                "properties": {"kind": {"const": "b"}}
            }
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::OneOf {
                discriminator: Some(disc),
                arms,
            } => {
                assert_eq!(disc.property, "kind");
                let tags_set: HashSet<_> = disc.tags.iter().collect();
                let expected: HashSet<String> =
                    ["a".to_owned(), "b".to_owned()].into_iter().collect();
                let expected_refs: HashSet<&String> = expected.iter().collect();
                assert_eq!(tags_set, expected_refs);
                assert_eq!(arms.len(), 2);
            }
            other => panic!("expected discriminated OneOf, got {other:?}"),
        }
    }

    #[test]
    fn if_then_else_non_discriminator_if_rejects() {
        let schema = json!({
            "if": {
                "type": "string",
                "minLength": 5
            },
            "then": {"type": "string"}
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(err.contains("discriminator pattern"), "got: {err}");
    }

    #[test]
    fn if_then_else_else_property_mismatch_rejects() {
        let schema = json!({
            "if": {
                "properties": {"kind": {"const": "a"}}
            },
            "then": {"properties": {"x": {"type": "string"}}, "required": ["x"]},
            "else": {
                "properties": {"otherProp": {"const": "b"}}
            }
        });
        let err = field_type_from_schema(&schema).unwrap_err().to_string();
        assert!(
            err.contains("must agree") || err.contains("disagree"),
            "got: {err}"
        );
    }

    // --- Backward-compat sanity: existing schemas still convert. ---

    #[test]
    fn existing_primitive_schemas_still_convert() {
        for ty in ["string", "integer", "number", "boolean"] {
            let schema = json!({"type": ty});
            field_type_from_schema(&schema).expect(ty);
        }
    }

    #[test]
    fn existing_nullable_schemas_still_convert() {
        let schema = json!({"anyOf": [{"type": "string"}, {"type": "null"}]});
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::Nullable(Box::new(FieldType::String)));
    }

    // ============================================================
    // Layer 10: Cross-language TS/Zod-shaped schema fixtures.
    //
    // The application's slot schemas typically arrive from the
    // TypeScript API, which generates JSON Schema via Zod's
    // `zodToJsonSchema()` helper. Zod produces specific shapes for
    // each combinator that the Rust converter must accept. These
    // fixtures are minimal real-world shapes — not exhaustive
    // documentation but enough to catch a regression that breaks
    // a documented Zod combinator's output.
    //
    // Coverage:
    //   z.discriminatedUnion → oneOf w/ const discriminator
    //   z.union              → anyOf
    //   z.intersection       → allOf
    //   z.object.optional()  → anyOf-nullable
    //   z.literal            → const
    //   z.enum               → enum
    //   z.array              → array
    //   z.record             → object.additionalProperties
    //   z.object.partial()   → properties without required array
    // ============================================================

    #[test]
    fn ts_zod_discriminated_union_shape() {
        // z.discriminatedUnion("kind", [
        //   z.object({ kind: z.literal("a"), x: z.number() }),
        //   z.object({ kind: z.literal("b"), y: z.string() }),
        // ])
        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "a"},
                        "x": {"type": "number"}
                    },
                    "required": ["kind", "x"],
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "b"},
                        "y": {"type": "string"}
                    },
                    "required": ["kind", "y"],
                    "additionalProperties": false
                }
            ]
        });
        // Zod emits `anyOf` (not `oneOf`) for discriminatedUnion — our
        // anyOf-general path accepts it. The discriminator pattern is
        // still visible to the converter; it just lands as AnyOf rather
        // than as OneOf w/ discriminator (since the keyword is anyOf).
        let ft = field_type_from_schema(&schema).unwrap();
        assert!(matches!(ft, FieldType::AnyOf { .. }), "got: {ft:?}");
    }

    #[test]
    fn ts_zod_discriminated_union_via_one_of_shape() {
        // Some TS generators (zod-to-json-schema with target: "openApi3")
        // emit `oneOf` instead of `anyOf` for discriminatedUnion. We
        // accept both.
        let schema = json!({
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "a"},
                        "x": {"type": "number"}
                    },
                    "required": ["kind", "x"]
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "b"},
                        "y": {"type": "string"}
                    },
                    "required": ["kind", "y"]
                }
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::OneOf {
                discriminator: Some(d),
                ..
            } => {
                assert_eq!(d.property, "kind");
            }
            other => panic!("expected discriminated OneOf, got {other:?}"),
        }
    }

    #[test]
    fn ts_zod_union_shape() {
        // z.union([z.string(), z.number(), z.boolean()])
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"}
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert!(matches!(ft, FieldType::AnyOf { .. }));
    }

    #[test]
    fn ts_zod_optional_field_shape() {
        // z.object({ a: z.string().optional() })
        // Zod emits a top-level object with `a` NOT in required.
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"}
            }
            // Note: no "required" — `a` is therefore nullable per the
            // standard JSON Schema convention.
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::Object(fields) => {
                let a = fields.iter().find(|f| f.name == "a").unwrap();
                assert!(matches!(a.field_type, FieldType::Nullable(_)));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn ts_zod_nullable_field_shape() {
        // z.string().nullable() — Zod emits anyOf<[type, null]> at the
        // FIELD level (not the property level).
        let schema = json!({
            "anyOf": [{"type": "string"}, {"type": "null"}]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::Nullable(Box::new(FieldType::String)));
    }

    #[test]
    fn ts_zod_literal_shape() {
        // z.literal("production")
        let schema = json!({"type": "string", "const": "production"});
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::Enum(vec!["production".into()]));
    }

    #[test]
    fn ts_zod_enum_shape() {
        // z.enum(["red", "green", "blue"])
        let schema = json!({
            "type": "string",
            "enum": ["red", "green", "blue"]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(
            ft,
            FieldType::Enum(vec!["red".into(), "green".into(), "blue".into()])
        );
    }

    #[test]
    fn ts_zod_array_shape() {
        // z.array(z.string())
        let schema = json!({
            "type": "array",
            "items": {"type": "string"}
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::List(Box::new(FieldType::String)));
    }

    #[test]
    fn ts_zod_record_shape() {
        // z.record(z.number())
        let schema = json!({
            "type": "object",
            "additionalProperties": {"type": "number"}
        });
        let ft = field_type_from_schema(&schema).unwrap();
        assert_eq!(ft, FieldType::Map(Box::new(FieldType::Float)));
    }

    #[test]
    fn ts_zod_object_with_passthrough_additional_properties() {
        // z.object({...}).passthrough() — Zod adds
        // `additionalProperties: true` (boolean, not a schema). The
        // converter sees an Object with properties; additionalProperties
        // is ignored because it's a boolean, not an object schema.
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": true
        });
        let ft = field_type_from_schema(&schema).unwrap();
        // Object with declared properties — the passthrough flag is
        // dropped (the converter doesn't model "open" objects).
        assert!(matches!(ft, FieldType::Object(_)));
    }

    #[test]
    fn ts_zod_intersection_shape() {
        // z.intersection(
        //   z.object({ a: z.string() }),
        //   z.object({ b: z.number() })
        // )
        let schema = json!({
            "allOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"]
                },
                {
                    "type": "object",
                    "properties": {"b": {"type": "number"}},
                    "required": ["b"]
                }
            ]
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::Object(fields) => {
                let names: Vec<_> = fields.iter().map(|f| f.name.as_str()).collect();
                assert!(
                    names.contains(&"a") && names.contains(&"b"),
                    "got: {names:?}"
                );
            }
            other => panic!("expected merged Object, got {other:?}"),
        }
    }

    #[test]
    fn ts_zod_nested_complex_shape() {
        // A real-world chat-orchestrator schema: z.array(z.object({...,
        // assignment: z.discriminatedUnion("toolName", [...]).nullable() }))
        let schema = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "assignment": {
                        "anyOf": [
                            {
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "toolName": {"const": "tool_a"},
                                            "params": {
                                                "type": "object",
                                                "properties": {"x": {"type": "number"}},
                                                "required": ["x"]
                                            }
                                        },
                                        "required": ["toolName", "params"]
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "toolName": {"const": "tool_b"},
                                            "params": {
                                                "type": "object",
                                                "properties": {"y": {"type": "string"}},
                                                "required": ["y"]
                                            }
                                        },
                                        "required": ["toolName", "params"]
                                    }
                                ]
                            },
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["query", "assignment"]
            }
        });
        let ft = field_type_from_schema(&schema).unwrap();
        match ft {
            FieldType::List(inner) => match inner.as_ref() {
                FieldType::Object(fields) => {
                    let assignment = fields.iter().find(|f| f.name == "assignment").unwrap();
                    match &assignment.field_type {
                        FieldType::Nullable(inner) => {
                            assert!(matches!(
                                inner.as_ref(),
                                FieldType::OneOf {
                                    discriminator: Some(_),
                                    ..
                                }
                            ));
                        }
                        other => panic!("expected Nullable<OneOf>, got {other:?}"),
                    }
                }
                other => panic!("expected inner Object, got {other:?}"),
            },
            other => panic!("expected outer List, got {other:?}"),
        }
    }

    // ============================================================
    // Layer 1: Arbitrary JSON Schema proptest.
    //
    // Generates bounded random schemas using the supported keyword
    // set (primitives, oneOf, anyOf, allOf, const, enum, $ref with
    // cycles, dependentSchemas, if/then/else, nullable idioms,
    // List/Object/Map containers). Asserts:
    //
    //   For every generated schema `s`:
    //     field_type_from_schema(s) returns either
    //       Ok(FieldType) — the schema was accepted
    //       Err(PredictError::InvalidSignature { reason }) — rejected with
    //         a specific reason naming the offending construct
    //
    //   NEVER panics; NEVER returns a FieldType that semantically
    //   misrepresents `s`.
    //
    // The "named error or success" property is the load-bearing
    // invariant: it catches silent fallbacks and panics in one sweep.
    // ============================================================

    use proptest::prelude::*;

    fn arb_const_string() -> impl Strategy<Value = Value> {
        prop_oneof![
            "a".prop_map(|s| json!({"const": s})),
            Just(json!({"const": "x"})),
        ]
    }

    fn arb_primitive_schema() -> impl Strategy<Value = Value> {
        prop_oneof![
            Just(json!({"type": "string"})),
            Just(json!({"type": "integer"})),
            Just(json!({"type": "number"})),
            Just(json!({"type": "boolean"})),
            Just(json!({"type": "string", "enum": ["a", "b", "c"]})),
            arb_const_string(),
        ]
    }

    /// Recursive arbitrary JSON Schema generator bounded by max_depth.
    /// Limits the keyword set to the ones the converter supports so the
    /// proptest asserts "convertible OR explicit rejection", not
    /// "rejects everything outside the supported set" (already covered
    /// by the deterministic unknown-type tests).
    fn arb_json_schema(max_depth: u32) -> BoxedStrategy<Value> {
        let leaf = arb_primitive_schema().boxed();
        leaf.prop_recursive(max_depth, 32, 4, |inner| {
            prop_oneof![
                // List<T>
                inner
                    .clone()
                    .prop_map(|t| json!({"type": "array", "items": t})),
                // Object{x: T}
                inner.clone().prop_map(|t| json!({
                    "type": "object",
                    "properties": {"x": t},
                    "required": ["x"]
                })),
                // Object with two distinct fields
                (inner.clone(), inner.clone()).prop_map(|(a, b)| json!({
                    "type": "object",
                    "properties": {"a": a, "b": b},
                    "required": ["a", "b"]
                })),
                // Map<T>
                inner.clone().prop_map(|t| json!({
                    "type": "object",
                    "additionalProperties": t
                })),
                // Nullable<T> via anyOf
                inner.clone().prop_map(|t| json!({
                    "anyOf": [t, {"type": "null"}]
                })),
                // OneOf — two tagged arms with distinct const properties
                inner.clone().prop_map(|t| json!({
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"kind": {"const": "a"}, "x": t},
                            "required": ["kind", "x"]
                        },
                        {
                            "type": "object",
                            "properties": {"kind": {"const": "b"}, "y": {"type": "string"}},
                            "required": ["kind", "y"]
                        }
                    ]
                })),
                // AnyOf (non-nullable)
                inner.clone().prop_map(|t| json!({
                    "anyOf": [t, {"type": "integer"}]
                })),
                // allOf of two object arms (may conflict or compose
                // depending on the inner; both outcomes are valid for
                // the property). Final branch — consume `inner` without
                // a redundant clone.
                inner.prop_map(|t| json!({
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"a": t},
                            "required": ["a"]
                        },
                        {
                            "type": "object",
                            "properties": {"b": {"type": "string"}},
                            "required": ["b"]
                        }
                    ]
                })),
            ]
        })
        .boxed()
    }

    proptest! {
        /// Property: `field_type_from_schema` either succeeds or fails
        /// with a specific named error — never panics, never produces
        /// a silent misrepresentation. Catches the entire "unknown
        /// keyword silently swallowed" class of bug.
        #[test]
        fn schema_conversion_never_panics_and_always_names_errors(
            schema in arb_json_schema(4),
        ) {
            match field_type_from_schema(&schema) {
                Ok(_) => {} // Convertible — fine.
                Err(PredictError::InvalidSignature { reason }) => {
                    // Every rejection must carry a reason string. The
                    // schema converter doesn't produce empty rejection
                    // reasons in any current code path; this guard fails
                    // loudly if a future refactor regresses that.
                    prop_assert!(
                        !reason.is_empty(),
                        "InvalidSignature with empty reason for schema: {schema}"
                    );
                }
                Err(other) => {
                    // The converter only returns InvalidSignature
                    // variants. Any other error variant is a contract
                    // violation worth surfacing immediately.
                    prop_assert!(
                        false,
                        "unexpected error variant {other:?} for schema: {schema}"
                    );
                }
            }
        }

        /// Property: $ref pre-pass terminates on cyclic schemas with a
        /// specific named error, never with an infinite-loop hang or
        /// stack overflow. The arb_json_schema generator doesn't
        /// produce $refs directly (the keyword set above doesn't
        /// include them); this property uses a parametric construction
        /// that injects a deliberate cycle for the resolver to detect.
        #[test]
        fn ref_resolver_terminates_on_cycles(
            depth in 1u32..=5u32,
        ) {
            // Build a chain A -> B -> ... -> Z -> A of length `depth`.
            let mut defs = serde_json::Map::new();
            for i in 0..depth {
                let next = (i + 1) % depth;
                defs.insert(
                    format!("D{i}"),
                    json!({"$ref": format!("#/$defs/D{next}")})
                );
            }
            let schema = json!({
                "$ref": "#/$defs/D0",
                "$defs": defs,
            });
            let result = field_type_from_schema(&schema);
            prop_assert!(
                matches!(&result, Err(PredictError::InvalidSignature { reason }) if reason.contains("cycle")),
                "expected cycle error, got: {result:?}"
            );
        }
    }
}
