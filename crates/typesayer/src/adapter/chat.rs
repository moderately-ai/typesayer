// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ChatAdapter` — formats prompts using `[[ ## field_name ## ]]` section delimiters.
//!
//! The system message contains field descriptions, a format template showing the
//! expected delimiter structure, and task instructions. Few-shot demos are formatted
//! as alternating user/assistant message pairs. Parsing uses finditer-style regex
//! scanning so markers embedded mid-line are found correctly.

use std::{
    collections::{BTreeMap, HashSet},
    sync::LazyLock,
};

use async_trait::async_trait;
use modelplease::{ContentPart, Message, Role};
use regex::Regex;
use typesayer_types::{
    error::{PredictError, Result},
    field::{FieldDef, FieldType, FieldValue},
    signature::Signature,
};

use crate::{
    adapter::{Adapter, Demo},
    format::{FieldDeserializer, FieldSerializer, JsonFieldDeserializer, JsonFieldSerializer},
};

/// Compiled field marker regex: matches `[[ ## field_name ## ]]` anywhere in text.
#[expect(
    clippy::expect_used,
    reason = "static LazyLock built from a literal regex: compile-time verifiable, \
              so a runtime parse failure would mean the binary was linked against \
              a broken `regex` crate and panicking at first-use is the only \
              meaningful recovery"
)]
static FIELD_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[ ## (\w+) ## \]\]").expect("static regex is always valid"));

/// Where [`ChatAdapter::format`] inserts the static-prefix cache breakpoint.
///
/// The static-prefix breakpoint is the per-message cache-control marker
/// the adapter emits so providers (Bedrock, Anthropic, etc.) can cache
/// the byte-identical prefix of the prompt. Picking where the prefix
/// ends is a customer-facing tuning knob: different placements optimise
/// for different cache-reuse patterns.
///
/// Per-input-field breakpoints (driven by [`FieldDef::cacheable`]) are
/// independent of this — they always emit when present, since they
/// target a separate cache window inside the live user message.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CachePlacement {
    /// Place the breakpoint at the end of the fully-static prefix:
    /// the system message when no demos are present, otherwise the
    /// last demo's assistant message. The cached span covers the
    /// system message AND every demo turn.
    ///
    /// Historical default. Best when demo content is stable across
    /// calls (so it benefits from being inside the cached span).
    #[default]
    PostStaticPrefix,
    /// Place the breakpoint at the end of the system message only.
    /// Demos sit AFTER the breakpoint and are NOT included in the
    /// cached span.
    ///
    /// Use when the system prompt is byte-stable across calls but
    /// demos vary, so demos drifting in/out of the cached span would
    /// cause spurious misses. Guarantees the cached span is the
    /// system message and nothing else.
    PostSystemPrompt,
    /// Don't emit a static-prefix cache breakpoint at all. Disables
    /// adapter-driven caching for this format call. Per-input
    /// cacheable-field breakpoints (driven by `FieldDef::cacheable`)
    /// are unaffected — they emit independently.
    None,
}

/// `ChatAdapter` formats prompts using `[[ ## field_name ## ]]` section delimiters.
///
/// Parameterized by a [`FieldSerializer`] (for formatting values into prompts) and
/// a [`FieldDeserializer`] (for parsing values from completions). Defaults to JSON
/// for both.
///
/// # Examples
///
/// ```rust
/// use typesayer::ChatAdapter;
///
/// // Default: JSON serialization and deserialization
/// let adapter = ChatAdapter::default();
///
/// // Custom formats
/// # use typesayer::{JsonFieldSerializer, JsonFieldDeserializer};
/// let adapter = ChatAdapter::with_formats(JsonFieldSerializer, JsonFieldDeserializer);
/// ```
pub struct ChatAdapter<S = JsonFieldSerializer, D = JsonFieldDeserializer>
where
    S: FieldSerializer,
    D: FieldDeserializer,
{
    serializer: S,
    deserializer: D,
    cache_placement: CachePlacement,
}

impl Default for ChatAdapter {
    fn default() -> Self {
        Self {
            serializer: JsonFieldSerializer,
            deserializer: JsonFieldDeserializer,
            cache_placement: CachePlacement::default(),
        }
    }
}

impl<S: FieldSerializer, D: FieldDeserializer> ChatAdapter<S, D> {
    /// Create a `ChatAdapter` with custom serializer and deserializer.
    /// Cache placement defaults to [`CachePlacement::PostStaticPrefix`];
    /// override afterwards via [`Self::with_cache_placement`].
    pub const fn with_formats(serializer: S, deserializer: D) -> Self {
        Self {
            serializer,
            deserializer,
            cache_placement: CachePlacement::PostStaticPrefix,
        }
    }

    /// Override the static-prefix cache placement. Chainable with
    /// [`Self::with_formats`] so a caller assembles the configured
    /// adapter in one expression.
    #[must_use]
    pub const fn with_cache_placement(mut self, placement: CachePlacement) -> Self {
        self.cache_placement = placement;
        self
    }

    /// Inspect the configured static-prefix cache placement.
    #[must_use]
    pub const fn cache_placement(&self) -> CachePlacement {
        self.cache_placement
    }

    /// Format the system message: field descriptions + optional variant
    /// shapes block + format template + instructions.
    ///
    /// The variant-shapes block fires only when at least one output field
    /// transitively contains a [`OneOf`](FieldType::OneOf) or
    /// [`AnyOf`](FieldType::AnyOf). It carries the per-arm structural
    /// details that the one-line type label can't — without it, the LLM
    /// has only the discriminator tag set and no per-arm field shape.
    fn format_system_message(signature: &Signature) -> String {
        let mut parts = Vec::new();

        // Field descriptions
        parts.push(Self::format_field_descriptions(signature));

        // Variant shapes — only when any output field transitively
        // contains a sum-type variant.
        if let Some(block) = Self::format_variant_shapes(signature) {
            parts.push(block);
        }

        // Format template
        parts.push(Self::format_template(signature));

        // Instructions
        parts.push(format!(
            "In adhering to this structure, your objective is:\n{}",
            signature.instructions()
        ));

        parts.join("\n\n")
    }

    /// Build the "Variant shapes" block. Returns `None` when no output
    /// field has a transitively-reachable [`OneOf`](FieldType::OneOf) or
    /// [`AnyOf`](FieldType::AnyOf); the system message then omits the
    /// block entirely so non-variant signatures stay byte-identical to
    /// the prior prompt shape.
    fn format_variant_shapes(signature: &Signature) -> Option<String> {
        let mut sections: Vec<String> = Vec::new();
        for field in signature.output_fields() {
            walk_variant_shapes(&field.name, &field.field_type, &mut sections);
        }
        if sections.is_empty() {
            None
        } else {
            Some(format!("Variant shapes:\n{}", sections.join("\n")))
        }
    }

    /// Format the field descriptions section.
    fn format_field_descriptions(signature: &Signature) -> String {
        let mut lines = Vec::new();

        lines.push("Your input fields are:".to_owned());
        for (i, field) in signature.input_fields().enumerate() {
            lines.push(format_field_line(i, field));
        }

        lines.push(String::new());
        lines.push("Your output fields are:".to_owned());
        for (i, field) in signature.output_fields().enumerate() {
            lines.push(format_field_line(i, field));
        }

        lines.join("\n")
    }

    /// Format the template section showing the expected delimiter structure.
    fn format_template(signature: &Signature) -> String {
        let mut lines = Vec::new();
        lines.push(
            "All interactions will be structured in the following way, \
             with the appropriate values filled in."
                .to_owned(),
        );
        // Tell the model that fields whose declared type is optional may
        // be omitted entirely from the response when they are not
        // applicable — `Nullable<T>` declared outputs are dropped from
        // the missing-fields check at parse time, and admitting that
        // here removes the most common cause of stranded-null outputs
        // (model emits `null` for an inapplicable optional field whose
        // schema doesn't accept null, triggering a deterministic retry
        // loop in the application validator).
        if signature
            .output_fields()
            .any(|f| matches!(f.field_type, FieldType::Nullable(_)))
        {
            lines.push(
                "Fields whose type is `optional[...]` may be omitted entirely when not \
                 applicable — drop the marker; do not emit a literal `null` unless the \
                 schema explicitly accepts it."
                    .to_owned(),
            );
        }
        lines.push(String::new());

        for field in signature.input_fields() {
            lines.push(format_marker(&field.name));
            lines.push(format!("{{{}}}", field.name));
            lines.push(String::new());
        }

        for field in signature.output_fields() {
            lines.push(format_marker(&field.name));
            match field.field_type.output_format_hint() {
                Some(hint) => lines.push(format!("{{{}}}        # note: {hint}", field.name)),
                None => lines.push(format!("{{{}}}", field.name)),
            }
            lines.push(String::new());
        }

        lines.push(format_marker("completed"));

        lines.join("\n")
    }

    /// A concise output-format reminder for the live user message. Restates
    /// each output field's marker and type note, ending with the `completed`
    /// marker. The system-message template can drift out of the model's
    /// attention as demos / history grow; this reminder rides in the final
    /// user turn where it is freshest. Mirrors DSPy's
    /// `user_message_output_requirements`.
    ///
    /// Wording is "Respond with these output fields:" (no "in order") —
    /// the buffered parser collects markers into a `BTreeMap` keyed by
    /// field name and is order-independent. Claiming order matters when
    /// it doesn't gives the model a constraint we don't enforce.
    fn output_requirements(signature: &Signature) -> String {
        let mut lines = vec!["Respond with these output fields:".to_owned()];
        for field in signature.output_fields() {
            let marker = format_marker(&field.name);
            match field.field_type.output_format_hint() {
                Some(hint) => lines.push(format!("{marker} — {hint}")),
                None => lines.push(marker),
            }
        }
        lines.push(format!("Then end with {}.", format_marker("completed")));
        lines.join("\n")
    }

    /// Serialize field values into delimited text content.
    fn format_delimited_fields(&self, fields: &[(&FieldDef, &FieldValue)]) -> Result<String> {
        let mut parts = Vec::new();
        for (field_def, value) in fields {
            let serialized = self.serializer.serialize(value, &field_def.field_type)?;
            parts.push(format!(
                "{}\n{}",
                format_marker(&field_def.name),
                serialized
            ));
        }
        Ok(parts.join("\n\n"))
    }

    /// Build the final user message's content-parts list, emitting a
    /// real `ContentPart::Image / Document / Audio / Video` for every
    /// media-typed input slot and keeping text slots in a delimited
    /// `[[ ## name ## ]]` envelope so the model sees the same prompt
    /// shape it would for an all-text signature.
    fn format_input_parts(&self, fields: &[(&FieldDef, &FieldValue)]) -> Result<Vec<ContentPart>> {
        let mut parts: Vec<ContentPart> = Vec::new();
        let mut text_buf = String::new();
        let mut emitted_first_text = false;

        // A cache breakpoint is inserted right after the last `cacheable`
        // input field's content, so a stable head (e.g. a constant
        // dataContext) is cached while the per-call tail is not. `None`
        // when no input slot is declared cacheable — then there is no
        // user-message breakpoint, only the static-prefix one.
        let last_cacheable_idx = fields
            .iter()
            .rposition(|(field_def, _)| field_def.cacheable);

        let flush_buf = |buf: &mut String, parts: &mut Vec<ContentPart>| {
            if !buf.is_empty() {
                parts.push(ContentPart::text(std::mem::take(buf)));
            }
        };

        for (idx, (field_def, value)) in fields.iter().enumerate() {
            if emitted_first_text {
                text_buf.push_str("\n\n");
            }
            if matches!(field_def.field_type, FieldType::Media { .. }) {
                let media = match value {
                    FieldValue::Media(m) => m,
                    other => {
                        return Err(PredictError::FieldTypeMismatch {
                            field: field_def.name.clone(),
                            expected: "media value".into(),
                            actual: format!("{other:?}"),
                        });
                    }
                };
                // Flush any text accumulated up to this point so the
                // model receives an interleaved sequence — the marker
                // for the media slot lands as the last line of the
                // preceding text part, immediately before the
                // ContentPart::<modality>.
                text_buf.push_str(&format_marker(&field_def.name));
                flush_buf(&mut text_buf, &mut parts);
                let part = match media.kind {
                    modelplease::MediaKind::Image => ContentPart::image(media.source.clone()),
                    modelplease::MediaKind::Document => {
                        ContentPart::document(media.source.clone(), None)
                    }
                    modelplease::MediaKind::Audio => ContentPart::audio(media.source.clone()),
                    modelplease::MediaKind::Video => ContentPart::video(media.source.clone()),
                };
                parts.push(part);
            } else {
                let serialized = self.serializer.serialize(value, &field_def.field_type)?;
                text_buf.push_str(&format_marker(&field_def.name));
                text_buf.push('\n');
                text_buf.push_str(&serialized);
            }
            emitted_first_text = true;

            // Close the cacheable head after the last cacheable field.
            // Flushing first makes that field's text its own content
            // part, so the breakpoint lands exactly at the head/tail
            // boundary (the concatenated text is unchanged vs. no split).
            if Some(idx) == last_cacheable_idx {
                flush_buf(&mut text_buf, &mut parts);
                parts.push(ContentPart::cache_breakpoint());
            }
        }
        flush_buf(&mut text_buf, &mut parts);
        Ok(parts)
    }
}

#[async_trait]
impl<S: FieldSerializer, D: FieldDeserializer> Adapter for ChatAdapter<S, D> {
    async fn format(
        &self,
        signature: &Signature,
        inputs: &BTreeMap<String, FieldValue>,
        demos: &[Demo],
    ) -> Result<Vec<Message>> {
        let mut messages = Vec::new();

        // System message
        messages.push(Message::system(Self::format_system_message(signature)));

        // Demo pairs
        for demo in demos {
            // User message: input fields
            let input_fields: Vec<_> = signature
                .input_fields()
                .filter_map(|f| demo.inputs.get(&f.name).map(|v| (f, v)))
                .collect();
            let content = self.format_delimited_fields(&input_fields)?;
            messages.push(Message::user(content));

            // Assistant message: output fields + completed marker
            let output_fields: Vec<_> = signature
                .output_fields()
                .filter_map(|f| demo.outputs.get(&f.name).map(|v| (f, v)))
                .collect();
            let mut content = self.format_delimited_fields(&output_fields)?;
            content.push_str("\n\n");
            content.push_str(&format_marker("completed"));
            messages.push(Message::assistant(content));
        }

        // Static-prefix cache breakpoint placement. Each variant of
        // [`CachePlacement`] documents the cached span it produces;
        // the dispatch here keeps the message tree consistent so a
        // downstream provider can translate the marker to its native
        // cachePoint without inspecting the policy. `None` skips the
        // marker entirely — per-input cacheable-field breakpoints
        // emitted inside the user-message path below are independent
        // and still fire when the signature declares them.
        match self.cache_placement {
            CachePlacement::PostStaticPrefix => {
                // Last static message: system if no demos, else the
                // last demo's assistant turn. Covers system + demos
                // in the cached span.
                if let Some(last) = messages.last_mut() {
                    last.content.push(ContentPart::cache_breakpoint());
                }
            }
            CachePlacement::PostSystemPrompt => {
                // System message only: demos sit AFTER the
                // breakpoint and don't participate in the cached
                // span. Index 0 is the system message — `push` since
                // it was just emitted as `Message::system(text)` and
                // has a single text part; adding the breakpoint
                // after preserves the text content.
                if let Some(system) = messages.first_mut() {
                    system.content.push(ContentPart::cache_breakpoint());
                }
            }
            CachePlacement::None => {
                // Adapter-driven caching disabled. Nothing to do.
            }
        }

        // Final user message: actual input values.
        //
        // Media-typed fields can't be text-serialized — they emit a
        // dedicated [`ContentPart`] (Image / Document / Audio / Video).
        // Build a `Vec<ContentPart>` interleaving text-field delimiters
        // with media parts so an image input lands as a real
        // multimodal content part the provider can route to the wire.
        let input_fields: Vec<_> = signature
            .input_fields()
            .filter_map(|f| inputs.get(&f.name).map(|v| (f, v)))
            .collect();
        let mut parts = self.format_input_parts(&input_fields)?;
        parts.push(ContentPart::text(Self::output_requirements(signature)));
        messages.push(Message::with_parts(Role::User, parts));

        // Operator-opt-in prompt visibility. The signature/serialization
        // expressions live inside the macro so they only run when the
        // target is enabled at TRACE — serializing a 30k-token prompt on
        // every disabled call would be catastrophic. See the crate-level
        // doc for the RUST_LOG / RUST_LOG invocation.
        tracing::trace!(
            target: "typesayer::adapter::chat::messages",
            signature = %{
                let inputs =
                    signature.input_fields().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", ");
                let outputs =
                    signature.output_fields().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", ");
                format!("{inputs} -> {outputs}")
            },
            messages_json = %serde_json::to_string(&messages).unwrap_or_default(),
            "assembled chat-adapter prompt",
        );

        Ok(messages)
    }

    async fn parse(
        &self,
        signature: &Signature,
        completion: &str,
    ) -> Result<BTreeMap<String, FieldValue>> {
        let output_fields: Vec<&FieldDef> = signature.output_fields().collect();
        let output_names: HashSet<&str> = output_fields.iter().map(|f| f.name.as_str()).collect();

        tracing::debug!(
            target: "typesayer::adapter::chat",
            output_field_count = output_fields.len(),
            completion_bytes = completion.len(),
            "parse start"
        );

        // Collect all marker positions and field names. Group 0 is the full
        // match and group 1 is the captured field name — both are guaranteed
        // present by the regex (see `FIELD_MARKER`). filter_map drops any
        // hypothetical misses rather than panicking.
        let captures: Vec<(usize, usize, String)> = FIELD_MARKER
            .captures_iter(completion)
            .filter_map(|cap| {
                let m = cap.get(0)?;
                let name = cap.get(1)?.as_str().to_owned();
                Some((m.start(), m.end(), name))
            })
            .collect();

        // The `completed` marker is a sentinel, not a field-naming marker.
        // Treat "only the completed sentinel present" the same as "no
        // markers at all" for single-output fallback purposes —
        // instruction-tuned models routinely emit the visible "I'm done"
        // sentinel while dropping the opening field marker as
        // boilerplate, which left otherwise-correct single-output
        // completions failing with spurious `MissingFields`.
        let has_real_marker = captures.iter().any(|(_, _, name)| name != "completed");

        if !has_real_marker {
            if output_fields.len() == 1 {
                let field = output_fields[0];
                // Silent heuristic by design — but tracing makes it
                // observable so an operator can confirm "yes, the
                // single-output completion was parsed via the no-marker
                // path" instead of guessing whether the model output
                // even had markers.
                tracing::debug!(
                    target: "typesayer::adapter::chat",
                    field = %field.name,
                    "single-output no-marker fallback fired"
                );
                let end = captures
                    .iter()
                    .find(|(_, _, n)| n == "completed")
                    .map_or(completion.len(), |(start, _, _)| *start);
                let raw = completion[..end].trim();
                let value = self
                    .deserializer
                    .deserialize(raw, &field.name, &field.field_type)?;
                return Ok(BTreeMap::from([(field.name.clone(), value)]));
            }
            // Multi-output completion with no real markers — sometimes
            // the model emits only the `completed` sentinel and drops
            // every field opener. Surfacing the raw excerpt at warn
            // gives early observability before the error climbs the
            // call stack and gets formatted for a user response.
            let only_completed =
                !captures.is_empty() && captures.iter().all(|(_, _, n)| n == "completed");
            tracing::warn!(
                target: "typesayer::adapter::chat",
                only_completed_sentinel = only_completed,
                output_field_count = output_fields.len(),
                completion_bytes = completion.len(),
                error_kind = "no_field_markers",
                "parse failed: no real field markers found"
            );
            return Err(PredictError::no_field_markers_from_parse(
                output_fields.iter().map(|f| format_marker(&f.name)),
                completion,
            ));
        }

        // Extract content between consecutive markers
        let mut fields: BTreeMap<String, FieldValue> = BTreeMap::new();

        for (i, (marker_start, marker_end, field_name)) in captures.iter().enumerate() {
            // Skip completed marker and non-output fields
            if field_name == "completed" || !output_names.contains(field_name.as_str()) {
                continue;
            }

            // First occurrence wins — duplicate marker gets observable so
            // an operator who sees a "wrong value won" outcome can trace
            // it to a duplicate emission instead of a parser bug.
            if fields.contains_key(field_name) {
                tracing::warn!(
                    target: "typesayer::adapter::chat",
                    field = %field_name,
                    duplicate_at_byte = marker_start,
                    "duplicate marker discarded; first occurrence wins"
                );
                continue;
            }

            // Content runs from end of this marker to start of next marker (or end of string)
            let content_end = captures
                .get(i + 1)
                .map_or(completion.len(), |(start, _, _)| *start);

            let raw = completion[*marker_end..content_end].trim();

            // Find the field definition to get the type. output_names is
            // derived from output_fields one line above, so a hit in the
            // HashSet guarantees the Vec lookup succeeds — continue rather
            // than panic in the impossible case.
            let Some(field_def) = output_fields.iter().find(|f| f.name == *field_name) else {
                continue;
            };

            let value =
                self.deserializer
                    .deserialize(raw, &field_def.name, &field_def.field_type)?;
            fields.insert(field_name.clone(), value);
        }

        // Check for missing fields — only fields whose declared type is
        // non-nullable count as required. Pipelines like the help-docs
        // search agent declare `sql` and `result` both as outputs and
        // expect the model to emit ONE based on the chosen `action`;
        // the schema-layer marks the conditional ones nullable. Walking
        // every output_field unconditionally produced spurious
        // "missing output fields: [sql]" errors when the model
        // correctly emitted only the action-relevant slot.
        let missing: Vec<String> = output_fields
            .iter()
            .filter(|f| !matches!(f.field_type, FieldType::Nullable(_)))
            .filter(|f| !fields.contains_key(&f.name))
            .map(|f| f.name.clone())
            .collect();

        if !missing.is_empty() {
            tracing::warn!(
                target: "typesayer::adapter::chat",
                missing_fields = ?missing,
                output_field_count = output_fields.len(),
                completion_bytes = completion.len(),
                error_kind = "missing_fields",
                "parse failed: model emitted at least one marker but \
                 missed required output fields"
            );
            return Err(PredictError::missing_fields_from_parse(
                missing,
                output_fields.iter().map(|f| f.name.clone()),
                completion,
            ));
        }

        Ok(fields)
    }
}

/// Format a field marker: `[[ ## name ## ]]`
fn format_marker(name: &str) -> String {
    format!("[[ ## {name} ## ]]")
}

/// Render one numbered field line in the system message's field-descriptions
/// section. Single-line descriptions inline after the type label as before;
/// multi-line descriptions break to their own indented block so the second
/// line doesn't visually attach to the next field's header. When the field
/// carries JSON Schema `examples`, those are rendered as their own indented
/// block under the description so the model sees concrete "good output"
/// without bloating the instructions string.
fn format_field_line(idx: usize, field: &FieldDef) -> String {
    let header = format!(
        "{}. `{}` ({})",
        idx + 1,
        field.name,
        field.field_type.type_label()
    );
    let mut out = if field.description.contains('\n') {
        let body = field
            .description
            .lines()
            .map(|line| format!("     {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("{header}:\n{body}")
    } else {
        format!("{header}: {}", field.description)
    };
    if !field.examples.is_empty() {
        out.push_str("\n     Examples:");
        for example in &field.examples {
            out.push_str("\n       ");
            out.push_str(&format_example_value(example));
        }
    }
    out
}

/// Per-example serialization budget — keeps a long example payload from
/// blowing out one field-line in the system prompt while still surfacing
/// enough of the shape that the model can pattern-match.
const EXAMPLE_RENDER_BYTES: usize = 200;

/// Format one example value for the prompt: compact JSON, truncated to
/// [`EXAMPLE_RENDER_BYTES`] on a char boundary with an ellipsis when
/// over budget. Falls back to a debug rendering if JSON serialization
/// fails — unreachable for any value sourced from `serde_json::Value`,
/// but the fallback keeps the prompt path infallible.
fn format_example_value(value: &serde_json::Value) -> String {
    let json = serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"));
    if json.len() <= EXAMPLE_RENDER_BYTES {
        return json;
    }
    let mut end = EXAMPLE_RENDER_BYTES;
    while end > 0 && !json.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… ({} bytes truncated)", &json[..end], json.len() - end)
}

/// Walk a FieldType tree rooted at `path` and append per-variant
/// description sections to `out` for every transitively-reachable
/// [`OneOf`](FieldType::OneOf) or [`AnyOf`](FieldType::AnyOf).
///
/// `path` follows a dotted-and-bracketed JSON-pointer-style convention so
/// each section names exactly which sub-field the variant lives at:
/// `assignment` (top-level), `results[].assignment` (inside a list),
/// `response.payload` (inside an object), etc. Composed cases like
/// `Nullable<List<OneOf<...>>>` recurse through the wrappers.
fn walk_variant_shapes(path: &str, ft: &FieldType, out: &mut Vec<String>) {
    use std::fmt::Write as _;

    use typesayer_types::field::VariantArm;

    // Indentation hierarchy:
    //   "  At `path` ..."         (2 spaces — section header)
    //   "    - condition: desc"   (4 spaces — arm bullet, deeper than header)
    //   "        {type-label}"    (8 spaces — payload shape, deeper than bullet)
    // The deepening makes each level scannable in the LLM context window
    // without relying on punctuation alone.
    fn render_arm(arm: &VariantArm, header: &str) -> String {
        format!(
            "    - {header}: {}\n        {}",
            arm.description,
            arm.field_type.type_label()
        )
    }

    match ft {
        FieldType::OneOf {
            arms,
            discriminator,
        } => {
            let mut section = String::new();
            if let Some(disc) = discriminator {
                let _ = write!(
                    section,
                    "  At `{path}` (discriminator: `{}`):",
                    disc.property
                );
                for (i, arm) in arms.iter().enumerate() {
                    let tag = disc.tags.get(i).map_or("?", String::as_str);
                    section.push('\n');
                    section.push_str(&render_arm(arm, &format!("{} == \"{tag}\"", disc.property)));
                }
            } else {
                let _ = write!(
                    section,
                    "  At `{path}` (untagged; emit a JSON value matching exactly one of the \
                     shapes below):"
                );
                for (i, arm) in arms.iter().enumerate() {
                    section.push('\n');
                    section.push_str(&render_arm(arm, &format!("Shape {}", i + 1)));
                }
            }
            out.push(section);

            // Recurse into each arm so a OneOf-of-OneOf surfaces every
            // level. The nested path conveys the arm selection condition:
            // `root[kind="outer_a"].nested` is JSON-path-style notation that
            // makes "this block applies when root.kind == outer_a" visible
            // in the path itself, avoiding a separate condition clause and
            // matching the way schema authors document conditional paths.
            for (i, arm) in arms.iter().enumerate() {
                let nested_path = discriminator.as_ref().map_or_else(
                    || format!("{path}[arm{i}]"),
                    |d| {
                        let tag = d.tags.get(i).map_or("?", String::as_str);
                        format!("{path}[{}=\"{tag}\"]", d.property)
                    },
                );
                walk_variant_shapes(&nested_path, &arm.field_type, out);
            }
        }
        FieldType::AnyOf { arms } => {
            let mut section = format!(
                "  At `{path}` (any-of; emit one value matching one of the shapes below — \
                 first match wins on parse):"
            );
            for (i, arm) in arms.iter().enumerate() {
                section.push('\n');
                section.push_str(&render_arm(arm, &format!("Shape {}", i + 1)));
            }
            out.push(section);
            for (i, arm) in arms.iter().enumerate() {
                let nested_path = format!("{path}[arm{i}]");
                walk_variant_shapes(&nested_path, &arm.field_type, out);
            }
        }
        FieldType::List(inner) => {
            walk_variant_shapes(&format!("{path}[]"), inner, out);
        }
        FieldType::Nullable(inner) => walk_variant_shapes(path, inner, out),
        FieldType::Map(value_type) => {
            walk_variant_shapes(&format!("{path}{{}}"), value_type, out);
        }
        FieldType::Object(fields) => {
            for f in fields {
                walk_variant_shapes(&format!("{path}.{}", f.name), &f.field_type, out);
            }
        }
        FieldType::String
        | FieldType::Int
        | FieldType::Float
        | FieldType::Bool
        | FieldType::Enum(_)
        | FieldType::Media { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use typesayer_types::field::FieldType;

    use super::*;

    fn qa_signature() -> Signature {
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

    fn multi_output_signature() -> Signature {
        Signature::builder("Answer with reasoning.")
            .input(FieldDef::input(
                "question",
                FieldType::String,
                "The user question",
            ))
            .output(FieldDef::output(
                "reasoning",
                FieldType::String,
                "Step by step reasoning",
            ))
            .output(FieldDef::output(
                "answer",
                FieldType::String,
                "The final answer",
            ))
            .build()
            .unwrap()
    }

    fn list_output_signature() -> Signature {
        Signature::builder("Generate search queries.")
            .input(FieldDef::input("topic", FieldType::String, "The topic"))
            .output(FieldDef::output(
                "queries",
                FieldType::List(Box::new(FieldType::String)),
                "Search queries",
            ))
            .build()
            .unwrap()
    }

    // --- output-format guidance tests ---

    #[tokio::test]
    async fn system_message_states_list_output_format() {
        let adapter = ChatAdapter::default();
        let sig = list_output_signature();
        let inputs = BTreeMap::from([("topic".into(), FieldValue::Str("rust".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        // The template must state the concrete serialization for the list
        // output, not just the `list[str]` type label.
        let system = messages[0].text();
        assert!(
            system.contains("JSON array"),
            "system message should state the array format:\n{system}"
        );
    }

    #[tokio::test]
    async fn user_message_includes_output_requirements() {
        let adapter = ChatAdapter::default();
        let sig = list_output_signature();
        let inputs = BTreeMap::from([("topic".into(), FieldValue::Str("rust".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        // A reminder rides in the live user message (fights context drift):
        // names each output marker + its format note, ending at `completed`.
        let user = messages.last().unwrap().text();
        assert!(
            user.contains("[[ ## queries ## ]]"),
            "user message should list the output marker:\n{user}"
        );
        assert!(
            user.contains("JSON array"),
            "user message reminder should carry the format note:\n{user}"
        );
        assert!(
            user.contains("[[ ## completed ## ]]"),
            "user message should reference the completed marker:\n{user}"
        );
    }

    // --- format tests ---

    #[tokio::test]
    async fn format_system_message_contains_field_descriptions() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("What is 2+2?".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        let system = messages[0].text();
        assert!(system.contains("`question` (str)"));
        assert!(system.contains("`answer` (str)"));
        assert!(system.contains("Your input fields are:"));
        assert!(system.contains("Your output fields are:"));
    }

    #[tokio::test]
    async fn format_system_message_contains_template() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        let system = messages[0].text();
        assert!(system.contains("[[ ## question ## ]]"));
        assert!(system.contains("[[ ## answer ## ]]"));
        assert!(system.contains("[[ ## completed ## ]]"));
    }

    #[tokio::test]
    async fn format_system_message_contains_instructions() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        let system = messages[0].text();
        assert!(system.contains("Answer the question."));
    }

    #[tokio::test]
    async fn format_user_message_contains_delimited_inputs() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("What is 2+2?".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        // Last message is the user message
        let user = messages.last().unwrap().text();
        assert!(user.contains("[[ ## question ## ]]"));
        assert!(user.contains("What is 2+2?"));
    }

    #[tokio::test]
    async fn format_no_demos_produces_system_and_user() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        assert_eq!(messages.len(), 2); // system + user
        assert_eq!(messages[0].role, modelplease::Role::System);
        assert_eq!(messages[1].role, modelplease::Role::User);
    }

    #[tokio::test]
    async fn format_demos_produce_alternating_user_assistant_pairs() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("What is 3+3?".into()))]);

        let demos = vec![
            Demo {
                inputs: BTreeMap::from([(
                    "question".into(),
                    FieldValue::Str("What is 1+1?".into()),
                )]),
                outputs: BTreeMap::from([("answer".into(), FieldValue::Str("2".into()))]),
            },
            Demo {
                inputs: BTreeMap::from([(
                    "question".into(),
                    FieldValue::Str("What is 2+2?".into()),
                )]),
                outputs: BTreeMap::from([("answer".into(), FieldValue::Str("4".into()))]),
            },
        ];

        let messages = adapter.format(&sig, &inputs, &demos).await.unwrap();

        // system(1) + 2 demos × (user + assistant)(4) + final user(1) = 6
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].role, modelplease::Role::System);
        assert_eq!(messages[1].role, modelplease::Role::User);
        assert_eq!(messages[2].role, modelplease::Role::Assistant);
        assert_eq!(messages[3].role, modelplease::Role::User);
        assert_eq!(messages[4].role, modelplease::Role::Assistant);
        assert_eq!(messages[5].role, modelplease::Role::User); // final user
    }

    fn cacheable_signature() -> Signature {
        let mut ctx = FieldDef::input("dataContext", FieldType::String, "stable context");
        ctx.cacheable = true;
        Signature::builder("Extract fields.")
            .input(ctx)
            .input(FieldDef::input(
                "toolInput",
                FieldType::String,
                "per-call input",
            ))
            .output(FieldDef::output("result", FieldType::String, "the result"))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn format_marks_cache_breakpoint_at_end_of_static_prefix_no_demos() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        // No demos: the system message is the whole static prefix; its
        // final content part is the cache breakpoint.
        assert!(messages[0].content.last().unwrap().is_cache_breakpoint());
    }

    #[tokio::test]
    async fn cache_placement_post_system_prompt_places_breakpoint_on_system_only_no_demos() {
        // No demos: PostSystemPrompt and PostStaticPrefix coincide on
        // the system message. The breakpoint must land on the system
        // message regardless of placement choice.
        let adapter = ChatAdapter::default().with_cache_placement(CachePlacement::PostSystemPrompt);
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        assert!(messages[0].content.last().unwrap().is_cache_breakpoint());
        // No breakpoint leaked into the final user message — that
        // would change the prompt size on every call and undo caching.
        assert!(
            !messages[1]
                .content
                .iter()
                .any(ContentPart::is_cache_breakpoint)
        );
    }

    #[tokio::test]
    async fn cache_placement_post_system_prompt_keeps_demos_outside_cached_span() {
        // Demos present: PostSystemPrompt MUST place the breakpoint
        // on the system message (not after demos), so the cached span
        // is exactly the system message. This is the customer-facing
        // guarantee downstream consumers need: byte-identical system prompts
        // get byte-identical cached spans even when demos vary.
        let adapter = ChatAdapter::default().with_cache_placement(CachePlacement::PostSystemPrompt);
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let demos = vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("q".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("a".into()))]),
        }];
        let messages = adapter.format(&sig, &inputs, &demos).await.unwrap();
        // [system, demoUser, demoAssistant, finalUser]: PostSystemPrompt
        // means messages[0] (system) carries the breakpoint, NOT
        // messages[2] (last demo assistant) as PostStaticPrefix would.
        assert!(
            messages[0].content.last().unwrap().is_cache_breakpoint(),
            "system message must carry the breakpoint under PostSystemPrompt"
        );
        assert!(
            !messages[2]
                .content
                .iter()
                .any(ContentPart::is_cache_breakpoint),
            "demo assistant must NOT carry the breakpoint under PostSystemPrompt"
        );
    }

    #[tokio::test]
    async fn cache_placement_none_emits_no_static_prefix_breakpoint() {
        // `None` disables adapter-driven caching entirely. Neither
        // the system message nor any demo message carries the
        // marker — per-input cacheable fields still emit
        // independently (covered by `format_inserts_user_breakpoint_*`
        // tests above), but the static prefix is uncached.
        let adapter = ChatAdapter::default().with_cache_placement(CachePlacement::None);
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let demos = vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("q".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("a".into()))]),
        }];
        let messages = adapter.format(&sig, &inputs, &demos).await.unwrap();
        for (i, msg) in messages.iter().enumerate() {
            assert!(
                !msg.content.iter().any(ContentPart::is_cache_breakpoint),
                "CachePlacement::None must not emit any cache breakpoint; \
                 message {i} ({:?}) carried one",
                msg.role
            );
        }
    }

    #[tokio::test]
    async fn cache_placement_default_matches_post_static_prefix_behavior() {
        // Sanity check: ChatAdapter::default() preserves the
        // historical placement so existing callers see no change
        // without opting in.
        let adapter = ChatAdapter::default();
        assert_eq!(adapter.cache_placement(), CachePlacement::PostStaticPrefix);
    }

    #[tokio::test]
    async fn format_marks_cache_breakpoint_on_last_demo_when_present() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let demos = vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("q".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("a".into()))]),
        }];
        let messages = adapter.format(&sig, &inputs, &demos).await.unwrap();
        // [system, demoUser, demoAssistant, finalUser]: the static prefix
        // ends at the demo assistant, which carries the breakpoint — the
        // system message does not.
        assert!(
            !messages[0]
                .content
                .iter()
                .any(ContentPart::is_cache_breakpoint)
        );
        assert!(messages[2].content.last().unwrap().is_cache_breakpoint());
    }

    #[tokio::test]
    async fn format_inserts_user_breakpoint_after_last_cacheable_field() {
        let adapter = ChatAdapter::default();
        let sig = cacheable_signature();
        let inputs = BTreeMap::from([
            ("dataContext".into(), FieldValue::Str("stable".into())),
            ("toolInput".into(), FieldValue::Str("dynamic".into())),
        ]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let user = messages.last().unwrap();
        let bp = user
            .content
            .iter()
            .position(ContentPart::is_cache_breakpoint)
            .expect("user message should carry a cache breakpoint");
        // The cacheable head mentions dataContext; the dynamic tail
        // (toolInput) sits after the breakpoint and must not leak into
        // the cached head.
        let head: String = user.content[..bp]
            .iter()
            .filter_map(ContentPart::as_text)
            .collect();
        let tail: String = user.content[bp + 1..]
            .iter()
            .filter_map(ContentPart::as_text)
            .collect();
        assert!(head.contains("dataContext"), "head: {head}");
        assert!(tail.contains("toolInput"), "tail: {tail}");
        assert!(
            !head.contains("toolInput"),
            "dynamic field leaked into cached head: {head}"
        );
    }

    #[tokio::test]
    async fn format_no_user_breakpoint_without_cacheable_field() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let user = messages.last().unwrap();
        assert!(!user.content.iter().any(ContentPart::is_cache_breakpoint));
    }

    #[tokio::test]
    async fn format_demo_assistant_contains_completed_marker() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);
        let demos = vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("What is 1+1?".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("2".into()))]),
        }];

        let messages = adapter.format(&sig, &inputs, &demos).await.unwrap();
        let assistant = messages[2].text();
        assert!(assistant.contains("[[ ## completed ## ]]"));
        assert!(assistant.contains("[[ ## answer ## ]]"));
        assert!(assistant.contains('2'));
    }

    #[tokio::test]
    async fn format_typed_values_serialized_correctly() {
        let sig = Signature::builder("Count items.")
            .input(FieldDef::input(
                "items",
                FieldType::List(Box::new(FieldType::String)),
                "Items to count",
            ))
            .output(FieldDef::output("count", FieldType::Int, "Number of items"))
            .build()
            .unwrap();

        let adapter = ChatAdapter::default();
        let inputs = BTreeMap::from([(
            "items".into(),
            FieldValue::List(vec![
                FieldValue::Str("a".into()),
                FieldValue::Str("b".into()),
            ]),
        )]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let user = messages.last().unwrap().text();
        assert!(user.contains(r#"["a","b"]"#));
    }

    // --- parse tests ---

    #[tokio::test]
    async fn parse_well_formed_completion() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let completion = "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(result["answer"], FieldValue::Str("Paris".into()));
    }

    #[tokio::test]
    async fn parse_markers_mid_line() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        // Marker appears mid-line — the core robustness fix over DSPy
        let completion = "Some prefix[[ ## answer ## ]]Paris[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(result["answer"], FieldValue::Str("Paris".into()));
    }

    #[tokio::test]
    async fn parse_missing_completed_marker() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        // No completed marker — content runs to end of string
        let completion = "[[ ## answer ## ]]\nParis";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(result["answer"], FieldValue::Str("Paris".into()));
    }

    #[tokio::test]
    async fn parse_single_output_no_markers_fallback() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        // MIPROv2 edge case: no markers at all, single output field
        let completion = "The answer is Paris";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(
            result["answer"],
            FieldValue::Str("The answer is Paris".into())
        );
    }

    #[tokio::test]
    async fn parse_single_output_only_completed_marker_fallback() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        // Instruction-tuned models routinely emit the closing sentinel
        // while dropping the opening field marker on single-output
        // signatures, treating the visible "I'm done" sentinel as
        // boilerplate. The single-output fallback must cover this case.
        let completion = "Paris is the capital of France.\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(
            result["answer"],
            FieldValue::Str("Paris is the capital of France.".into())
        );
    }

    #[tokio::test]
    async fn parse_no_markers_multiple_outputs_fails() {
        let adapter = ChatAdapter::default();
        let sig = multi_output_signature();
        let completion = "Some text without markers";
        let result = adapter.parse(&sig, completion).await;
        match result.unwrap_err() {
            PredictError::NoFieldMarkers {
                expected_markers,
                raw_excerpt,
                raw_bytes_total,
            } => {
                // Surfacing what the parser hoped for + what the model emitted
                // is the whole point — assert both reach the operator.
                assert!(expected_markers.iter().any(|m| m.contains("reasoning")));
                assert!(expected_markers.iter().any(|m| m.contains("answer")));
                assert_eq!(raw_bytes_total, completion.len());
                assert!(raw_excerpt.contains("Some text"));
            }
            other => panic!("expected NoFieldMarkers, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parse_missing_fields_error() {
        let adapter = ChatAdapter::default();
        let sig = multi_output_signature();
        // Only answer field present, reasoning missing
        let completion = "[[ ## answer ## ]]\n42\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            PredictError::MissingFields {
                fields,
                expected,
                raw_excerpt,
                raw_bytes_total,
            } => {
                assert!(fields.contains(&"reasoning".to_owned()));
                // expected carries the full output set so the operator sees
                // "missing 1 of 2" context, not just the bare miss.
                assert!(expected.contains(&"reasoning".to_owned()));
                assert!(expected.contains(&"answer".to_owned()));
                assert_eq!(raw_bytes_total, completion.len());
                assert!(raw_excerpt.contains("[[ ## answer ## ]]"));
            }
            other => panic!("expected MissingFields, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parse_skips_nullable_outputs_when_missing() {
        // Pipelines with action-conditional outputs (conditional_search
        // emits `sql` OR `result` based on action, never both) declare
        // the conditional slots nullable. The parser should accept a
        // completion that includes only the present subset, not flag
        // the absent ones as missing.
        let sig = Signature::builder("Pick an action.")
            .input(FieldDef::input("query", FieldType::String, "User query"))
            .output(FieldDef::output(
                "action",
                FieldType::String,
                "Chosen action",
            ))
            .output(FieldDef::output(
                "sql",
                FieldType::Nullable(Box::new(FieldType::String)),
                "SQL to run when action is execute_sql",
            ))
            .output(FieldDef::output(
                "result",
                FieldType::Nullable(Box::new(FieldType::String)),
                "Final answer when action is finish",
            ))
            .build()
            .unwrap();

        let adapter = ChatAdapter::default();
        // Model picked `finish` and emitted only `result`. `sql` is
        // intentionally absent — that should not raise MissingFields.
        let completion =
            "[[ ## action ## ]]\nfinish\n[[ ## result ## ]]\nDone.\n[[ ## completed ## ]]";
        let fields = adapter.parse(&sig, completion).await.unwrap();
        assert!(fields.contains_key("action"));
        assert!(fields.contains_key("result"));
        assert!(!fields.contains_key("sql"));
    }

    #[tokio::test]
    async fn parse_type_mismatch_error() {
        let sig = Signature::builder("Get age.")
            .input(FieldDef::input("name", FieldType::String, "Name"))
            .output(FieldDef::output("age", FieldType::Int, "Age"))
            .build()
            .unwrap();

        let adapter = ChatAdapter::default();
        let completion = "[[ ## age ## ]]\nnot_a_number\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await;
        assert!(matches!(
            result,
            Err(PredictError::FieldTypeMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn parse_ignores_echoed_input_fields() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        // Completion echoes the input field — should be ignored
        let completion =
            "[[ ## question ## ]]\nWhat is 2+2?\n[[ ## answer ## ]]\n4\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result["answer"], FieldValue::Str("4".into()));
    }

    #[tokio::test]
    async fn parse_first_occurrence_wins() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        // Duplicate answer markers — first one wins
        let completion =
            "[[ ## answer ## ]]\nFirst\n[[ ## answer ## ]]\nSecond\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(result["answer"], FieldValue::Str("First".into()));
    }

    #[tokio::test]
    async fn parse_multiple_output_fields() {
        let adapter = ChatAdapter::default();
        let sig = multi_output_signature();
        let completion = "[[ ## reasoning ## ]]\n2+2=4 because math\n[[ ## answer ## ]]\n4\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(
            result["reasoning"],
            FieldValue::Str("2+2=4 because math".into())
        );
        assert_eq!(result["answer"], FieldValue::Str("4".into()));
    }

    #[tokio::test]
    async fn parse_typed_output() {
        let sig = Signature::builder("Count.")
            .input(FieldDef::input("text", FieldType::String, "text"))
            .output(FieldDef::output("count", FieldType::Int, "count"))
            .build()
            .unwrap();

        let adapter = ChatAdapter::default();
        let completion = "[[ ## count ## ]]\n42\n[[ ## completed ## ]]";
        let result = adapter.parse(&sig, completion).await.unwrap();
        assert_eq!(result["count"], FieldValue::Int(42));
    }

    // --- format + parse round-trip ---

    #[tokio::test]
    async fn format_parse_round_trip() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();

        // Format a demo's assistant message
        let demo_output: BTreeMap<String, FieldValue> =
            BTreeMap::from([("answer".into(), FieldValue::Str("Paris".into()))]);
        let output_fields: Vec<_> = sig
            .output_fields()
            .filter_map(|f| demo_output.get(&f.name).map(|v| (f, v)))
            .collect();
        let mut content = adapter.format_delimited_fields(&output_fields).unwrap();
        content.push_str("\n\n");
        content.push_str(&format_marker("completed"));

        // Parse it back
        let result = adapter.parse(&sig, &content).await.unwrap();
        assert_eq!(result["answer"], FieldValue::Str("Paris".into()));
    }

    // --- Media-aware format() ---

    #[tokio::test]
    async fn format_emits_image_content_part_for_media_field() {
        use enumset::EnumSet;
        use modelplease::{HttpsUrl, MediaKind, MediaSource, Role, SourceKind};

        let sig = Signature::builder("Describe the page in detail.")
            .input(FieldDef::input(
                "instructions",
                FieldType::String,
                "What the user wants extracted",
            ))
            .input(FieldDef::input(
                "page",
                FieldType::Media {
                    kind: MediaKind::Image,
                    accepted_sources: EnumSet::all(),
                },
                "The image to describe",
            ))
            .output(FieldDef::output(
                "summary",
                FieldType::String,
                "Description",
            ))
            .build()
            .unwrap();

        let inputs: BTreeMap<String, FieldValue> = BTreeMap::from([
            ("instructions".into(), FieldValue::Str("Be brief.".into())),
            (
                "page".into(),
                FieldValue::Media(typesayer_types::field::MediaValue {
                    kind: MediaKind::Image,
                    source: MediaSource::Url {
                        url: HttpsUrl::parse("https://example.com/page.png").unwrap(),
                    },
                }),
            ),
        ]);

        let adapter = ChatAdapter::default();
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();

        // system + final user message
        assert_eq!(messages.len(), 2);
        let user = &messages[1];
        assert_eq!(user.role, Role::User);

        // User message has 3 parts: Text (delimiter + serialized text input
        // + delimiter for the media slot), the Image content part, then the
        // trailing output-requirements reminder.
        assert_eq!(user.content.len(), 3);
        let first = user.content[0].as_text().expect("first part is text");
        assert!(first.contains("[[ ## instructions ## ]]"));
        assert!(first.contains("Be brief."));
        assert!(first.contains("[[ ## page ## ]]"));
        assert_eq!(user.content[1].media_kind(), Some(MediaKind::Image));
        match &user.content[1] {
            modelplease::ContentPart::Image {
                source: MediaSource::Url { url },
            } => {
                assert_eq!(url.as_str(), "https://example.com/page.png");
            }
            other => panic!("expected ContentPart::Image with Url source, got {other:?}"),
        }
        // Trailing reminder restates the output requirements after the inputs.
        let reminder = user.content[2]
            .as_text()
            .expect("third part is the reminder text");
        assert!(reminder.contains("[[ ## summary ## ]]"));
        // The default-EnumSet accepted_sources must not narrow at the
        // adapter — narrowing is the application preflight's job.
        let _ = SourceKind::Url;
    }

    // ============================================================
    // Phase 4: Variant shapes prompt-rendering tests
    // ============================================================

    use typesayer_types::field::{ObjectField, OneOfDiscriminator, VariantArm};

    fn tagged_oneof_assignment_signature() -> Signature {
        Signature::builder("Pick a tool.")
            .input(FieldDef::input("query", FieldType::String, "User query"))
            .output(FieldDef::output(
                "assignment",
                FieldType::OneOf {
                    arms: vec![
                        VariantArm {
                            description: "Monthly spend by category".into(),
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
                            description: "Top N merchants".into(),
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
                },
                "The selected tool",
            ))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn variant_shapes_block_appears_for_tagged_oneof_output() {
        let adapter = ChatAdapter::default();
        let sig = tagged_oneof_assignment_signature();
        let inputs = BTreeMap::from([("query".into(), FieldValue::Str("how much?".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let system = messages[0].content[0].as_text().expect("system text");

        assert!(
            system.contains("Variant shapes:"),
            "block missing: {system}"
        );
        assert!(
            system.contains("toolName"),
            "discriminator missing: {system}"
        );
        assert!(system.contains("monthly"), "tag missing: {system}");
        assert!(system.contains("ranked_items"), "tag missing: {system}");
        assert!(
            system.contains("Monthly spend by category"),
            "arm description missing: {system}"
        );
    }

    #[tokio::test]
    async fn variant_shapes_block_omitted_for_non_variant_signature() {
        let adapter = ChatAdapter::default();
        let sig = qa_signature();
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let system = messages[0].content[0].as_text().expect("system text");
        assert!(
            !system.contains("Variant shapes:"),
            "block leaked into non-variant sig"
        );
    }

    #[tokio::test]
    async fn variant_shapes_block_path_names_list_nesting() {
        let adapter = ChatAdapter::default();
        let inner_oneof = tagged_oneof_assignment_signature()
            .fields()
            .iter()
            .find(|f| f.name == "assignment")
            .unwrap()
            .field_type
            .clone();
        let sig = Signature::builder("Pick tools.")
            .input(FieldDef::input(
                "queries",
                FieldType::String,
                "User queries",
            ))
            .output(FieldDef::output(
                "results",
                FieldType::List(Box::new(inner_oneof)),
                "Selected tools",
            ))
            .build()
            .unwrap();
        let inputs = BTreeMap::from([("queries".into(), FieldValue::Str("how much?".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let system = messages[0].content[0].as_text().expect("system text");
        assert!(
            system.contains("`results[]`"),
            "list-path notation missing: {system}"
        );
    }

    // ============================================================
    // Layer 7: snapshot tests for the Variant-shapes prompt block.
    //
    // The block format is load-bearing for LLM accuracy. Snapshot
    // tests pin the exact rendered output so any format drift under
    // refactor produces an `insta` failure with a diff, forcing the
    // refactorer to acknowledge the change instead of letting it
    // slip past `contains()` assertions.
    // ============================================================

    fn extract_system_body(messages: &[modelplease::Message]) -> String {
        let system = messages
            .iter()
            .find(|m| m.role == modelplease::Role::System)
            .expect("system message present");
        system
            .content
            .iter()
            .filter_map(|p| match p {
                modelplease::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn rendered_system_for(sig: Signature) -> String {
        let adapter = ChatAdapter::default();
        let inputs = BTreeMap::from([("q".into(), FieldValue::Str("?".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        extract_system_body(&messages)
    }

    fn one_input_then_output(output_name: &str, output_type: FieldType) -> Signature {
        Signature::builder("Demo.")
            .input(FieldDef::input("q", FieldType::String, "Q"))
            .output(FieldDef::output(output_name, output_type, "the answer"))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn snapshot_top_level_tagged_oneof() {
        let ft = FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "Monthly spend".into(),
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
                    description: "Top N merchants".into(),
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
        };
        let body = rendered_system_for(one_input_then_output("assignment", ft)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_top_level_untagged_oneof() {
        let ft = FieldType::OneOf {
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
        };
        let body = rendered_system_for(one_input_then_output("result", ft)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_top_level_anyof() {
        let ft = FieldType::AnyOf {
            arms: vec![
                VariantArm {
                    description: "labeled".into(),
                    field_type: FieldType::String,
                },
                VariantArm {
                    description: "numeric".into(),
                    field_type: FieldType::Int,
                },
            ],
        };
        let body = rendered_system_for(one_input_then_output("data", ft)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_list_of_oneof() {
        let ft = FieldType::List(Box::new(FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "tool a".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "kind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec!["a".into()]),
                    }]),
                },
                VariantArm {
                    description: "tool b".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "kind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec!["b".into()]),
                    }]),
                },
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["a".into(), "b".into()],
            }),
        }));
        let body = rendered_system_for(one_input_then_output("results", ft)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_nullable_of_oneof() {
        let ft = FieldType::Nullable(Box::new(FieldType::OneOf {
            arms: vec![VariantArm {
                description: "single tool".into(),
                field_type: FieldType::Object(vec![ObjectField {
                    name: "kind".into(),
                    description: String::new(),
                    field_type: FieldType::Enum(vec!["only".into()]),
                }]),
            }],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["only".into()],
            }),
        }));
        let body = rendered_system_for(one_input_then_output("maybe_tool", ft)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_object_with_oneof_and_anyof_siblings() {
        let ft = FieldType::Object(vec![
            ObjectField {
                name: "a".into(),
                description: String::new(),
                field_type: FieldType::OneOf {
                    arms: vec![VariantArm {
                        description: "a-only".into(),
                        field_type: FieldType::Object(vec![ObjectField {
                            name: "kind".into(),
                            description: String::new(),
                            field_type: FieldType::Enum(vec!["x".into()]),
                        }]),
                    }],
                    discriminator: Some(OneOfDiscriminator {
                        property: "kind".into(),
                        tags: vec!["x".into()],
                    }),
                },
            },
            ObjectField {
                name: "b".into(),
                description: String::new(),
                field_type: FieldType::AnyOf {
                    arms: vec![
                        VariantArm {
                            description: "str".into(),
                            field_type: FieldType::String,
                        },
                        VariantArm {
                            description: "int".into(),
                            field_type: FieldType::Int,
                        },
                    ],
                },
            },
        ]);
        let body = rendered_system_for(one_input_then_output("composite", ft)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_nested_oneof_of_oneof() {
        let inner = FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "inner a".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "subkind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec!["alpha".into()]),
                    }]),
                },
                VariantArm {
                    description: "inner b".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "subkind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec!["beta".into()]),
                    }]),
                },
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "subkind".into(),
                tags: vec!["alpha".into(), "beta".into()],
            }),
        };
        let outer = FieldType::OneOf {
            arms: vec![
                VariantArm {
                    description: "outer with nested".into(),
                    field_type: FieldType::Object(vec![
                        ObjectField {
                            name: "kind".into(),
                            description: String::new(),
                            field_type: FieldType::Enum(vec!["outer_a".into()]),
                        },
                        ObjectField {
                            name: "nested".into(),
                            description: String::new(),
                            field_type: inner,
                        },
                    ]),
                },
                VariantArm {
                    description: "outer plain".into(),
                    field_type: FieldType::Object(vec![ObjectField {
                        name: "kind".into(),
                        description: String::new(),
                        field_type: FieldType::Enum(vec!["outer_b".into()]),
                    }]),
                },
            ],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["outer_a".into(), "outer_b".into()],
            }),
        };
        let body = rendered_system_for(one_input_then_output("root", outer)).await;
        insta::assert_snapshot!(body);
    }

    #[tokio::test]
    async fn snapshot_output_requirements_reminder_for_oneof() {
        // The user-message reminder is a separate surface from the
        // system block; pin its OneOf rendering too.
        let ft = FieldType::OneOf {
            arms: vec![VariantArm {
                description: "tool".into(),
                field_type: FieldType::Object(vec![ObjectField {
                    name: "kind".into(),
                    description: String::new(),
                    field_type: FieldType::Enum(vec!["x".into()]),
                }]),
            }],
            discriminator: Some(OneOfDiscriminator {
                property: "kind".into(),
                tags: vec!["x".into()],
            }),
        };
        let sig = one_input_then_output("assignment", ft);
        let adapter = ChatAdapter::default();
        let inputs = BTreeMap::from([("q".into(), FieldValue::Str("?".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let user = messages
            .iter()
            .find(|m| m.role == modelplease::Role::User)
            .expect("user message present");
        let reminder = user
            .content
            .iter()
            .filter_map(|p| match p {
                modelplease::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .next_back()
            .expect("text part present")
            .to_owned();
        insta::assert_snapshot!(reminder);
    }

    #[tokio::test]
    async fn field_with_examples_renders_indented_examples_block() {
        // JSON Schema `examples` on an output field surface in the
        // system prompt as an indented block under the description.
        // The block is the closest typesayer equivalent to
        // "this is what good output looks like" without bloating the
        // free-text instructions.
        let adapter = ChatAdapter::default();
        let sig = Signature::builder("Pick a bucket.")
            .input(FieldDef::input("q", FieldType::String, "Q"))
            .output(
                FieldDef::output(
                    "groupBy",
                    FieldType::Nullable(Box::new(FieldType::String)),
                    "How to bucket",
                )
                .with_examples(vec![
                    serde_json::json!("month"),
                    serde_json::json!("day"),
                    serde_json::json!(null),
                ]),
            )
            .build()
            .unwrap();
        let inputs = BTreeMap::from([("q".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let system = messages[0].content[0].as_text().expect("system text");
        // Examples header sits between this field's description and the
        // next section, indented to visually attach to the field.
        assert!(system.contains("`groupBy` (optional[str]): How to bucket\n     Examples:"));
        // Every example rendered as compact JSON on its own indented line.
        assert!(system.contains("       \"month\""));
        assert!(system.contains("       \"day\""));
        assert!(system.contains("       null"));
    }

    #[tokio::test]
    async fn multi_line_field_description_breaks_to_indented_block() {
        // A description containing newlines used to render the second
        // line flush-left and visually attached to the next field's
        // header line. The renderer now breaks to an indented block
        // when newlines are present so each line clearly belongs to
        // its declared field.
        let adapter = ChatAdapter::default();
        let multi = "How to bucket aggregation results.\n\
                     Use \"month\" for the most common case.\n\
                     Set to null when the result does not apply.";
        let sig = Signature::builder("Pick a bucketing.")
            .input(FieldDef::input("query", FieldType::String, "Q"))
            .output(FieldDef::output("groupBy", FieldType::String, multi))
            .build()
            .unwrap();
        let inputs = BTreeMap::from([("query".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let system = messages[0].content[0].as_text().expect("system text");
        // Header line ends with the colon — no trailing inline content.
        assert!(
            system.contains("`groupBy` (str):\n     How to bucket"),
            "header line should break to indented body:\n{system}"
        );
        // Every body line indented uniformly so they stay visually
        // attached to the header rather than the next field.
        assert!(system.contains("     Use \"month\""));
        assert!(system.contains("     Set to null"));
        // Single-line descriptions stay inline (unchanged) — assert on
        // the input field's render so we know the fast path still fires.
        assert!(system.contains("`query` (str): Q"));
    }

    #[tokio::test]
    async fn variant_shapes_block_renders_untagged_with_first_match_caveat() {
        let adapter = ChatAdapter::default();
        let sig = Signature::builder("Pick a payload.")
            .input(FieldDef::input("query", FieldType::String, "Q"))
            .output(FieldDef::output(
                "result",
                FieldType::AnyOf {
                    arms: vec![
                        VariantArm {
                            description: "string arm".into(),
                            field_type: FieldType::String,
                        },
                        VariantArm {
                            description: "int arm".into(),
                            field_type: FieldType::Int,
                        },
                    ],
                },
                "The result",
            ))
            .build()
            .unwrap();
        let inputs = BTreeMap::from([("query".into(), FieldValue::Str("hi".into()))]);
        let messages = adapter.format(&sig, &inputs, &[]).await.unwrap();
        let system = messages[0].content[0].as_text().expect("system text");
        assert!(system.contains("any-of"), "anyof framing missing: {system}");
        assert!(
            system.contains("first match"),
            "first-match caveat missing: {system}"
        );
    }
}
