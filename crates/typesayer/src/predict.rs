// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Predict — orchestrates format → generate → parse.

use std::collections::BTreeMap;

use modelplease::{GenerateRequest, LanguageModelConfig};
use typesayer_types::{
    PredictError,
    error::Result,
    field::{FieldDef, FieldType, FieldValue},
    signature::Signature,
};

use crate::{adapter::Demo, context::Context, prediction::Prediction};

/// Orchestrates a structured prediction: format a prompt from a signature,
/// call a language model, and parse the response into typed fields.
///
/// # Examples
///
/// ```rust
/// use std::{collections::BTreeMap, sync::Arc};
///
/// use modelplease::{DummyLM, ModelId};
/// use typesayer::{
///     ChatAdapter, Context, FieldDef, FieldType, FieldValue, Predict, Signature,
/// };
///
/// # async fn example() -> typesayer::Result<()> {
/// let sig = Signature::builder("Answer the question.")
///     .input(FieldDef::input("question", FieldType::String, "The question"))
///     .output(FieldDef::output("answer", FieldType::String, "The answer"))
///     .build()?;
///
/// let lm = DummyLM::sequential(vec!["[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into()]);
/// let ctx = Context {
///     provider: Arc::new(lm),
///     model: ModelId::new("test"),
///     adapter: Arc::new(ChatAdapter::default()),
/// };
///
/// let prediction = Predict::new(sig)
///     .call(
///         &BTreeMap::from([("question".into(), FieldValue::Str("Capital of France?".into()))]),
///         &ctx,
///     )
///     .await?;
///
/// assert_eq!(prediction.get::<String>("answer")?, "Paris");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Predict {
    pub signature: Signature,
    pub demos: Vec<Demo>,
    pub config: LanguageModelConfig,
}

impl Predict {
    /// Create a new Predict with the given signature, an empty demo list,
    /// and a default `LanguageModelConfig`. Override `demos` / `config`
    /// via struct-update syntax:
    ///
    /// ```ignore
    /// Predict { demos, ..Predict::new(sig) }
    /// ```
    #[must_use]
    pub fn new(signature: Signature) -> Self {
        Self {
            signature,
            demos: Vec::new(),
            config: LanguageModelConfig::default(),
        }
    }

    /// Construct a `Predict` whose signature is `signature` with a leading
    /// `reasoning: String` output field. Replaces a separate
    /// `ChainOfThought` type — calling this constructor instead of [`Self::new`]
    /// is the way to opt into chain-of-thought prompting.
    #[must_use]
    pub fn chain_of_thought(signature: Signature) -> Self {
        let signature = signature.with_prepended_output(FieldDef::output(
            "reasoning",
            FieldType::String,
            "Step-by-step reasoning process",
        ));
        Self::new(signature)
    }

    /// The signature this predict will use (including any reasoning field).
    #[must_use]
    pub const fn signature(&self) -> &Signature {
        &self.signature
    }

    /// The current demos.
    #[must_use]
    pub fn demos(&self) -> &[Demo] {
        &self.demos
    }

    /// The current language model configuration.
    #[must_use]
    pub const fn config(&self) -> &LanguageModelConfig {
        &self.config
    }

    /// Replace the demos on this predictor.
    pub fn set_demos(&mut self, demos: Vec<Demo>) {
        self.demos = demos;
    }

    /// Replace the language model configuration.
    pub fn set_config(&mut self, config: LanguageModelConfig) {
        self.config = config;
    }

    /// Update the instructions on the underlying signature.
    pub fn set_instructions(&mut self, instructions: impl Into<String>) {
        self.signature.set_instructions(instructions);
    }

    /// Serialize this predictor's state to a JSON value.
    ///
    /// The format includes `demos`, `signature` (with instructions and field
    /// descriptors), and DSPy-compatible keys (`traces`, `train`, `lm`).
    ///
    /// # Errors
    ///
    /// Returns a `PredictError` wrapping `serde_json` if a demo value fails
    /// to serialize (not expected for well-formed `FieldValue`s).
    pub fn dump_state(&self) -> Result<serde_json::Value> {
        let demo_values: Vec<serde_json::Value> = self
            .demos
            .iter()
            .map(|d| {
                // Flatten demo into a single dict for DSPy compat
                let mut flat = BTreeMap::new();
                flat.extend(d.inputs.iter().map(|(k, v)| (k.clone(), v.clone())));
                flat.extend(d.outputs.iter().map(|(k, v)| (k.clone(), v.clone())));
                serde_json::to_value(flat)
            })
            .collect::<std::result::Result<_, _>>()?;

        let sig_fields: Vec<serde_json::Value> = self
            .signature
            .fields()
            .iter()
            .map(|f| {
                serde_json::json!({
                    "prefix": format!("{}:", capitalize(&f.name)),
                    "description": format!("${{{}}}", f.name),
                })
            })
            .collect();

        Ok(serde_json::json!({
            "traces": [],
            "train": [],
            "demos": demo_values,
            "signature": {
                "instructions": self.signature.instructions(),
                "fields": sig_fields,
            },
            "lm": null,
        }))
    }

    /// Load state from a JSON value (handles both native and `DSPy` formats).
    ///
    /// Updates demos and instructions from the loaded state. Fields not present
    /// in the state are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns a `PredictError` when a demo value is not a JSON object or
    /// when a demo's flat format contains values that don't parse as
    /// `FieldValue`.
    pub fn load_state(&mut self, state: &serde_json::Value) -> Result<()> {
        // Load instructions from signature
        if let Some(sig) = state.get("signature")
            && let Some(instructions) = sig.get("instructions").and_then(|v| v.as_str())
        {
            self.signature.set_instructions(instructions);
        }

        // Load demos
        if let Some(demos_val) = state.get("demos").and_then(|v| v.as_array()) {
            let mut demos = Vec::new();
            for demo_val in demos_val {
                let demo = self.parse_demo(demo_val)?;
                demos.push(demo);
            }
            self.demos = demos;
        }

        Ok(())
    }

    /// Parse a single demo from JSON, handling both native and `DSPy` flat formats.
    fn parse_demo(&self, val: &serde_json::Value) -> Result<Demo> {
        let obj = val.as_object().ok_or_else(|| {
            typesayer_types::PredictError::invalid_signature("demo must be a JSON object")
        })?;

        // Native format: has "inputs" and "outputs" keys
        if obj.contains_key("inputs") && obj.contains_key("outputs") {
            return serde_json::from_value(val.clone()).map_err(Into::into);
        }

        // DSPy flat format: split using signature's input field names
        let input_names: std::collections::HashSet<&str> = self
            .signature
            .input_fields()
            .map(|f| f.name.as_str())
            .collect();

        let flat: BTreeMap<String, FieldValue> = serde_json::from_value(val.clone())?;
        let mut inputs = BTreeMap::new();
        let mut outputs = BTreeMap::new();

        for (key, value) in flat {
            // Skip DSPy internal keys
            if key.starts_with("dspy_") || key == "augmented" {
                continue;
            }
            if input_names.contains(key.as_str()) {
                inputs.insert(key, value);
            } else {
                outputs.insert(key, value);
            }
        }

        Ok(Demo { inputs, outputs })
    }

    /// Execute the prediction: format → generate → parse.
    ///
    /// Uses the predictor's stored [`LanguageModelConfig`]. For a per-call
    /// override, use [`call_with_config`](Self::call_with_config).
    ///
    /// # Errors
    ///
    /// Returns [`PredictError`](typesayer_types::PredictError) if formatting, generation,
    /// or parsing fails.
    pub async fn call(
        &self,
        inputs: &BTreeMap<String, FieldValue>,
        ctx: &Context,
    ) -> Result<Prediction> {
        self.call_with_config(inputs, ctx, &self.config).await
    }

    /// Execute the prediction with an explicit config override.
    ///
    /// Like [`call`](Self::call) but uses the provided config instead of the
    /// predictor's stored config.
    ///
    /// # Errors
    ///
    /// Returns [`PredictError`](typesayer_types::PredictError) if the adapter fails to
    /// format the prompt, the language model returns an error, or the
    /// response cannot be parsed back into typed fields.
    pub async fn call_with_config(
        &self,
        inputs: &BTreeMap<String, FieldValue>,
        ctx: &Context,
        config: &LanguageModelConfig,
    ) -> Result<Prediction> {
        // Format messages from signature + inputs + demos
        let messages = ctx
            .adapter()
            .format(&self.signature, inputs, &self.demos)
            .await?;

        // Call the language model provider
        let response = ctx
            .provider()
            .generate(GenerateRequest {
                model: ctx.model(),
                messages: &messages,
                config,
            })
            .await?;

        // If the model stopped early because it ran out of room or hit a
        // safety filter, the completion is truncated and the chat adapter
        // would produce a generic "missing output fields" error. Surface
        // the real root cause instead. `EndTurn` and `StopSequence` are
        // natural terminations; `ToolUse` is not used on this path
        // (predict doesn't drive tool calls), but if it appears we treat
        // it the same as `EndTurn` since the content block we got is
        // still complete relative to its own message.
        if let Some(reason) = &response.stop_reason {
            match reason {
                modelplease::StopReason::EndTurn
                | modelplease::StopReason::StopSequence
                | modelplease::StopReason::ToolUse => {}
                modelplease::StopReason::MaxTokens
                | modelplease::StopReason::ContentFilter
                | modelplease::StopReason::Other(_) => {
                    return Err(PredictError::truncated(
                        reason,
                        response.usage.map(|u| u.output_tokens),
                    ));
                }
            }
        }

        // Parse the completion into typed fields. If the parse errors
        // out (missing fields, no markers, type mismatch) emit a single
        // diagnostic log line carrying the stop_reason, content length,
        // and a short head of the actual content. This is the only
        // place predict sees both sides — the upstream API response and
        // the structured-field expectation — so it is the right layer
        // to attribute parse failures back to the real upstream signal.
        let fields = match ctx
            .adapter()
            .parse(&self.signature, &response.content)
            .await
        {
            Ok(fields) => fields,
            Err(e) => {
                let head: String = response.content.chars().take(300).collect();
                tracing::warn!(
                    stop_reason = ?response.stop_reason,
                    content_bytes = response.content.len(),
                    output_tokens = response.usage.map(|u| u.output_tokens),
                    error = %e,
                    content_head = %head,
                    "predict parse failed; logging upstream context for debugging"
                );
                return Err(e);
            }
        };

        Ok(Prediction::new(fields, response.usage))
    }

    /// Execute the prediction and return both the result and a trace entry.
    ///
    /// Like [`call()`](Self::call) but additionally returns a [`TraceEntry`](crate::TraceEntry)
    /// recording the predictor's inputs and output. The `predictor_name`
    /// must be supplied by the caller (typically the module that owns this
    /// predictor, matching the name from [`Module::named_predictors()`](crate::Module::named_predictors)).
    ///
    /// # Errors
    ///
    /// Returns [`PredictError`](typesayer_types::PredictError) propagated from
    /// [`call()`](Self::call).
    pub async fn call_traced(
        &self,
        inputs: &BTreeMap<String, FieldValue>,
        ctx: &Context,
        predictor_name: &str,
    ) -> Result<(Prediction, crate::trace::TraceEntry)> {
        let prediction = self.call(inputs, ctx).await?;
        let entry = crate::trace::TraceEntry {
            predictor_name: predictor_name.to_owned(),
            inputs: inputs.clone(),
            prediction: prediction.clone(),
        };
        Ok((prediction, entry))
    }
}

/// Capitalize the first character of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map_or_else(String::new, |c| {
        c.to_uppercase().to_string() + chars.as_str()
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use modelplease::{DummyLM, ModelId};

    use super::*;
    use crate::adapter::ChatAdapter;

    fn qa_signature() -> Signature {
        Signature::builder("Answer the question.")
            .input(FieldDef::input(
                "question",
                FieldType::String,
                "The question",
            ))
            .output(FieldDef::output("answer", FieldType::String, "The answer"))
            .build()
            .unwrap()
    }

    fn make_ctx(answers: Vec<String>) -> Context {
        Context {
            provider: Arc::new(DummyLM::sequential(answers)),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        }
    }

    /// Provider stub that returns a fixed `LanguageModelResponse`. Used
    /// to drive predict against a response whose `stop_reason` we control
    /// — `DummyLM` always reports `EndTurn`, so it can't exercise the
    /// truncation path.
    struct CannedLM {
        response: modelplease::LanguageModelResponse,
    }

    #[async_trait::async_trait]
    impl modelplease::LanguageModelProvider for CannedLM {
        fn name(&self) -> &'static str {
            "canned"
        }
        fn capabilities(
            &self,
            _model: &modelplease::ModelId,
        ) -> Option<modelplease::ModelCapabilities> {
            None
        }
        async fn list_models(
            &self,
        ) -> std::result::Result<Vec<modelplease::ChatModelInfo>, modelplease::LanguageModelError>
        {
            Ok(Vec::new())
        }
        async fn generate(
            &self,
            _request: modelplease::GenerateRequest<'_>,
        ) -> std::result::Result<modelplease::LanguageModelResponse, modelplease::LanguageModelError>
        {
            Ok(self.response.clone())
        }
        async fn generate_stream(
            &self,
            _request: modelplease::GenerateRequest<'_>,
        ) -> std::result::Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = std::result::Result<
                                modelplease::StreamDelta,
                                modelplease::LanguageModelError,
                            >,
                        > + Send,
                >,
            >,
            modelplease::LanguageModelError,
        > {
            unimplemented!("stream not used in this test")
        }
    }

    #[tokio::test]
    async fn call_surfaces_truncation_instead_of_missing_fields() {
        // Mimic Haiku running out of room mid-output: it emitted the
        // start of a [[ ## answer ## ]] block but never closed it.
        // Without the early-return check, parse() would report
        // "missing output fields: [answer]". With the check, predict
        // returns Truncated{ stop_reason: "max_tokens" } so operators
        // see the real root cause.
        let response = modelplease::LanguageModelResponse {
            content: "[[ ## answer ## ]]\nThe capital of Fr".into(),
            thinking: None,
            usage: Some(modelplease::Usage {
                input_tokens: 47,
                output_tokens: 8192,
                ..modelplease::Usage::default()
            }),
            model: Some("claude-haiku-4-5".into()),
            stop_reason: Some(modelplease::StopReason::MaxTokens),
        };
        let ctx = Context {
            provider: Arc::new(CannedLM { response }),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let predict = Predict::new(qa_signature());
        let inputs = BTreeMap::from([(
            "question".into(),
            FieldValue::Str("Capital of France?".into()),
        )]);

        let err = predict.call(&inputs, &ctx).await.unwrap_err();
        match err {
            PredictError::Truncated {
                stop_reason,
                output_tokens,
            } => {
                assert_eq!(stop_reason, "max_tokens");
                assert_eq!(output_tokens, Some(8192));
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_returns_expected_output() {
        let ctx = make_ctx(vec![
            "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into(),
        ]);
        let predict = Predict::new(qa_signature());
        let inputs = BTreeMap::from([(
            "question".into(),
            FieldValue::Str("Capital of France?".into()),
        )]);

        let prediction = predict.call(&inputs, &ctx).await.unwrap();
        assert_eq!(prediction.get::<String>("answer").unwrap(), "Paris");
    }

    #[tokio::test]
    async fn call_with_demos() {
        let ctx = make_ctx(vec!["[[ ## answer ## ]]\n6\n[[ ## completed ## ]]".into()]);

        let demos = vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("2+2?".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("4".into()))]),
        }];

        let predict = Predict {
            demos,
            ..Predict::new(qa_signature())
        };
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("3+3?".into()))]);

        let prediction = predict.call(&inputs, &ctx).await.unwrap();
        assert_eq!(prediction.get::<String>("answer").unwrap(), "6");
    }

    #[tokio::test]
    async fn with_reasoning_adds_reasoning_field() {
        let ctx = make_ctx(vec![
            "[[ ## reasoning ## ]]\nI need to think...\n[[ ## answer ## ]]\n42\n[[ ## completed ## ]]"
                .into(),
        ]);

        let predict = Predict::chain_of_thought(qa_signature());

        // Verify the signature has reasoning as the first output field
        let output_names: Vec<_> = predict
            .signature()
            .output_fields()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(output_names, vec!["reasoning", "answer"]);

        let inputs = BTreeMap::from([(
            "question".into(),
            FieldValue::Str("What is the meaning?".into()),
        )]);
        let prediction = predict.call(&inputs, &ctx).await.unwrap();
        assert_eq!(
            prediction.get::<String>("reasoning").unwrap(),
            "I need to think..."
        );
        assert_eq!(prediction.get::<String>("answer").unwrap(), "42");
    }

    #[tokio::test]
    async fn different_contexts_use_different_lms() {
        let ctx1 = make_ctx(vec![
            "[[ ## answer ## ]]\nFirst\n[[ ## completed ## ]]".into(),
        ]);
        let ctx2 = make_ctx(vec![
            "[[ ## answer ## ]]\nSecond\n[[ ## completed ## ]]".into(),
        ]);

        let predict = Predict::new(qa_signature());
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);

        let p1 = predict.call(&inputs, &ctx1).await.unwrap();
        let p2 = predict.call(&inputs, &ctx2).await.unwrap();

        assert_eq!(p1.get::<String>("answer").unwrap(), "First");
        assert_eq!(p2.get::<String>("answer").unwrap(), "Second");
    }

    #[tokio::test]
    async fn call_propagates_lm_error() {
        // DummyLM with no answers → EmptyResponse on first call
        let ctx = make_ctx(vec![]);
        let predict = Predict::new(qa_signature());
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);

        let err = predict.call(&inputs, &ctx).await.unwrap_err();
        assert!(matches!(
            err,
            typesayer_types::PredictError::LanguageModel(_)
        ));
    }

    #[tokio::test]
    async fn prediction_carries_usage() {
        let ctx = make_ctx(vec![
            "[[ ## answer ## ]]\ntest\n[[ ## completed ## ]]".into(),
        ]);
        let predict = Predict::new(qa_signature());
        let inputs = BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]);

        let prediction = predict.call(&inputs, &ctx).await.unwrap();
        assert!(prediction.usage().is_some());
    }

    #[test]
    fn clone_produces_independent_copy() {
        let demos = vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("1+1?".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("2".into()))]),
        }];
        let predict = Predict {
            demos,
            ..Predict::new(qa_signature())
        };

        let mut cloned = predict.clone();
        cloned.set_demos(vec![]); // clear demos on clone

        assert_eq!(predict.demos().len(), 1); // original unchanged
        assert!(cloned.demos().is_empty());
    }

    #[tokio::test]
    async fn call_traced_returns_prediction_and_entry() {
        let ctx = make_ctx(vec![
            "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into(),
        ]);
        let predict = Predict::new(qa_signature());
        let inputs = BTreeMap::from([(
            "question".into(),
            FieldValue::Str("Capital of France?".into()),
        )]);

        let (prediction, entry) = predict.call_traced(&inputs, &ctx, "qa").await.unwrap();

        assert_eq!(prediction.get::<String>("answer").unwrap(), "Paris");
        assert_eq!(entry.predictor_name, "qa");
        assert_eq!(
            entry.inputs["question"],
            FieldValue::Str("Capital of France?".into())
        );
        assert_eq!(entry.prediction.get::<String>("answer").unwrap(), "Paris");
    }
}
