// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/unwrap/expect target production code, not example binaries"
)]

//! Module composition, state persistence, and few-shot optimization.
//!
//! Demonstrates:
//! - Implementing the Module trait for a multi-step LLM program
//! - Few-shot demos on Predict
//! - Saving and loading module state
//! - `LabeledFewShot` optimizer for automatic demo injection
//! - Named predictor introspection
//!
//! Usage:
//!   dotenvx run -f .env.local -- cargo run -p typesayer --example
//! `module_and_optimization`

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
mod common;

use common::build_http_client;
use modelplease::{ApiKey, ModelId, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};
use typesayer::{ChatAdapter, Context, Demo, Example, LabeledFewShot, Module, Predict, Prediction};
use typesayer_types::{FieldDef, FieldType, FieldValue, Signature};

/// A two-step module: first classifies a question's topic, then answers it.
struct ClassifyAndAnswer {
    classify: Predict,
    answer: Predict,
}

impl ClassifyAndAnswer {
    fn new() -> typesayer_types::Result<Self> {
        let classify_sig = Signature::builder("Classify the question into a topic category.")
            .input(FieldDef::input(
                "question",
                FieldType::String,
                "The user's question",
            ))
            .output(FieldDef::output(
                "topic",
                FieldType::Enum(vec![
                    "science".into(),
                    "history".into(),
                    "geography".into(),
                    "math".into(),
                    "other".into(),
                ]),
                "The topic category",
            ))
            .build()?;

        let answer_sig =
            Signature::builder("Answer the question concisely, considering its topic category.")
                .input(FieldDef::input(
                    "question",
                    FieldType::String,
                    "The user's question",
                ))
                .input(FieldDef::input(
                    "topic",
                    FieldType::String,
                    "The classified topic",
                ))
                .output(FieldDef::output(
                    "answer",
                    FieldType::String,
                    "A concise, accurate answer",
                ))
                .build()?;

        Ok(Self {
            classify: Predict::new(classify_sig),
            answer: Predict::new(answer_sig),
        })
    }
}

#[async_trait]
impl Module for ClassifyAndAnswer {
    async fn forward(
        &self,
        inputs: BTreeMap<String, FieldValue>,
        ctx: &Context,
    ) -> typesayer_types::Result<Prediction> {
        // Step 1: Classify the topic
        let classification = self.classify.call(&inputs, ctx).await?;
        let topic: String = classification.get("topic")?;

        // Step 2: Answer with topic context
        let mut answer_inputs = inputs;
        answer_inputs.insert("topic".into(), FieldValue::Str(topic));
        self.answer.call(&answer_inputs, ctx).await
    }

    fn named_predictors(&self) -> Vec<(String, &Predict)> {
        vec![
            ("classify".to_owned(), &self.classify),
            ("answer".to_owned(), &self.answer),
        ]
    }

    fn named_predictors_mut(&mut self) -> Vec<(String, &mut Predict)> {
        vec![
            ("classify".to_owned(), &mut self.classify),
            ("answer".to_owned(), &mut self.answer),
        ]
    }

    fn deep_clone(&self) -> Box<dyn Module> {
        Box::new(Self {
            classify: self.classify.clone(),
            answer: self.answer.clone(),
        })
    }
}

#[tokio::main]
async fn main() -> typesayer_types::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Arc::new(build_http_client().expect("build reqwest client"));
    let lm = OpenAiLanguageModel::new(
        OpenAiDeps { client },
        OpenAiConfig {
            api_key: ApiKey::parse(api_key).unwrap(),
            base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
            retry_config: RetryConfig::default(),
        },
    );
    let ctx = Context {
        provider: Arc::new(lm),
        model: ModelId::new("gpt-5-nano"),
        adapter: Arc::new(ChatAdapter::default()),
    };

    // --- Predictor introspection ---
    println!("=== Predictor Introspection ===");
    let module = ClassifyAndAnswer::new()?;
    for (name, predict) in module.named_predictors() {
        println!(
            "Predictor '{}': {} inputs, {} outputs",
            name,
            predict.signature().input_fields().count(),
            predict.signature().output_fields().count(),
        );
    }

    // --- Manual demos ---
    println!("\n=== Manual Few-Shot Demos ===");
    let mut module = ClassifyAndAnswer::new()?;
    module.classify.set_demos(vec![
        Demo {
            inputs: BTreeMap::from([(
                "question".into(),
                FieldValue::Str("What is photosynthesis?".into()),
            )]),
            outputs: BTreeMap::from([("topic".into(), FieldValue::Str("science".into()))]),
        },
        Demo {
            inputs: BTreeMap::from([(
                "question".into(),
                FieldValue::Str("When did World War 2 end?".into()),
            )]),
            outputs: BTreeMap::from([("topic".into(), FieldValue::Str("history".into()))]),
        },
    ]);

    let inputs = BTreeMap::from([(
        "question".into(),
        FieldValue::Str("What is the speed of light?".into()),
    )]);
    let prediction = module.forward(inputs, &ctx).await?;
    println!("Answer: {}", prediction.get::<String>("answer")?);

    // --- State save/load ---
    println!("\n=== State Save/Load ===");
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let state_path = dir.path().join("module_state.json");

    module.save(&state_path)?;
    println!("Saved state to: {}", state_path.display());

    // Load into a fresh module
    let mut fresh_module = ClassifyAndAnswer::new()?;
    assert!(fresh_module.classify.demos().is_empty());
    fresh_module.load(&state_path)?;
    println!(
        "Loaded: classify has {} demos, answer has {} demos",
        fresh_module.classify.demos().len(),
        fresh_module.answer.demos().len(),
    );

    // Verify loaded module still works
    let inputs = BTreeMap::from([(
        "question".into(),
        FieldValue::Str("What is the capital of Brazil?".into()),
    )]);
    let prediction = fresh_module.forward(inputs, &ctx).await?;
    println!(
        "Answer (from loaded module): {}",
        prediction.get::<String>("answer")?
    );

    // --- LabeledFewShot optimizer ---
    println!("\n=== LabeledFewShot Optimizer ===");
    let mut module = ClassifyAndAnswer::new()?;

    let trainset = vec![
        Example::new(
            BTreeMap::from([
                (
                    "question".into(),
                    FieldValue::Str("What causes earthquakes?".into()),
                ),
                ("topic".into(), FieldValue::Str("science".into())),
                (
                    "answer".into(),
                    FieldValue::Str("Tectonic plate movement".into()),
                ),
            ]),
            HashSet::from(["question".into()]),
        ),
        Example::new(
            BTreeMap::from([
                (
                    "question".into(),
                    FieldValue::Str("Who built the pyramids?".into()),
                ),
                ("topic".into(), FieldValue::Str("history".into())),
                ("answer".into(), FieldValue::Str("Ancient Egyptians".into())),
            ]),
            HashSet::from(["question".into()]),
        ),
        Example::new(
            BTreeMap::from([
                (
                    "question".into(),
                    FieldValue::Str("What is the tallest mountain?".into()),
                ),
                ("topic".into(), FieldValue::Str("geography".into())),
                ("answer".into(), FieldValue::Str("Mount Everest".into())),
            ]),
            HashSet::from(["question".into()]),
        ),
    ];

    LabeledFewShot::new(3).compile_labeled(&mut module, &trainset)?;
    println!(
        "After optimization: classify has {} demos, answer has {} demos",
        module.classify.demos().len(),
        module.answer.demos().len(),
    );

    let inputs = BTreeMap::from([(
        "question".into(),
        FieldValue::Str("How many planets are in our solar system?".into()),
    )]);
    let prediction = module.forward(inputs, &ctx).await?;
    println!(
        "Answer (optimized): {}",
        prediction.get::<String>("answer")?
    );

    Ok(())
}
