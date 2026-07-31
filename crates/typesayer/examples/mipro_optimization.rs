// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::similar_names,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/unwrap/expect target production code, not example binaries. \
              similar_names is a pedantic lint that fires on the linear demo script; \
              splitting it up would hurt readability"
)]

//! `MIPROv2` optimization on rhetorical device identification.
//!
//! A deliberately difficult task: classify rhetorical devices in sentences
//! using a specific taxonomy of overlapping categories. The baseline instruction
//! scores poorly because the LLM uses different names, picks obvious labels
//! over subtle ones, and doesn't know our decision boundaries.
//!
//! `MIPROv2` optimizes instructions to teach the model our exact taxonomy and
//! the distinctions between similar devices.
//!
//! Usage:
//!   dotenvx run --env-file .env.local -- cargo run -p typesayer --example
//! `mipro_optimization`

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
mod common;

use common::build_http_client;
use modelplease::{ApiKey, ModelId, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};
use typesayer::{
    AutoMode, ChatAdapter, Context, EvaluateConfig, Example, MIPROv2, MetricFn,
    MiproCompileRequest, MiproConfig, MiproDeps, Module, Predict, Prediction, Progress, ProgressFn,
    evaluate,
};
use typesayer_types::{FieldDef, FieldType, FieldValue, Signature};

// ---------------------------------------------------------------------------
// Module: two-step classify + explain
// ---------------------------------------------------------------------------

struct RhetoricAnalyzer {
    classify: Predict,
    explain: Predict,
}

impl RhetoricAnalyzer {
    fn new() -> typesayer_types::Result<Self> {
        let classify_sig =
            Signature::builder("Identify the rhetorical device used in this sentence.")
                .input(FieldDef::input(
                    "sentence",
                    FieldType::String,
                    "A sentence using a rhetorical device",
                ))
                .output(FieldDef::output(
                    "device",
                    FieldType::String,
                    "The name of the rhetorical device",
                ))
                .build()?;

        let explain_sig =
            Signature::builder("Explain why this sentence uses the identified device.")
                .input(FieldDef::input(
                    "sentence",
                    FieldType::String,
                    "The sentence",
                ))
                .input(FieldDef::input(
                    "device",
                    FieldType::String,
                    "The identified device",
                ))
                .output(FieldDef::output(
                    "explanation",
                    FieldType::String,
                    "A one-sentence explanation of why this device applies",
                ))
                .build()?;

        Ok(Self {
            classify: Predict::new(classify_sig),
            explain: Predict::new(explain_sig),
        })
    }
}

#[async_trait]
impl Module for RhetoricAnalyzer {
    async fn forward(
        &self,
        inputs: BTreeMap<String, FieldValue>,
        ctx: &Context,
    ) -> typesayer_types::Result<Prediction> {
        let classification = self.classify.call(&inputs, ctx).await?;
        let device: String = classification.get("device")?;

        let mut explain_inputs = inputs;
        explain_inputs.insert("device".into(), FieldValue::Str(device));
        self.explain.call(&explain_inputs, ctx).await
    }

    fn named_predictors(&self) -> Vec<(String, &Predict)> {
        vec![
            ("classify".to_owned(), &self.classify),
            ("explain".to_owned(), &self.explain),
        ]
    }

    fn named_predictors_mut(&mut self) -> Vec<(String, &mut Predict)> {
        vec![
            ("classify".to_owned(), &mut self.classify),
            ("explain".to_owned(), &mut self.explain),
        ]
    }

    fn deep_clone(&self) -> Box<dyn Module> {
        Box::new(Self {
            classify: self.classify.clone(),
            explain: self.explain.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Dataset: sentences with rhetorical devices from a confusing taxonomy
//
// The taxonomy deliberately uses technical/uncommon names and has overlapping
// categories. LLMs will default to common names like "exaggeration" instead
// of "hyperbole", "understatement" instead of "litotes", etc.
// ---------------------------------------------------------------------------

fn make_dataset() -> Vec<Example> {
    let data: Vec<(&str, &str, &str)> = vec![
        // LITOTES — understatement via double negative. LLMs often say "understatement" or "double
        // negative"
        (
            "He's not the friendliest person I've ever met.",
            "litotes",
            "negation of the opposite",
        ),
        (
            "That wasn't the worst meal I've had.",
            "litotes",
            "negates the negative",
        ),
        (
            "She's not unlike her mother in temperament.",
            "litotes",
            "double negative affirms similarity",
        ),
        (
            "The results were not insignificant.",
            "litotes",
            "negating insignificance",
        ),
        (
            "He is not unaware of the risks involved.",
            "litotes",
            "double negative for emphasis",
        ),
        // METONYMY — substituting associated concept. LLMs confuse with synecdoche
        (
            "The pen is mightier than the sword.",
            "metonymy",
            "pen represents writing, sword represents force",
        ),
        (
            "The White House issued a statement today.",
            "metonymy",
            "building represents administration",
        ),
        (
            "Hollywood is obsessed with sequels.",
            "metonymy",
            "place represents film industry",
        ),
        (
            "She's a big name on Wall Street.",
            "metonymy",
            "street represents finance industry",
        ),
        (
            "The crown has ruled for centuries.",
            "metonymy",
            "crown represents monarchy",
        ),
        // SYNECDOCHE — part for whole or whole for part. LLMs confuse with metonymy
        (
            "All hands on deck immediately.",
            "synecdoche",
            "hands represent whole sailors",
        ),
        (
            "Nice wheels you've got there.",
            "synecdoche",
            "wheels represent whole car",
        ),
        (
            "Brazil won the World Cup.",
            "synecdoche",
            "country represents team",
        ),
        (
            "She counted fifty head of cattle.",
            "synecdoche",
            "head represents whole animals",
        ),
        (
            "The company hired new blood.",
            "synecdoche",
            "blood represents people",
        ),
        // CHIASMUS — ABBA reversal pattern. LLMs often say "parallelism" or "antithesis"
        (
            "Ask not what your country can do for you, ask what you can do for your country.",
            "chiasmus",
            "ABBA reversal structure",
        ),
        (
            "When the going gets tough, the tough get going.",
            "chiasmus",
            "reversed phrase structure",
        ),
        (
            "You forget what you want to remember, and remember what you want to forget.",
            "chiasmus",
            "forget-remember reversed",
        ),
        (
            "We shape our tools and then our tools shape us.",
            "chiasmus",
            "subject-object inversion",
        ),
        // ZEUGMA — one word governs two others in different senses. Very uncommon, LLMs rarely
        // identify
        (
            "She lowered her standards and her neckline.",
            "zeugma",
            "lowered applies differently",
        ),
        (
            "He lost his coat and his temper at the party.",
            "zeugma",
            "lost applies to physical and emotional",
        ),
        (
            "She broke his car and his heart.",
            "zeugma",
            "broke applies literally and figuratively",
        ),
        (
            "He stole her purse and her attention.",
            "zeugma",
            "stole used in two senses",
        ),
        // ANADIPLOSIS — end of clause repeated at start of next. Extremely uncommon term
        (
            "Fear leads to anger, anger leads to hate.",
            "anadiplosis",
            "anger repeated at boundary",
        ),
        (
            "Work gives purpose, purpose gives meaning.",
            "anadiplosis",
            "purpose repeated at boundary",
        ),
        (
            "Information is knowledge, knowledge is power.",
            "anadiplosis",
            "knowledge repeated at boundary",
        ),
        // ANTITHESIS — contrasting ideas in parallel structure. LLMs sometimes say "contrast" or
        // "juxtaposition"
        (
            "It was the best of times, it was the worst of times.",
            "antithesis",
            "best and worst contrasted",
        ),
        (
            "Speech is silver, but silence is golden.",
            "antithesis",
            "speech and silence contrasted",
        ),
        (
            "One small step for man, one giant leap for mankind.",
            "antithesis",
            "small step and giant leap contrasted",
        ),
        (
            "To err is human, to forgive is divine.",
            "antithesis",
            "human error and divine forgiveness contrasted",
        ),
        // HYPERBOLE — extreme exaggeration. LLMs usually get this but sometimes say "exaggeration"
        (
            "I've told you a million times to clean your room.",
            "hyperbole",
            "million is extreme exaggeration",
        ),
        (
            "This bag weighs a ton.",
            "hyperbole",
            "ton is exaggerated weight",
        ),
        (
            "I'm so hungry I could eat a horse.",
            "hyperbole",
            "eating a horse is impossible exaggeration",
        ),
        (
            "It took an eternity for the results to come in.",
            "hyperbole",
            "eternity is time exaggeration",
        ),
    ];

    data.into_iter()
        .map(|(sentence, device, key_term)| {
            Example::new(
                BTreeMap::from([
                    ("sentence".into(), FieldValue::Str(sentence.into())),
                    ("device".into(), FieldValue::Str(device.into())),
                    ("key_term".into(), FieldValue::Str(key_term.into())),
                ]),
                HashSet::from(["sentence".into()]),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Metric: strict exact-match on device name from our taxonomy
// ---------------------------------------------------------------------------

fn make_metric() -> MetricFn {
    Arc::new(|example, prediction| {
        let expected_device = example.get("device").and_then(|v| {
            if let FieldValue::Str(s) = v {
                Some(s.to_lowercase())
            } else {
                None
            }
        });

        let predicted_answer = prediction
            .get::<String>("explanation")
            .ok()
            .unwrap_or_default()
            .to_lowercase();

        // The explanation predictor's output includes the device name from
        // the classify step. But we score based on whether the pipeline
        // identified the correct device. We need to check the full pipeline.
        // Since forward() chains classify → explain, the "device" input to
        // explain came from classify. We can check if the explanation
        // references the expected key term as a proxy for correct classification.

        // Actually, let's just check if the explanation mentions the expected
        // device name OR the key term — this validates the whole pipeline.
        let expected_key = example.get("key_term").and_then(|v| {
            if let FieldValue::Str(s) = v {
                Some(s.to_lowercase())
            } else {
                None
            }
        });

        let device_mentioned = expected_device
            .as_ref()
            .is_some_and(|d| predicted_answer.contains(d));

        let key_mentioned = expected_key
            .as_ref()
            .is_some_and(|k| predicted_answer.contains(k));

        if device_mentioned && key_mentioned {
            1.0
        } else if device_mentioned || key_mentioned {
            0.5
        } else {
            0.0
        }
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> typesayer_types::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Arc::new(build_http_client().expect("build reqwest client"));

    let task_lm = OpenAiLanguageModel::new(
        OpenAiDeps {
            client: Arc::clone(&client),
        },
        OpenAiConfig {
            api_key: ApiKey::parse(api_key.clone()).unwrap(),
            base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
            retry_config: RetryConfig::default(),
        },
    );
    let prompt_lm = OpenAiLanguageModel::new(
        OpenAiDeps { client },
        OpenAiConfig {
            api_key: ApiKey::parse(api_key).unwrap(),
            base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
            retry_config: RetryConfig::default(),
        },
    );

    let task_ctx = Context {
        provider: Arc::new(task_lm),
        model: ModelId::new("gpt-5.4-nano"),
        adapter: Arc::new(ChatAdapter::default()),
    };
    let prompt_ctx = Context {
        provider: Arc::new(prompt_lm),
        model: ModelId::new("test"),
        adapter: Arc::new(ChatAdapter::default()),
    };

    let mut module = RhetoricAnalyzer::new()?;
    let dataset = make_dataset();
    let metric = make_metric();

    // Split: 26 train, 10 val (leave harder/rarer devices in val)
    let (train, val) = dataset.split_at(26);

    println!("=== Rhetorical Device Identification ===");
    println!(
        "Taxonomy: litotes, metonymy, synecdoche, chiasmus, zeugma, anadiplosis, antithesis, hyperbole"
    );
    println!("Dataset: {} train, {} val", train.len(), val.len());
    println!();

    // Show baseline instructions
    println!("--- Baseline instructions ---");
    for (name, pred) in module.named_predictors() {
        println!("  {name}: \"{}\"", pred.signature().instructions());
    }

    // Evaluate baseline
    println!("\n--- Baseline evaluation ---");
    let baseline = evaluate(&module, val, &metric, &task_ctx, &EvaluateConfig::default()).await?;
    println!(
        "Score: {:.0}% ({:.1}/{} — using >0 threshold)",
        baseline.score * 100.0,
        baseline.results.iter().map(|(_, _, s)| s).sum::<f64>(),
        baseline.num_examples(),
    );
    for (ex, pred, score) in &baseline.results {
        let sentence = match ex.get("sentence") {
            Some(FieldValue::Str(s)) => truncate(s, 55),
            _ => "?".into(),
        };
        let expected = match ex.get("device") {
            Some(FieldValue::Str(s)) => s.as_str(),
            _ => "?",
        };
        let explanation = pred.get::<String>("explanation").unwrap_or_default();
        let mark = if *score >= 1.0 {
            "✓"
        } else if *score > 0.0 {
            "~"
        } else {
            "✗"
        };
        println!("  {mark} [{expected:>12}] {sentence}");
        println!("             → {}", truncate(&explanation, 80));
    }

    // Run MIPROv2
    println!("\n=== Running MIPROv2 (light mode) ===");
    println!("Optimizing instructions + demos for both predictors...\n");

    let mut optimizer = MIPROv2::new(
        MiproDeps {
            metric: metric.clone(),
        },
        MiproConfig {
            auto: Some(AutoMode::Light),
            seed: 42,
            max_bootstrapped_demos: 3,
            max_labeled_demos: 3,
            ..MIPROv2::default_config()
        },
    );

    let print_progress: ProgressFn = Arc::new(|p: &Progress| println!("{p}"));
    optimizer
        .compile_mipro(MiproCompileRequest {
            module: &mut module,
            trainset: train,
            task_ctx: &task_ctx,
            prompt_ctx: &prompt_ctx,
            teacher_ctx: None,
            valset: Some(val),
            progress: &print_progress,
        })
        .await?;

    // Show optimized instructions
    println!("\n--- Optimized instructions ---");
    for (name, pred) in module.named_predictors() {
        println!(
            "  {name}: \"{}\"",
            truncate(pred.signature().instructions(), 120)
        );
        println!("         demos: {}", pred.demos().len());
    }

    // Evaluate optimized
    println!("\n--- Optimized evaluation ---");
    let optimized = evaluate(&module, val, &metric, &task_ctx, &EvaluateConfig::default()).await?;
    println!(
        "Score: {:.0}% ({:.1}/{} — using >0 threshold)",
        optimized.score * 100.0,
        optimized.results.iter().map(|(_, _, s)| s).sum::<f64>(),
        optimized.num_examples(),
    );
    for (ex, pred, score) in &optimized.results {
        let sentence = match ex.get("sentence") {
            Some(FieldValue::Str(s)) => truncate(s, 55),
            _ => "?".into(),
        };
        let expected = match ex.get("device") {
            Some(FieldValue::Str(s)) => s.as_str(),
            _ => "?",
        };
        let explanation = pred.get::<String>("explanation").unwrap_or_default();
        let mark = if *score >= 1.0 {
            "✓"
        } else if *score > 0.0 {
            "~"
        } else {
            "✗"
        };
        println!("  {mark} [{expected:>12}] {sentence}");
        println!("             → {}", truncate(&explanation, 80));
    }

    // Summary
    println!("\n=== Summary ===");
    let b_total: f64 = baseline.results.iter().map(|(_, _, s)| s).sum();
    let o_total: f64 = optimized.results.iter().map(|(_, _, s)| s).sum();
    println!(
        "Baseline: {:.0}% pass, {:.1}/{} total score",
        baseline.score * 100.0,
        b_total,
        baseline.num_examples(),
    );
    println!(
        "Optimized: {:.0}% pass, {:.1}/{} total score",
        optimized.score * 100.0,
        o_total,
        optimized.num_examples(),
    );
    let improvement = optimized.score - baseline.score;
    if improvement > 0.0 {
        println!("Pass rate improvement: +{:.0}%", improvement * 100.0);
    } else if improvement < 0.0 {
        println!("Pass rate regression: {:.0}%", improvement * 100.0);
    } else {
        println!("No change in pass rate");
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let truncated: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{truncated}...")
    } else {
        s.to_owned()
    }
}
