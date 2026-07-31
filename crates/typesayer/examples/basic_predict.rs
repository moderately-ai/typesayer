// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::unwrap_used,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/unwrap/expect target production code, not example binaries"
)]

//! Basic prediction: Signature → Predict → typed Prediction access.
//!
//! Demonstrates:
//! - Building a Signature with typed input/output fields
//! - Creating a Context with a real LM (Ollama)
//! - Running `Predict::call()` to get a Prediction
//! - Typed access via `prediction.get::`<T>()
//! - Chain-of-thought with `Predict::chain_of_thought(...)`
//!
//! Usage:
//!   dotenvx run -f .env.local -- cargo run -p typesayer --example `basic_predict`

use std::{collections::BTreeMap, sync::Arc};

mod common;

use common::build_http_client;
use modelplease::{ApiKey, ModelId, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};
use typesayer::{ChatAdapter, Context, Predict};
use typesayer_types::{FieldDef, FieldType, FieldValue, Signature};

#[tokio::main]
async fn main() -> typesayer_types::Result<()> {
    let client = Arc::new(build_http_client().unwrap());
    let lm = OpenAiLanguageModel::new(
        OpenAiDeps { client },
        OpenAiConfig {
            api_key: ApiKey::parse("testing").unwrap(),
            base_url: "http://localhost:11434".to_string(),
            retry_config: RetryConfig::default(),
        },
    );
    let ctx = Context {
        provider: Arc::new(lm),
        model: ModelId::new("gemma4"),
        adapter: Arc::new(ChatAdapter::default()),
    };

    // --- Simple Q&A ---
    println!("=== Simple Q&A ===");

    let sig = Signature::builder("Answer the question concisely.")
        .input(FieldDef::input(
            "question",
            FieldType::String,
            "The question to answer",
        ))
        .output(FieldDef::output(
            "answer",
            FieldType::String,
            "A concise answer",
        ))
        .build()?;

    let predict = Predict::new(sig);
    let inputs = BTreeMap::from([(
        "question".into(),
        FieldValue::Str("What is the largest planet in our solar system?".into()),
    )]);

    let prediction = predict.call(&inputs, &ctx).await?;
    let answer: String = prediction.get("answer")?;
    println!("Answer: {answer}");

    if let Some(usage) = prediction.usage() {
        println!(
            "Tokens: {} input, {} output",
            usage.input_tokens, usage.output_tokens
        );
    }

    // --- Chain of Thought ---
    println!("\n=== Chain of Thought ===");

    let sig = Signature::builder("Solve the math problem step by step.")
        .input(FieldDef::input(
            "problem",
            FieldType::String,
            "A math problem",
        ))
        .output(FieldDef::output(
            "answer",
            FieldType::String,
            "The final numeric answer",
        ))
        .build()?;

    let predict = Predict::chain_of_thought(sig);
    let inputs = BTreeMap::from([(
        "problem".into(),
        FieldValue::Str("If a train travels 120km in 2 hours, what is its speed in m/s?".into()),
    )]);

    let prediction = predict.call(&inputs, &ctx).await?;
    let reasoning: String = prediction.get("reasoning")?;
    let answer: String = prediction.get("answer")?;
    println!("Reasoning: {reasoning}");
    println!("Answer: {answer}");

    Ok(())
}
