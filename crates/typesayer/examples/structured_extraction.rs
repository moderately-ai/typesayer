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

//! Structured data extraction with JSON Schema signatures.
//!
//! Demonstrates:
//! - Dynamic signature construction from JSON Schema
//! - Structured output via `ResponseFormat::JsonSchema`
//! - Multi-field extraction from unstructured text
//! - Enum field types for constrained classification
//!
//! Usage:
//!   dotenvx run -f .env.local -- cargo run -p typesayer --example `structured_extraction`

use std::{collections::BTreeMap, sync::Arc};

mod common;

use common::build_http_client;
use modelplease::{ApiKey, ModelId, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};
use typesayer::{ChatAdapter, Context, Predict, signature_from_json_schema};
use typesayer_types::{FieldDef, FieldType, FieldValue, Signature};

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

    // --- Entity extraction with typed fields ---
    println!("=== Entity Extraction ===");

    let sig = Signature::builder("Extract structured information about the person from the text.")
        .input(FieldDef::input(
            "text",
            FieldType::String,
            "The text to analyze",
        ))
        .output(FieldDef::output(
            "name",
            FieldType::String,
            "The person's full name",
        ))
        .output(FieldDef::output("age", FieldType::Int, "The person's age"))
        .output(FieldDef::output(
            "occupation",
            FieldType::String,
            "The person's job or role",
        ))
        .build()?;

    let predict = Predict::new(sig);
    let inputs = BTreeMap::from([(
        "text".into(),
        FieldValue::Str(
            "Dr. Sarah Chen, a 42-year-old neuroscientist at MIT, published \
             groundbreaking research on memory formation last month."
                .into(),
        ),
    )]);

    let prediction = predict.call(&inputs, &ctx).await?;
    println!("Name:       {}", prediction.get::<String>("name")?);
    println!("Age:        {}", prediction.get::<i64>("age")?);
    println!("Occupation: {}", prediction.get::<String>("occupation")?);

    // --- Sentiment classification with enum ---
    println!("\n=== Sentiment Classification ===");

    let sig = Signature::builder("Classify the sentiment of the text.")
        .input(FieldDef::input(
            "text",
            FieldType::String,
            "The text to classify",
        ))
        .output(FieldDef::output(
            "sentiment",
            FieldType::Enum(vec!["positive".into(), "negative".into(), "neutral".into()]),
            "The sentiment classification",
        ))
        .output(FieldDef::output(
            "confidence",
            FieldType::Float,
            "Confidence score from 0.0 to 1.0",
        ))
        .build()?;

    let reviews = [
        "This product exceeded all my expectations. Absolutely love it!",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It works as described. Nothing special but gets the job done.",
    ];

    let predict = Predict::new(sig);
    for review in &reviews {
        let inputs = BTreeMap::from([("text".into(), FieldValue::Str((*review).into()))]);
        let prediction = predict.call(&inputs, &ctx).await?;
        println!(
            "  {:?} → {} (confidence: {})",
            &review[..40.min(review.len())],
            prediction.get::<String>("sentiment")?,
            prediction.get::<f64>("confidence")?,
        );
    }

    // --- Dynamic signature from JSON Schema ---
    println!("\n=== JSON Schema Signature ===");

    let sig = signature_from_json_schema(
        &serde_json::json!({
            "email": {"type": "string", "description": "The raw email text"}
        }),
        &serde_json::json!({
            "subject": {"type": "string", "description": "Email subject line"},
            "sender_intent": {
                "type": "string",
                "enum": ["request", "complaint", "inquiry", "feedback"],
                "description": "The sender's primary intent"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Suggested priority level"
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of action items extracted from the email"
            }
        }),
        "Analyze the email and extract structured metadata.",
    )?;

    let predict = Predict::new(sig);
    let inputs = BTreeMap::from([(
        "email".into(),
        FieldValue::Str(
            "Hi team,\n\nOur API has been returning 500 errors since this morning. \
             Multiple customers have reported issues. We need to:\n\
             1. Investigate the root cause immediately\n\
             2. Roll back the latest deployment if needed\n\
             3. Send a status update to affected customers\n\n\
             This is urgent.\n\nBest,\nJohn"
                .into(),
        ),
    )]);

    let prediction = predict.call(&inputs, &ctx).await?;
    println!("Subject:     {}", prediction.get::<String>("subject")?);
    println!(
        "Intent:      {}",
        prediction.get::<String>("sender_intent")?
    );
    println!("Priority:    {}", prediction.get::<String>("priority")?);
    println!(
        "Actions:     {:?}",
        prediction.get::<Vec<FieldValue>>("action_items")?
    );

    Ok(())
}
