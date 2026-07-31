// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dataset summarization via iterative LLM observation.

use std::collections::BTreeMap;

use modelplease::LanguageModelConfig;
use typesayer_types::{
    error::Result,
    field::{FieldDef, FieldType, FieldValue},
    signature::Signature,
};

use super::format_examples_batch;
use crate::{context::Context, example::Example, predict::Predict};

/// Iteratively build a concise summary of a training dataset.
///
/// Shows batches of examples to an LLM, accumulates observations about
/// trends and patterns, then distills into a 2-3 sentence summary.
///
/// # Arguments
///
/// * `trainset` — Training examples to summarize.
/// * `ctx` — Context with the LLM to use for summarization.
/// * `batch_size` — Examples per batch (default: 10).
/// * `max_iterations` — Maximum observation rounds (default: 10).
///
/// # Errors
///
/// Returns a `PredictError` when a signature fails to build, when a
/// `Predict::call()` fails, or when the parsed prediction is missing the
/// expected `observations` / `summary` field.
pub async fn summarize_dataset(
    trainset: &[Example],
    ctx: &Context,
    batch_size: usize,
    max_iterations: usize,
) -> Result<String> {
    if trainset.is_empty() {
        return Ok("No training data available.".to_owned());
    }

    let config = LanguageModelConfig {
        temperature: Some(1.0),
        ..Default::default()
    };

    // Phase 1: Initial observation on first batch
    let first_batch_end = trainset.len().min(batch_size);
    let batch_text = format_examples_batch(&trainset[..first_batch_end]);

    let observer_sig = build_observer_signature()?;
    let observer = Predict {
        config: config.clone(),
        ..Predict::new(observer_sig)
    };
    let inputs = BTreeMap::from([("examples".into(), FieldValue::Str(batch_text))]);
    let prediction = observer.call(&inputs, ctx).await?;
    let mut observations = prediction.get::<String>("observations")?;

    // Phase 2: Iterate over remaining batches
    let mut complete_count = 0;
    let mut iteration = 0;

    let mut batch_start = batch_size;
    while batch_start < trainset.len() && iteration < max_iterations {
        let batch_end = trainset.len().min(batch_start + batch_size);
        let batch_text = format_examples_batch(&trainset[batch_start..batch_end]);

        let prior_sig = build_observer_with_prior_signature()?;
        let prior_observer = Predict {
            config: config.clone(),
            ..Predict::new(prior_sig)
        };
        let inputs = BTreeMap::from([
            ("examples".into(), FieldValue::Str(batch_text)),
            (
                "prior_observations".into(),
                FieldValue::Str(observations.clone()),
            ),
        ]);

        let prediction = prior_observer.call(&inputs, ctx).await?;
        let new_obs = prediction.get::<String>("observations")?;

        if new_obs.to_uppercase().starts_with("COMPLETE") {
            complete_count += 1;
            if complete_count >= 5 {
                break;
            }
        } else {
            observations.push_str("\n\n");
            observations.push_str(&new_obs);
        }

        batch_start += batch_size;
        iteration += 1;
    }

    // Phase 3: Summarize accumulated observations
    let summarizer_sig = build_summarizer_signature()?;
    let summarizer = Predict {
        config,
        ..Predict::new(summarizer_sig)
    };
    let inputs = BTreeMap::from([("observations".into(), FieldValue::Str(observations))]);
    let prediction = summarizer.call(&inputs, ctx).await?;
    let summary = prediction.get::<String>("summary")?;

    Ok(super::strip_instruction_prefix(&summary))
}

fn build_observer_signature() -> Result<Signature> {
    Signature::builder(
        "Given several examples from a dataset, write observations about trends \
         that hold for most or all of the samples. Consider topics, content, syntax, \
         and conciseness. Make an educated guess about what task this dataset enables.",
    )
    .input(FieldDef::input(
        "examples",
        FieldType::String,
        "Sample data points from the dataset",
    ))
    .output(FieldDef::output(
        "observations",
        FieldType::String,
        "Observations that hold true for most or all of the data",
    ))
    .build()
}

fn build_observer_with_prior_signature() -> Result<Signature> {
    Signature::builder(
        "Given several examples from a dataset, write observations about trends \
         that hold for most or all of the samples. I will also provide prior observations. \
         Please add your own observations, or if the observations are comprehensive, \
         say 'COMPLETE'. Consider topics, content, syntax, and conciseness.",
    )
    .input(FieldDef::input(
        "examples",
        FieldType::String,
        "Sample data points from the dataset",
    ))
    .input(FieldDef::input(
        "prior_observations",
        FieldType::String,
        "Prior observations already made about the data",
    ))
    .output(FieldDef::output(
        "observations",
        FieldType::String,
        "Additional observations or COMPLETE if nothing to add",
    ))
    .build()
}

fn build_summarizer_signature() -> Result<Signature> {
    Signature::builder(
        "Given a series of observations about a dataset, summarize them into a \
         brief 2-3 sentence summary highlighting only the most important details.",
    )
    .input(FieldDef::input(
        "observations",
        FieldType::String,
        "Observations made about the dataset",
    ))
    .output(FieldDef::output(
        "summary",
        FieldType::String,
        "Two to three sentence summary of the most significant highlights",
    ))
    .build()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use modelplease::{DummyLM, ModelId};

    use super::*;
    use crate::adapter::ChatAdapter;

    fn make_examples(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new(
                    BTreeMap::from([
                        ("question".into(), FieldValue::Str(format!("Question {i}"))),
                        ("answer".into(), FieldValue::Str(format!("Answer {i}"))),
                    ]),
                    HashSet::from(["question".into()]),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn summarize_single_batch() {
        let lm = DummyLM::sequential(vec![
            // Observer response
            "[[ ## observations ## ]]\nDataset contains Q&A pairs.\n[[ ## completed ## ]]".into(),
            // Summarizer response
            "[[ ## summary ## ]]\nA Q&A dataset for testing.\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let examples = make_examples(3);

        let summary = summarize_dataset(&examples, &ctx, 10, 10).await.unwrap();
        assert!(summary.contains("Q&A"));
    }

    #[tokio::test]
    async fn summarize_stops_on_complete() {
        let lm = DummyLM::sequential(vec![
            // Observer response (batch 1)
            "[[ ## observations ## ]]\nInitial observations.\n[[ ## completed ## ]]".into(),
            // Prior observer (batch 2) — says COMPLETE
            "[[ ## observations ## ]]\nCOMPLETE\n[[ ## completed ## ]]".into(),
            "[[ ## observations ## ]]\nCOMPLETE\n[[ ## completed ## ]]".into(),
            "[[ ## observations ## ]]\nCOMPLETE\n[[ ## completed ## ]]".into(),
            "[[ ## observations ## ]]\nCOMPLETE\n[[ ## completed ## ]]".into(),
            "[[ ## observations ## ]]\nCOMPLETE\n[[ ## completed ## ]]".into(),
            // Summarizer response
            "[[ ## summary ## ]]\nA concise summary.\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let examples = make_examples(100);

        let summary = summarize_dataset(&examples, &ctx, 5, 20).await.unwrap();
        assert!(!summary.is_empty());
    }

    #[tokio::test]
    async fn summarize_empty_trainset() {
        let lm = DummyLM::sequential(vec![]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let summary = summarize_dataset(&[], &ctx, 10, 10).await.unwrap();
        assert!(summary.contains("No training data"));
    }
}
