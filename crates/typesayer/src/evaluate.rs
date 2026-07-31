// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dataset evaluation runner for scoring module performance.
//!
//! Runs a [`Module`](crate::Module) on a dataset of [`Example`](crate::Example)s,
//! scores each prediction with a metric function, and aggregates the results.
//! Used by optimizers to compare candidate programs.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use typesayer_types::{
    error::{PredictError, Result},
    field::FieldValue,
};

use crate::{
    context::Context, example::Example, module::Module, optimizer::MetricFn, prediction::Prediction,
};

/// Configuration for the evaluation runner.
#[derive(Debug, Clone)]
pub struct EvaluateConfig {
    /// Maximum number of concurrent evaluation tasks.
    ///
    /// Defaults to 1 (sequential execution).
    pub num_threads: usize,
    /// Maximum number of forward errors before aborting.
    ///
    /// Defaults to 5. Set to `usize::MAX` to never abort.
    pub max_errors: usize,
}

impl Default for EvaluateConfig {
    fn default() -> Self {
        Self {
            num_threads: 1,
            max_errors: 5,
        }
    }
}

impl EvaluateConfig {
    /// Create a config with the given concurrency level. Override
    /// `max_errors` via struct-update syntax:
    ///
    /// ```ignore
    /// EvaluateConfig { max_errors: 0, ..EvaluateConfig::new(4) }
    /// ```
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }
}

/// The result of an evaluation run.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Aggregate score: percentage of examples that scored > 0.
    pub score: f64,
    /// Per-example results in input order: (example, prediction, `metric_score`).
    pub results: Vec<(Example, Prediction, f64)>,
}

impl EvaluationResult {
    /// Number of examples evaluated (excludes errors).
    #[must_use]
    pub fn num_examples(&self) -> usize {
        self.results.len()
    }

    /// Number of examples that scored > 0.
    #[must_use]
    pub fn num_correct(&self) -> usize {
        self.results
            .iter()
            .filter(|(_, _, score)| *score > 0.0)
            .count()
    }
}

/// Evaluate a module on a dataset using a metric function.
///
/// Runs `module.forward()` on each example's input fields, scores the
/// resulting prediction with the metric, and aggregates results.
///
/// # Concurrency
///
/// When `config.num_threads > 1`, examples are evaluated concurrently
/// via `tokio::spawn`. The module is deep-cloned once and shared across
/// tasks via `Arc`. Concurrency limiting is delegated to the language
/// model implementation — this function does not throttle.
///
/// # Error handling
///
/// Individual `forward()` errors are counted and skipped. If errors
/// exceed `config.max_errors`, evaluation aborts early.
///
/// # Errors
///
/// Returns [`PredictError::Evaluation`] if too many examples fail.
pub async fn evaluate(
    module: &dyn Module,
    examples: &[Example],
    metric: &MetricFn,
    ctx: &Context,
    config: &EvaluateConfig,
) -> Result<EvaluationResult> {
    if examples.is_empty() {
        return Ok(EvaluationResult {
            score: 0.0,
            results: vec![],
        });
    }

    if config.num_threads <= 1 {
        return evaluate_sequential(module, examples, metric, ctx, config).await;
    }

    // Parallel path: deep_clone once, share via Arc
    let module_arc: Arc<dyn Module> = Arc::from(module.deep_clone());
    let ctx = ctx.clone();
    let metric = metric.clone();
    let max_errors = config.max_errors;
    let error_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = examples
        .iter()
        .enumerate()
        .map(|(idx, example)| {
            let m = Arc::clone(&module_arc);
            let c = ctx.clone();
            let met = Arc::clone(&metric);
            let errs = Arc::clone(&error_count);
            let ex = example.clone();
            tokio::spawn(async move {
                if errs.load(Ordering::Relaxed) >= max_errors {
                    return (idx, None);
                }
                let inputs: BTreeMap<String, FieldValue> = ex
                    .inputs()
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v.clone()))
                    .collect();
                m.forward(inputs, &c).await.map_or_else(
                    |_| {
                        errs.fetch_add(1, Ordering::Relaxed);
                        (idx, None)
                    },
                    |pred| {
                        let score = met(&ex, &pred);
                        (idx, Some((ex, pred, score)))
                    },
                )
            })
        })
        .collect();

    let raw_results = futures::future::join_all(handles).await;

    // Check error budget
    let errors = error_count.load(Ordering::Relaxed);
    if errors >= max_errors {
        return Err(PredictError::evaluation(format!(
            "exceeded max errors ({errors}/{max_errors})"
        )));
    }

    // Collect successful results, preserving input order
    let mut successful: Vec<(usize, Example, Prediction, f64)> = raw_results
        .into_iter()
        .filter_map(std::result::Result::ok) // unwrap JoinHandle
        .filter_map(|(idx, opt)| opt.map(|(ex, pred, score)| (idx, ex, pred, score)))
        .collect();
    successful.sort_by_key(|(idx, _, _, _)| *idx);

    let results: Vec<(Example, Prediction, f64)> = successful
        .into_iter()
        .map(|(_, ex, pred, score)| (ex, pred, score))
        .collect();

    let num_correct = results.iter().filter(|(_, _, s)| *s > 0.0).count();
    let score = accuracy(num_correct, results.len());

    Ok(EvaluationResult { score, results })
}

/// Compute `correct / total` as an f64, returning 0.0 for an empty batch.
#[expect(
    clippy::cast_precision_loss,
    reason = "usize→f64 loses precision only above ~2^53 items, far beyond any \
              realistic evaluation batch; the output is a 0.0–1.0 ratio reported \
              to humans so sub-epsilon rounding is invisible"
)]
fn accuracy(num_correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        num_correct as f64 / total as f64
    }
}

/// Sequential evaluation (used when `num_threads` <= 1).
async fn evaluate_sequential(
    module: &dyn Module,
    examples: &[Example],
    metric: &MetricFn,
    ctx: &Context,
    config: &EvaluateConfig,
) -> Result<EvaluationResult> {
    let mut successful: Vec<(Example, Prediction, f64)> = Vec::new();
    let mut error_count: usize = 0;

    for example in examples {
        if error_count >= config.max_errors {
            return Err(PredictError::evaluation(format!(
                "exceeded max errors ({error_count}/{max})",
                max = config.max_errors
            )));
        }

        let inputs: BTreeMap<String, FieldValue> = example
            .inputs()
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.clone()))
            .collect();

        match module.forward(inputs, ctx).await {
            Ok(prediction) => {
                let score = metric(example, &prediction);
                successful.push((example.clone(), prediction, score));
            }
            Err(_) => {
                error_count += 1;
            }
        }
    }

    let num_correct = successful.iter().filter(|(_, _, s)| *s > 0.0).count();
    let score = accuracy(num_correct, successful.len());

    Ok(EvaluationResult {
        score,
        results: successful,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use async_trait::async_trait;
    use modelplease::{DummyLM, ModelId};
    use typesayer_types::{
        field::{FieldDef, FieldType},
        signature::Signature,
    };

    use super::*;
    use crate::{adapter::ChatAdapter, predict::Predict};

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

    struct TestModule {
        qa: Predict,
    }

    #[async_trait]
    impl Module for TestModule {
        async fn forward(
            &self,
            inputs: BTreeMap<String, FieldValue>,
            ctx: &Context,
        ) -> Result<Prediction> {
            self.qa.call(&inputs, ctx).await
        }

        fn named_predictors(&self) -> Vec<(String, &Predict)> {
            vec![("qa".to_owned(), &self.qa)]
        }

        fn named_predictors_mut(&mut self) -> Vec<(String, &mut Predict)> {
            vec![("qa".to_owned(), &mut self.qa)]
        }

        fn deep_clone(&self) -> Box<dyn Module> {
            Box::new(Self {
                qa: self.qa.clone(),
            })
        }
    }

    fn make_examples(questions: &[&str]) -> Vec<Example> {
        questions
            .iter()
            .map(|q| {
                Example::new(
                    BTreeMap::from([
                        ("question".into(), FieldValue::Str((*q).into())),
                        ("answer".into(), FieldValue::Str("expected".into())),
                    ]),
                    HashSet::from(["question".into()]),
                )
            })
            .collect()
    }

    fn always_correct() -> MetricFn {
        Arc::new(|_example, _prediction| 1.0)
    }

    fn always_wrong() -> MetricFn {
        Arc::new(|_example, _prediction| 0.0)
    }

    fn match_expected() -> MetricFn {
        Arc::new(|example, prediction| {
            let expected = example.get("answer").and_then(|v| {
                if let FieldValue::Str(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            });
            let actual = prediction.get::<String>("answer").ok();
            if expected == actual.as_deref() {
                1.0
            } else {
                0.0
            }
        })
    }

    #[tokio::test]
    async fn evaluate_all_correct() {
        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nexpected\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nexpected\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let examples = make_examples(&["q1", "q2"]);

        let result = evaluate(
            &module,
            &examples,
            &always_correct(),
            &ctx,
            &EvaluateConfig::default(),
        )
        .await
        .unwrap();

        assert!((result.score - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.num_examples(), 2);
        assert_eq!(result.num_correct(), 2);
    }

    #[tokio::test]
    async fn evaluate_all_wrong() {
        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nwrong\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nwrong\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let examples = make_examples(&["q1", "q2"]);

        let result = evaluate(
            &module,
            &examples,
            &always_wrong(),
            &ctx,
            &EvaluateConfig::default(),
        )
        .await
        .unwrap();

        assert!(result.score.abs() < f64::EPSILON);
        assert_eq!(result.num_correct(), 0);
    }

    #[tokio::test]
    async fn evaluate_partial() {
        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nexpected\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nwrong\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let examples = make_examples(&["q1", "q2"]);

        let result = evaluate(
            &module,
            &examples,
            &match_expected(),
            &ctx,
            &EvaluateConfig::default(),
        )
        .await
        .unwrap();

        assert!((result.score - 0.5).abs() < f64::EPSILON);
        assert_eq!(result.num_correct(), 1);
    }

    #[tokio::test]
    async fn evaluate_empty_dataset() {
        let lm = DummyLM::sequential(vec![]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };

        let result = evaluate(
            &module,
            &[],
            &always_correct(),
            &ctx,
            &EvaluateConfig::default(),
        )
        .await
        .unwrap();

        assert!(result.score.abs() < f64::EPSILON);
        assert!(result.results.is_empty());
    }

    #[tokio::test]
    async fn evaluate_respects_max_errors() {
        // DummyLM with no answers → EmptyResponse on every call
        let lm = DummyLM::sequential(vec![]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let examples = make_examples(&["q1", "q2", "q3"]);

        let config = EvaluateConfig {
            max_errors: 2,
            ..EvaluateConfig::default()
        };
        let result = evaluate(&module, &examples, &always_correct(), &ctx, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeded max errors"));
    }

    #[tokio::test]
    async fn evaluate_concurrent_same_as_sequential() {
        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nexpected\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nwrong\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nexpected\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let examples = make_examples(&["q1", "q2", "q3"]);

        let config = EvaluateConfig::new(3); // 3 concurrent
        let result = evaluate(&module, &examples, &match_expected(), &ctx, &config)
            .await
            .unwrap();

        assert_eq!(result.num_examples(), 3);
        // 2 out of 3 match "expected"
        assert_eq!(result.num_correct(), 2);
    }
}
