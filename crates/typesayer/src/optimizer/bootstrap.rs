// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Automatic few-shot demo selection via teacher-model bootstrapping.

use std::collections::BTreeMap;

use async_trait::async_trait;
use typesayer_types::{
    error::{PredictError, Result},
    field::FieldValue,
};

use super::{CompileRequest, MetricFn, Optimizer};
use crate::{adapter::Demo, context::Context, example::Example, module::Module, trace::TraceEntry};

/// Automatic few-shot demo selection via teacher-model bootstrapping.
///
/// Runs training examples through a teacher model (or the student itself),
/// scores predictions with a metric function, and keeps high-scoring results
/// as demos. Remaining unbootstrapped examples fill demo slots as raw labeled
/// demos up to `max_labeled_demos`.
///
/// # Per-predictor attribution
///
/// When the module implements [`forward_traced()`](crate::Module::forward_traced),
/// bootstrapped demos are attributed to specific predictors based on execution
/// traces. Each predictor receives only the demos it produced. When tracing is
/// not available (default `forward_traced()` returns empty traces), demos are
/// broadcast to all predictors (backward-compatible behavior).
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
///
/// use typesayer::{BootstrapFewShot, MetricFn};
///
/// let metric: MetricFn =
///     Arc::new(
///         |_example, prediction| {
///             if prediction.get_value("answer").is_some() { 1.0 } else { 0.0 }
///         },
///     );
///
/// let optimizer = BootstrapFewShot {
///     max_bootstrapped_demos: 4,
///     max_labeled_demos: 8,
///     ..BootstrapFewShot::new(metric)
/// };
/// ```
pub struct BootstrapFewShot {
    pub metric: MetricFn,
    pub metric_threshold: f64,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub max_errors: usize,
}

impl BootstrapFewShot {
    pub const DEFAULT_METRIC_THRESHOLD: f64 = 0.0;
    pub const DEFAULT_MAX_BOOTSTRAPPED_DEMOS: usize = 4;
    pub const DEFAULT_MAX_LABELED_DEMOS: usize = 16;
    pub const DEFAULT_MAX_ROUNDS: usize = 1;
    pub const DEFAULT_MAX_ERRORS: usize = 5;

    /// Create a new optimizer with the given metric function and the
    /// shipping defaults for every tunable. Override any field via
    /// struct-update syntax (see the type docs).
    pub fn new(metric: MetricFn) -> Self {
        Self {
            metric,
            metric_threshold: Self::DEFAULT_METRIC_THRESHOLD,
            max_bootstrapped_demos: Self::DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
            max_labeled_demos: Self::DEFAULT_MAX_LABELED_DEMOS,
            max_rounds: Self::DEFAULT_MAX_ROUNDS,
            max_errors: Self::DEFAULT_MAX_ERRORS,
        }
    }

    /// Run the bootstrap optimization loop.
    ///
    /// For each training example, runs the module's `forward_traced()` through
    /// the teacher context (or student context if no teacher), scores with the
    /// metric, and collects passing predictions as demos.
    ///
    /// When traces are available, demos are attributed per-predictor. When
    /// traces are empty (default), demos are broadcast to all predictors.
    ///
    /// # Errors
    ///
    /// Returns an error if teacher errors exceed `max_errors`.
    pub async fn compile_bootstrap(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
        ctx: &Context,
        teacher_ctx: Option<&Context>,
    ) -> Result<()> {
        let effective_ctx = teacher_ctx.unwrap_or(ctx);

        // Per-predictor bootstrapped demos
        let mut per_predictor_demos: BTreeMap<String, Vec<Demo>> = BTreeMap::new();
        // Flat bootstrapped demos (fallback when no traces)
        let mut flat_bootstrapped_demos: Vec<Demo> = Vec::new();
        let mut bootstrapped_indices: Vec<bool> = vec![false; trainset.len()];
        let mut error_count: usize = 0;
        let mut has_traces = false;
        let mut total_bootstrapped = 0;

        // Phase 1: Bootstrap — run teacher with tracing, collect successful demos
        for (idx, example) in trainset.iter().enumerate() {
            if total_bootstrapped >= self.max_bootstrapped_demos {
                break;
            }

            let inputs: BTreeMap<String, FieldValue> = example
                .inputs()
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.clone()))
                .collect();

            for _round in 0..self.max_rounds {
                if let Ok((prediction, trace)) =
                    module.forward_traced(inputs.clone(), effective_ctx).await
                {
                    let score = (self.metric)(example, &prediction);
                    if score > self.metric_threshold
                        || (self.metric_threshold == 0.0 && score > 0.0)
                    {
                        if trace.is_empty() {
                            // No traces — collect flat demo for broadcast
                            flat_bootstrapped_demos.push(Demo {
                                inputs: inputs.clone(),
                                outputs: prediction.into_fields(),
                            });
                        } else {
                            // Traces available — attribute per predictor
                            has_traces = true;
                            for entry in trace.entries() {
                                let demo = trace_entry_to_demo(entry);
                                per_predictor_demos
                                    .entry(entry.predictor_name.clone())
                                    .or_default()
                                    .push(demo);
                            }
                        }
                        bootstrapped_indices[idx] = true;
                        total_bootstrapped += 1;
                        break;
                    }
                } else {
                    error_count += 1;
                    if error_count >= self.max_errors {
                        return Err(PredictError::optimizer(format!(
                            "BootstrapFewShot: exceeded max errors ({error_count}/{max})",
                            max = self.max_errors
                        )));
                    }
                }
            }
        }

        // Phase 2: Collect non-bootstrapped examples for labeled fill
        let validation: Vec<&Example> = trainset
            .iter()
            .enumerate()
            .filter(|(idx, _)| !bootstrapped_indices[*idx])
            .map(|(_, ex)| ex)
            .collect();

        // Phase 3: Inject demos into predictors
        if has_traces {
            // Per-predictor injection
            for (name, predict) in module.named_predictors_mut() {
                let mut demos = per_predictor_demos.remove(&name).unwrap_or_default();
                // Fill remaining capacity with labeled examples
                let labeled_budget = self.max_labeled_demos.saturating_sub(demos.len());
                let labeled_count = validation.len().min(labeled_budget);
                let labeled: Vec<Demo> = validation[..labeled_count]
                    .iter()
                    .map(|ex| (*ex).clone().into())
                    .collect();
                demos.extend(labeled);
                predict.set_demos(demos);
            }
        } else {
            // Broadcast to all predictors (backward-compatible)
            let labeled_budget = self
                .max_labeled_demos
                .saturating_sub(flat_bootstrapped_demos.len());
            let labeled_count = validation.len().min(labeled_budget);
            let labeled_demos: Vec<Demo> = validation[..labeled_count]
                .iter()
                .map(|ex| (*ex).clone().into())
                .collect();

            let mut all_demos = flat_bootstrapped_demos;
            all_demos.extend(labeled_demos);

            for (_name, predict) in module.named_predictors_mut() {
                predict.set_demos(all_demos.clone());
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Optimizer for BootstrapFewShot {
    async fn compile(&self, args: CompileRequest<'_>) -> Result<()> {
        self.compile_bootstrap(args.module, args.trainset, args.ctx, args.teacher_ctx)
            .await
    }
}

/// Convert a trace entry into a demo (inputs + outputs from the prediction).
fn trace_entry_to_demo(entry: &TraceEntry) -> Demo {
    Demo {
        inputs: entry.inputs.clone(),
        outputs: entry.prediction.fields().clone(),
    }
}

/// Generate N variant demo sets for each predictor, for use by `MIPROv2`'s
/// candidate generation phase.
///
/// Returns `predictor_name → [demo_set_0, ..., demo_set_n]` where:
/// - Set 0: zero-shot (empty demos)
/// - Set 1: labeled-only (raw training examples, no bootstrapping)
/// - Sets 2..n: bootstrapped with different demo counts
///
/// Student / teacher context pair for [`create_n_demo_sets`]. Bundling
/// avoids the silent transposition that two consecutive `&Context`-shaped
/// args would otherwise allow.
pub struct BootstrapContexts<'a> {
    pub student: &'a Context,
    pub teacher: Option<&'a Context>,
}

/// Behavior-bearing dependencies for [`create_n_demo_sets`]. Holds the
/// metric closure so two `&MetricFn`-shaped function args can't drift
/// past each other.
pub struct BootstrapCandidatesDeps<'a> {
    pub metric: &'a MetricFn,
}

/// Counts that drive [`create_n_demo_sets`]. Three adjacent `usize` args
/// (n / max_bootstrapped_demos / max_labeled_demos) used to be positional
/// — silent transposition would have produced wrong-shaped optimizer
/// search spaces with no compile-time signal.
pub struct BootstrapCandidatesConfig {
    pub n: usize,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
}

/// Each bootstrapped set uses a different slice of the trainset to produce
/// diversity in demo selection.
///
/// # Errors
///
/// Returns a `PredictError` when a bootstrap task fails (propagates from
/// `BootstrapFewShot::compile_bootstrap`) or when the spawned task panics.
pub async fn create_n_demo_sets(
    module: &dyn Module,
    trainset: &[Example],
    contexts: BootstrapContexts<'_>,
    deps: BootstrapCandidatesDeps<'_>,
    config: BootstrapCandidatesConfig,
) -> Result<BTreeMap<String, Vec<Vec<Demo>>>> {
    let BootstrapContexts {
        student: ctx,
        teacher: teacher_ctx,
    } = contexts;
    let BootstrapCandidatesDeps { metric } = deps;
    let BootstrapCandidatesConfig {
        n,
        max_bootstrapped_demos,
        max_labeled_demos,
    } = config;
    let predictor_names: Vec<String> = module
        .named_predictors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    // Initialize result: each predictor gets N demo sets, pre-seeded with
    // the zero-shot set (empty demos).
    let mut result: BTreeMap<String, Vec<Vec<Demo>>> = predictor_names
        .iter()
        .map(|name| (name.clone(), vec![vec![]]))
        .collect();

    // Set 1: labeled-only
    if n > 1 {
        let count = trainset.len().min(max_labeled_demos);
        let labeled: Vec<Demo> = trainset[..count]
            .iter()
            .map(|ex| ex.clone().into())
            .collect();
        for sets in result.values_mut() {
            sets.push(labeled.clone());
        }
    }

    // Sets 2..n: bootstrap in parallel — each task gets its own module clone
    if n > 2 {
        let ctx_owned = ctx.clone();
        let teacher_owned = teacher_ctx.cloned();

        let handles: Vec<_> = (2..n)
            .map(|set_idx| {
                let mut cloned_module = module.deep_clone();
                let m = metric.clone();
                let c = ctx_owned.clone();
                let tc = teacher_owned.clone();

                let offset = (set_idx - 2) % trainset.len().max(1);
                let mut rotated = trainset.to_vec();
                rotated.rotate_left(offset);

                let demo_count = if max_bootstrapped_demos <= 1 {
                    1
                } else {
                    1 + ((set_idx - 2) % max_bootstrapped_demos)
                };

                tokio::spawn(async move {
                    let optimizer = BootstrapFewShot {
                        max_bootstrapped_demos: demo_count,
                        max_labeled_demos,
                        max_rounds: 1,
                        max_errors: rotated.len(),
                        ..BootstrapFewShot::new(m)
                    };

                    optimizer
                        .compile_bootstrap(cloned_module.as_mut(), &rotated, &c, tc.as_ref())
                        .await?;

                    let demos: BTreeMap<String, Vec<Demo>> = cloned_module
                        .named_predictors()
                        .into_iter()
                        .map(|(n, p)| (n, p.demos().to_vec()))
                        .collect();
                    Ok::<_, typesayer_types::PredictError>(demos)
                })
            })
            .collect();

        let bootstrap_results = futures::future::join_all(handles).await;
        for join_result in bootstrap_results {
            let demos_per_predictor = join_result.map_err(|e| {
                typesayer_types::PredictError::optimizer(format!("bootstrap task panicked: {e}"))
            })??;
            for (name, sets) in &mut result {
                let demos = demos_per_predictor.get(name).cloned().unwrap_or_default();
                sets.push(demos);
            }
        }
    }

    Ok(result)
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
    use crate::{adapter::ChatAdapter, predict::Predict, prediction::Prediction};

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

    fn make_trainset() -> Vec<Example> {
        vec![
            Example::new(
                BTreeMap::from([
                    ("question".into(), FieldValue::Str("What is 1+1?".into())),
                    ("answer".into(), FieldValue::Str("2".into())),
                ]),
                HashSet::from(["question".into()]),
            ),
            Example::new(
                BTreeMap::from([
                    ("question".into(), FieldValue::Str("What is 2+2?".into())),
                    ("answer".into(), FieldValue::Str("4".into())),
                ]),
                HashSet::from(["question".into()]),
            ),
            Example::new(
                BTreeMap::from([
                    ("question".into(), FieldValue::Str("What is 3+3?".into())),
                    ("answer".into(), FieldValue::Str("6".into())),
                ]),
                HashSet::from(["question".into()]),
            ),
        ]
    }

    fn always_pass_metric() -> MetricFn {
        Arc::new(|_example, _prediction| 1.0)
    }

    fn threshold_metric() -> MetricFn {
        Arc::new(
            |_example, prediction| match prediction.get_value("answer") {
                Some(FieldValue::Str(s)) if s == "correct" => 1.0,
                _ => 0.0,
            },
        )
    }

    #[tokio::test]
    async fn bootstrap_with_always_pass_metric() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\n2\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\n4\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\n6\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer = BootstrapFewShot {
            max_bootstrapped_demos: 3,
            max_labeled_demos: 3,
            ..BootstrapFewShot::new(always_pass_metric())
        };

        optimizer
            .compile_bootstrap(&mut module, &trainset, &ctx, None)
            .await
            .unwrap();

        assert_eq!(module.qa.demos().len(), 3);
    }

    #[tokio::test]
    async fn bootstrap_respects_max_bootstrapped_demos() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\n2\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\n4\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\n6\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer = BootstrapFewShot {
            max_bootstrapped_demos: 1,
            max_labeled_demos: 4,
            ..BootstrapFewShot::new(always_pass_metric())
        };

        optimizer
            .compile_bootstrap(&mut module, &trainset, &ctx, None)
            .await
            .unwrap();

        assert!(module.qa.demos().len() <= 3);
        assert!(!module.qa.demos().is_empty());
    }

    #[tokio::test]
    async fn bootstrap_with_threshold_metric() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\ncorrect\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nwrong\n[[ ## completed ## ]]".into(),
            "[[ ## answer ## ]]\nwrong\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer = BootstrapFewShot {
            metric_threshold: 0.5,
            max_bootstrapped_demos: 3,
            max_labeled_demos: 3,
            ..BootstrapFewShot::new(threshold_metric())
        };

        optimizer
            .compile_bootstrap(&mut module, &trainset, &ctx, None)
            .await
            .unwrap();

        let demos = module.qa.demos();
        assert!(!demos.is_empty());
        assert_eq!(
            demos[0].outputs["answer"],
            FieldValue::Str("correct".into())
        );
    }

    #[tokio::test]
    async fn bootstrap_with_teacher_context() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let student_lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nstudent_answer\n[[ ## completed ## ]]".into(),
        ]);
        let student_ctx = Context {
            provider: Arc::new(student_lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let teacher_lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nteacher_answer\n[[ ## completed ## ]]".into(),
        ]);
        let teacher_ctx = Context {
            provider: Arc::new(teacher_lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer = BootstrapFewShot {
            max_bootstrapped_demos: 1,
            max_labeled_demos: 1,
            ..BootstrapFewShot::new(always_pass_metric())
        };

        optimizer
            .compile_bootstrap(&mut module, &trainset, &student_ctx, Some(&teacher_ctx))
            .await
            .unwrap();

        let demos = module.qa.demos();
        assert!(!demos.is_empty());
        assert_eq!(
            demos[0].outputs["answer"],
            FieldValue::Str("teacher_answer".into())
        );
    }

    #[tokio::test]
    async fn bootstrap_error_counting() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer = BootstrapFewShot {
            max_errors: 2,
            max_rounds: 1,
            ..BootstrapFewShot::new(always_pass_metric())
        };

        let result = optimizer
            .compile_bootstrap(&mut module, &trainset, &ctx, None)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeded max errors"));
    }

    #[tokio::test]
    async fn bootstrap_empty_trainset() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };

        let lm = DummyLM::sequential(vec![]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer = BootstrapFewShot::new(always_pass_metric());
        optimizer
            .compile_bootstrap(&mut module, &[], &ctx, None)
            .await
            .unwrap();

        assert!(module.qa.demos().is_empty());
    }

    #[tokio::test]
    async fn bootstrap_works_via_optimizer_trait() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec!["[[ ## answer ## ]]\n2\n[[ ## completed ## ]]".into()]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let optimizer: Box<dyn Optimizer> = Box::new(BootstrapFewShot {
            max_bootstrapped_demos: 1,
            max_labeled_demos: 3,
            ..BootstrapFewShot::new(always_pass_metric())
        });
        optimizer
            .compile(CompileRequest {
                module: &mut module,
                trainset: &trainset,
                ctx: &ctx,
                teacher_ctx: None,
                valset: None,
                progress: &crate::optimizer::no_progress(),
            })
            .await
            .unwrap();

        assert!(!module.qa.demos().is_empty());
    }

    #[tokio::test]
    async fn create_n_demo_sets_basic() {
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        // Need enough LM answers for multiple bootstrap rounds
        let mut answers = Vec::new();
        for _ in 0..20 {
            answers.push("[[ ## answer ## ]]\nok\n[[ ## completed ## ]]".to_string());
        }
        let lm = DummyLM::sequential(answers);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let sets = create_n_demo_sets(
            &module,
            &trainset,
            BootstrapContexts {
                student: &ctx,
                teacher: None,
            },
            BootstrapCandidatesDeps {
                metric: &always_pass_metric(),
            },
            BootstrapCandidatesConfig {
                n: 4, // 4 sets: zero-shot, labeled, 2 bootstrapped
                max_bootstrapped_demos: 2,
                max_labeled_demos: 2,
            },
        )
        .await
        .unwrap();

        // Should have entry for "qa" predictor
        assert!(sets.contains_key("qa"));
        let qa_sets = &sets["qa"];
        assert_eq!(qa_sets.len(), 4);

        // Set 0: zero-shot (empty)
        assert!(qa_sets[0].is_empty());

        // Set 1: labeled (up to max_labeled_demos)
        assert!(!qa_sets[1].is_empty());
        assert!(qa_sets[1].len() <= 2);
    }
}
