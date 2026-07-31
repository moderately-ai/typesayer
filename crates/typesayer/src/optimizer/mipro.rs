// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MIPROv2` — joint instruction and demo optimization via Bayesian search.
//!
//! Three-phase algorithm:
//! 1. Bootstrap N demo candidate sets per predictor
//! 2. Propose N instruction candidates per predictor via `GroundedProposer`
//! 3. Search over (instruction, demo) combinations using TPE

use std::collections::BTreeMap;

use async_trait::async_trait;
use parzen::{
    Direction, FrozenTrial, GammaStrategy, ParamValue, Study, TpeSampler, TpeSamplerConfig,
    TpeSamplerDeps,
};
use rand::{SeedableRng, rngs::StdRng};
use typesayer_types::error::Result;

use super::{
    CompileRequest, MetricFn, Optimizer, Progress, ProgressFn,
    bootstrap::{
        BootstrapCandidatesConfig, BootstrapCandidatesDeps, BootstrapContexts, create_n_demo_sets,
    },
};
use crate::{
    adapter::Demo,
    context::Context,
    evaluate::{EvaluateConfig, evaluate},
    example::Example,
    module::Module,
    propose::{GroundedProposer, ProposeRequest, summarize_dataset},
};

const MIN_MINIBATCH_SIZE: usize = 50;

/// Auto mode presets matching `DSPy` conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoMode {
    /// 6 candidates, ~100 valset, quick iteration.
    Light,
    /// 12 candidates, ~300 valset, balanced.
    Medium,
    /// 18 candidates, ~1000 valset, thorough.
    Heavy,
}

impl AutoMode {
    const fn n_candidates(self) -> usize {
        match self {
            Self::Light => 6,
            Self::Medium => 12,
            Self::Heavy => 18,
        }
    }

    const fn valset_cap(self) -> usize {
        match self {
            Self::Light => 100,
            Self::Medium => 300,
            Self::Heavy => 1000,
        }
    }

    const fn n_startup_trials(self) -> usize {
        match self {
            Self::Light => 5,
            Self::Medium => 8,
            Self::Heavy => 10,
        }
    }
}

/// Behavior-bearing dependencies for [`MIPROv2`]. The metric closure is
/// the only thing here — every other knob lives on [`MiproConfig`].
pub struct MiproDeps {
    /// User-supplied scoring function. Called per-prediction to grade
    /// candidate outputs against ground truth.
    pub metric: MetricFn,
}

/// Pure-value tuning for [`MIPROv2`].
///
/// Per-field `MIPROv2::DEFAULT_*` constants document the shipping
/// defaults; [`MIPROv2::default_config`] assembles them in one go for
/// callers who only need to override a few fields via struct-update
/// syntax.
pub struct MiproConfig {
    pub auto: Option<AutoMode>,
    pub n_instruction_candidates: Option<usize>,
    pub n_demo_candidates: Option<usize>,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    /// Number of training examples sampled per minibatch evaluation.
    pub minibatch_examples: usize,
    /// Number of optimizer steps between full-trainset evaluations.
    pub full_eval_interval_steps: usize,
    pub seed: u64,
    /// The prompt model context for instruction generation. If `None`,
    /// the task context is used for both instruction generation and
    /// evaluation. Genuinely optional override per `Optimizer` contract.
    pub prompt_ctx: Option<Context>,
}

/// `MIPROv2` optimizer — joint instruction and demo optimization.
///
/// Generates instruction and demo candidate sets, then uses TPE (Bayesian
/// optimization) to search over combinations. Requires a separate prompt
/// model context for instruction generation.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
///
/// use typesayer::{AutoMode, MIPROv2, MetricFn, MiproConfig, MiproDeps};
///
/// let metric: MetricFn =
///     Arc::new(|_ex, pred| if pred.get_value("answer").is_some() { 1.0 } else { 0.0 });
///
/// let optimizer = MIPROv2::new(
///     MiproDeps { metric },
///     MiproConfig { auto: Some(AutoMode::Light), seed: 42, ..MIPROv2::default_config() },
/// );
/// ```
pub struct MIPROv2 {
    pub deps: MiproDeps,
    pub config: MiproConfig,
}

impl MIPROv2 {
    pub const DEFAULT_MAX_BOOTSTRAPPED_DEMOS: usize = 4;
    pub const DEFAULT_MAX_LABELED_DEMOS: usize = 4;
    pub const DEFAULT_MINIBATCH_EXAMPLES: usize = 35;
    pub const DEFAULT_FULL_EVAL_INTERVAL_STEPS: usize = 5;
    pub const DEFAULT_SEED: u64 = 0;

    /// Create a new `MIPROv2` optimizer from explicit deps + config.
    #[must_use]
    pub const fn new(deps: MiproDeps, config: MiproConfig) -> Self {
        Self { deps, config }
    }

    /// Default [`MiproConfig`] — all tunables at their shipping
    /// defaults and `prompt_ctx: None`. Use with struct-update syntax
    /// when overriding a few fields.
    #[must_use]
    pub const fn default_config() -> MiproConfig {
        MiproConfig {
            auto: None,
            n_instruction_candidates: None,
            n_demo_candidates: None,
            max_bootstrapped_demos: Self::DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
            max_labeled_demos: Self::DEFAULT_MAX_LABELED_DEMOS,
            minibatch_examples: Self::DEFAULT_MINIBATCH_EXAMPLES,
            full_eval_interval_steps: Self::DEFAULT_FULL_EVAL_INTERVAL_STEPS,
            seed: Self::DEFAULT_SEED,
            prompt_ctx: None,
        }
    }

    /// Resolve the effective number of candidates from auto mode or explicit settings.
    fn resolve_n_candidates(&self) -> (usize, usize) {
        let n = self.config.auto.map_or(12, AutoMode::n_candidates);
        let n_inst = self.config.n_instruction_candidates.unwrap_or(n);
        let n_demo = self.config.n_demo_candidates.unwrap_or(n);
        (n_inst, n_demo)
    }

    /// Phase A + B: bootstrap demo candidate sets and propose instruction
    /// candidates. Returns `(demos_by_predictor, instructions_by_predictor)`.
    async fn generate_candidates(
        &self,
        args: CandidateGenArgs<'_>,
    ) -> Result<(
        BTreeMap<String, Vec<Vec<Demo>>>,
        BTreeMap<String, Vec<String>>,
    )> {
        let CandidateGenArgs {
            module,
            effective_trainset,
            task_ctx,
            prompt_ctx,
            teacher_ctx,
            n_inst,
            n_demo,
            progress,
        } = args;

        progress(&Progress {
            phase: "bootstrap".into(),
            step: 0,
            total: 0,
            message: format!("bootstrapping {n_demo} demo candidate sets"),
            best_score: None,
        });
        let demo_candidates = create_n_demo_sets(
            module,
            effective_trainset,
            BootstrapContexts {
                student: task_ctx,
                teacher: teacher_ctx,
            },
            BootstrapCandidatesDeps {
                metric: &self.deps.metric,
            },
            BootstrapCandidatesConfig {
                n: n_demo,
                max_bootstrapped_demos: self.config.max_bootstrapped_demos,
                max_labeled_demos: self.config.max_labeled_demos,
            },
        )
        .await?;

        progress(&Progress {
            phase: "propose".into(),
            step: 0,
            total: 0,
            message: format!("summarizing dataset and proposing {n_inst} instruction candidates"),
            best_score: None,
        });
        let dataset_summary = summarize_dataset(effective_trainset, prompt_ctx, 10, 10).await?;
        let mut proposer = GroundedProposer::new(self.config.seed);
        proposer.dataset_summary = Some(dataset_summary);
        let instruction_candidates = proposer
            .propose(ProposeRequest {
                module,
                trainset: effective_trainset,
                n_candidates: n_inst,
                ctx: prompt_ctx,
                instruction_history: None,
            })
            .await?;

        Ok((demo_candidates, instruction_candidates))
    }

    /// Run the Bayesian search loop. Returns the best full-eval score and its
    /// associated param combo (or `None` if no full eval beat the baseline).
    async fn run_search_trials(&self, args: SearchArgs<'_>) -> Result<SearchOutcome> {
        let SearchArgs {
            module,
            task_ctx,
            eval_config,
            study,
            predictor_names,
            instruction_candidates,
            demo_candidates,
            effective_valset,
            baseline_score,
            num_trials,
            use_minibatch,
            progress,
        } = args;

        let mut best_score = baseline_score;
        let mut best_params: Option<BTreeMap<String, ParamValue>> = None;
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut param_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for trial_idx in 0..num_trials {
            let trial_params = suggest_trial_params(
                study,
                predictor_names,
                instruction_candidates,
                demo_candidates,
            );

            let mut candidate = module.deep_clone();
            apply_params(
                candidate.as_mut(),
                &trial_params,
                instruction_candidates,
                demo_candidates,
            );

            let (score, is_full_eval) = if use_minibatch
                && (trial_idx + 1) % (self.config.full_eval_interval_steps + 1) != 0
            {
                let batch =
                    sample_minibatch(effective_valset, self.config.minibatch_examples, &mut rng);
                let result = evaluate(
                    candidate.as_ref(),
                    &batch,
                    &self.deps.metric,
                    task_ctx,
                    eval_config,
                )
                .await?;
                (result.score, false)
            } else {
                let result = evaluate(
                    candidate.as_ref(),
                    effective_valset,
                    &self.deps.metric,
                    task_ctx,
                    eval_config,
                )
                .await?;
                (result.score, true)
            };

            study.complete_trial(score);

            let eval_type = if is_full_eval { "full" } else { "mini" };
            progress(&Progress {
                phase: "search".into(),
                step: trial_idx + 1,
                total: num_trials,
                message: format!(
                    "trial {}/{num_trials} ({eval_type}): score={:.1}%",
                    trial_idx + 1,
                    score * 100.0
                ),
                best_score: Some(best_score),
            });

            param_scores
                .entry(format_param_key(&trial_params))
                .or_default()
                .push(score);

            if is_full_eval && score > best_score {
                best_score = score;
                best_params = Some(trial_params.clone());
                progress(&Progress {
                    phase: "search".into(),
                    step: trial_idx + 1,
                    total: num_trials,
                    message: format!("new best! score={:.1}%", score * 100.0),
                    best_score: Some(best_score),
                });
            }
        }

        Ok(SearchOutcome {
            best_score,
            best_params,
        })
    }

    /// Calculate the number of trials based on candidate counts and predictor count.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "f64 round-trip is a budgeting heuristic: on overflow / precision \
                  loss the worst outcome is a clamped trial count; the optimizer \
                  still converges, just slower"
    )]
    fn calculate_num_trials(n_candidates: usize, num_predictors: usize) -> usize {
        let num_vars = num_predictors * 2; // instruction + demos per predictor
        let log_based = (2.0 * num_vars as f64 * (n_candidates as f64).log2()).ceil() as usize;
        let linear = ((1.5 * n_candidates as f64).ceil()) as usize;
        log_based.max(linear).max(1)
    }

    /// Run the full `MIPROv2` optimization.
    ///
    /// See [`MiproCompileRequest`] for the per-call inputs.
    ///
    /// # Errors
    ///
    /// Returns a `PredictError` when demo generation, instruction proposal,
    /// or trial evaluation fails, or when the initial baseline evaluation
    /// cannot run (e.g., language model is unreachable).
    pub async fn compile_mipro(&mut self, args: MiproCompileRequest<'_>) -> Result<()> {
        let MiproCompileRequest {
            module,
            trainset,
            task_ctx,
            prompt_ctx,
            teacher_ctx,
            valset,
            progress,
        } = args;

        let (n_inst, n_demo) = self.resolve_n_candidates();
        let num_predictors = module.named_predictors().len();

        // Dataset splitting: if no valset, use 80% for val, 20% for train (DSPy convention)
        let (effective_trainset, effective_valset): (Vec<Example>, Vec<Example>) = valset
            .map_or_else(
                || {
                    let split = trainset.len() * 80 / 100;
                    let val = trainset[..split].to_vec();
                    let train = trainset[split..].to_vec();
                    (train, val)
                },
                |vs| (trainset.to_vec(), vs.to_vec()),
            );

        // Cap valset size per auto mode
        let valset_cap = self.config.auto.map_or(usize::MAX, AutoMode::valset_cap);
        let effective_valset: Vec<Example> = if effective_valset.len() > valset_cap {
            effective_valset[..valset_cap].to_vec()
        } else {
            effective_valset
        };

        let (demo_candidates, instruction_candidates) = self
            .generate_candidates(CandidateGenArgs {
                module: &*module,
                effective_trainset: &effective_trainset,
                task_ctx,
                prompt_ctx,
                teacher_ctx,
                n_inst,
                n_demo,
                progress,
            })
            .await?;

        // ===== Phase C: Bayesian search =====
        let num_trials = Self::calculate_num_trials(n_inst.max(n_demo), num_predictors);
        let n_startup = self.config.auto.map_or(10, AutoMode::n_startup_trials);
        let use_minibatch = effective_valset.len() > MIN_MINIBATCH_SIZE;

        progress(&Progress {
            phase: "search".into(),
            step: 0,
            total: num_trials,
            message: format!(
                "starting Bayesian search ({num_trials} trials, {} valset examples{})",
                effective_valset.len(),
                if use_minibatch {
                    ", minibatch mode"
                } else {
                    ""
                }
            ),
            best_score: None,
        });

        let sampler = TpeSampler::new(
            TpeSamplerDeps {
                gamma_strategy: GammaStrategy::Default,
            },
            TpeSamplerConfig {
                seed: self.config.seed,
                n_startup_trials: n_startup,
                prior_weight: TpeSamplerConfig::DEFAULT_PRIOR_WEIGHT,
            },
        );
        let mut study = Study::new(Direction::Maximize, sampler);
        // Use parallel evaluation — concurrency limiting delegated to LM crate
        let eval_config = EvaluateConfig {
            max_errors: effective_valset.len(),
            ..EvaluateConfig::new(effective_valset.len())
        };

        // Predictor names in consistent order
        let predictor_names: Vec<String> = module
            .named_predictors()
            .into_iter()
            .map(|(n, _)| n)
            .collect();

        // Evaluate baseline (all indices = 0)
        let baseline_score = {
            let result = evaluate(
                module,
                &effective_valset,
                &self.deps.metric,
                task_ctx,
                &eval_config,
            )
            .await?;
            result.score
        };

        // Inject baseline trial
        let mut baseline_params = BTreeMap::new();
        for name in &predictor_names {
            baseline_params.insert(format!("{name}_instruction"), ParamValue::Categorical(0));
            baseline_params.insert(format!("{name}_demos"), ParamValue::Categorical(0));
        }
        study.add_trial(FrozenTrial {
            number: 0,
            params: baseline_params,
            value: baseline_score,
        });

        let search = self
            .run_search_trials(SearchArgs {
                module,
                task_ctx,
                eval_config: &eval_config,
                study: &mut study,
                predictor_names: &predictor_names,
                instruction_candidates: &instruction_candidates,
                demo_candidates: &demo_candidates,
                effective_valset: &effective_valset,
                baseline_score,
                num_trials,
                use_minibatch,
                progress,
            })
            .await?;

        let best_score = search.best_score;
        let best_params = search.best_params;

        // Apply best params to the original module
        if let Some(params) = best_params {
            apply_params(module, &params, &instruction_candidates, &demo_candidates);
            progress(&Progress {
                phase: "complete".into(),
                step: 0,
                total: 0,
                message: "optimization complete, best params applied".into(),
                best_score: Some(best_score),
            });
        } else {
            progress(&Progress {
                phase: "complete".into(),
                step: 0,
                total: 0,
                message: "no improvement found, keeping baseline".into(),
                best_score: Some(baseline_score),
            });
        }

        Ok(())
    }
}

#[async_trait]
impl Optimizer for MIPROv2 {
    async fn compile(&self, args: CompileRequest<'_>) -> Result<()> {
        let CompileRequest {
            module,
            trainset,
            ctx,
            teacher_ctx,
            valset,
            progress,
        } = args;
        // Clone self to get &mut for compile_mipro
        let prompt_ctx = self.config.prompt_ctx.as_ref().unwrap_or(ctx);
        let mut mipro = Self {
            deps: MiproDeps {
                metric: self.deps.metric.clone(),
            },
            config: MiproConfig {
                auto: self.config.auto,
                n_instruction_candidates: self.config.n_instruction_candidates,
                n_demo_candidates: self.config.n_demo_candidates,
                max_bootstrapped_demos: self.config.max_bootstrapped_demos,
                max_labeled_demos: self.config.max_labeled_demos,
                minibatch_examples: self.config.minibatch_examples,
                full_eval_interval_steps: self.config.full_eval_interval_steps,
                seed: self.config.seed,
                prompt_ctx: self.config.prompt_ctx.clone(),
            },
        };
        mipro
            .compile_mipro(MiproCompileRequest {
                module,
                trainset,
                task_ctx: ctx,
                prompt_ctx,
                teacher_ctx,
                valset,
                progress,
            })
            .await
    }
}

/// Apply instruction and demo selections to a module's predictors.
fn apply_params(
    module: &mut dyn Module,
    params: &BTreeMap<String, ParamValue>,
    instruction_candidates: &BTreeMap<String, Vec<String>>,
    demo_candidates: &BTreeMap<String, Vec<Vec<Demo>>>,
) {
    for (name, predict) in module.named_predictors_mut() {
        if let Some(ParamValue::Categorical(inst_idx)) = params.get(&format!("{name}_instruction"))
            && let Some(candidates) = instruction_candidates.get(&name)
            && let Some(instruction) = candidates.get(*inst_idx as usize)
        {
            predict.set_instructions(instruction.clone());
        }

        if let Some(ParamValue::Categorical(demo_idx)) = params.get(&format!("{name}_demos"))
            && let Some(candidates) = demo_candidates.get(&name)
            && let Some(demos) = candidates.get(*demo_idx as usize)
        {
            predict.set_demos(demos.clone());
        }
    }
}

/// Per-call inputs for [`MIPROv2::compile_mipro`].
///
/// Bundles the three same-shape `&Context` args (task / prompt / teacher) so
/// transposition is impossible: positional adjacency would compile cleanly
/// while routing instruction-generation prompts to the student model.
pub struct MiproCompileRequest<'a> {
    /// Module to optimize (mutated with best params).
    pub module: &'a mut dyn Module,
    /// Training examples for demo generation.
    pub trainset: &'a [Example],
    /// Student model context for evaluation.
    pub task_ctx: &'a Context,
    /// Stronger model context for instruction generation.
    pub prompt_ctx: &'a Context,
    /// Optional teacher model for demo bootstrapping.
    pub teacher_ctx: Option<&'a Context>,
    /// Validation set for scoring. If `None`, splits from `trainset`.
    pub valset: Option<&'a [Example]>,
    /// Progress reporter callback.
    pub progress: &'a ProgressFn,
}

/// Inputs to `MIPROv2::generate_candidates` (Phase A + B). Bundled to dodge
/// `clippy::too_many_arguments` and to disambiguate the three same-shape
/// `&Context` positional args.
struct CandidateGenArgs<'a> {
    module: &'a dyn Module,
    effective_trainset: &'a [Example],
    task_ctx: &'a Context,
    prompt_ctx: &'a Context,
    teacher_ctx: Option<&'a Context>,
    n_inst: usize,
    n_demo: usize,
    progress: &'a ProgressFn,
}

/// Arguments bundle for `MIPROv2::run_search_trials` — packed together to
/// dodge `clippy::too_many_arguments` while still being explicit about what
/// the trial loop reads vs mutates.
struct SearchArgs<'a> {
    module: &'a mut dyn Module,
    task_ctx: &'a Context,
    eval_config: &'a EvaluateConfig,
    study: &'a mut Study,
    predictor_names: &'a [String],
    instruction_candidates: &'a BTreeMap<String, Vec<String>>,
    demo_candidates: &'a BTreeMap<String, Vec<Vec<Demo>>>,
    effective_valset: &'a [Example],
    baseline_score: f64,
    num_trials: usize,
    use_minibatch: bool,
    progress: &'a ProgressFn,
}

/// The outcome of running all search trials: the best full-eval score plus
/// the param combo that produced it (if any full eval beat the baseline).
struct SearchOutcome {
    best_score: f64,
    best_params: Option<BTreeMap<String, ParamValue>>,
}

/// Ask the study to suggest per-predictor instruction + demo indices. The
/// index space is `[0, n_choices)` where `n_choices` is the number of
/// pre-generated candidates for that predictor (index 0 always preserves
/// the baseline instruction/demo set).
fn suggest_trial_params(
    study: &mut Study,
    predictor_names: &[String],
    instruction_candidates: &BTreeMap<String, Vec<String>>,
    demo_candidates: &BTreeMap<String, Vec<Vec<Demo>>>,
) -> BTreeMap<String, ParamValue> {
    let mut trial_params = BTreeMap::new();
    for name in predictor_names {
        let n_inst_choices = instruction_candidates
            .get(name)
            .map_or(1, std::vec::Vec::len);
        let n_demo_choices = demo_candidates.get(name).map_or(1, std::vec::Vec::len);

        let inst_idx = study.suggest_categorical(&format!("{name}_instruction"), n_inst_choices);
        let demo_idx = study.suggest_categorical(&format!("{name}_demos"), n_demo_choices);

        trial_params.insert(
            format!("{name}_instruction"),
            ParamValue::Categorical(u32::try_from(inst_idx).unwrap_or(u32::MAX)),
        );
        trial_params.insert(
            format!("{name}_demos"),
            ParamValue::Categorical(u32::try_from(demo_idx).unwrap_or(u32::MAX)),
        );
    }
    trial_params
}

/// Sample a random minibatch from the valset.
fn sample_minibatch(valset: &[Example], size: usize, rng: &mut StdRng) -> Vec<Example> {
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..valset.len()).collect();
    indices.shuffle(rng);
    indices.truncate(size);
    indices.into_iter().map(|i| valset[i].clone()).collect()
}

/// Create a string key from trial params for tracking.
fn format_param_key(params: &BTreeMap<String, ParamValue>) -> String {
    params
        .iter()
        .map(|(k, v)| match v {
            ParamValue::Categorical(idx) => format!("{k}={idx}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use async_trait::async_trait;
    use typesayer_types::{
        field::{FieldDef, FieldType, FieldValue},
        signature::Signature,
    };

    use super::*;
    use crate::{predict::Predict, prediction::Prediction};

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

    fn make_trainset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new(
                    BTreeMap::from([
                        ("question".into(), FieldValue::Str(format!("Q{i}"))),
                        ("answer".into(), FieldValue::Str(format!("A{i}"))),
                    ]),
                    HashSet::from(["question".into()]),
                )
            })
            .collect()
    }

    #[test]
    fn auto_presets() {
        let light = AutoMode::Light;
        assert_eq!(light.n_candidates(), 6);
        assert_eq!(light.valset_cap(), 100);

        let medium = AutoMode::Medium;
        assert_eq!(medium.n_candidates(), 12);
        assert_eq!(medium.valset_cap(), 300);

        let heavy = AutoMode::Heavy;
        assert_eq!(heavy.n_candidates(), 18);
        assert_eq!(heavy.valset_cap(), 1000);
    }

    #[test]
    fn trial_count_calculation() {
        // 1 predictor, 6 candidates: num_vars=2, max(ceil(2*2*log2(6)), ceil(1.5*6))
        // = max(ceil(10.34), ceil(9.0)) = max(11, 9) = 11
        let t1 = MIPROv2::calculate_num_trials(6, 1);
        assert!(t1 >= 9, "expected >= 9, got {t1}");

        // 2 predictors, 12 candidates: num_vars=4, max(ceil(2*4*log2(12)), ceil(1.5*12))
        // = max(ceil(28.7), ceil(18)) = max(29, 18) = 29
        let t2 = MIPROv2::calculate_num_trials(12, 2);
        assert!(t2 >= 18, "expected >= 18, got {t2}");
    }

    #[test]
    fn apply_params_sets_instruction_and_demos() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };

        let instruction_candidates: BTreeMap<String, Vec<String>> = BTreeMap::from([(
            "qa".to_owned(),
            vec!["Original".to_owned(), "Improved".to_owned()],
        )]);

        let demo_candidates: BTreeMap<String, Vec<Vec<Demo>>> = BTreeMap::from([(
            "qa".to_owned(),
            vec![
                vec![], // empty demos
                vec![Demo {
                    inputs: BTreeMap::from([("question".into(), FieldValue::Str("Q".into()))]),
                    outputs: BTreeMap::from([("answer".into(), FieldValue::Str("A".into()))]),
                }],
            ],
        )]);

        let params = BTreeMap::from([
            ("qa_instruction".into(), ParamValue::Categorical(1)),
            ("qa_demos".into(), ParamValue::Categorical(1)),
        ]);

        apply_params(
            &mut module,
            &params,
            &instruction_candidates,
            &demo_candidates,
        );

        assert_eq!(module.qa.signature().instructions(), "Improved");
        assert_eq!(module.qa.demos().len(), 1);
    }

    #[test]
    fn sample_minibatch_respects_size() {
        let examples = make_trainset(100);
        let mut rng = StdRng::seed_from_u64(42);
        let batch = sample_minibatch(&examples, 35, &mut rng);
        assert_eq!(batch.len(), 35);
    }
}
