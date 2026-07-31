// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data-aware instruction candidate generation for `MIPROv2`.

use std::collections::BTreeMap;

use modelplease::LanguageModelConfig;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use typesayer_types::{
    error::Result,
    field::{FieldDef, FieldType, FieldValue},
    signature::Signature,
};

use super::{describe_module_structure, format_examples_batch, strip_instruction_prefix};
use crate::{context::Context, example::Example, module::Module, predict::Predict};

/// Default tips for instruction generation diversity (matching `DSPy`).
const TIPS: &[(&str, &str)] = &[
    ("none", ""),
    (
        "creative",
        "Don't be afraid to be creative when creating the new instruction!",
    ),
    ("simple", "Keep the instruction clear and concise."),
    (
        "description",
        "Make sure your instruction is very informative and descriptive.",
    ),
    (
        "high_stakes",
        "The instruction should include a high stakes scenario in which the LM must solve the task!",
    ),
    (
        "persona",
        "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\").",
    ),
];

/// A previous instruction attempt with its average score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionAttempt {
    /// The instruction text.
    pub instruction: String,
    /// The average score achieved with this instruction.
    pub avg_score: f64,
}

/// History of instruction attempts for history-aware proposal generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstructionHistory {
    /// Previous instruction attempts, ordered by score (lowest first).
    pub entries: Vec<InstructionAttempt>,
}

impl InstructionHistory {
    /// Format the history as a readable string for inclusion in prompts.
    ///
    /// Format: `"instruction text" | Score: 0.75\n\n`
    #[must_use]
    pub fn format(&self) -> String {
        self.entries
            .iter()
            .map(|e| format!("\"{}\" | Score: {:.2}", e.instruction, e.avg_score))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Per-call inputs to [`GroundedProposer::propose`]. Bundled into a
/// struct so the `&dyn Module` / `&[Example]` / `Option<&InstructionHistory>`
/// reference args land at the call site as named fields rather than in
/// positional order — six args including `&mut self` was over the
/// rust.md "4+ is a smell" threshold.
pub struct ProposeRequest<'a> {
    /// Module whose predictors should receive candidate instructions.
    pub module: &'a dyn Module,
    /// Training examples used to ground the candidate prompts.
    pub trainset: &'a [Example],
    /// Number of candidates to generate per predictor (the original
    /// instruction is always candidate 0, so this is the upper bound on
    /// the LLM-generated tail length).
    pub n_candidates: usize,
    /// Predict context (`LanguageModelProvider`, `ModelId`, `Adapter`)
    /// used by every internal proposal call.
    pub ctx: &'a Context,
    /// Optional history of past attempts with scores, used to bias the
    /// LLM toward novel candidates.
    pub instruction_history: Option<&'a InstructionHistory>,
}

/// Data-aware instruction candidate generator for `MIPROv2`.
///
/// Generates N instruction candidates per predictor by calling an LLM with
/// context about the dataset, module structure, and task examples.
pub struct GroundedProposer {
    /// Optional dataset summary (typically from [`summarize_dataset`]).
    /// When set, the proposer threads it into every candidate prompt.
    pub dataset_summary: Option<String>,
    rng: StdRng,
}

impl GroundedProposer {
    /// Create a new proposer with the given random seed and no dataset
    /// summary. Set `dataset_summary` directly via field assignment when
    /// one is available — see [`summarize_dataset`].
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            dataset_summary: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate N instruction candidates per predictor.
    ///
    /// Returns `predictor_name → [candidate_0, ..., candidate_n]` where
    /// candidate 0 is always the original instruction (baseline anchor).
    ///
    /// # Arguments
    ///
    /// * `module` — The module to generate instructions for (read-only, for structure).
    /// * `trainset` — Training examples for task demos.
    /// * `n_candidates` — Number of candidates to generate per predictor.
    /// * `ctx` — Context with the prompt model (typically a stronger LM).
    /// * `instruction_history` — Optional history of past attempts with scores.
    ///
    /// # Errors
    ///
    /// Returns a `PredictError` when the proposer signature fails to build,
    /// when a proposer call fails, or when the prediction does not include
    /// the expected `proposed_instruction` field.
    pub async fn propose(
        &mut self,
        request: ProposeRequest<'_>,
    ) -> Result<BTreeMap<String, Vec<String>>> {
        let ProposeRequest {
            module,
            trainset,
            n_candidates,
            ctx,
            instruction_history,
        } = request;
        // Pre-generated task descriptor: declared at the top of the function
        // so `tasks.push(ProposalTask { ... })` below reads cleanly.
        struct ProposalTask {
            predictor_name: String,
            current_instruction: String,
            tip_text: String,
            demos_text: String,
            dataset_summary: Option<String>,
            history_text: Option<String>,
        }

        let module_desc = describe_module_structure(module);
        let predictors = module.named_predictors();
        let config = LanguageModelConfig {
            temperature: Some(1.0),
            ..Default::default()
        };

        let mut tasks: Vec<ProposalTask> = Vec::new();
        let mut original_instructions: BTreeMap<String, String> = BTreeMap::new();

        for (name, predict) in &predictors {
            let current_instruction = predict.signature().instructions().to_owned();
            original_instructions.insert(name.clone(), current_instruction.clone());

            for i in 0..n_candidates.saturating_sub(1) {
                let tip_idx = rand::Rng::random_range(&mut self.rng, 0..TIPS.len());
                let tip_text = TIPS[tip_idx].1.to_owned();

                let demo_start = (i * 3) % trainset.len().max(1);
                let demo_end = trainset.len().min(demo_start + 3);
                let demos_text = if trainset.is_empty() {
                    "No task demos available.".to_owned()
                } else {
                    format_examples_batch(&trainset[demo_start..demo_end])
                };

                tasks.push(ProposalTask {
                    predictor_name: name.clone(),
                    current_instruction: current_instruction.clone(),
                    tip_text,
                    demos_text,
                    dataset_summary: self.dataset_summary.clone(),
                    history_text: instruction_history.map(InstructionHistory::format),
                });
            }
        }

        // Spawn all proposal LLM calls in parallel
        let ctx_owned = ctx.clone();
        let handles: Vec<_> = tasks
            .into_iter()
            .map(|task| {
                let c = ctx_owned.clone();
                let desc = module_desc.clone();
                let cfg = config.clone();
                let has_tip = !task.tip_text.is_empty();

                tokio::spawn(async move {
                    let sig = match build_proposal_signature(ProposalSignatureFields {
                        dataset: task.dataset_summary.is_some(),
                        history: task.history_text.is_some(),
                        tip: has_tip,
                    }) {
                        Ok(s) => s,
                        Err(e) => return (task.predictor_name, Err(e)),
                    };

                    let mut inputs = BTreeMap::new();
                    inputs.insert("module_description".into(), FieldValue::Str(desc));
                    inputs.insert("task_demos".into(), FieldValue::Str(task.demos_text));
                    inputs.insert(
                        "basic_instruction".into(),
                        FieldValue::Str(task.current_instruction),
                    );

                    if let Some(summary) = task.dataset_summary {
                        inputs.insert("dataset_description".into(), FieldValue::Str(summary));
                    }
                    if let Some(history) = task.history_text {
                        inputs.insert("previous_instructions".into(), FieldValue::Str(history));
                    }
                    if has_tip {
                        inputs.insert("tip".into(), FieldValue::Str(task.tip_text));
                    }

                    let predictor = Predict {
                        config: cfg,
                        ..Predict::new(sig)
                    };
                    let result = predictor.call(&inputs, &c).await;
                    (task.predictor_name, result)
                })
            })
            .collect();

        let results = futures::future::join_all(handles).await;

        // Assemble results — original instruction first, then generated
        let mut result: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, instruction) in &original_instructions {
            result.insert(name.clone(), vec![instruction.clone()]);
        }

        for join_result in results.into_iter().flatten() {
            let (pred_name, call_result) = join_result;
            if let Ok(prediction) = call_result
                && let Ok(instruction) = prediction.get::<String>("proposed_instruction")
            {
                let cleaned = strip_instruction_prefix(&instruction);
                if !cleaned.is_empty() {
                    result.entry(pred_name).or_default().push(cleaned);
                }
            }
        }

        Ok(result)
    }
}

/// Which optional input fields to include in the proposer's signature.
/// Bundled into a struct so callers can't transpose three adjacent `bool`
/// args at the only call site.
struct ProposalSignatureFields {
    dataset: bool,
    history: bool,
    tip: bool,
}

/// Build the instruction proposal signature with conditional fields.
fn build_proposal_signature(fields: ProposalSignatureFields) -> Result<Signature> {
    let ProposalSignatureFields {
        dataset: with_dataset,
        history: with_history,
        tip: with_tip,
    } = fields;
    let mut builder = Signature::builder(
        "Use the information below to learn about a task solved using a Language Model, \
         then generate a new instruction that will better guide the model to solve the task.",
    );

    if with_dataset {
        builder = builder.input(FieldDef::input(
            "dataset_description",
            FieldType::String,
            "A description of the dataset being used",
        ));
    }

    builder = builder
        .input(FieldDef::input(
            "module_description",
            FieldType::String,
            "Description of the language model program and its predictors",
        ))
        .input(FieldDef::input(
            "task_demos",
            FieldType::String,
            "Example inputs/outputs for the task",
        ));

    if with_history {
        builder = builder.input(FieldDef::input(
            "previous_instructions",
            FieldType::String,
            "Previous instructions attempted with their scores",
        ));
    }

    builder = builder.input(FieldDef::input(
        "basic_instruction",
        FieldType::String,
        "The current basic instruction",
    ));

    if with_tip {
        builder = builder.input(FieldDef::input(
            "tip",
            FieldType::String,
            "A suggestion for generating the new instruction",
        ));
    }

    builder
        .output(FieldDef::output(
            "proposed_instruction",
            FieldType::String,
            "A new instruction to better guide the language model",
        ))
        .build()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use async_trait::async_trait;
    use modelplease::{DummyLM, ModelId};

    use super::*;
    use crate::{adapter::ChatAdapter, prediction::Prediction};

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
                    ("question".into(), FieldValue::Str("What is 2+2?".into())),
                    ("answer".into(), FieldValue::Str("4".into())),
                ]),
                HashSet::from(["question".into()]),
            ),
            Example::new(
                BTreeMap::from([
                    (
                        "question".into(),
                        FieldValue::Str("Capital of France?".into()),
                    ),
                    ("answer".into(), FieldValue::Str("Paris".into())),
                ]),
                HashSet::from(["question".into()]),
            ),
        ]
    }

    #[tokio::test]
    async fn propose_generates_candidates() {
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        // LLM returns instruction candidates
        let lm = DummyLM::sequential(vec![
            "[[ ## proposed_instruction ## ]]\nBe precise and concise.\n[[ ## completed ## ]]"
                .into(),
            "[[ ## proposed_instruction ## ]]\nThink step by step.\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let mut proposer = GroundedProposer::new(42);
        let candidates = proposer
            .propose(ProposeRequest {
                module: &module,
                trainset: &trainset,
                n_candidates: 3,
                ctx: &ctx,
                instruction_history: None,
            })
            .await
            .unwrap();

        assert!(candidates.contains_key("qa"));
        let qa_candidates = &candidates["qa"];
        // Candidate 0 = original instruction, then up to 2 generated
        assert!(qa_candidates.len() >= 2);
        assert_eq!(qa_candidates[0], "Answer the question.");
    }

    #[tokio::test]
    async fn first_candidate_is_original_instruction() {
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![
            "[[ ## proposed_instruction ## ]]\nNew instruction.\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let mut proposer = GroundedProposer::new(42);
        let candidates = proposer
            .propose(ProposeRequest {
                module: &module,
                trainset: &trainset,
                n_candidates: 2,
                ctx: &ctx,
                instruction_history: None,
            })
            .await
            .unwrap();

        assert_eq!(candidates["qa"][0], "Answer the question.");
    }

    #[tokio::test]
    async fn propose_with_dataset_summary() {
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![
            "[[ ## proposed_instruction ## ]]\nWith context.\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let mut proposer = GroundedProposer::new(42);
        proposer.dataset_summary = Some("A Q&A dataset.".into());
        let candidates = proposer
            .propose(ProposeRequest {
                module: &module,
                trainset: &trainset,
                n_candidates: 2,
                ctx: &ctx,
                instruction_history: None,
            })
            .await
            .unwrap();

        assert!(candidates["qa"].len() >= 2);
    }

    #[tokio::test]
    async fn propose_with_instruction_history() {
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let lm = DummyLM::sequential(vec![
            "[[ ## proposed_instruction ## ]]\nImproved.\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let history = InstructionHistory {
            entries: vec![
                InstructionAttempt {
                    instruction: "Old instruction".into(),
                    avg_score: 0.3,
                },
                InstructionAttempt {
                    instruction: "Better instruction".into(),
                    avg_score: 0.7,
                },
            ],
        };

        let mut proposer = GroundedProposer::new(42);
        let candidates = proposer
            .propose(ProposeRequest {
                module: &module,
                trainset: &trainset,
                n_candidates: 2,
                ctx: &ctx,
                instruction_history: Some(&history),
            })
            .await
            .unwrap();

        assert!(candidates["qa"].len() >= 2);
    }

    #[tokio::test]
    async fn propose_skips_failed_candidates() {
        let module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        // First call fails (empty LM), but should still return original instruction
        let lm = DummyLM::sequential(vec![]);
        let ctx = Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };

        let mut proposer = GroundedProposer::new(42);
        let candidates = proposer
            .propose(ProposeRequest {
                module: &module,
                trainset: &trainset,
                n_candidates: 3,
                ctx: &ctx,
                instruction_history: None,
            })
            .await
            .unwrap();

        // Should have at least the original instruction
        assert!(!candidates["qa"].is_empty());
        assert_eq!(candidates["qa"][0], "Answer the question.");
    }

    #[test]
    fn instruction_history_format() {
        let history = InstructionHistory {
            entries: vec![
                InstructionAttempt {
                    instruction: "Do the thing".into(),
                    avg_score: 0.45,
                },
                InstructionAttempt {
                    instruction: "Do it better".into(),
                    avg_score: 0.78,
                },
            ],
        };
        let formatted = history.format();
        assert!(formatted.contains("\"Do the thing\" | Score: 0.45"));
        assert!(formatted.contains("\"Do it better\" | Score: 0.78"));
    }
}
