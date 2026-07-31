// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instruction proposer infrastructure for prompt optimization.
//!
//! Proposers are LLM programs that generate instruction candidates for
//! optimizers. [`GroundedProposer`] generates data-aware candidates for
//! `MIPROv2`. [`summarize_dataset`] builds a concise dataset summary
//! for use by proposers.

pub mod dataset_summary;
pub mod grounded;

pub use dataset_summary::summarize_dataset;
pub use grounded::{GroundedProposer, ProposeRequest};
use typesayer_types::field::FieldValue;

use crate::{example::Example, format::serialize_value, module::Module};

/// Format an [`Example`]'s fields as readable text for LLM consumption.
///
/// ```text
/// question: What causes earthquakes?
/// answer: Tectonic plate movement
/// ```
#[must_use]
pub fn format_example_as_text(example: &Example) -> String {
    example
        .fields()
        .iter()
        .map(|(k, v)| format!("{k}: {}", format_field_value(v)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format a batch of examples as numbered text.
#[must_use]
pub fn format_examples_batch(examples: &[Example]) -> String {
    examples
        .iter()
        .enumerate()
        .map(|(i, ex)| format!("Example {}:\n{}", i + 1, format_example_as_text(ex)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Describe a module's structure from its predictors and signatures.
///
/// ```text
/// Module with 2 predictors:
/// 1. "classify" — inputs: question (str) → outputs: topic (str)
///    Current instruction: "Classify the topic."
/// 2. "answer" — inputs: question (str), topic (str) → outputs: answer (str)
///    Current instruction: "Answer the question."
/// ```
pub fn describe_module_structure(module: &dyn Module) -> String {
    let predictors = module.named_predictors();
    let mut lines = vec![format!("Module with {} predictor(s):", predictors.len())];

    for (i, (name, predict)) in predictors.iter().enumerate() {
        let sig = predict.signature();

        let inputs: Vec<String> = sig
            .input_fields()
            .map(|f| format!("{} ({})", f.name, f.field_type.type_label()))
            .collect();

        let outputs: Vec<String> = sig
            .output_fields()
            .map(|f| format!("{} ({})", f.name, f.field_type.type_label()))
            .collect();

        lines.push(format!(
            "{}. \"{}\" — inputs: {} → outputs: {}",
            i + 1,
            name,
            inputs.join(", "),
            outputs.join(", "),
        ));
        lines.push(format!(
            "   Current instruction: \"{}\"",
            sig.instructions()
        ));
    }

    lines.join("\n")
}

/// Strip common prefixes from LLM-generated instructions.
///
/// Removes patterns like "Instruction:", "New instruction:", "Here is the
/// instruction:", etc., plus surrounding quotes and whitespace.
#[must_use]
pub fn strip_instruction_prefix(instruction: &str) -> String {
    let trimmed = instruction.trim();

    // Strip common prefixes (case-insensitive check on first ~30 chars)
    let lower = trimmed.to_lowercase();
    let stripped = if lower.starts_with("new instruction:") {
        &trimmed["new instruction:".len()..]
    } else if lower.starts_with("instruction:") {
        &trimmed["instruction:".len()..]
    } else if lower.starts_with("here is the instruction:") {
        &trimmed["here is the instruction:".len()..]
    } else if lower.starts_with("proposed instruction:") {
        &trimmed["proposed instruction:".len()..]
    } else {
        trimmed
    };

    // Strip surrounding quotes
    let stripped = stripped.trim();
    let stripped = stripped.strip_prefix('"').unwrap_or(stripped);
    let stripped = stripped.strip_suffix('"').unwrap_or(stripped);
    stripped.trim().to_owned()
}

/// Convert a [`FieldValue`] to a readable string.
fn format_field_value(value: &FieldValue) -> String {
    serialize_value(value)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use async_trait::async_trait;
    use typesayer_types::{
        error::Result,
        field::{FieldDef, FieldType},
        signature::Signature,
    };

    use super::*;
    use crate::{context::Context, predict::Predict, prediction::Prediction};

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

    struct TwoStepModule {
        classify: Predict,
        answer: Predict,
    }

    #[async_trait]
    impl Module for TwoStepModule {
        async fn forward(
            &self,
            _inputs: BTreeMap<String, FieldValue>,
            _ctx: &Context,
        ) -> Result<Prediction> {
            unimplemented!("not needed for structure tests")
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

    #[test]
    fn format_example_basic() {
        let ex = Example::new(
            BTreeMap::from([
                ("question".into(), FieldValue::Str("What is 2+2?".into())),
                ("answer".into(), FieldValue::Str("4".into())),
            ]),
            HashSet::from(["question".into()]),
        );
        let text = format_example_as_text(&ex);
        assert!(text.contains("answer: 4"));
        assert!(text.contains("question: What is 2+2?"));
    }

    #[test]
    fn format_example_handles_types() {
        let ex = Example::new(
            BTreeMap::from([
                ("count".into(), FieldValue::Int(42)),
                ("active".into(), FieldValue::Bool(true)),
                ("name".into(), FieldValue::Str("test".into())),
                ("empty".into(), FieldValue::Null),
            ]),
            HashSet::from(["name".into()]),
        );
        let text = format_example_as_text(&ex);
        assert!(text.contains("count: 42"));
        assert!(text.contains("active: true"));
        assert!(text.contains("name: test"));
        assert!(text.contains("empty: null"));
    }

    #[test]
    fn describe_module_single_predictor() {
        struct SingleModule {
            qa: Predict,
        }

        #[async_trait]
        impl Module for SingleModule {
            async fn forward(
                &self,
                inputs: BTreeMap<String, FieldValue>,
                _ctx: &Context,
            ) -> Result<Prediction> {
                // No-op: echo inputs back as the prediction. `describe_module_structure`
                // never exercises this path, but the trait contract requires a real
                // implementation — returning the inputs keeps the module usable in
                // any test that does call `forward` without maintaining a separate
                // mock layer.
                Ok(Prediction::new(inputs, None))
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

        let module = SingleModule {
            qa: Predict::new(qa_signature()),
        };
        let desc = describe_module_structure(&module);
        assert!(desc.contains("1 predictor"));
        assert!(desc.contains("\"qa\""));
        assert!(desc.contains("question (str)"));
        assert!(desc.contains("answer (str)"));
        assert!(desc.contains("Answer the question."));
    }

    #[test]
    fn describe_module_two_predictors() {
        let classify_sig = Signature::builder("Classify the topic.")
            .input(FieldDef::input("question", FieldType::String, "q"))
            .output(FieldDef::output("topic", FieldType::String, "t"))
            .build()
            .unwrap();

        let answer_sig = Signature::builder("Answer given topic.")
            .input(FieldDef::input("question", FieldType::String, "q"))
            .input(FieldDef::input("topic", FieldType::String, "t"))
            .output(FieldDef::output("answer", FieldType::String, "a"))
            .build()
            .unwrap();

        let module = TwoStepModule {
            classify: Predict::new(classify_sig),
            answer: Predict::new(answer_sig),
        };

        let desc = describe_module_structure(&module);
        assert!(desc.contains("2 predictor"));
        assert!(desc.contains("\"classify\""));
        assert!(desc.contains("\"answer\""));
        assert!(desc.contains("topic (str)"));
    }

    #[test]
    fn strip_prefix_instruction() {
        assert_eq!(
            strip_instruction_prefix("Instruction: Do the thing"),
            "Do the thing"
        );
    }

    #[test]
    fn strip_prefix_new_instruction() {
        assert_eq!(
            strip_instruction_prefix("New instruction: Do the thing"),
            "Do the thing"
        );
    }

    #[test]
    fn strip_prefix_with_quotes() {
        assert_eq!(strip_instruction_prefix("\"Do the thing\""), "Do the thing");
    }

    #[test]
    fn strip_prefix_clean_input() {
        assert_eq!(strip_instruction_prefix("Do the thing"), "Do the thing");
    }

    #[test]
    fn strip_prefix_case_insensitive() {
        assert_eq!(
            strip_instruction_prefix("INSTRUCTION: Do the thing"),
            "Do the thing"
        );
    }

    #[test]
    fn strip_prefix_combined() {
        assert_eq!(
            strip_instruction_prefix("New Instruction: \"Do the thing\""),
            "Do the thing"
        );
    }
}
