// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Simple few-shot demo assignment from labeled training data.

use async_trait::async_trait;
use typesayer_types::error::Result;

use super::{CompileRequest, Optimizer};
use crate::{adapter::Demo, example::Example, module::Module};

/// Simple few-shot demo assignment from labeled training data.
///
/// Assigns up to `max_demos` training examples as demos to each predictor
/// in the module. No LLM calls, no metric evaluation — just direct injection.
///
/// # Examples
///
/// ```rust
/// use typesayer::LabeledFewShot;
///
/// let optimizer = LabeledFewShot::new(4);
/// ```
pub struct LabeledFewShot {
    max_demos: usize,
}

impl LabeledFewShot {
    /// Create a new optimizer that assigns up to `max_demos` labeled examples.
    #[must_use]
    pub const fn new(max_demos: usize) -> Self {
        Self { max_demos }
    }

    /// Assign training examples as demos to each predictor in the module.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result<()>` to match the optimizer
    /// trait signature and future-proof for validation additions.
    pub fn compile_labeled(&self, module: &mut dyn Module, trainset: &[Example]) -> Result<()> {
        let count = trainset.len().min(self.max_demos);
        let demos: Vec<Demo> = trainset[..count]
            .iter()
            .map(|ex| ex.clone().into())
            .collect();

        for (_name, predict) in module.named_predictors_mut() {
            predict.set_demos(demos.clone());
        }

        Ok(())
    }
}

#[async_trait]
impl Optimizer for LabeledFewShot {
    async fn compile(&self, args: CompileRequest<'_>) -> Result<()> {
        self.compile_labeled(args.module, args.trainset)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, HashSet},
        sync::Arc,
    };

    use async_trait::async_trait;
    use modelplease::{DummyLM, ModelId};
    use typesayer_types::{
        field::{FieldDef, FieldType, FieldValue},
        signature::Signature,
    };

    use super::*;
    use crate::{adapter::ChatAdapter, context::Context, predict::Predict, prediction::Prediction};

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

    #[test]
    fn labeled_assigns_demos() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        LabeledFewShot::new(2)
            .compile_labeled(&mut module, &trainset)
            .unwrap();
        assert_eq!(module.qa.demos().len(), 2);
    }

    #[test]
    fn labeled_empty_trainset() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        LabeledFewShot::new(4)
            .compile_labeled(&mut module, &[])
            .unwrap();
        assert!(module.qa.demos().is_empty());
    }

    #[test]
    fn labeled_respects_max_demos() {
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        LabeledFewShot::new(1)
            .compile_labeled(&mut module, &trainset)
            .unwrap();
        assert_eq!(module.qa.demos().len(), 1);
    }

    #[tokio::test]
    async fn labeled_works_via_optimizer_trait() {
        let lm = DummyLM::sequential(vec![]);
        let ctx = crate::Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        };
        let mut module = TestModule {
            qa: Predict::new(qa_signature()),
        };
        let trainset = make_trainset();

        let optimizer: Box<dyn Optimizer> = Box::new(LabeledFewShot::new(2));
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
        assert_eq!(module.qa.demos().len(), 2);
    }
}
