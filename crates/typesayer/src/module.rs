// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Module trait for composing multi-step LLM programs.
//!
//! A [`Module`] is a composable unit of LLM-powered logic that contains one or
//! more [`Predict`] instances. The trait provides predictor introspection for
//! optimizer access and state serialization for saving/loading trained parameters.

use std::{collections::BTreeMap, path::Path};

use async_trait::async_trait;
use typesayer_types::{error::Result, field::FieldValue};

use crate::{
    context::Context,
    predict::Predict,
    prediction::Prediction,
    state::{load_module_from_path, save_module_to_path},
};

/// A composable unit of LLM-powered logic.
///
/// Implement this trait for any struct that contains one or more [`Predict`]
/// instances. The trait provides predictor introspection for demo injection
/// and optimizer access, plus state serialization for saving/loading
/// trained parameters.
///
/// # Naming Convention
///
/// Predictor names use dotted paths: `"qa"` for a top-level field named `qa`,
/// `"chain.qa"` for a `qa` field inside a sub-module named `chain`.
///
/// # Example
///
/// ```rust
/// use std::collections::BTreeMap;
///
/// use async_trait::async_trait;
/// use typesayer::{
///     Context, FieldDef, FieldType, FieldValue, Module, Predict, Prediction, Signature,
/// };
///
/// struct QAModule {
///     qa: Predict,
/// }
///
/// impl QAModule {
///     fn new() -> typesayer::Result<Self> {
///         let sig = Signature::builder("Answer the question.")
///             .input(FieldDef::input("question", FieldType::String, "The question"))
///             .output(FieldDef::output("answer", FieldType::String, "The answer"))
///             .build()?;
///         Ok(Self { qa: Predict::new(sig) })
///     }
/// }
///
/// #[async_trait]
/// impl Module for QAModule {
///     async fn forward(
///         &self,
///         inputs: BTreeMap<String, FieldValue>,
///         ctx: &Context,
///     ) -> typesayer::Result<Prediction> {
///         self.qa.call(&inputs, ctx).await
///     }
///
///     fn named_predictors(&self) -> Vec<(String, &Predict)> {
///         vec![("qa".to_owned(), &self.qa)]
///     }
///
///     fn named_predictors_mut(&mut self) -> Vec<(String, &mut Predict)> {
///         vec![("qa".to_owned(), &mut self.qa)]
///     }
///
///     fn deep_clone(&self) -> Box<dyn Module> {
///         Box::new(QAModule { qa: self.qa.clone() })
///     }
/// }
/// ```
#[async_trait]
pub trait Module: Send + Sync {
    /// Execute the module's primary logic.
    async fn forward(
        &self,
        inputs: BTreeMap<String, FieldValue>,
        ctx: &Context,
    ) -> Result<Prediction>;

    /// Return all [`Predict`] instances with their dotted path names, for read access.
    ///
    /// Nested sub-modules must be traversed manually. A top-level `Predict` field
    /// named `qa` should appear as `"qa"`. A `Predict` nested inside a sub-module
    /// field named `chain` should appear as `"chain.qa"`.
    fn named_predictors(&self) -> Vec<(String, &Predict)>;

    /// Return all [`Predict`] instances with their dotted path names, for mutable access.
    ///
    /// Must return the same names in the same order as
    /// [`named_predictors`](Module::named_predictors).
    fn named_predictors_mut(&mut self) -> Vec<(String, &mut Predict)>;

    /// Create an independent deep copy of this module as a boxed trait object.
    ///
    /// Each concrete module type must implement this by cloning all inner
    /// state (predictors, sub-modules, etc.). Used by optimizers that need
    /// independent copies for parallel candidate evaluation.
    fn deep_clone(&self) -> Box<dyn Module>;

    /// Execute the module's primary logic with execution tracing.
    ///
    /// Returns both the final prediction and an [`ExecutionTrace`](crate::ExecutionTrace) recording
    /// every predictor invocation. The default implementation calls
    /// [`forward()`](Module::forward) and returns an empty trace.
    ///
    /// Concrete modules should override this to thread trace collection
    /// through their predictor calls using [`Predict::call_traced()`].
    async fn forward_traced(
        &self,
        inputs: BTreeMap<String, FieldValue>,
        ctx: &Context,
    ) -> Result<(Prediction, crate::trace::ExecutionTrace)> {
        let prediction = self.forward(inputs, ctx).await?;
        Ok((prediction, crate::trace::ExecutionTrace::new()))
    }

    /// Save all predictor states to a JSON file.
    ///
    /// The file format is compatible with `DSPy`'s saved module format.
    ///
    /// # Errors
    ///
    /// Propagates errors from the module state serializer.
    fn save(&self, path: &Path) -> Result<()>
    where
        Self: Sized,
    {
        save_module_to_path(self, path)
    }

    /// Load predictor states from a JSON file.
    ///
    /// Predictors named in the file that are present in this module have their
    /// demos and instructions updated. Predictors missing from the file are
    /// unchanged. Unknown names in the file are silently ignored.
    ///
    /// # Errors
    ///
    /// Propagates errors from the module state loader.
    fn load(&mut self, path: &Path) -> Result<()>
    where
        Self: Sized,
    {
        load_module_from_path(self, path)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use modelplease::{DummyLM, ModelId};
    use typesayer_types::{
        field::{FieldDef, FieldType},
        signature::Signature,
    };

    use super::*;
    use crate::adapter::Demo;

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

    // --- Flat module with a single Predict ---

    struct FlatModule {
        qa: Predict,
    }

    #[async_trait]
    impl Module for FlatModule {
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

    // --- Nested module with two sub-modules ---

    struct NestedModule {
        first: FlatModule,
        second: FlatModule,
    }

    #[async_trait]
    impl Module for NestedModule {
        async fn forward(
            &self,
            inputs: BTreeMap<String, FieldValue>,
            ctx: &Context,
        ) -> Result<Prediction> {
            self.first.forward(inputs, ctx).await
        }

        fn named_predictors(&self) -> Vec<(String, &Predict)> {
            let mut predictors = Vec::new();
            for (name, pred) in self.first.named_predictors() {
                predictors.push((format!("first.{name}"), pred));
            }
            for (name, pred) in self.second.named_predictors() {
                predictors.push((format!("second.{name}"), pred));
            }
            predictors
        }

        fn named_predictors_mut(&mut self) -> Vec<(String, &mut Predict)> {
            let mut predictors = Vec::new();
            for (name, pred) in self.first.named_predictors_mut() {
                predictors.push((format!("first.{name}"), pred));
            }
            for (name, pred) in self.second.named_predictors_mut() {
                predictors.push((format!("second.{name}"), pred));
            }
            predictors
        }

        fn deep_clone(&self) -> Box<dyn Module> {
            Box::new(Self {
                first: FlatModule {
                    qa: self.first.qa.clone(),
                },
                second: FlatModule {
                    qa: self.second.qa.clone(),
                },
            })
        }
    }

    #[test]
    fn flat_module_named_predictors() {
        let module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        let predictors = module.named_predictors();
        assert_eq!(predictors.len(), 1);
        assert_eq!(predictors[0].0, "qa");
    }

    #[test]
    fn nested_module_named_predictors() {
        let module = NestedModule {
            first: FlatModule {
                qa: Predict::new(qa_signature()),
            },
            second: FlatModule {
                qa: Predict::new(qa_signature()),
            },
        };
        let predictors = module.named_predictors();
        assert_eq!(predictors.len(), 2);
        assert_eq!(predictors[0].0, "first.qa");
        assert_eq!(predictors[1].0, "second.qa");
    }

    #[tokio::test]
    async fn flat_module_forward() {
        let module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = crate::Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(crate::adapter::ChatAdapter::default()),
        };
        let inputs = BTreeMap::from([(
            "question".into(),
            FieldValue::Str("Capital of France?".into()),
        )]);
        let prediction = module.forward(inputs, &ctx).await.unwrap();
        assert_eq!(prediction.get::<String>("answer").unwrap(), "Paris");
    }

    #[test]
    fn save_load_round_trip() {
        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };

        // Add demos
        module.qa.set_demos(vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("1+1?".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("2".into()))]),
        }]);

        // Save
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        module.save(&path).unwrap();

        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            saved["metadata"]["dependency_versions"]["typesayer"],
            env!("CARGO_PKG_VERSION")
        );

        // Load into a fresh module
        let mut fresh = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        assert!(fresh.qa.demos().is_empty());
        fresh.load(&path).unwrap();
        assert_eq!(fresh.qa.demos().len(), 1);
        assert_eq!(
            fresh.qa.demos()[0].inputs["question"],
            FieldValue::Str("1+1?".into())
        );
    }

    #[test]
    fn loads_legacy_catalyzed_predict_metadata() {
        let source = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        let state = serde_json::json!({
            "qa": source.qa.dump_state().unwrap(),
            "metadata": {
                "dependency_versions": {"catalyzed_predict": "0.1.0"}
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy-state.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();

        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        module.load(&path).unwrap();
    }

    #[test]
    fn ignores_unknown_metadata_keys() {
        let state = serde_json::json!({
            "metadata": {
                "dependency_versions": {"future-runtime": "99.0.0"},
                "unknown": {"nested": true}
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unknown-metadata.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();

        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        module.load(&path).unwrap();
    }

    #[test]
    fn invalid_module_state_preserves_structured_error() {
        let state = serde_json::json!({"qa": {"demos": [42]}});
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid-state.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();

        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        let error = module.load(&path).unwrap_err();
        assert!(matches!(
            error,
            typesayer_types::PredictError::InvalidSignature { .. }
        ));
    }

    #[test]
    fn load_dspy_format() {
        let dspy_json = serde_json::json!({
            "qa": {
                "traces": [],
                "train": [],
                "demos": [
                    {"question": "What is 2+2?", "answer": "4"}
                ],
                "signature": {
                    "instructions": "Answer the question concisely.",
                    "fields": [
                        {"prefix": "Question:", "description": "${question}"},
                        {"prefix": "Answer:", "description": "${answer}"}
                    ]
                },
                "lm": null
            },
            "metadata": {
                "dependency_versions": {"python": "3.13", "dspy": "3.0.0"}
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dspy_state.json");
        std::fs::write(&path, serde_json::to_string_pretty(&dspy_json).unwrap()).unwrap();

        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        module.load(&path).unwrap();

        // Demos should be reconstructed with correct split
        assert_eq!(module.qa.demos().len(), 1);
        assert_eq!(
            module.qa.demos()[0].inputs["question"],
            FieldValue::Str("What is 2+2?".into())
        );
        assert_eq!(
            module.qa.demos()[0].outputs["answer"],
            FieldValue::Str("4".into())
        );

        // Instructions should be updated
        assert_eq!(
            module.qa.signature().instructions(),
            "Answer the question concisely."
        );
    }

    #[test]
    fn named_predictors_mut_injects_demos() {
        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };

        for (_name, predict) in module.named_predictors_mut() {
            predict.set_demos(vec![Demo {
                inputs: BTreeMap::from([("question".into(), FieldValue::Str("test".into()))]),
                outputs: BTreeMap::from([("answer".into(), FieldValue::Str("test".into()))]),
            }]);
        }

        assert_eq!(module.qa.demos().len(), 1);
    }

    #[test]
    fn deep_clone_produces_independent_copy() {
        let mut module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        module.qa.set_demos(vec![Demo {
            inputs: BTreeMap::from([("question".into(), FieldValue::Str("1+1?".into()))]),
            outputs: BTreeMap::from([("answer".into(), FieldValue::Str("2".into()))]),
        }]);

        let cloned = module.deep_clone();

        // Modify original — clone should be unaffected
        module.qa.set_demos(vec![]);

        let cloned_demos = cloned.named_predictors()[0].1.demos();
        assert_eq!(cloned_demos.len(), 1);
        assert!(module.qa.demos().is_empty());
    }

    #[tokio::test]
    async fn forward_traced_default_returns_empty_trace() {
        let module = FlatModule {
            qa: Predict::new(qa_signature()),
        };
        let lm = DummyLM::sequential(vec![
            "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into(),
        ]);
        let ctx = crate::Context {
            provider: Arc::new(lm),
            model: ModelId::new("test"),
            adapter: Arc::new(crate::adapter::ChatAdapter::default()),
        };
        let inputs = BTreeMap::from([(
            "question".into(),
            FieldValue::Str("Capital of France?".into()),
        )]);

        let (prediction, trace) = module.forward_traced(inputs, &ctx).await.unwrap();
        assert_eq!(prediction.get::<String>("answer").unwrap(), "Paris");
        assert!(trace.is_empty()); // default impl returns empty trace
    }
}
