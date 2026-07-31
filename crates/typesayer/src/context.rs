// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prediction context — explicit parameter, not global state.

use std::sync::Arc;

use modelplease::{LanguageModelProvider, ModelId};

use crate::adapter::Adapter;

/// Holds the language model provider, target model, and adapter for
/// prediction calls.
///
/// Passed explicitly to [`Predict::call()`](crate::Predict::call) — there is no
/// global configuration or thread-local state.
///
/// Constructed with struct-literal syntax — the named fields prevent
/// any caller from silently transposing `provider` and `adapter`,
/// which would compile cleanly through a positional constructor since
/// both are `Arc<dyn _>` trait objects.
#[derive(Clone)]
pub struct Context {
    pub provider: Arc<dyn LanguageModelProvider>,
    pub model: ModelId,
    pub adapter: Arc<dyn Adapter>,
}

impl Context {
    /// The language model provider.
    #[must_use]
    pub fn provider(&self) -> &dyn LanguageModelProvider {
        self.provider.as_ref()
    }

    /// The model identifier this context dispatches to.
    #[must_use]
    pub const fn model(&self) -> &ModelId {
        &self.model
    }

    /// The adapter.
    #[must_use]
    pub fn adapter(&self) -> &dyn Adapter {
        self.adapter.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use modelplease::DummyLM;

    use super::*;
    use crate::adapter::ChatAdapter;

    fn make_ctx(answers: Vec<String>) -> Context {
        Context {
            provider: Arc::new(DummyLM::sequential(answers)),
            model: ModelId::new("test"),
            adapter: Arc::new(ChatAdapter::default()),
        }
    }

    #[test]
    fn new_context() {
        let ctx = make_ctx(vec!["test".into()]);
        assert!(ctx.usage_compiles());
    }

    impl Context {
        #[cfg(test)]
        fn usage_compiles(&self) -> bool {
            let _provider: &dyn LanguageModelProvider = self.provider();
            let _model: &ModelId = self.model();
            let _adapter: &dyn Adapter = self.adapter();
            true
        }
    }
}
