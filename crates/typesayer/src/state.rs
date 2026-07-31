// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Module save/load functions for DSPy-compatible JSON state format.
//!
//! `Predict::dump_state()` and `Predict::load_state()` are the source of
//! truth for the wire shape (they round-trip `serde_json::Value`); these
//! helpers just iterate over a module's named predictors and read/write
//! the file.

use std::path::Path;

use typesayer_types::error::Result;

/// Save all predictor states from a module to a JSON file.
///
/// The file format is a JSON object keyed by predictor dotted-path names,
/// with a `"metadata"` key containing version information.
///
/// # Errors
///
/// Returns a `PredictError` when a predictor's `dump_state()` fails, when
/// the final JSON serialization fails, or when the file cannot be written.
pub fn save_module_to_path(module: &dyn crate::module::Module, path: &Path) -> Result<()> {
    let predictors = module.named_predictors();
    let mut state = serde_json::Map::new();

    for (name, predict) in predictors {
        let predictor_state = predict.dump_state()?;
        state.insert(name, predictor_state);
    }

    state.insert(
        "metadata".to_owned(),
        serde_json::json!({
            "dependency_versions": {
                "typesayer": env!("CARGO_PKG_VERSION"),
            }
        }),
    );

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(state))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load predictor states from a JSON file into a module.
///
/// Predictors named in the file that are present in the module have their
/// demos and instructions updated. Predictors missing from the file are
/// unchanged. Unknown names in the file are silently ignored.
///
/// # Errors
///
/// Returns a `PredictError` when the file cannot be read, is not valid
/// JSON, is not a JSON object, or when a predictor's `load_state()` fails.
pub fn load_module_from_path(module: &mut dyn crate::module::Module, path: &Path) -> Result<()> {
    let json = std::fs::read_to_string(path)?;
    let state: serde_json::Value = serde_json::from_str(&json)?;

    let state_obj = state.as_object().ok_or_else(|| {
        typesayer_types::PredictError::invalid_signature("module state must be a JSON object")
    })?;

    for (name, predict) in module.named_predictors_mut() {
        if let Some(predictor_state) = state_obj.get(&name) {
            predict.load_state(predictor_state)?;
        }
    }

    Ok(())
}
