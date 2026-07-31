// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prompt optimizers: automatic few-shot demo selection and instruction tuning.
//!
//! The [`Optimizer`] trait provides a uniform interface for all optimization
//! strategies. [`BootstrapFewShot`] runs training examples through a teacher
//! model and keeps high-scoring predictions as demos. [`LabeledFewShot`] is a
//! simpler baseline that assigns raw training examples directly.

pub mod bootstrap;
pub mod labeled;
pub mod mipro;

use std::{fmt, sync::Arc};

use async_trait::async_trait;
pub use bootstrap::BootstrapFewShot;
pub use labeled::LabeledFewShot;
pub use mipro::{AutoMode, MIPROv2, MiproCompileRequest, MiproConfig, MiproDeps};
use typesayer_types::error::Result;

use crate::{context::Context, example::Example, module::Module, prediction::Prediction};

/// A metric function that scores a prediction against an expected example.
///
/// Returns a score in the range 0.0–1.0. Higher is better.
pub type MetricFn = Arc<dyn Fn(&Example, &Prediction) -> f64 + Send + Sync>;

/// A progress callback invoked during optimization.
///
/// Receives a [`Progress`] update at each significant step.
pub type ProgressFn = Arc<dyn Fn(&Progress) + Send + Sync>;

/// A progress update from an optimizer.
#[derive(Clone)]
pub struct Progress {
    /// Which phase of optimization is running.
    pub phase: String,
    /// Current step within the phase (1-indexed).
    pub step: usize,
    /// Total steps in the phase (0 if unknown).
    pub total: usize,
    /// Human-readable description of what's happening.
    pub message: String,
    /// Current best score (0.0–1.0), if applicable.
    pub best_score: Option<f64>,
}

impl fmt::Display for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.total > 0 {
            write!(
                f,
                "[{} {}/{}] {}",
                self.phase, self.step, self.total, self.message
            )?;
        } else {
            write!(f, "[{}] {}", self.phase, self.message)?;
        }
        if let Some(score) = self.best_score {
            write!(f, " (best: {:.1}%)", score * 100.0)?;
        }
        Ok(())
    }
}

/// Returns a no-op progress callback (used by tests that don't care
/// about per-step progress output).
#[cfg(test)]
#[must_use]
pub fn no_progress() -> ProgressFn {
    Arc::new(|_| {})
}

/// Per-call inputs for [`Optimizer::compile`].
///
/// Bundled so additive fields don't break-change every impl, and so the
/// adjacent `&Context` / `Option<&Context>` pair can't be transposed at the
/// call site. Optimizers that don't need every field simply ignore the
/// destructured locals.
pub struct CompileRequest<'a> {
    /// The module to optimize (mutated in place).
    pub module: &'a mut dyn Module,
    /// Training examples for demo generation / selection.
    pub trainset: &'a [Example],
    /// The student context (LM + adapter for evaluation).
    pub ctx: &'a Context,
    /// Optional stronger LM context for demo generation.
    pub teacher_ctx: Option<&'a Context>,
    /// Optional validation set for scoring candidates.
    pub valset: Option<&'a [Example]>,
    /// Callback for progress updates.
    pub progress: &'a ProgressFn,
}

/// A prompt optimizer that improves a module's performance by modifying its
/// predictors' demos, instructions, or both.
///
/// All optimizers implement this trait so they can be used interchangeably
/// (e.g., chained by `BetterTogether`).
#[async_trait]
pub trait Optimizer: Send + Sync {
    /// Compile the module by optimizing its predictors.
    ///
    /// Mutates the module in place (setting demos, instructions, etc.).
    /// Fields on [`CompileRequest`] that aren't needed by a specific optimizer
    /// are ignored.
    async fn compile(&self, args: CompileRequest<'_>) -> Result<()>;
}
