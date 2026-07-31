// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapter trait and implementations for formatting/parsing LM interactions.
//!
//! An [`Adapter`] converts between a [`Signature`](crate::Signature) with typed
//! input values and a sequence of [`Message`]s, and parses LM completions back
//! into typed [`FieldValue`](crate::FieldValue)s.

pub mod chat;

use std::collections::BTreeMap;

use async_trait::async_trait;
pub use chat::ChatAdapter;
use modelplease::Message;
use typesayer_types::{error::Result, field::FieldValue, signature::Signature};

/// A few-shot demonstration example with typed values.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Demo {
    /// Input field values for this example.
    pub inputs: BTreeMap<String, FieldValue>,
    /// Output field values for this example.
    pub outputs: BTreeMap<String, FieldValue>,
}

/// Converts between signatures with typed input values and LM message sequences.
#[async_trait]
pub trait Adapter: Send + Sync {
    /// Format a signature, typed input values, and few-shot demos into a sequence
    /// of messages ready to send to an LM.
    ///
    /// # Errors
    ///
    /// Returns [`PredictError`](typesayer_types::PredictError) if any field value cannot be
    /// serialized into the prompt format.
    async fn format(
        &self,
        signature: &Signature,
        inputs: &BTreeMap<String, FieldValue>,
        demos: &[Demo],
    ) -> Result<Vec<Message>>;

    /// Parse an LM completion string into a map of output field names to typed values.
    ///
    /// # Errors
    ///
    /// - [`PredictError::NoFieldMarkers`](typesayer_types::PredictError::NoFieldMarkers) if the completion
    ///   has no `[[ ## ... ## ]]` markers and multiple output fields
    /// - [`PredictError::MissingFields`](typesayer_types::PredictError::MissingFields) if not all output
    ///   fields are present
    /// - [`PredictError::FieldTypeMismatch`](typesayer_types::PredictError::FieldTypeMismatch) if a value
    ///   cannot be coerced to the declared type
    async fn parse(
        &self,
        signature: &Signature,
        completion: &str,
    ) -> Result<BTreeMap<String, FieldValue>>;
}
