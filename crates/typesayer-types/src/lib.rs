// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod error;
pub mod field;
pub mod signature;

pub use error::{PredictError, Result};
pub use field::{
    FieldDef, FieldKind, FieldType, FieldValue, ObjectField, OneOfDiscriminator, VariantArm,
};
pub use signature::{Signature, SignatureBuilder};
