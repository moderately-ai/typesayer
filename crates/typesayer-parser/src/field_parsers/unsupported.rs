// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level FieldParser placeholder for output FieldTypes that have
//! no streaming support yet (currently only `Media`). Emits a clear
//! [`ParseEvent::StreamError`] on first push so the consumer returns
//! a non-retryable failure with a useful error message.
//!
//! Distinct from [`super::json::unsupported::JsonUnsupportedParser`]
//! which serves the same role one layer down inside the JSON value
//! parser dispatch — this one is for top-level marker-delimited
//! fields whose entire FieldType lacks a parser; the JSON-side one
//! is for unsupported INNER types inside containers.

use typesayer_types::field::{FieldType, FieldValue};

use super::{super::event::ParseEvent, FieldParser};

pub(super) struct UnsupportedParser {
    path: String,
    field_type: FieldType,
    errored: bool,
}

impl UnsupportedParser {
    pub(super) const fn new(path: String, field_type: FieldType) -> Self {
        Self {
            path,
            field_type,
            errored: false,
        }
    }
}

impl FieldParser for UnsupportedParser {
    fn push(&mut self, _chunk: &str) -> Vec<ParseEvent> {
        if self.errored {
            return Vec::new();
        }
        self.errored = true;
        vec![ParseEvent::StreamError {
            path: Some(self.path.clone()),
            message: format!(
                "streaming parser does not yet support field type '{}'",
                self.field_type
            ),
        }]
    }

    fn finish(self: Box<Self>) -> (Vec<ParseEvent>, FieldValue) {
        if self.errored {
            (Vec::new(), FieldValue::Null)
        } else {
            (
                vec![ParseEvent::StreamError {
                    path: Some(self.path),
                    message: format!(
                        "streaming parser does not yet support field type '{}'",
                        self.field_type
                    ),
                }],
                FieldValue::Null,
            )
        }
    }
}
