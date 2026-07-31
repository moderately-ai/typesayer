// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! JSON value-parser layer — incremental, type-aware, recursive.
//!
//! Each [`JsonValueParser`] parses ONE complete JSON value matching
//! the expected [`FieldType`](typesayer_types::field::FieldType). Self-terminating on
//! the value's syntactic close (`"` ends a string, `]` ends an array,
//! `}` ends an object, transition to non-digit ends a number, etc.) —
//! the parser reports `consumed: usize` so the caller knows where the
//! value ended and what remains belongs to whatever comes next (a
//! sibling element, a closing bracket, trailing whitespace).
//!
//! Containers ([`array::JsonArrayParser`], `object::JsonObjectParser`)
//! recurse into [`json_value_parser_for`] for their element / value
//! types, so arbitrary nesting (`List<List<Object<...>>>`) flows
//! through the same dispatch without container-specific glue.

pub(super) mod array;
pub(super) mod bool;
pub(super) mod enum_;
pub(super) mod map;
pub(super) mod multi_arm;
pub(super) mod null;
pub(super) mod nullable;
pub(super) mod number;
pub(super) mod object;
pub(super) mod skip;
pub(super) mod string;
pub(super) mod unsupported;

use typesayer_types::field::{FieldType, FieldValue};

use super::super::event::ParseEvent;

/// Outcome of one push of input into a [`JsonValueParser`].
///
/// `consumed` is the number of bytes from the pushed chunk that the
/// parser took. The caller passes anything after that to whatever
/// comes next — typically the parent parser's main state machine
/// (e.g. an array parser sees a child element finish, then resumes
/// looking for `,` or `]`).
#[derive(Debug)]
pub(super) struct JsonStep {
    pub(super) consumed: usize,
    pub(super) events: Vec<ParseEvent>,
    pub(super) completion: JsonCompletion,
}

#[derive(Debug)]
pub(super) enum JsonCompletion {
    /// Parser consumed up to `consumed` bytes and needs more input
    /// before it can complete its value.
    NeedMore,
    /// Parser successfully built a value. The parent should advance
    /// past `consumed` bytes and resume its own state machine.
    Done(FieldValue),
    /// Parser hit a terminal grammar error. The parent should also
    /// terminate. `consumed` reports how many bytes were consumed
    /// before the error so the parent can include context.
    Errored(String),
}

/// One-complete-JSON-value parser, parameterised on the expected
/// [`FieldType`](typesayer_types::field::FieldType). Holds its own state across pushes.
pub(super) trait JsonValueParser: Send {
    /// Feed input. Returns events + the parser's status. Once a
    /// parser returns `Done` or `Errored`, calling `push` again is
    /// a contract violation by the parent; the implementation may
    /// panic or no-op (impls in this module no-op for forward-
    /// compat).
    fn push(&mut self, chunk: &str) -> JsonStep;

    /// Finalize at end-of-input (the wrapping field's content
    /// finished without the JSON value self-closing). Returns the
    /// best-effort value (partial accumulator state) PLUS a stream
    /// error event identifying where the parser was stuck.
    fn finish(self: Box<Self>, path: &str) -> (Vec<ParseEvent>, FieldValue);
}

/// Build a JSON value parser for a given expected `FieldType`. Used
/// by container parsers to dispatch their inner element / value
/// parsers — this is what makes recursive nesting work uniformly.
///
/// FieldTypes not yet implemented funnel to
/// [`unsupported::JsonUnsupportedParser`], which emits a clear
/// [`ParseEvent::StreamError`] on first push rather than silently
/// accepting input. Adding a new variant means writing a new
/// `JsonValueParser` impl and adding an arm here.
pub(super) fn json_value_parser_for(
    field_type: &FieldType,
    path: String,
) -> Box<dyn JsonValueParser> {
    match field_type {
        FieldType::String => Box::new(string::JsonStringParser::new(path)),
        FieldType::Int => Box::new(number::JsonNumberParser::int(path)),
        FieldType::Float => Box::new(number::JsonNumberParser::float(path)),
        FieldType::Bool => Box::new(bool::JsonBoolParser::new(path)),
        FieldType::Enum(variants) => Box::new(enum_::JsonEnumParser::new(path, variants.clone())),
        FieldType::Nullable(inner) => {
            Box::new(nullable::JsonNullableParser::new(path, (**inner).clone()))
        }
        FieldType::List(inner) => Box::new(array::JsonArrayParser::new(path, (**inner).clone())),
        FieldType::Object(fields) => Box::new(object::JsonObjectParser::new(path, fields.clone())),
        FieldType::Map(value_type) => {
            Box::new(map::JsonMapParser::new(path, (**value_type).clone()))
        }
        other @ FieldType::Media { .. } => {
            Box::new(unsupported::JsonUnsupportedParser::new(path, other.clone()))
        }
        FieldType::OneOf {
            arms,
            discriminator,
        } => {
            // Tagged OneOf uses the fast-path scanner to identify the
            // arm before any per-arm parser sees bytes; untagged falls
            // back to the parallel-arm path.
            match discriminator {
                Some(d) => Box::new(multi_arm::JsonMultiArmParser::tagged_oneof(
                    path,
                    arms,
                    d.property.clone(),
                    d.tags.clone(),
                )),
                None => Box::new(multi_arm::JsonMultiArmParser::oneof(path, arms)),
            }
        }
        FieldType::AnyOf { arms } => Box::new(multi_arm::JsonMultiArmParser::anyof(path, arms)),
    }
}

/// Helper: count bytes of leading ASCII whitespace in `s`. Used by
/// every state-machine arm that allows whitespace between syntactic
/// tokens (the `whitespace?` slots in the JSON grammar). Returns the
/// byte index of the first non-whitespace char, or `s.len()` if `s`
/// is all whitespace.
pub(super) fn skip_whitespace(s: &str) -> usize {
    s.bytes()
        .take_while(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        .count()
}
