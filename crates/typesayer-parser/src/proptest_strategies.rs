// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-based testing strategies for the streaming chat-adapter
//! codec.
//!
//! Bounded random `(FieldType, FieldValue)` pairs + arbitrary chunk
//! splits drive smoke property tests against `JsonStringParser`,
//! `JsonArrayParser`, and the top-level [`crate::ChatStreamParser`].
//!
//! ## Bounds (per the streaming-output plan)
//!
//! - **Depth**: up to 10 nested `List<List<…>>`. Enforced by [`arb_supported_field_type`]'s
//!   `prop_recursive` and bounded incidentally by the leaf-budget (see below) once a value tree is
//!   materialised.
//! - **Total leaves per value**: up to `MAX_LEAF_BUDGET = 1000`. Enforced by
//!   [`arb_value_for_type`]'s budget-aware recursion — each `List<T>` claims at most
//!   `min(remaining_budget, MAX_COLLECTION_SIZE)` elements and divides the remaining budget evenly
//!   among children. By construction, the leaf count of any generated value is ≤ the root budget.
//!   Without this, `MAX_COLLECTION_SIZE^MAX_DEPTH = 20^10` leaves were possible at worst case — the
//!   original observed cost was ~42 s for the array round-trip.
//! - **Collection size at any level**: up to `MAX_COLLECTION_SIZE = 20`. Hard cap independent of
//!   remaining budget — top-level lists can still be 20 wide, but deep nesting fans out smaller
//!   (the budget geometric-shrinks naturally).
//! - **String length**: 0–100 chars from the safe alphabet.
//!
//! These bounds are "large but bounded": the worst-case JSON payload
//! per case is ~100 KB (1000 leaves × 100 chars + JSON syntax), the
//! typical case is far smaller because proptest's own case sizing
//! prefers small generations early. Across the default 256 cases,
//! the worst-case is ~25 MB of generated content — well within the
//! tractable envelope for a smoke test in debug builds.
//!
//! ## Coverage
//!
//! [`arb_supported_field_type`] and [`arb_value_for_type`] cover the
//! FieldTypes the streaming codec currently supports: top-level
//! `String` and `List<T>` recursive where `T` eventually reaches
//! `String`. Unsupported variants (`Int`, `Float`, `Bool`, `Map`,
//! `Object`, `Enum`, `Nullable`, `Media`) are exercised by
//! [`arb_unsupported_field_type`] feeding a separate property that
//! asserts the parser emits `StreamError` without panic.
//!
//! ## Alphabet restriction
//!
//! [`arb_safe_string`] generates strings from a restricted ASCII
//! alphabet (`[a-zA-Z0-9 .,!?-]`) so generated values cannot
//! accidentally form `[[ ## ` marker prefixes that the parser would
//! interpret as field boundaries. It also avoids `"` and `\` so
//! generated JSON encodings stay free of escape sequences (escape
//! handling has its own dedicated unit tests).

use std::collections::BTreeMap;

use proptest::{collection::vec, prelude::*, sample::select};
use typesayer_types::{
    FieldDef,
    field::{FieldType, FieldValue, ObjectField, OneOfDiscriminator, VariantArm},
    signature::Signature,
};

/// Max depth of nested `List<List<...>>` types. Per the plan.
const MAX_DEPTH: u32 = 10;
/// Soft target for total node count across one generated type tree.
/// Used by `prop_recursive` to size the FieldType skeleton — separate
/// from the value-tree leaf budget below.
const MAX_TYPE_NODES: u32 = 1000;
/// Approximate branching factor for `prop_recursive` sizing on the
/// type tree.
const EXPECTED_BRANCH_SIZE: u32 = 5;
/// Hard cap on collection size at any single nesting level. Combined
/// with `MAX_LEAF_BUDGET` this prevents both shape-pathological cases
/// (one list with 10k elements) and depth-pathological cases
/// (10-deep nesting with no width cap).
const MAX_COLLECTION_SIZE: usize = 20;
/// Total leaf budget per generated value tree. The budget-aware
/// recursion in [`arb_value_for_type`] divides this among children
/// at each level, so the total leaf count of any generated value is
/// `≤ MAX_LEAF_BUDGET`. Matches the plan's `max_total_nodes` target.
const MAX_LEAF_BUDGET: usize = 1000;
/// Max char count of a generated string. The leaf-budget bound and
/// per-leaf string length together cap the worst-case JSON payload
/// size per generated value at `MAX_LEAF_BUDGET * (MAX_STRING_LEN +
/// JSON syntax overhead)` ≈ 100 KB.
const MAX_STRING_LEN: usize = 100;

/// Restricted alphabet for generated string values — printable ASCII
/// without `[` (would risk accidentally forming the `[[ ## ` marker
/// prefix the parser scans for), `"` / `\` (would require escape-
/// sequence generation), or control characters (RFC 8259 §7 rejects
/// unescaped controls inside JSON strings). `&'static [char]` so
/// `select(SAFE_ALPHABET)` just stores a slice reference per call.
#[rustfmt::skip]
const SAFE_ALPHABET: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ' ', '.', ',', '!', '?', '-',
];

/// Restricted alphabet for generated string values; see `SAFE_ALPHABET` for the included
/// characters and rationale.
///
/// Cheap to construct on each call — `select(SAFE_ALPHABET)` stores
/// a `&'static [char]` reference, no regex parse or compile fires.
///
/// Strings are trimmed before being returned. The buffered parser
/// (`format.rs:216`) calls `trimmed.to_owned()` when decoding a JSON
/// string into a `FieldType::String`, even inside nested containers —
/// so a generated `" R0..."` round-trips through buffered as `"R0..."`
/// while streaming preserves the leading space verbatim. This is a
/// known buffered quirk that loses data inside nested strings; rather
/// than match it (which would make the streaming parser also lose
/// data) we filter the generator to exclude strings whose ends would
/// be trimmed. The core parsing logic is unaffected — any non-trim
/// content is still exercised.
pub fn arb_safe_string() -> impl Strategy<Value = String> {
    vec(select(SAFE_ALPHABET), 0..=MAX_STRING_LEN)
        .prop_map(|chars| chars.into_iter().collect::<String>().trim().to_owned())
}

/// Recursive strategy producing any FieldType the streaming codec supports as of Step 4.
///
/// Every variant except `Media` — `String`, `Int`, `Float`, `Bool`, `Enum` at leaves;
/// `List<T>`, `Nullable<T>`, `Map<T>`, `Object<fields>` at branches.
///
/// Nullable<Nullable<...>> is explicitly excluded — the JSON-Schema →
/// FieldType converter never produces double-nullable (nullability is
/// propagated via a single Nullable wrapper), and double-nullable
/// creates a wire-format ambiguity where buffered uses the raw-text
/// path while streaming uses the JSON value layer.
pub fn arb_supported_field_type() -> impl Strategy<Value = FieldType> {
    let leaf = prop_oneof![
        Just(FieldType::String),
        Just(FieldType::Int),
        Just(FieldType::Float),
        Just(FieldType::Bool),
        Just(FieldType::Enum(vec!["a".into(), "b".into(), "c".into()])),
    ];
    leaf.prop_recursive(MAX_DEPTH, MAX_TYPE_NODES, EXPECTED_BRANCH_SIZE, |inner| {
        prop_oneof![
            inner.clone().prop_map(|t| FieldType::List(Box::new(t))),
            // Don't nest Nullable inside Nullable.
            inner
                .clone()
                .prop_filter("non-nested-nullable", |t| !matches!(
                    t,
                    FieldType::Nullable(_)
                ))
                .prop_map(|t| FieldType::Nullable(Box::new(t))),
            inner.clone().prop_map(|t| FieldType::Map(Box::new(t))),
            inner.prop_map(|t| FieldType::Object(vec![ObjectField {
                name: "f1".into(),
                description: String::new(),
                field_type: t,
            }])),
        ]
    })
}

/// Given a supported FieldType, generate a structurally-matching
/// FieldValue with total leaf count bounded by `MAX_LEAF_BUDGET`.
/// Top-level entry point; delegates to `arb_value_for_type_with_budget`.
pub fn arb_value_for_type(field_type: FieldType) -> BoxedStrategy<FieldValue> {
    arb_value_for_type_with_budget(field_type, MAX_LEAF_BUDGET)
}

/// Budget-aware value generator. At each `List<T>` level:
///
/// 1. Choose `n ∈ [0, min(budget, MAX_COLLECTION_SIZE)]` elements.
/// 2. Each child claims `max(budget / n, 1)` of the remaining budget.
/// 3. Recurse.
///
/// **Bound proof**: by induction. A `String` is 1 leaf. A `List<T>`
/// with `n` children each producing at most `budget/n` leaves
/// produces at most `n * (budget/n) = budget` leaves. So a value
/// tree rooted at `arb_value_for_type` has at most `MAX_LEAF_BUDGET`
/// leaves. (Strict overshoot is possible only when forced size-1
/// children at budget 0 produce 1-leaf strings — bounded by the
/// total node count of the type tree, itself bounded by
/// `MAX_TYPE_NODES`.)
///
/// **Cost shape**: with `MAX_LEAF_BUDGET=1000` and `MAX_STRING_LEN=100`,
/// worst-case JSON payload is ~100 KB per case. Typical cases are
/// much smaller — proptest's case-size scaling prefers small
/// generations during early cases.
fn arb_value_for_type_with_budget(
    field_type: FieldType,
    budget: usize,
) -> BoxedStrategy<FieldValue> {
    match field_type {
        FieldType::String => arb_safe_string().prop_map(FieldValue::Str).boxed(),
        // Restrict numeric ranges so the serialised form round-trips
        // cleanly through serde_json (no scientific notation surprises
        // for tiny/huge floats; no f64 NaN/Infinity which i64/f64
        // from_str rejects anyway).
        FieldType::Int => any::<i32>()
            .prop_map(|i| FieldValue::Int(i64::from(i)))
            .boxed(),
        FieldType::Float => (-1e9_f64..1e9_f64)
            .prop_filter("finite", |f| f.is_finite())
            .prop_map(FieldValue::Float)
            .boxed(),
        FieldType::Bool => any::<bool>().prop_map(FieldValue::Bool).boxed(),
        FieldType::Enum(variants) => select(variants).prop_map(FieldValue::Str).boxed(),
        FieldType::Nullable(inner) => {
            let inner_strategy = arb_value_for_type_with_budget((*inner).clone(), budget);
            // 50/50 null vs a value of the inner type. The actual
            // ratio doesn't matter for coverage — both branches must
            // round-trip.
            //
            // Filter out ambiguous values that buffered conflates with
            // Null per `format.rs:288-294`: empty string and the
            // literal "null" string are treated as the Null sentinel
            // inside Nullable<String>/<Enum>. Streaming preserves these
            // as Str(""), so a generator that produced them under
            // Nullable would fail parity. Coverage of empty strings is
            // preserved for non-Nullable String/Enum contexts.
            prop_oneof![Just(FieldValue::Null), inner_strategy]
                .prop_filter("non-ambiguous-null", |v| match v {
                    FieldValue::Str(s) => !s.is_empty() && !s.eq_ignore_ascii_case("null"),
                    _ => true,
                })
                .boxed()
        }
        FieldType::Map(value_type) => {
            let cap = budget.min(MAX_COLLECTION_SIZE);
            (0..=cap)
                .prop_flat_map(move |n| {
                    let per_elem_budget = budget.checked_div(n).map_or(0, |b| b.max(1));
                    let value_strategy =
                        arb_value_for_type_with_budget((*value_type).clone(), per_elem_budget);
                    // Use a small fixed key alphabet to keep BTreeMap
                    // collision-free and keep the test surface small.
                    let key_strategy =
                        arb_safe_string().prop_filter("non-empty", |s| !s.is_empty());
                    vec((key_strategy, value_strategy), n..=n).prop_map(|pairs| {
                        let map: BTreeMap<String, FieldValue> = pairs.into_iter().collect();
                        FieldValue::Object(map)
                    })
                })
                .boxed()
        }
        FieldType::Object(fields) => {
            // For each declared field, generate a matching value with a
            // share of the budget; assemble into a BTreeMap. Object
            // values always supply every declared field (the parser's
            // "missing nullable → Null" semantics is unit-tested
            // separately).
            let per_field_budget = budget.checked_div(fields.len()).map_or(0, |b| b.max(1));
            let field_strategies: Vec<BoxedStrategy<(String, FieldValue)>> = fields
                .into_iter()
                .map(|f| {
                    let name = f.name.clone();
                    arb_value_for_type_with_budget(f.field_type, per_field_budget)
                        .prop_map(move |v| (name.clone(), v))
                        .boxed()
                })
                .collect();
            field_strategies
                .prop_map(|pairs| {
                    let map: BTreeMap<String, FieldValue> = pairs.into_iter().collect();
                    FieldValue::Object(map)
                })
                .boxed()
        }
        FieldType::List(inner) => {
            let cap = budget.min(MAX_COLLECTION_SIZE);
            // `(0..=cap)` chooses the element count; `prop_flat_map`
            // then builds a vec strategy of that exact length, each
            // element seeded with its share of the remaining budget.
            (0..=cap)
                .prop_flat_map(move |n| {
                    // `checked_div` collapses both the n==0 and (pathologically)
                    // n>usize::MAX cases into a `None` we then default to 0;
                    // the `.max(1)` clamps the n>0 path so each child still gets
                    // at least one leaf of budget.
                    let per_elem_budget = budget.checked_div(n).map_or(0, |b| b.max(1));
                    let elem_strategy =
                        arb_value_for_type_with_budget((*inner).clone(), per_elem_budget);
                    vec(elem_strategy, n..=n).prop_map(FieldValue::List)
                })
                .boxed()
        }
        FieldType::Media { .. } => {
            unreachable!("arb_value_for_type called with Media (Media is unsupported)")
        }
        // OneOf / AnyOf value generation lands as part of Phase 5
        // streaming work — `arb_supported_field_type` does not yet
        // produce them, so this arm is unreachable in current sweeps.
        // Phase 5 extends the generator and replaces this arm with a
        // real strategy.
        FieldType::OneOf { .. } | FieldType::AnyOf { .. } => {
            unreachable!(
                "arb_value_for_type called with OneOf/AnyOf (Phase 5 extends \
                 arb_supported_field_type with variant generation)"
            )
        }
    }
}

/// Joint strategy: produce a `(FieldType, FieldValue)` pair where
/// the value structurally matches the type. Cuts out the `flat_map`
/// boilerplate at every property-test call site.
pub fn arb_supported_type_and_value() -> impl Strategy<Value = (FieldType, FieldValue)> {
    arb_supported_field_type().prop_flat_map(|ft| {
        let v = arb_value_for_type(ft.clone());
        (Just(ft), v)
    })
}

/// Joint strategy variant for the `JsonArrayParser` round-trip.
///
/// The outer type is guaranteed to be `List<inner>` (wrapping any supported `inner`),
/// so the test can always cast the resulting completion as a `List`.
pub fn arb_supported_list_type_and_value() -> impl Strategy<Value = (FieldType, FieldValue)> {
    arb_supported_field_type()
        .prop_map(|inner| FieldType::List(Box::new(inner)))
        .prop_flat_map(|list_ft| {
            let v = arb_value_for_type(list_ft.clone());
            (Just(list_ft), v)
        })
}

/// Strategy producing FieldType variants the streaming codec does NOT yet support.
///
/// Fuel for the "unsupported-types-degrade-gracefully" property that asserts `StreamError`
/// instead of panic. Shrinks as each step lands; after the full FieldType-coverage work
/// completes, only `Media` remains here.
pub fn arb_unsupported_field_type() -> impl Strategy<Value = FieldType> {
    // After Step 4 only `Media` remains unsupported. Wrap in a `List`
    // so the unsupported variant is reachable via the value-parser
    // dispatch (top-level Media is rejected at create-time per the
    // upcoming validate_configuration hook).
    prop_oneof![Just(FieldType::List(Box::new(FieldType::Media {
        kind: modelplease::MediaKind::Image,
        accepted_sources: enumset::EnumSet::all(),
    }))),]
}

/// Build a single-output `Signature` for a given FieldType. Used by
/// the top-level ChatStreamParser round-trip property.
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "test-support helper; infallible by construction"
)]
pub fn single_output_signature(name: &str, field_type: FieldType) -> Signature {
    Signature::builder("proptest")
        .input(FieldDef::input("q", FieldType::String, "question"))
        .output(FieldDef::output(name, field_type, "output"))
        .build()
        .expect("single-output signature must build")
}

/// Serialize a `(FieldType, FieldValue)` pair into the LM-style wire
/// format the chat adapter produces — marker-delimited single output
/// field with content either passed through (top-level `String`) or
/// JSON-encoded (everything else).
/// Recursive helper that produces just the content body (no marker
/// envelope) for any `(FieldType, FieldValue)` pair. Used by
/// `serialize_completion` and by the Nullable inner-delegation arm.
#[expect(
    clippy::expect_used,
    reason = "test-support helper; FieldValue is Serialize by construction"
)]
fn serialize_completion_body(field_type: &FieldType, value: &FieldValue) -> String {
    match (field_type, value) {
        (FieldType::String | FieldType::Enum(_), FieldValue::Str(s)) => s.clone(),
        (FieldType::Nullable(inner), FieldValue::Null)
            if matches!(**inner, FieldType::String | FieldType::Enum(_)) =>
        {
            "null".to_owned()
        }
        (FieldType::Nullable(inner), v)
            if matches!(**inner, FieldType::String | FieldType::Enum(_)) =>
        {
            serialize_completion_body(inner, v)
        }
        (_, v) => serde_json::to_string(v).expect("FieldValue is Serialize"),
    }
}

#[must_use]
pub fn serialize_completion(
    field_name: &str,
    field_type: &FieldType,
    value: &FieldValue,
) -> String {
    serialize_completion_with_drift(field_name, field_type, value, DriftPerturbation::None)
}

/// A drift perturbation applied to the wire-format content body to exercise the streaming parser's
/// tolerance for LM-output drift.
///
/// Each variant maps to one of the buffered behaviors in
/// `format.rs::prepare_complex_json`: code fence, surrounding prose,
/// single-key wrapper, or a combination.
///
/// Perturbations apply ONLY to container fields (`List`, `Object`,
/// `Map`, `Nullable<container>`). For scalars and raw-text fields the
/// perturbation is silently a no-op so the buffered side doesn't
/// reject content the buffered codec wouldn't accept either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftPerturbation {
    None,
    /// Wrap content in ```json\n...\n```.
    FenceJson,
    /// Wrap content in ```\n...\n``` (bare fence).
    FenceBare,
    /// Prepend prose before the value: "Here is your <field_name>: ...".
    ProsePrefix,
    /// Wrap as `{"<field_name>": <value>}`.
    SingleKeyWrapper,
    /// Fence ```json + single-key wrapper.
    FenceAndWrapper,
}

pub fn arb_drift_perturbation() -> impl Strategy<Value = DriftPerturbation> {
    prop_oneof![
        Just(DriftPerturbation::None),
        Just(DriftPerturbation::FenceJson),
        Just(DriftPerturbation::FenceBare),
        Just(DriftPerturbation::ProsePrefix),
        Just(DriftPerturbation::SingleKeyWrapper),
        Just(DriftPerturbation::FenceAndWrapper),
    ]
}

/// Whether drift perturbation applies to this field type. Mirrors
/// `field_parsers::json_field::drift_applicable` — only container
/// types get drift handling in the streaming parser AND in the
/// buffered parser (`prepare_complex_json` is only called from the
/// List/Object/Map arms).
fn drift_applicable(ft: &FieldType) -> bool {
    match ft {
        FieldType::List(_)
        | FieldType::Object(_)
        | FieldType::Map(_)
        | FieldType::OneOf { .. }
        | FieldType::AnyOf { .. } => true,
        FieldType::Nullable(inner) => drift_applicable(inner),
        _ => false,
    }
}

/// Whether the single-key wrapper perturbation is safe for parity
/// testing on this field type. The streaming parser deliberately
/// skips wrapper unwrap for Object/Map declared types (the leading
/// `{` is the value's own opener — disambiguation is impossible
/// without parsing the full object first, which a streaming parser
/// can't do without buffering everything). The buffered parser DOES
/// unwrap there, which would cause a parity divergence; we skip the
/// perturbation in those cases.
///
/// Same diverge risk applies to tagged OneOf (every arm is an Object,
/// so the leading `{` is the variant value's opener) and any OneOf/
/// AnyOf whose arms include Object/Map shapes. Conservatively skip the
/// wrapper for variant types — they aren't safe even when all arms
/// happen to be non-Object.
fn single_key_wrapper_safe(ft: &FieldType) -> bool {
    // Variants disambiguate by structural validation per arm; the
    // streaming parser can't tell a wrapper from a tagged-arm object
    // opener without buffering everything. They fall under the
    // wildcard `false` arm — kept here as documentation of the
    // reasoning, but folded into the wildcard for clippy
    // (match_same_arms).
    match ft {
        FieldType::List(_) => true,
        FieldType::Nullable(inner) => single_key_wrapper_safe(inner),
        _ => false,
    }
}

#[must_use]
#[expect(
    clippy::expect_used,
    reason = "test-support helper; FieldValue is Serialize by construction"
)]
pub fn serialize_completion_with_drift(
    field_name: &str,
    field_type: &FieldType,
    value: &FieldValue,
    perturb: DriftPerturbation,
) -> String {
    let mut content_body = match (field_type, value) {
        // Top-level String and Enum are both raw passthrough at the
        // top level: StringParser doesn't JSON-decode, and EnumParser
        // consumes a bare variant name (no quotes) per
        // `FieldType::output_format_hint`.
        (FieldType::String | FieldType::Enum(_), FieldValue::Str(s)) => s.clone(),
        // Top-level Nullable<String> / Nullable<Enum>: raw-text path
        // matches buffered's null-sentinel rule. Null serialises as
        // the literal word "null"; non-null delegates to the inner
        // type's top-level form.
        (FieldType::Nullable(inner), FieldValue::Null)
            if matches!(**inner, FieldType::String | FieldType::Enum(_)) =>
        {
            "null".to_owned()
        }
        (FieldType::Nullable(inner), v)
            if matches!(**inner, FieldType::String | FieldType::Enum(_)) =>
        {
            // Reuse the inner type's serialisation path.
            serialize_completion_body(inner, v)
        }
        // Everything else is the JSON-value path (JsonFieldParser).
        (_, v) => serde_json::to_string(v).expect("FieldValue is Serialize"),
    };
    if drift_applicable(field_type) {
        // Guard wrapper-based perturbations against Object/Map types
        // where streaming and buffered diverge. The other perturbations
        // (fence, prose) are safe across all container types.
        let safe_perturb = match perturb {
            DriftPerturbation::SingleKeyWrapper | DriftPerturbation::FenceAndWrapper
                if !single_key_wrapper_safe(field_type) =>
            {
                DriftPerturbation::None
            }
            other => other,
        };
        content_body = apply_drift(&content_body, field_name, safe_perturb);
    }
    format!("[[ ## {field_name} ## ]]\n{content_body}\n[[ ## completed ## ]]")
}

fn apply_drift(content: &str, field_name: &str, perturb: DriftPerturbation) -> String {
    match perturb {
        DriftPerturbation::None => content.to_owned(),
        DriftPerturbation::FenceJson => format!("```json\n{content}\n```"),
        DriftPerturbation::FenceBare => format!("```\n{content}\n```"),
        DriftPerturbation::ProsePrefix => format!("Here is your {field_name}: {content}"),
        DriftPerturbation::SingleKeyWrapper => format!("{{\"{field_name}\": {content}}}"),
        DriftPerturbation::FenceAndWrapper => {
            format!("```json\n{{\"{field_name}\": {content}}}\n```")
        }
    }
}

/// Split `s` into chunks at byte positions derived from `raw_positions`.
///
/// Each raw byte is mapped to a position in `[0, len)` via modulo,
/// then filtered to char boundaries. Sorts + dedupes the result so
/// each chunking deterministically reproduces from the same input.
///
/// Returns at least one chunk. Empty input yields `vec![""]` so
/// callers don't need to special-case empty completions.
///
/// Returns `Vec<&str>` (borrowed) rather than `Vec<String>` so each
/// proptest case avoids one `to_owned()` per chunk. With chunks up to
/// 11 per case × hundreds of cases per test, the saved allocations
/// add up when the proptest suite runs under nextest.
#[must_use]
pub fn chunk_at_positions<'a>(s: &'a str, raw_positions: &[u8]) -> Vec<&'a str> {
    if s.is_empty() {
        return vec![""];
    }
    let len = s.len();
    let mut points: Vec<usize> = raw_positions
        .iter()
        .map(|&p| (p as usize) % len)
        .filter(|&p| p > 0 && s.is_char_boundary(p))
        .collect();
    points.sort_unstable();
    points.dedup();

    let mut chunks: Vec<&str> = Vec::with_capacity(points.len() + 1);
    let mut last = 0;
    for &p in &points {
        chunks.push(&s[last..p]);
        last = p;
    }
    chunks.push(&s[last..]);
    chunks
}

// ============================================================
// Phase 5: OneOf / AnyOf strategies.
//
// Variant types compose recursively with the existing supported types
// but combining them with the existing `arb_supported_field_type` would
// explode the proptest state space. They live in a dedicated strategy
// `arb_variant_type_and_value` driven by its own parity property so the
// existing 10k-case sweep stays tractable and the variant coverage
// remains visible as its own test surface.
//
// The arm sets below are chosen so that every generated value is
// **unambiguous**: it parses successfully for exactly one arm. This
// matches JSON Schema `oneOf` semantics and lets the parity property
// assert byte-for-byte equality between buffered and streaming. AnyOf
// shares the same arm sets — under unambiguous arms its first-match-wins
// rule coincides with OneOf's exactly-one-match rule.
// ============================================================

/// Index identifying one arm shape used by [`arb_variant_type_and_value`].
/// Encoded as a numeric tag rather than a closure so the arm-shape set
/// is enumerable and the value generator can dispatch deterministically.
#[derive(Clone, Copy, Debug)]
enum ArmShape {
    Int,
    Bool,
    StringList,
    IntList,
    SingleFieldObject,
}

impl ArmShape {
    fn field_type(self) -> FieldType {
        match self {
            Self::Int => FieldType::Int,
            Self::Bool => FieldType::Bool,
            Self::StringList => FieldType::List(Box::new(FieldType::String)),
            Self::IntList => FieldType::List(Box::new(FieldType::Int)),
            Self::SingleFieldObject => FieldType::Object(vec![ObjectField {
                name: "x".into(),
                description: String::new(),
                field_type: FieldType::Int,
            }]),
        }
    }

    fn arb_value(self) -> BoxedStrategy<FieldValue> {
        match self {
            Self::Int => any::<i32>()
                .prop_map(|i| FieldValue::Int(i64::from(i)))
                .boxed(),
            Self::Bool => any::<bool>().prop_map(FieldValue::Bool).boxed(),
            Self::StringList => vec(arb_safe_string(), 0..=4)
                .prop_map(|items| {
                    FieldValue::List(items.into_iter().map(FieldValue::Str).collect())
                })
                .boxed(),
            Self::IntList => vec(any::<i32>(), 0..=4)
                .prop_map(|items| {
                    FieldValue::List(
                        items
                            .into_iter()
                            .map(|i| FieldValue::Int(i64::from(i)))
                            .collect(),
                    )
                })
                .boxed(),
            Self::SingleFieldObject => any::<i32>()
                .prop_map(|i| {
                    FieldValue::Object(BTreeMap::from([(
                        "x".into(),
                        FieldValue::Int(i64::from(i)),
                    )]))
                })
                .boxed(),
        }
    }
}

/// Arm-set choices: each set's shapes are mutually-disjoint at the
/// JSON-syntactic level so values are unambiguous. Coupled to the
/// fixed [`ArmShape`] enum rather than randomly composed because the
/// disjointness property is structural — generating arbitrary pairs of
/// arm shapes would routinely produce ambiguity.
fn arm_set_choices() -> Vec<Vec<ArmShape>> {
    vec![
        vec![ArmShape::Int, ArmShape::StringList],
        vec![ArmShape::Int, ArmShape::SingleFieldObject],
        vec![ArmShape::Bool, ArmShape::StringList],
        vec![ArmShape::Bool, ArmShape::SingleFieldObject],
        vec![ArmShape::StringList, ArmShape::SingleFieldObject],
        vec![ArmShape::IntList, ArmShape::SingleFieldObject],
        vec![ArmShape::Int, ArmShape::Bool, ArmShape::StringList],
        vec![ArmShape::Int, ArmShape::Bool, ArmShape::SingleFieldObject],
    ]
}

/// Joint strategy producing `(FieldType, FieldValue)` pairs for OneOf
/// (untagged) and AnyOf where the value matches exactly one arm. Used
/// by the variant parity property.
pub fn arb_variant_type_and_value() -> impl Strategy<Value = (FieldType, FieldValue)> {
    let arm_sets = arm_set_choices();
    let arm_count = arm_sets.len();
    (0..arm_count, any::<bool>(), any::<u32>()).prop_flat_map(
        move |(set_idx, use_anyof, picker)| {
            let shapes = arm_sets[set_idx].clone();
            let chosen_idx = (picker as usize) % shapes.len();
            let chosen_shape = shapes[chosen_idx];
            let arms: Vec<VariantArm> = shapes
                .iter()
                .map(|s| VariantArm {
                    description: format!("{s:?}"),
                    field_type: s.field_type(),
                })
                .collect();
            let field_type = if use_anyof {
                FieldType::AnyOf { arms }
            } else {
                FieldType::OneOf {
                    arms,
                    discriminator: None,
                }
            };
            chosen_shape.arb_value().prop_map(move |inner| {
                (
                    field_type.clone(),
                    FieldValue::Variant {
                        arm_index: chosen_idx,
                        value: Box::new(inner),
                    },
                )
            })
        },
    )
}

/// Joint strategy producing `(FieldType, FieldValue)` pairs for tagged OneOf.
///
/// Each arm is an Object containing a `kind` discriminator (const per arm) plus one payload
/// field. Values disambiguate by the discriminator alone, so the strategy doesn't need
/// disjoint-shape constraints — any Object arms work as long as they all share the `kind`
/// discriminator with distinct const values.
pub fn arb_tagged_oneof_type_and_value() -> impl Strategy<Value = (FieldType, FieldValue)> {
    // Three fixed arms with distinct tag values + simple payload fields.
    let arms_template: Vec<(&'static str, &'static str, FieldType)> = vec![
        ("kind_a", "alpha", FieldType::Int),
        ("kind_b", "beta", FieldType::String),
        ("kind_c", "gamma", FieldType::List(Box::new(FieldType::Int))),
    ];
    let arms_field_types: Vec<FieldType> = arms_template
        .iter()
        .map(|(tag, _, payload_ty)| {
            FieldType::Object(vec![
                ObjectField {
                    name: "kind".into(),
                    description: String::new(),
                    field_type: FieldType::Enum(vec![(*tag).to_string()]),
                },
                ObjectField {
                    name: "payload".into(),
                    description: String::new(),
                    field_type: payload_ty.clone(),
                },
            ])
        })
        .collect();
    let arms: Vec<VariantArm> = arms_template
        .iter()
        .zip(arms_field_types.iter())
        .map(|((_, desc, _), ft)| VariantArm {
            description: (*desc).to_string(),
            field_type: ft.clone(),
        })
        .collect();
    let tags: Vec<String> = arms_template
        .iter()
        .map(|(tag, _, _)| (*tag).to_string())
        .collect();
    let field_type = FieldType::OneOf {
        arms,
        discriminator: Some(OneOfDiscriminator {
            property: "kind".into(),
            tags: tags.clone(),
        }),
    };

    (0_usize..arms_template.len(), any::<u32>()).prop_flat_map(move |(idx, seed)| {
        let tag = tags[idx].clone();
        let payload_strategy: BoxedStrategy<FieldValue> = match arms_template[idx].2 {
            FieldType::Int => any::<i32>()
                .prop_map(|i| FieldValue::Int(i64::from(i)))
                .boxed(),
            FieldType::String => arb_safe_string().prop_map(FieldValue::Str).boxed(),
            FieldType::List(_) => vec(any::<i32>(), 0..=3)
                .prop_map(|items| {
                    FieldValue::List(
                        items
                            .into_iter()
                            .map(|i| FieldValue::Int(i64::from(i)))
                            .collect(),
                    )
                })
                .boxed(),
            _ => unreachable!("template payload types are fixed"),
        };
        let _ = seed;
        let ft = field_type.clone();
        payload_strategy.prop_map(move |payload| {
            let inner = FieldValue::Object(BTreeMap::from([
                ("kind".into(), FieldValue::Str(tag.clone())),
                ("payload".into(), payload),
            ]));
            (
                ft.clone(),
                FieldValue::Variant {
                    arm_index: idx,
                    value: Box::new(inner),
                },
            )
        })
    })
}

/// Strategy producing a chunking specification — a vector of raw byte positions to feed into
/// [`chunk_at_positions`].
///
/// Bounded at 10 splits per case to keep individual cases tractable while still exercising
/// mid-token boundaries.
pub fn arb_chunking_positions() -> impl Strategy<Value = Vec<u8>> {
    vec(any::<u8>(), 0..=10)
}
