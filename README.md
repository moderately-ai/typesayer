# typesayer

Typed structured prediction for Rust language-model applications.

Typesayer turns a declared input/output signature into provider-neutral messages, parses the
response into typed values, supports incremental streaming, evaluates programs, persists state,
and optimizes instructions and demonstrations.

```bash
cargo add typesayer typesayer-types modelplease
```

## Crates

| Crate | Purpose |
|---|---|
| `typesayer` | Prediction, adapters, evaluation, modules, state, and optimizers |
| `typesayer-types` | Fields, values, signatures, media values, and errors |
| `typesayer-parser` | Push-driven streaming structured-output parser |

Bayesian optimization is supplied by the independent
[`parzen`](https://github.com/moderately-ai/parzen) crate. Provider clients and shared model types
come from [`modelplease`](https://github.com/moderately-ai/modelplease).

```text
modelplease ──→ typesayer-types ──→ typesayer-parser
     │                 │
     └─────────────────┴─────────→ typesayer ←── parzen
```

## Structured prediction

```rust
use std::{collections::BTreeMap, sync::Arc};

use modelplease::{DummyLM, ModelId};
use typesayer::{ChatAdapter, Context, Predict};
use typesayer_types::{FieldDef, FieldType, FieldValue, Signature};

# async fn run() -> typesayer_types::Result<()> {
let signature = Signature::builder("Answer the question.")
    .input(FieldDef::input("question", FieldType::String, "Question"))
    .output(FieldDef::output("answer", FieldType::String, "Answer"))
    .build()?;
let context = Context {
    provider: Arc::new(DummyLM::sequential(vec![
        "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]".into(),
    ])),
    model: ModelId::new("test"),
    adapter: Arc::new(ChatAdapter::default()),
};
let prediction = Predict::new(signature)
    .call(
        &BTreeMap::from([("question".into(), FieldValue::Str("Capital of France?".into()))]),
        &context,
    )
    .await?;
assert_eq!(prediction.get::<String>("answer")?, "Paris");
# Ok(())
# }
```

## OpenAI-backed applications

Enable `typesayer/openai` and construct an `OpenAiLanguageModel` from `modelplease`. Typesayer
accepts any `Arc<dyn LanguageModelProvider>`, so Anthropic, Ollama, Bedrock, custom providers, and
test doubles use the same prediction API.

```toml
[dependencies]
typesayer = { version = "0.1.1", features = ["openai"] }
modelplease = { version = "0.1.1", features = ["openai"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls"] }
```

```rust,no_run
use std::sync::Arc;

use modelplease::{ApiKey, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};

# fn provider() -> Result<OpenAiLanguageModel, Box<dyn std::error::Error>> {
let provider = OpenAiLanguageModel::new(
    OpenAiDeps { client: Arc::new(reqwest::Client::new()) },
    OpenAiConfig {
        api_key: ApiKey::parse(std::env::var("OPENAI_API_KEY")?)?,
        base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
        retry_config: RetryConfig::default(),
    },
);
# Ok(provider)
# }
```

## Streaming parsing

```rust
use typesayer_parser::ChatStreamParser;
use typesayer_types::{FieldDef, FieldType, Signature};

let signature = Signature::builder("Answer")
    .input(FieldDef::input("question", FieldType::String, "Question"))
    .output(FieldDef::output("answer", FieldType::String, "Answer"))
    .build()
    .unwrap();
let mut parser = ChatStreamParser::new(&signature);
let mut events = parser.push("[[ ## answer ## ]]\nPar");
events.extend(parser.push("is\n[[ ## completed ## ]]"));
events.extend(parser.finish());
assert!(!events.is_empty());
```

## Optimization and state

`BootstrapFewShot`, `LabeledFewShot`, and `MIPROv2` optimize demonstrations and instructions.
The complete deterministic MIPRO example is runnable with:

```bash
cargo run -p typesayer --example mipro_optimization --features openai
```

Modules save DSPy-compatible JSON through `Module::save` and restore it through `Module::load`.
New files record the `typesayer` crate version. Typesayer 0.1 also reads state emitted by the
original prediction crate, ignores unknown metadata keys, and preserves the existing state wire
format.

## Compatibility and license

The synchronized crate family requires Rust 1.88. Licensed under either MIT or Apache-2.0 at your
option.
