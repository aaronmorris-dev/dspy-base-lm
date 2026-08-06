# DSPy 3.3 feature matrix

This matrix describes the reference package, not every possible backend. Provider-dependent
rows become supported only when that provider has a native probe and a DSPy scenario.

| Feature | Classification | Owner and evidence |
| --- | --- | --- |
| Unconfigured construction | Supported | Bare `CustomLM` raises `LMNotConfiguredError`. |
| Runtime provider injection | Supported | Application injects an `LMProvider`; runtime is absent from saved state. |
| Reconstructable provider inference | Supported | `EchoLM.infer_provider()` and trusted `BaseLM.load_state`. |
| Typed sync and async calls | Supported | `LMRequest -> LMResponse` through `forward` and `aforward`. |
| Normalized public calls | Supported | Inherited current `BaseLM` normalization. |
| Predict | Supported | Deterministic provider integration scenario. |
| ChainOfThought | Supported | Deterministic reasoning-formatted scenario. |
| ChatAdapter | Supported | Deterministic adapter scenario. |
| JSONAdapter | Supported | Provider declares `response_format`; deterministic JSON scenario. |
| Callbacks | Supported | Inherited DSPy callback wrapper, exactly one start/end. |
| Typed history | Supported | Inherited `BaseLM` finalization, disabled by DSPy context when requested. |
| Usage tracking | Supported | `LMUsage` flows to `dspy.track_usage()`. |
| DSPy request cache | Supported | Final safe responses only; policy removed from identity; rollout retained; provider instances partition entries. |
| Configuration safety | Supported | Runtime objects, cycles, and credential-shaped keys are rejected before dispatch, cache, history, or saved state. |
| Response cache safety | Supported | Unsafe native/runtime response data is returned live and bypasses cache storage. |
| Copy | Supported | DSPy-owned mutable state is isolated; the shared provider owns concurrency safety, and replacements receive their provider's cache partition. |
| Save/load | Provider-dependent | Reconstructable subclasses supported; injected runtime must be reattached. |
| Function calling | Provider-dependent | Provider capability plus typed tool request/response translation required. |
| Response schema | Provider-dependent | Provider capability and `response_format` support required. |
| Native reasoning | Provider-dependent | Provider capability and typed reasoning translation required. |
| Multimodal input/output | Provider-dependent | Provider must map applicable DSPy parts without flattening. |
| Multiple candidates and rich output | Supported | Rich-response fidelity scenario preserves candidates and typed parts. |
| Known error mapping | Provider-dependent | Provider maps backend failures to specific DSPy `LMError` subclasses. |
| Unknown error containment | Supported | `CustomLM` chains unknown boundary failures as `LMUnexpectedError`. |
| Retries | Provider-dependent | `num_retries` is passed once; provider owns the policy. |
| Fine-tuning | Provider-dependent | DSPy `TrainingJob` is used when `provider.finetunable` is true. |
| BootstrapFinetune per-LM mapping | Blocked by DSPy 3.3 | Upstream mapping recognizes concrete `dspy.LM` keys. |
| Typed custom-LM streaming | Blocked by DSPy 3.3 | No stable public typed custom-`BaseLM` stream integration hook. |
| Provider launch/kill on CustomLM | Blocked by DSPy 3.3 | Upstream provider signatures name concrete `dspy.LM`. |
| Reinforcement on CustomLM | Blocked by DSPy 3.3 | Upstream `ReinforceJob` owns a concrete `dspy.LM`. |
| LiteLLM legacy calls | Not applicable | Intentionally outside the package contract. |

“Blocked by DSPy” means the package does not monkey-patch, subclass `dspy.LM`, mirror
private helpers, or add a compatibility layer to make the feature appear supported.
