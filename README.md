# DSPy Custom BaseLM Reference

A lightweight reference for building typed custom language models on DSPy’s `LMRequest -> LMResponse` `BaseLM` contract. `CustomLM` reuses native DSPy behavior and adds only the custom inference boundary needed for future SDK, API, subprocess, or local-runtime integrations, without LiteLLM-specific compatibility or parallel framework machinery.
