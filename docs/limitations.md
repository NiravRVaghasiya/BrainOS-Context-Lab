# Limitations and non-goals

This project is an experimental evaluation platform, not a claim that BrainOS provides human-like cognition or eliminates context rot.

## Initial limitations

- The first release supports OpenAI and generic OpenAI-compatible endpoints only.
- BrainOS remains an upstream dependency whose exact revision and runtime API must be validated before integration claims are made.
- Initial memory extraction is conservative and may miss useful information.
- Token estimates may begin as approximate counts until provider/model-specific tokenizers are integrated.
- Synthetic context-rot tasks may not represent real user conversations.
- Provider behavior, model changes, and stochastic generation can affect results.
- Session-local persistence is not a multi-user account system.
- Evaluation results are only meaningful when baselines use the same models, prompts, and scoring rules.

## Explicitly out of scope for v1

- user accounts and billing,
- fine-tuning in the Space,
- multi-agent systems,
- complex vector databases,
- custom model hosting,
- mobile applications and browser extensions,
- permanent storage of user API keys, and
- broad claims about solving context rot.
