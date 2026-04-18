# `llama-moe-analyze`

`llama-moe-analyze` profiles how a MoE model routes tokens to experts.

## Goal

This tool exists to answer one practical question for distributed MoE serving:

"If we restrict a node to a subset of experts, how much routing mass do we
lose?"

That is the key input for deciding whether an expert split is viable and which
experts should stay together.

## What It Does

- Loads a MoE GGUF model.
- Runs prompts through the model.
- Captures router probability tensors during eval.
- Aggregates per-expert routing mass and selection counts.
- Exports a ranking CSV that downstream tooling can reuse.

In the Mesh-LLM workflow, this ranking is the canonical artifact that drives:

- MoE planning
- expert placement
- `llama-moe-split`
- published expert components

## Why It Exists

Heuristic splits are not enough for real MoE placement. We want splits based on
observed routing behavior, not just contiguous expert IDs.

`llama-moe-analyze` gives us a reusable ranking of "hot" and "cold" experts so
the rest of the pipeline can make better decisions.

## Typical Output

The main output is a CSV ranking file, which Mesh-LLM publishes as:

- `ranking.csv`

This file is then consumed by:

- `mesh-llm moe plan`
- `mesh-llm moe share`
- `llama-moe-split --ranking-file`

## Relationship To The Other MoE Tools

- `llama-moe-analyze` measures routing behavior.
- `llama-moe-split` builds runnable topology-specific shards.
- `llama-moe-components` extracts topology-independent components and
  reassembles local shards later.

So the flow is:

1. Analyze routing.
2. Decide expert assignment.
3. Either split directly for a topology or publish reusable components.

## Notes

This is a Mesh-LLM fork tool, not a general upstream llama.cpp feature yet.
The output format is intended to support Mesh-LLM's MoE planning and sharing
workflow.
