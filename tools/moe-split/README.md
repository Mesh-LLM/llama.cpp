# `llama-moe-split`

`llama-moe-split` builds runnable MoE shard GGUFs.

## Goal

This tool takes a full MoE GGUF and emits one or more node-specific GGUFs that
can be launched directly by `llama-server`.

It is the stable "make me a runnable shard now" path.

## What It Does

For each output shard, it keeps:

- the shared trunk tensors
- router tensors
- only the selected expert tensors for that shard
- updated metadata describing the reduced expert set

The result is a normal GGUF that downstream runtime can launch directly.

## Supported Split Modes

- Contiguous groups
- Balanced groups
- Ranking-driven groups via `--ranking-file`
- Explicit custom expert lists
- Group maps loaded from a file

This makes it useful both for simple experiments and for Mesh-LLM's planner.

## Why It Exists

Mesh-LLM needs a reliable fallback path that does not depend on published
artifacts. Even if no expert components exist remotely, a node should still be
able to split locally and start serving.

That fallback is `llama-moe-split`.

## Design Position

`llama-moe-split` should stay focused on one job:

"Given a model and an expert assignment, write a runnable shard GGUF."

We intentionally keep that responsibility separate from the new component
workflow so this tool remains stable and easy to reason about.

## Relationship To The Other MoE Tools

- `llama-moe-analyze` produces the expert ranking data.
- `llama-moe-split` consumes an assignment and writes runnable shards.
- `llama-moe-components` handles the topology-independent
  `trunk + experts -> local shard` workflow.

If the component workflow is unavailable or invalid at runtime, Mesh-LLM falls
back to `llama-moe-split`.

## Notes

This is a Mesh-LLM fork tool. It exists to support distributed MoE serving and
published-ranking-driven expert placement.
