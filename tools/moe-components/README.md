# `llama-moe-components`

`llama-moe-components` extracts and reassembles topology-independent MoE model
components.

## Goal

The purpose of this tool is to avoid publishing a separate full shard set for
every possible node count.

Instead of storing:

- 2-node shards
- 3-node shards
- 4-node shards
- and so on

we store one canonical decomposition:

- one `trunk`
- one file per expert

Then a node downloads only the experts it needs and materializes a local shard
on demand.

## What It Does

This tool currently supports three operations:

- `extract-trunk`
- `extract-expert`
- `assemble`

That gives us a round-trip:

1. Start from a full GGUF.
2. Extract reusable MoE components.
3. Reassemble `trunk + selected experts` into a runnable local shard.

## Why It Exists

Publishing topology-specific split outputs does not scale well. Every supported
node count creates another family of large artifacts.

`llama-moe-components` is the storage-efficient alternative:

- publish once per model revision
- reuse across many mesh topologies
- download only the experts a node actually needs

## Design Position

This tool is intentionally separate from `llama-moe-split`.

- `llama-moe-split` is the stable direct-to-runnable-shard tool.
- `llama-moe-components` is the topology-independent component workflow.

They share internal ideas, but they solve different product problems.

Keeping them separate helps us:

- preserve the existing fallback path
- experiment on the new workflow safely
- avoid turning one binary into an overloaded kitchen-sink interface

## Mesh-LLM Workflow

In Mesh-LLM this tool backs the published expert-components flow:

1. `mesh-llm moe share MODEL --with-experts`
   generates `manifest.json`, `trunk.gguf`, and `expert-XYZ.gguf`.
2. The files are uploaded to the MoE dataset on Hugging Face.
3. `mesh-llm serve` resolves the node's expert assignment.
4. Runtime downloads `trunk + needed experts`.
5. Runtime assembles a local runnable shard.
6. If anything is missing or invalid, Mesh-LLM falls back to
   `llama-moe-split`.

## Relationship To The Other MoE Tools

- `llama-moe-analyze` tells us which experts matter most.
- `llama-moe-split` writes topology-specific runnable shards directly.
- `llama-moe-components` publishes reusable building blocks and reassembles a
  shard later.

## Notes

This tool is part of the Mesh-LLM llama.cpp fork. It was added specifically to
support topology-independent expert sharing and on-demand shard materialization.
