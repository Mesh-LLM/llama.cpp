# `llama-layer-split`

`llama-layer-split` builds runnable layer-range GGUF shards.

## Goal

This tool takes a full GGUF and emits one or more node-specific GGUFs that keep
only a contiguous range of repeating transformer blocks, while duplicating the
shared tensors each shard still needs to run.

It is the artifact path for budget-real layer sharding such as:

- `4 x 8 GiB` nodes
- one contiguous block range per node
- same owners used for both prefill and decode

## What It Does

For each output shard, it keeps:

- shared input/global tensors
- only the requested `blk.<i>.` layer tensors
- output head tensors only on the final shard by default
- shard metadata describing the owned layer range

The result is a normal GGUF with explicit shard metadata that the runtime can
load directly.

## Supported Modes

- single range via `--layer-start` and `--layer-end`
- evenly partitioned multi-shard output via `--stages N --output-prefix`
- optional `--include-output` override for non-final shards
- optional `--smoke-load` validation after write
- `--dry-run` planning mode

## Why It Exists

The staged runtime work proved the execution topology for:

- `4-owner prefill -> 4-owner decode`

but those proofs still used full-model contexts with restricted compute ranges.
That proves correctness, not memory budget.

`llama-layer-split` is the artifact step that makes the budget real.

## Design Position

`llama-layer-split` should stay focused on one job:

"Given a model and a layer assignment, write runnable layer shards."

It is intentionally separate from the runtime orchestration tools so the shard
writer stays simple, inspectable, and reusable.

## Examples

Single shard:

```bash
llama-layer-split \
  --model /path/to/model.gguf \
  --layer-start 0 \
  --layer-end 10 \
  --output /tmp/node-0.gguf
```

Four contiguous shards:

```bash
llama-layer-split \
  --model /path/to/model.gguf \
  --stages 4 \
  --output-prefix /tmp/quad-small
```

Dry run:

```bash
llama-layer-split \
  --model /path/to/model.gguf \
  --stages 4 \
  --output-prefix /tmp/quad-small \
  --dry-run
```

Write and validate a shard loads into a real context:

```bash
llama-layer-split \
  --model /path/to/model.gguf \
  --layer-start 30 \
  --layer-end 40 \
  --output /tmp/node-3.gguf \
  --smoke-load
```

## Notes

- Layer ranges are half-open: `[layer_start, layer_end)`.
- Output tensors are written only to the final shard unless `--include-output`
  is specified.
- The writer supports input GGUFs that are already split across multiple files.
