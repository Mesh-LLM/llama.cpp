#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def import_gguf(repo: Path):
    sys.path.insert(0, str(repo / "gguf-py"))
    import gguf  # type: ignore

    return gguf


def map_name(tensor_map, name: str) -> str | None:
    return tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GLM-5.1 native MTP metadata and GGUF tensor mapping.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="llama.cpp checkout")
    parser.add_argument("--meta-dir", type=Path, default=Path("/tmp/glm51-meta"), help="directory with config and safetensor index")
    args = parser.parse_args()

    config = load_json(args.meta_dir / "config.json")
    index = load_json(args.meta_dir / "model.safetensors.index.json")
    weight_names = set(index["weight_map"].keys())

    gguf = import_gguf(args.repo)
    block_count = int(config["num_hidden_layers"]) + int(config.get("num_nextn_predict_layers", 0))
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GLM_DSA, block_count)

    failures: list[str] = []

    require(config.get("architectures") == ["GlmMoeDsaForCausalLM"], "unexpected architectures", failures)
    require(config.get("model_type") == "glm_moe_dsa", "unexpected model_type", failures)
    require(config.get("num_hidden_layers") == 78, "unexpected num_hidden_layers", failures)
    require(config.get("num_nextn_predict_layers") == 1, "unexpected num_nextn_predict_layers", failures)

    mtp_layer = int(config["num_hidden_layers"])
    require(f"model.layers.{mtp_layer}.eh_proj.weight" in weight_names, "missing MTP eh_proj", failures)
    require(f"model.layers.{mtp_layer}.enorm.weight" in weight_names, "missing MTP enorm", failures)
    require(f"model.layers.{mtp_layer}.hnorm.weight" in weight_names, "missing MTP hnorm", failures)
    require(f"model.layers.{mtp_layer}.shared_head.norm.weight" in weight_names, "missing MTP shared head norm", failures)
    require(f"model.layers.{mtp_layer}.embed_tokens.weight" not in weight_names, "unexpected MTP-specific embed_tokens", failures)
    require(f"model.layers.{mtp_layer}.shared_head.head.weight" not in weight_names, "unexpected MTP-specific shared head", failures)

    expected_mappings = {
        f"model.layers.{mtp_layer}.eh_proj.weight": f"blk.{mtp_layer}.nextn.eh_proj.weight",
        f"model.layers.{mtp_layer}.enorm.weight": f"blk.{mtp_layer}.nextn.enorm.weight",
        f"model.layers.{mtp_layer}.hnorm.weight": f"blk.{mtp_layer}.nextn.hnorm.weight",
        f"model.layers.{mtp_layer}.shared_head.norm.weight": f"blk.{mtp_layer}.nextn.shared_head_norm.weight",
        f"model.layers.{mtp_layer}.input_layernorm.weight": f"blk.{mtp_layer}.attn_norm.weight",
        f"model.layers.{mtp_layer}.self_attn.indexer.k_norm.weight": f"blk.{mtp_layer}.indexer.k_norm.weight",
        f"model.layers.{mtp_layer}.self_attn.indexer.k_norm.bias": f"blk.{mtp_layer}.indexer.k_norm.bias",
        f"model.layers.{mtp_layer}.self_attn.indexer.weights_proj.weight": f"blk.{mtp_layer}.indexer.proj.weight",
        f"model.layers.{mtp_layer}.self_attn.indexer.wk.weight": f"blk.{mtp_layer}.indexer.attn_k.weight",
        f"model.layers.{mtp_layer}.self_attn.indexer.wq_b.weight": f"blk.{mtp_layer}.indexer.attn_q_b.weight",
        f"model.layers.{mtp_layer}.mlp.gate.weight": f"blk.{mtp_layer}.ffn_gate_inp.weight",
        f"model.layers.{mtp_layer}.mlp.gate.e_score_correction_bias": f"blk.{mtp_layer}.exp_probs_b",
        f"model.layers.{mtp_layer}.mlp.shared_experts.down_proj.weight": f"blk.{mtp_layer}.ffn_down_shexp.weight",
        f"model.layers.{mtp_layer}.mlp.shared_experts.gate_proj.weight": f"blk.{mtp_layer}.ffn_gate_shexp.weight",
        f"model.layers.{mtp_layer}.mlp.shared_experts.up_proj.weight": f"blk.{mtp_layer}.ffn_up_shexp.weight",
    }

    for src, expected in expected_mappings.items():
        require(src in weight_names, f"missing source tensor {src}", failures)
        actual = map_name(tensor_map, src)
        require(actual == expected, f"mapping mismatch {src}: expected {expected}, got {actual}", failures)

    expert_src = f"model.layers.{mtp_layer}.mlp.experts.down_proj.weight"
    expert_expected = f"blk.{mtp_layer}.ffn_down_exps.weight"
    expert_actual = map_name(tensor_map, expert_src)
    require(expert_actual == expert_expected, f"merged expert mapping mismatch: expected {expert_expected}, got {expert_actual}", failures)

    print(f"repo={args.repo}")
    print(f"meta_dir={args.meta_dir}")
    print(f"architecture={config['architectures'][0]}")
    print(f"model_type={config['model_type']}")
    print(f"trunk_layers={config['num_hidden_layers']}")
    print(f"nextn_layers={config.get('num_nextn_predict_layers', 0)}")
    print(f"total_blocks={block_count}")
    print(f"mtp_layer={mtp_layer}")
    print(f"tensor_count={len(weight_names)}")
    print(f"source_total_size={index.get('metadata', {}).get('total_size')}")

    if failures:
        print("status=FAIL")
        for failure in failures:
            print(f"failure={failure}")
        return 1

    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
