#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf


GLM_PREFIX = "glm-dsa."


def field_value(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"missing required GGUF field: {key}")
    return field.contents()


def add_metadata(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    layer_count: int,
    top_k: int,
    context_length: int,
    vocab_size: int,
) -> None:
    indexer_types = field_value(reader, f"{GLM_PREFIX}attention.indexer.types")
    if len(indexer_types) < layer_count:
        raise ValueError("fixture requests more layers than the GLM indexer schedule contains")

    dense_layer_count = int(field_value(reader, f"{GLM_PREFIX}leading_dense_block_count"))
    overrides = {
        f"{GLM_PREFIX}vocab_size": (gguf.GGUFValueType.UINT32, vocab_size, None),
        f"{GLM_PREFIX}block_count": (gguf.GGUFValueType.UINT32, layer_count, None),
        f"{GLM_PREFIX}context_length": (gguf.GGUFValueType.UINT32, context_length, None),
        f"{GLM_PREFIX}leading_dense_block_count": (
            gguf.GGUFValueType.UINT32,
            min(dense_layer_count, layer_count),
            None,
        ),
        f"{GLM_PREFIX}nextn_predict_layers": (gguf.GGUFValueType.UINT32, 0, None),
        f"{GLM_PREFIX}attention.indexer.top_k": (gguf.GGUFValueType.UINT32, top_k, None),
        f"{GLM_PREFIX}attention.indexer.types": (
            gguf.GGUFValueType.ARRAY,
            indexer_types[:layer_count],
            gguf.GGUFValueType.STRING,
        ),
    }

    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        if not field.name.startswith(("general.", "tokenizer.", GLM_PREFIX)):
            continue
        if field.name in overrides:
            continue

        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    for key, (value_type, value, sub_type) in overrides.items():
        writer.add_key_value(key, value, value_type, sub_type=sub_type)


def fixture_tensor_data(tensor: gguf.ReaderTensor, vocab_size: int):
    if tensor.name in {"token_embd.weight", "output.weight"}:
        if tensor.data.ndim != 2 or tensor.data.shape[0] < vocab_size:
            raise ValueError(f"cannot trim padded vocabulary rows from {tensor.name}")
        return tensor.data[:vocab_size]
    return tensor.data


def add_tensor_info(writer: gguf.GGUFWriter, readers: list[gguf.GGUFReader], vocab_size: int) -> None:
    for reader in readers:
        for tensor in reader.tensors:
            data = fixture_tensor_data(tensor, vocab_size)
            writer.add_tensor_info(
                tensor.name,
                data.shape,
                data.dtype,
                data.nbytes,
                tensor.tensor_type,
            )


def write_tensor_data(writer: gguf.GGUFWriter, readers: list[gguf.GGUFReader], vocab_size: int) -> None:
    for reader in readers:
        for tensor in reader.tensors:
            data = fixture_tensor_data(tensor, vocab_size)
            print(f"writing {tensor.name}: {data.nbytes / (1024 * 1024):.2f} MiB", flush=True)
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)


def package_component_paths(package: Path, layer_count: int) -> tuple[Path, list[Path]]:
    manifest_path = package / "model-package.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing model package manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "layer-package":
        raise ValueError(f"unsupported model package format: {manifest.get('format')}")

    layers = {entry["layer_index"]: package / entry["path"] for entry in manifest["layers"]}
    missing_layers = [layer for layer in range(layer_count) if layer not in layers]
    if missing_layers:
        raise ValueError(f"model package is missing requested layers: {missing_layers}")

    metadata = package / "shared" / "metadata.gguf"
    paths = [package / "shared" / "embeddings.gguf"]
    paths.extend(layers[layer] for layer in range(layer_count))
    paths.append(package / "shared" / "output.gguf")
    return metadata, paths


def make_fixture(
    package: Path,
    output: Path,
    layer_count: int,
    top_k: int,
    context_length: int,
) -> None:
    metadata_path, paths = package_component_paths(package, layer_count)
    missing = [path for path in paths if not path.is_file()]
    if not metadata_path.is_file():
        missing.append(metadata_path)
    if missing:
        raise FileNotFoundError(f"missing GGUF component files: {missing}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    metadata = gguf.GGUFReader(metadata_path, "r")
    readers = [gguf.GGUFReader(path, "r") for path in paths]
    architecture = field_value(metadata, gguf.Keys.General.ARCHITECTURE)
    vocab_size = len(field_value(metadata, gguf.Keys.Tokenizer.LIST))
    writer = gguf.GGUFWriter(output, arch=architecture, endianess=metadata.endianess)

    alignment = metadata.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    add_metadata(metadata, writer, layer_count, top_k, context_length, vocab_size)
    add_tensor_info(writer, readers, vocab_size)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    write_tensor_data(writer, readers, vocab_size)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a real-weight GLM-DSA parity fixture from a layer package."
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=131072)
    args = parser.parse_args()

    if args.layers < 1:
        parser.error("--layers must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.context_length < 1:
        parser.error("--context-length must be positive")
    make_fixture(args.package, args.output, args.layers, args.top_k, args.context_length)


if __name__ == "__main__":
    main()
