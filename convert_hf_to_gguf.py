#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
    print_registered_models,
    _mistral_common_installed,
    _mistral_import_error_msg,
)

SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"


def _enable_traceback_watchdog() -> None:
    raw = os.environ.get("CONVERT_TRACEBACK_SECONDS", "0")
    try:
        seconds = int(raw)
    except ValueError:
        seconds = 0
    if seconds <= 0:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(seconds, repeat=True)
    logger.info("convert_phase=traceback_watchdog_enabled seconds=%d", seconds)


def _log_phase(phase: str, **fields: object) -> float:
    if fields:
        suffix = " " + " ".join(f"{key}={value}" for key, value in fields.items())
    else:
        suffix = ""
    logger.info("convert_phase=%s%s", phase, suffix)
    return time.time()


def _log_phase_done(phase: str, started_at: float, **fields: object) -> None:
    fields = {"elapsed_seconds": f"{time.time() - started_at:.3f}", **fields}
    _log_phase(f"{phase}_done", **fields)


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def _hub_prefix(prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/" if prefix else ""


def _split_shard_paths(fname_out: Path, total_shards: int) -> list[Path]:
    if total_shards <= 1:
        return [fname_out]
    return [
        fname_out.with_name(SHARD_NAME_FORMAT.format(fname_out.stem, i + 1, total_shards))
        for i in range(total_shards)
    ]


def _materialized_shard_paths(fname_out: Path, start_shard: int, stop_shard: int, total_shards: int) -> list[Path]:
    paths = _split_shard_paths(fname_out, total_shards)
    selected: list[Path] = []
    for shard_no, path in enumerate(paths, start=1):
        if shard_no < start_shard:
            continue
        if stop_shard > 0 and shard_no > stop_shard:
            continue
        if path.is_file():
            selected.append(path)
    return selected


def _uploaded_split_shards(repo_id: str, prefix: str, fname_out: Path) -> tuple[set[int], int]:
    from huggingface_hub import HfApi

    repo_prefix = _hub_prefix(prefix)
    escaped_stem = re.escape(fname_out.stem)
    pattern = re.compile(rf"^{re.escape(repo_prefix)}{escaped_stem}-(\d{{5}})-of-(\d{{5}})\.gguf$")
    uploaded: set[int] = set()
    expected_total = 0
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        logger.warning("hub_resume_list_failed repo=%s reason=%s", repo_id, exc)
        return uploaded, expected_total

    for path in files:
        match = pattern.match(path)
        if match is None:
            continue
        uploaded.add(int(match.group(1)))
        expected_total = max(expected_total, int(match.group(2)))
    return uploaded, expected_total


def _first_missing_shard(uploaded: set[int], expected_total: int) -> int:
    if expected_total <= 0:
        return 1
    for shard_no in range(1, expected_total + 1):
        if shard_no not in uploaded:
            return shard_no
    return expected_total + 1


def _upload_materialized_shards(
    *,
    repo_id: str,
    prefix: str,
    paths: list[Path],
    private: bool,
    delete_uploaded: bool,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    repo_prefix = _hub_prefix(prefix)
    for path in paths:
        rel_path = f"{repo_prefix}{path.name}"
        size = path.stat().st_size
        logger.info("hub_shard_upload_start path=%s size_bytes=%d repo=%s", rel_path, size, repo_id)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("hub_shard_upload_done path=%s size_bytes=%d repo=%s", rel_path, size, repo_id)
        if delete_uploaded:
            path.unlink()
            logger.info("hub_shard_delete_done path=%s size_bytes=%d", path, size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="auto",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=str,
        help="directory containing model file or huggingface repository ID (if --remote)",
        nargs="?",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--skip-output-shards-before", type=int, default=1,
        help="metadata-plan but do not materialize tensors assigned to output shards before this 1-based shard number",
    )
    parser.add_argument(
        "--stop-output-shards-after", type=int, default=0,
        help="metadata-plan but do not materialize tensors assigned to output shards after this 1-based shard number",
    )
    parser.add_argument(
        "--materialize-output-shard-window-size", type=int, default=0,
        help="materialize split output in repeated windows of this many shards; intended for upload/delete workflows that must bound local disk",
    )
    parser.add_argument(
        "--upload-finalized-shards-to-repo", type=str, default="",
        help="upload each materialized split GGUF shard to this Hugging Face model repo after its window completes",
    )
    parser.add_argument(
        "--upload-finalized-shards-prefix", type=str, default="",
        help="optional path prefix in the target Hugging Face repo for uploaded split GGUF shards",
    )
    parser.add_argument(
        "--upload-finalized-shards-private", action="store_true",
        help="create the target Hugging Face model repo as private if it does not already exist",
    )
    parser.add_argument(
        "--resume-uploaded-shards", action="store_true",
        help="when uploading split shards, inspect the target repo and resume from the first missing split shard",
    )
    parser.add_argument(
        "--delete-uploaded-shards", action="store_true",
        help="delete each local split GGUF shard after it uploads successfully",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="do not add tensors to the first split (disabled by default)"
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Specify the path for an authorship metadata override file"
    )
    parser.add_argument(
        "--print-supported-models", action="store_true",
        help="Print the supported models"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj", action="store_true",
        help="Export multimodal projector (mmproj) for vision models. This will only work on some vision models. An 'mmproj-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--mtp", action="store_true",
        help="Export only the multi-token prediction (MTP) head as a separate GGUF, suitable for use as a speculative draft. An 'mtp-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="Exclude the multi-token prediction (MTP) head from the converted GGUF. Pair with --mtp on a second run to publish trunk and MTP as two files. Note: the split form duplicates embeddings, but even though the bundled default is more space-efficient overall, this allows differing quantization which may be more performant.",
    )
    parser.add_argument(
        "--mistral-format", action="store_true",
        help="Whether the model is stored following the Mistral format.",
    )
    parser.add_argument(
        "--disable-mistral-community-chat-template", action="store_true",
        help=(
            "Whether to disable usage of Mistral community chat templates. If set, use the Mistral official `mistral-common` library for tokenization and detokenization of Mistral models. "
            "Using `mistral-common` ensure correctness and zero-day support of tokenization for models converted from the Mistral format but requires to manually setup the tokenization server."
        )
    )

    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help=("Whether to include sentence-transformers dense modules. "
              "It can be used for sentence-transformers models, like google/embeddinggemma-300m. "
              "Default these modules are not included.")
    )

    parser.add_argument(
        "--fuse-gate-up-exps", action="store_true",
        help="Fuse gate_exps and up_exps tensors into a single gate_up_exps tensor for MoE models.",
    )
    parser.add_argument(
        "--fp8-as-q8", action="store_true",
        help="Store tensors dequantized from FP8 as Q8_0 instead of BF16/F16.",
    )

    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def main() -> None:
    args = parse_args()

    if args.print_supported_models:
        logger.error("Supported models:")
        print_registered_models()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _enable_traceback_watchdog()
    _log_phase("start", model=args.model, remote=args.remote, outfile=args.outfile)

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download
        allowed_patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"]
        if args.sentence_transformers_dense_modules:
            # include sentence-transformers dense modules safetensors files
            allowed_patterns.append("*.safetensors")
        phase_started_at = _log_phase("snapshot_download_start", repo_id=hf_repo_id, allowed_patterns=",".join(allowed_patterns))
        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=allowed_patterns)
        _log_phase_done("snapshot_download", phase_started_at, local_dir=local_dir)
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)
        _log_phase("local_model_selected", dir_model=dir_model)

    if not dir_model.is_dir():
        logger.error(f'Error: {dir_model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if is_split and not args.use_temp_file:
        logger.info("Split output requested; using a temporary tensor spool to avoid retaining all tensors in memory")
        args.use_temp_file = True

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    is_mistral_format = args.mistral_format
    if is_mistral_format and not _mistral_common_installed:
        raise ImportError(_mistral_import_error_msg)
    disable_mistral_community_chat_template = args.disable_mistral_community_chat_template

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        phase_started_at = _log_phase("load_hparams_start", dir_model=dir_model)
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
        _log_phase_done("load_hparams", phase_started_at, keys=len(hparams))
        if not is_mistral_format:
            phase_started_at = _log_phase("architecture_detect_start", model_type=model_type.name)
            model_architecture = get_model_architecture(hparams, model_type)
            _log_phase_done("architecture_detect", phase_started_at, architecture=model_architecture)
            logger.info(f"Model architecture: {model_architecture}")
            try:
                phase_started_at = _log_phase("model_class_lookup_start", architecture=model_architecture)
                model_class = get_model_class(model_architecture, mmproj=(model_type == ModelType.MMPROJ))
                _log_phase_done("model_class_lookup", phase_started_at, model_class=model_class.__name__)
            except NotImplementedError:
                logger.error(f"Model {model_architecture} is not supported")
                sys.exit(1)
        elif args.mmproj:
            assert hparams.get("vision_encoder") is not None, "This model does not support multimodal"
            from conversion.pixtral import PixtralModel
            model_class = PixtralModel
        elif hparams.get("moe") is not None:
            from conversion.mistral import MistralMoeModel
            model_class = MistralMoeModel
        else:
            from conversion.mistral import MistralModel
            model_class = MistralModel

        if args.mtp and args.no_mtp:
            logger.error("--mtp and --no-mtp are mutually exclusive")
            sys.exit(1)

        if args.mtp or args.no_mtp:
            from conversion.qwen import _Qwen35MtpMixin
            from conversion.step3 import Step35Model
            if not (issubclass(model_class, _Qwen35MtpMixin) or issubclass(model_class, Step35Model)):
                logger.error("--mtp / --no-mtp are only supported for Qwen3.5/3.6 and Step3.5 text variants today")
                sys.exit(1)
            if args.no_mtp:
                model_class.no_mtp = True
            if args.mtp:
                model_class.mtp_only = True

        def build_model_instance(skip_output_shards_before: int, stop_output_shards_after: int) -> ModelBase:
            phase_started_at = _log_phase(
                "model_init_start",
                model_class=model_class.__name__,
                remote_hf_model_id=hf_repo_id,
                split_max_size=args.split_max_size,
                skip_output_shards_before=skip_output_shards_before,
                stop_output_shards_after=stop_output_shards_after,
            )
            model_instance = model_class(dir_model, output_type, fname_out,
                                         is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                         eager=args.no_lazy,
                                         metadata_override=args.metadata, model_name=args.model_name,
                                         split_max_tensors=args.split_max_tensors,
                                         split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                         small_first_shard=args.no_tensor_first_split,
                                         remote_hf_model_id=hf_repo_id, disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                                         sentence_transformers_dense_modules=args.sentence_transformers_dense_modules,
                                         fuse_gate_up_exps=args.fuse_gate_up_exps,
                                         fp8_as_q8=args.fp8_as_q8,
                                         skip_output_shards_before=skip_output_shards_before,
                                         stop_output_shards_after=stop_output_shards_after,
                                         )
            _log_phase_done("model_init", phase_started_at, model_class=model_class.__name__)
            return model_instance

        def write_model_window(skip_output_shards_before: int, stop_output_shards_after: int) -> ModelBase:
            model_instance = build_model_instance(skip_output_shards_before, stop_output_shards_after)
            phase_started_at = _log_phase(
                "write_model_start",
                skip_output_shards_before=skip_output_shards_before,
                stop_output_shards_after=stop_output_shards_after,
            )
            logger.info("Exporting model...")
            model_instance.write()
            _log_phase_done("write_model", phase_started_at)
            is_split_output = len(model_instance.gguf_writer.tensors) > 1
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split_output else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")
            return model_instance

        if args.vocab_only:
            if args.materialize_output_shard_window_size > 0:
                logger.error("--materialize-output-shard-window-size cannot be combined with --vocab-only")
                sys.exit(1)
            model_instance = build_model_instance(args.skip_output_shards_before, args.stop_output_shards_after)
            phase_started_at = _log_phase("write_vocab_start")
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            _log_phase_done("write_vocab", phase_started_at)
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
            return

        if args.materialize_output_shard_window_size <= 0:
            model_instance = write_model_window(args.skip_output_shards_before, args.stop_output_shards_after)
            if args.upload_finalized_shards_to_repo:
                total_shards = len(model_instance.gguf_writer.tensors)
                paths = _materialized_shard_paths(
                    model_instance.fname_out,
                    args.skip_output_shards_before,
                    args.stop_output_shards_after,
                    total_shards,
                )
                _upload_materialized_shards(
                    repo_id=args.upload_finalized_shards_to_repo,
                    prefix=args.upload_finalized_shards_prefix,
                    paths=paths,
                    private=args.upload_finalized_shards_private,
                    delete_uploaded=args.delete_uploaded_shards,
                )
            return

        if args.materialize_output_shard_window_size < 1:
            logger.error("--materialize-output-shard-window-size must be positive")
            sys.exit(1)

        start_shard = max(args.skip_output_shards_before, 1)
        if args.resume_uploaded_shards and args.upload_finalized_shards_to_repo:
            uploaded, expected_total = _uploaded_split_shards(
                args.upload_finalized_shards_to_repo,
                args.upload_finalized_shards_prefix,
                fname_out,
            )
            first_missing = _first_missing_shard(uploaded, expected_total)
            start_shard = max(start_shard, first_missing)
            logger.info(
                "hub_shard_resume repo=%s uploaded_count=%d expected_total=%d first_missing=%d start_shard=%d",
                args.upload_finalized_shards_to_repo,
                len(uploaded),
                expected_total,
                first_missing,
                start_shard,
            )

        stop_limit = args.stop_output_shards_after
        current_shard = start_shard
        total_shards = 0
        while True:
            window_stop = current_shard + args.materialize_output_shard_window_size - 1
            if stop_limit > 0:
                window_stop = min(window_stop, stop_limit)
            logger.info(
                "materialize_output_shard_window_start start=%d stop=%d",
                current_shard,
                window_stop,
            )
            model_instance = write_model_window(current_shard, window_stop)
            total_shards = len(model_instance.gguf_writer.tensors)
            materialized = _materialized_shard_paths(
                model_instance.fname_out,
                current_shard,
                window_stop,
                total_shards,
            )
            logger.info(
                "materialize_output_shard_window_done start=%d stop=%d total_shards=%d materialized_count=%d",
                current_shard,
                window_stop,
                total_shards,
                len(materialized),
            )
            if args.upload_finalized_shards_to_repo:
                _upload_materialized_shards(
                    repo_id=args.upload_finalized_shards_to_repo,
                    prefix=args.upload_finalized_shards_prefix,
                    paths=materialized,
                    private=args.upload_finalized_shards_private,
                    delete_uploaded=args.delete_uploaded_shards,
                )
            if current_shard > total_shards:
                break
            if window_stop >= total_shards:
                break
            if stop_limit > 0 and window_stop >= stop_limit:
                break
            current_shard = window_stop + 1
        logger.info("materialize_output_shard_windows_complete total_shards=%d", total_shards)


if __name__ == '__main__':
    main()
