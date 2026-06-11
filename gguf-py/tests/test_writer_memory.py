#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf


class TestWriterMemoryOptimizations(unittest.TestCase):
    def test_bf16_passthrough_bytes_match_torch_storage(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            self.skipTest("torch and transformers are required for conversion helper tests")

        import torch
        from conversion.base import ModelBase

        tensor = torch.tensor(
            [[1.0, -2.5, 3.25, 4.5], [0.0, 7.75, -8.5, 9.0]],
            dtype=torch.float32,
        ).to(torch.bfloat16)
        expected = tensor.contiguous().view(torch.uint8).reshape(2, 8).numpy()
        legacy = gguf.quants.quantize(tensor.float().numpy(), gguf.GGMLQuantizationType.BF16)

        got = ModelBase._bf16_tensor_to_gguf_bytes(tensor)

        self.assertEqual(got.dtype, np.uint8)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(got, legacy)

    def test_split_temp_file_writer_matches_in_memory_writer(self):
        tensors = {
            "tensor_a": np.arange(8, dtype=np.float32).reshape(2, 4),
            "tensor_b": np.arange(12, dtype=np.float32).reshape(3, 4),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct_dir = root / "direct"
            temp_dir = root / "temp"
            direct_dir.mkdir()
            temp_dir.mkdir()

            direct_files = self._write_split_model(direct_dir / "model.gguf", tensors, use_temp_file=False)
            temp_files = self._write_split_model(temp_dir / "model.gguf", tensors, use_temp_file=True)

            self.assertEqual([p.name for p in direct_files], [p.name for p in temp_files])
            for direct, temp in zip(direct_files, temp_files):
                self.assertEqual(direct.read_bytes(), temp.read_bytes())

    @staticmethod
    def _write_split_model(path: Path, tensors: dict[str, np.ndarray], *, use_temp_file: bool) -> list[Path]:
        writer = gguf.GGUFWriter(
            path=None,
            arch="llama",
            use_temp_file=use_temp_file,
            split_max_tensors=1,
        )
        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor)

        writer.write_header_to_file(path=path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        return sorted(path.parent.glob(f"{path.stem}-*.gguf"))


if __name__ == "__main__":
    unittest.main()
