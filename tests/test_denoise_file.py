"""Offline tests for src/denoise_file.py (no TensorFlow)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import denoise_file as d


class DenoiseFile(unittest.TestCase):
    def test_windows_rebuild_input(self):
        x = np.random.default_rng(0).normal(size=(5, 15000))
        self.assertLess(np.abs(d.windowed(lambda v: v, x) - x).max(), 1e-12)

    def test_short_input(self):
        x = np.random.default_rng(1).normal(size=(2, 3000))
        y = d.windowed(lambda v: v * 2, x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.allclose(y, 2 * x))

    def test_read_npy_and_resample(self):
        x = np.random.default_rng(2).normal(size=(12, 2500))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.npy"
            np.save(p, x)
            raw, fs = d.read(p, 250, None, False, 2, 6)
            self.assertEqual(raw.shape, (12, 1500))
            self.assertEqual(d.prepare(raw, fs).shape, (12, 600))


if __name__ == "__main__":
    unittest.main()
