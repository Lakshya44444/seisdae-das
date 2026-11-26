"""Offline tests for the onset SNR (no TensorFlow, no data)."""
import unittest

import numpy as np

from src import onset_snr as o


class OnsetSNR(unittest.TestCase):
    def test_arrival_raises_snr(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 6001))
        quiet = o.onset_snr_db(x)
        x[:, 1050:1150] += 20 * np.sin(np.arange(100) / 3)
        self.assertGreater(o.onset_snr_db(x), quiet + 15)

    def test_definition(self):
        x = np.zeros((1, 600))
        x[0, :400] = np.tile([1.0, -1.0], 200)
        x[0, 400] = 9.0
        s = o.continuous_snr(x)
        self.assertAlmostEqual(s[0, 400], 9.0)
        self.assertAlmostEqual(s[0, 399], 10.0)
        self.assertTrue(np.isnan(s[0, 100]))


if __name__ == "__main__":
    unittest.main()
