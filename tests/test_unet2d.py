"""Offline tests for the 2-D U-Net data pipeline (no TensorFlow)."""
import unittest

import numpy as np

from src import unet2d


class UNet2DData(unittest.TestCase):
    def test_pairs_have_target_inside_input(self):
        rng = np.random.default_rng(0)
        sig = [rng.normal(size=(96, 6001)).astype("float32")]
        noise = [np.zeros((96, 6001), dtype="float32")]
        x, y = unet2d.sample(sig, noise, 6, rng)
        self.assertEqual(x.shape, (6, unet2d.CROP_CH, unet2d.CROP_T, 1))
        self.assertTrue(np.allclose(x, y, atol=1e-5))
        self.assertTrue(np.allclose(x.std(axis=(1, 2, 3)), 1, atol=1e-3))

    def test_requested_snr(self):
        rng = np.random.default_rng(1)
        sig = [rng.normal(size=(96, 6001)).astype("float32")]
        noise = [rng.normal(size=(96, 6001)).astype("float32") * 7]
        x, y = unet2d.sample(sig, noise, 200, rng)
        snr = 10 * np.log10(np.sum(y ** 2, axis=(1, 2, 3)) / np.sum((x - y) ** 2, axis=(1, 2, 3)))
        self.assertGreaterEqual(snr.min(), -10.01)
        self.assertLessEqual(snr.max(), 10.01)

    def test_whole_window_padding(self):
        class StandIn(unet2d.UNet2D):
            def __init__(self):
                class M:
                    def predict(self, z, verbose=0):
                        assert z.shape[1] % 8 == 0 and z.shape[2] % 64 == 0
                        return z
                self.model = M()

        g = np.random.default_rng(2).normal(size=(96, 6001))
        self.assertLess(np.abs(StandIn()(g) - g).max(), 1e-5)


if __name__ == "__main__":
    unittest.main()
