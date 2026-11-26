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

    def test_matched_mixing_and_noise_only(self):
        rng = np.random.default_rng(3)
        sig = [rng.normal(size=(96, 6001)).astype("float32")]
        noise = [rng.normal(size=(96, 6001)).astype("float32")]
        x, y = unet2d.sample(sig, noise, 400, rng, matched=True, noise_only=0.25)
        empty = np.all(y == 0, axis=(1, 2, 3))
        self.assertTrue(0.15 < empty.mean() < 0.35)
        self.assertTrue(np.all(np.abs(x[empty]).sum(axis=(1, 2, 3)) > 0))

    def test_separate_targets(self):
        rng = np.random.default_rng(4)
        sig = [rng.normal(size=(96, 6001)).astype("float32")]
        noise = [np.zeros((96, 6001), dtype="float32")]
        tgt = [0.5 * sig[0]]
        x, y = unet2d.sample(sig, noise, 5, rng, targets=tgt)
        self.assertTrue(np.allclose(y, 0.5 * x, atol=1e-5))

    def test_same_draws_without_targets(self):
        sig = [np.random.default_rng(5).normal(size=(96, 6001)).astype("float32")]
        noise = [np.random.default_rng(6).normal(size=(96, 6001)).astype("float32")]
        a = unet2d.sample(sig, noise, 4, np.random.default_rng(7))
        b = unet2d.sample(sig, noise, 4, np.random.default_rng(7), targets=sig)
        self.assertTrue(np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]))

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
