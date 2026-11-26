"""Offline tests for the jDAS wrapper (no TensorFlow: a stand-in replaces the network)."""
import unittest

import numpy as np
from scipy.signal import resample_poly

from src import jdas


def copy_hidden(samples, masks):
    """Stand-in network that returns the hidden channel exactly."""
    return samples * (1 - masks)


class JDASWrapper(unittest.TestCase):
    def test_each_channel_hidden_once(self):
        data = np.random.default_rng(0).normal(size=(20, jdas.NT))
        samples, masks = jdas.masked_blocks(data)
        self.assertEqual(samples.shape, (20, jdas.NSUB, jdas.NT, 1))
        hidden = [int(np.argmin(masks[i, :, 0, 0])) for i in range(20)]
        self.assertEqual(hidden[:6], [0, 1, 2, 3, 4, 5])
        self.assertEqual(hidden[-1], 10)
        for i in range(20):
            self.assertEqual((masks[i] == 0).sum(), jdas.NT)
            self.assertTrue(np.allclose(samples[i, hidden[i], :, 0], data[i]))

    def test_blocks_and_crossfade_rebuild_input(self):
        x = np.random.default_rng(1).normal(size=(96, 3001))
        self.assertLess(np.abs(jdas.chunked(copy_hidden, x) - x).max(), 1e-5)

    def test_100hz_in_100hz_out(self):
        class StandIn(jdas.JDAS):
            def __init__(self):
                self.postfilter = False

            def predict(self, samples, masks):
                return copy_hidden(samples, masks)

        g = np.random.default_rng(2).normal(size=(30, 6001))
        out = StandIn()(g)
        ref = resample_poly(resample_poly(g, 1, 2, axis=1), 2, 1, axis=1)[:, :6001]
        self.assertEqual(out.shape, g.shape)
        self.assertLess(np.abs(out - ref).max(), 1e-5)

    def test_training_target_is_hidden_channel(self):
        g = [np.random.default_rng(3).normal(size=(96, 3001)).astype("float32")]
        (s, m), t = jdas.patches(g, 8, np.random.default_rng(4))
        self.assertEqual(s.shape, (8, jdas.NSUB, jdas.NT, 1))
        self.assertTrue(np.all((m == 0).sum(axis=(1, 2, 3)) == jdas.NT))
        self.assertTrue(np.allclose(t, s * (1 - m)))


if __name__ == "__main__":
    unittest.main()
