import unittest
import numpy as np
import pandas as pd
from src.data import ROOT, load, SEED, mix
from src.v2.metrics import si_sdr, arrival
from src.v2.sensitivity import pad_input, other_donors
from src.v2.training import folds
from src.v2.stats import aggregate


class ShapeMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.s = rng.normal(size=(4, 70))
        self.x = self.s + rng.normal(size=self.s.shape)

    def test_si_invariant_to_attenuation_offset_and_sign(self):
        base = si_sdr(self.s, self.x)
        for scale in [0.001, 0.47, -2.3]:
            np.testing.assert_allclose(
                si_sdr(self.s, self.x * scale + 7), base, atol=1e-9
            )

    def test_si_projection(self):
        a = self.s - self.s.mean(1, keepdims=True)
        b = self.x - self.x.mean(1, keepdims=True)
        target = a * (a * b).sum(1, keepdims=True) / (a * a).sum(1, keepdims=True)
        expected = 10 * np.log10((target**2).sum(1) / ((b - target) ** 2).sum(1))
        np.testing.assert_allclose(si_sdr(a, b), expected)

    def test_degenerate(self):
        self.assertTrue(np.isnan(si_sdr(self.s, np.zeros_like(self.s))).all())

    def test_polarity_and_peaks(self):
        pos = np.repeat(14, 4)
        for scale, hits in [(0.5, 4), (-0.5, 0), (0, 0)]:
            a = arrival(self.s, scale * self.s, pos)
            self.assertEqual(a["polarity_hits"], hits)
            self.assertAlmostEqual(a["peak_ratio"], abs(scale))


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events, cls.p = load(ROOT / "data/prepared")

    def test_folds(self):
        dev = self.events[self.events.split.isin(["train", "validation"])]
        scores = []
        for tr, va, te in folds(dev):
            self.assertEqual(len(set(tr) | set(va) | set(te)), len(dev))
            self.assertEqual((len(tr), len(va), len(te)), (16, 4, 10))
            self.assertEqual(dev.loc[te, "array"].nunique(), 2)
            self.assertEqual(dev.loc[te, "array"].value_counts().tolist(), [5, 5])
            scores.extend(te)
        self.assertEqual(sorted(scores), sorted(dev.index))

    def test_noise_donors(self):
        val = self.events[self.events.split == "validation"]
        donors = other_donors(val)
        self.assertEqual(set(donors.values()), set(val.index))
        for i, j in donors.items():
            self.assertNotEqual(i, j)
            self.assertEqual(val.at[i, "array"], val.at[j, "array"])
        manifest = pd.read_csv(ROOT / "data/noise_v2/other_blocks.csv")
        self.assertTrue(
            (abs(manifest.signal_first_row - manifest.noise_first_row) >= 24).all()
        )

    def test_padding(self):
        t = np.arange(70)
        x = np.stack([np.sin(2 * np.pi * t / 35)] * 2)
        for mode in ["zero", "reflect", "noise", "tile", "stretch4"]:
            full, crop = pad_input(x, mode, SEED)
            self.assertEqual(full.shape, (2, 6001))
            y = crop(full)
            self.assertEqual(y.shape, x.shape)
            np.testing.assert_allclose(y, x, atol=0.03 if mode == "stretch4" else 1e-12)
            np.testing.assert_array_equal(pad_input(x, mode, SEED)[0], full)

    def test_bootstrap_counts(self):
        d = pd.DataFrame(
            dict(
                event_id=[1, 2],
                array=["a", "b"],
                level=[0, 0],
                matched_picks=[1, 2],
                reference_picks=[1, 4],
            )
        )
        out = aggregate(d, ["level"], ["pick_recovery"]).iloc[0]
        self.assertAlmostEqual(out.pick_recovery, 0.6)
        self.assertEqual(out.pick_recovery_lo, out.pick_recovery_hi)

    def test_aperture_levels(self):
        for width in [8, 16, 24]:
            k = (24 - width) // 2
            s = self.p["reference"][0, k : k + width]
            n = self.p["noise"][0, k : k + width]
            for db in [-5, 0, 5, 10]:
                x = mix(s, n, db)
                self.assertAlmostEqual(
                    10 * np.log10(np.sum(s * s) / np.sum((x - s) ** 2)), db
                )
