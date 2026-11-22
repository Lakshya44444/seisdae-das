import hashlib
import unittest

import numpy as np

from src.data import LEVELS, ROOT, load, mix
from src.evaluate import fit_scalar
from src.summary import paired_intervals, summarise
from src.metrics import bandpass, fk, scores, wavelet
from src.neural import decode, encode


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events, cls.arrays = load(ROOT / "data/prepared")

    def test_no_event_reuse(self):
        e = self.events
        self.assertTrue(e.event_id.is_unique)
        self.assertEqual(
            e.split.value_counts().to_dict(),
            {"train": 24, "validation": 6, "pilot": 6, "test": 145},
        )
        for site, g in e.groupby("array"):
            self.assertEqual(
                g.split.value_counts().to_dict(),
                {
                    "train": 12,
                    "validation": 3,
                    "pilot": 3,
                    "test": 133 if site == "longvalley" else 12,
                },
            )

    def test_requested_snr(self):
        for s, n in zip(self.arrays["reference"], self.arrays["noise"]):
            for level in LEVELS:
                x = mix(s, n, level)
                measured = 10 * np.log10(np.sum(s * s) / np.sum((x - s) ** 2))
                self.assertAlmostEqual(measured, level, places=9)

    def test_identity_metrics_and_filters(self):
        s, n, p = (self.arrays[key][0] for key in ("reference", "noise", "nominal_p"))
        x = mix(s, n, 0)
        self.assertAlmostEqual(scores(s, x, x, p)["gain_db"], 0)
        for fun in (bandpass, fk, wavelet):
            y = fun(x)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(np.isfinite(y).all())

    def test_mask_roundtrip(self):
        x = self.arrays["reference"][0]
        feat, z, start = encode(x)
        self.assertEqual(feat.shape, (24, 100, 121, 2))
        np.testing.assert_allclose(decode(z, 1, start, 70), x, atol=1e-6, rtol=1e-5)

    def test_scalar_excludes_validation_pilot_and_test(self):
        baseline = fit_scalar(self.events, self.arrays)
        altered = {k: v.copy() for k, v in self.arrays.items()}
        altered["noise"][self.events.split != "train"] = np.nan
        self.assertEqual(fit_scalar(self.events, altered), baseline)
        self.assertTrue(0.0 < baseline < 1.0)

    def test_eventwise_summaries(self):
        import pandas as pd

        d = pd.DataFrame(
            {
                "event_id": [1, 2, 1, 2],
                "array": ["A", "B", "A", "B"],
                "level": [0] * 4,
                "method": ["Zero-shot"] * 2 + ["Fine-tuned"] * 2,
                "gain_db": [1.0, 2.0, 2.0, 3.0],
                "reference_picks": [4] * 4,
                "matched_picks": [1, 3, 2, 4],
                "pick_abs_ms_sum": [10, 60, 10, 60],
                "cc": [1.0] * 4,
                "rmse": [1.0] * 4,
                "nrmse": [1.0] * 4,
                "ms_per_trace": [1.0] * 4,
            }
        )
        row = summarise(d, ["level", "method"]).loc[(0, "Zero-shot")]
        self.assertEqual(row["gain_db"], 1.5)
        self.assertEqual(row["onset_mae_ms"], 17.5)
        self.assertEqual(row["pick_recovery"], 0.5)
        ci = paired_intervals(d, draws=50).iloc[0]
        self.assertEqual((ci.ci95_low_db, ci.ci95_high_db), (1.0, 1.0))

    def test_checkpoint_identity(self):
        with (ROOT / "models/gr_mixed_stft.h5").open("rb") as f:
            self.assertEqual(
                hashlib.file_digest(f, "sha256").hexdigest(),
                "ec61038e62259630bb35bed6217797b44c3b2bddcfeec9c9fc90b8748c54a63a",
            )


if __name__ == "__main__":
    unittest.main()
