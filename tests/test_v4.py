"""Offline tests for the V4 Ridgecrest pipeline (no network, no TensorFlow)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.v4 import build, catalog, das, methods
from src.v4.evaluate import pairs

ROOT = Path(__file__).resolve().parents[1]


class Archive(unittest.TestCase):
    def test_byte_offsets_match_v3_downloads(self):
        manifest = json.loads((ROOT / "data/noise_v2/continuous.json").read_text())
        n = manifest["duration_s"] * manifest["source_hz"]
        for seg in manifest["segments"]:
            first = das.byte_offset(seg["trace_row"], manifest["sample_start"])
            self.assertEqual(first, seg["start_byte"])
            self.assertEqual(first + 4 * n - 1, seg["end_byte"])

    def test_split_by_hour(self):
        start = pd.Timestamp("2020-06-24T17:59:50Z")
        pieces = das.split_by_hour(start, 20 * 250)
        self.assertEqual([p[1:] for p in pieces], [(das.PER_HOUR - 2500, 2500), (0, 2500)])
        self.assertEqual(pieces[1][0], pd.Timestamp("2020-06-24T18:00Z"))

    def test_common_mode_removes_simultaneous_arrival(self):
        t = np.arange(16010) / 250
        quake = np.exp(-((t - 30) * 2) ** 2) * np.sin(2 * np.pi * 3 * t)
        raw = np.tile(quake, (96, 1)) + 0.01 * np.random.default_rng(6).normal(size=(96, t.size))
        energy = {}
        for flag in (True, False):
            with mock.patch.dict(das.CFG["run"], {"common_mode_removal": flag}):
                energy[flag] = np.sum(das.preprocess(raw) ** 2)
        self.assertLess(energy[True], 0.01 * energy[False])

    def test_window_length(self):
        def fake(start, seconds, rows):
            return np.random.default_rng(0).normal(size=(len(rows), int(round(seconds * 250))))
        with mock.patch.object(das, "read_raw", fake):
            x = das.read_window(pd.Timestamp("2020-06-24T17:10Z"), list(range(96)))
        self.assertEqual(x.shape, (96, 6001))


class Geometry(unittest.TestCase):
    def test_located_runs(self):
        self.assertEqual(das.located_runs([5, 6, 7, 10, 11, 20], 2), [(5, 7, 3), (10, 11, 2)])

    def test_geometry_uses_index_column(self):
        first = das.block_rows()[0]
        rows = np.r_[first - 100:first - 50, first - 10:first + 200]
        info = pd.DataFrame(dict(index=rows, latitude=35.0 + 0 * rows, longitude=-117.0 + rows * 1e-4,
                                 elevation_m=0.0))
        with mock.patch.object(das, "das_info", lambda: info):
            lat, lon, dx = das.geometry()
        self.assertAlmostEqual(lon, -117.0 + np.mean(das.block_rows()) * 1e-4, places=9)
        self.assertAlmostEqual(dx, 9.1, delta=0.1)
        gap = info[(info["index"] < first + 20) | (info["index"] > first + 30)]
        with mock.patch.object(das, "das_info", lambda: gap):
            with self.assertRaises(KeyError):
                das.geometry()


class Catalogue(unittest.TestCase):
    def test_distance_filter(self):
        d = pd.DataFrame(dict(event_id=["near", "edge", "far"], latitude=[35.70, 36.20, 61.0],
                              longitude=[-117.70, -117.736, -150.0]))
        kept = catalog.within(d, 35.652, -117.736, 0.6)
        self.assertEqual(list(kept.event_id), ["near", "edge"])


class Dataset(unittest.TestCase):
    def test_day_split_cycle(self):
        days = pd.date_range("2020-06-23", periods=10, freq="D", tz="UTC")
        got = [build.split_of(d + pd.Timedelta(hours=5)) for d in days]
        self.assertEqual(got, ["train"] * 3 + ["validation", "test"] + ["train"] * 3 + ["validation", "test"])

    def test_snr(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 6001))
        x[:, 1000:2500] *= 10
        snr = build.channel_snr_db(x, 1000, 1400)
        self.assertTrue(np.all(snr > 15))

    def test_export_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            g = tmp / "gathers"
            g.mkdir()
            rng = np.random.default_rng(2)
            np.savez_compressed(g / "eq_1.npz", data=rng.normal(size=(8, 6001)).astype("float32"))
            np.save(g / "eq_1_snr.npy", np.array([20, 20, 5, 20, 20, 20, 20, 20.0]))
            np.savez_compressed(g / "noise_a.npz", data=rng.normal(size=(8, 6001)).astype("float32"))
            pd.DataFrame([dict(id="eq_1", kind="strong", split="train", start="2020-06-23T01:00Z", itp=1000, its=1500),
                          dict(id="noise_a", kind="noise", split="train", start="2020-06-23T02:00Z")]).to_csv(
                tmp / "manifest.csv", index=False)
            with mock.patch.multiple(build, GATHERS=g, MANIFEST=tmp / "manifest.csv", UPSTREAM=tmp / "up"):
                build.export()
            sig = sorted((tmp / "up/signal").glob("*.npz"))
            stride, low = build.CFG["export"]["channel_stride"], 2
            expected = [f"eq_1_c{c:03d}" for c in range(0, 8, stride) if c != low]
            self.assertEqual([f.stem for f in sig], expected)
            with np.load(sig[0]) as z:
                self.assertEqual(z["data"].shape, (6001,))
                self.assertEqual(int(z["itp"]), 1000)
            self.assertEqual(len(list((tmp / "up/noise").glob("*.npz"))), 2)


class Scores(unittest.TestCase):
    def test_mix_level(self):
        rng = np.random.default_rng(3)
        s, n = rng.normal(size=(5, 6001)), rng.normal(size=(5, 6001))
        for level in (-10, 0, 10):
            x = methods.mix(s, n, level)
            self.assertAlmostEqual(10 * np.log10(np.sum(s ** 2) / np.sum((x - s) ** 2)), level, places=6)

    def test_scalar_has_no_si_gain(self):
        rng = np.random.default_rng(4)
        s = rng.normal(size=(5, 6001))
        x = methods.mix(s, rng.normal(size=s.shape), 0)
        r = methods.scores(s, x, 0.5 * x, slice(1000, 2000))
        self.assertGreater(r["gain_db"], 0)
        self.assertAlmostEqual(r["si_gain_db"], 0, places=9)
        self.assertAlmostEqual(methods.scores(s, x, s, slice(None))["cc"], 1, places=9)

    def test_fk_keeps_fast_removes_slow(self):
        t = np.arange(6001) / 100
        nc, dx = 96, 8.0
        pulse = lambda delay: np.exp(-((t[None, :] - 20 - delay[:, None]) * 8) ** 2) * np.sin(2 * np.pi * 8 * (t[None, :] - delay[:, None]))
        offsets = np.arange(nc) * dx
        fast, slow = pulse(offsets / 5000.0), pulse(offsets / 300.0)
        keep = methods.fk(fast, dx, 500)
        kill = methods.fk(slow, dx, 500)
        mid = slice(20, 76)
        self.assertGreater(np.mean(methods.correlation(fast[mid], keep[mid])), 0.95)
        self.assertLess(np.sum(kill[mid] ** 2), 0.1 * np.sum(slow[mid] ** 2))

    def test_stft_identity_mask(self):
        m = methods.SeisDAE.__new__(methods.SeisDAE)
        m.n, m.spec = 6001, dict(fs=100, nfft=198, nperseg=99)
        from scipy.signal import istft
        x = np.random.default_rng(5).normal(size=(3, 6001))
        _, z = m.encode(x)
        _, y = istft(z, **m.spec)
        self.assertTrue(np.allclose(y[:, :6001], x - x.mean(1, keepdims=True), atol=1e-4))

    def test_config_with_windows_line_endings(self):
        raw = (ROOT / "models/gr_mixed_stft.config").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crlf.config"
            path.write_bytes(raw.replace(b"\n", b"\r\n"))
            self.assertEqual(methods.read_config(path), methods.read_config(ROOT / "models/gr_mixed_stft.config"))

    def test_pairs_deterministic(self):
        ev = pd.DataFrame(dict(id=["a", "b", "c"]))
        nz = pd.DataFrame(dict(id=["n1", "n2", "n3", "n4"]))
        one = [(r.id, n, d) for r, n, d in pairs(ev, nz, 2, 7)]
        two = [(r.id, n, d) for r, n, d in pairs(ev, nz, 2, 7)]
        self.assertEqual(one, two)
        self.assertTrue(all(len({n for e, n, _ in one if e == k}) == 2 for k in "abc"))


if __name__ == "__main__":
    unittest.main()
