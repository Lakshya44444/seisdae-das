"""Regression tests for the restored research pipelines, independent of golden slide text."""
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src import neural
from src.data import ROOT, LEVELS, load
from src.v2.training import folds
from src.v3.experiments import CFG, classical, clusters, context
from src.v3.objective import synthesis_basis
from src.v3.scratch_model import fold_ids
from src.v3.stretch import select

MODULES = ("boundary crossfit deck experiments extra_plots mechanism multi_data noise objective "
           "release report scene scratch_model sensitivity stead stretch visuals").split()


class FinalStudy(unittest.TestCase):
    def test_module_inventory_and_imports(self):
        for module in MODULES:
            with self.subTest(module=module):
                importlib.import_module("src.v3." + module)
        self.assertEqual(CFG["seed"], 2026)

    def test_zero_padding_roundtrip_and_svd(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((24,70))
        padded, central = context(x)
        self.assertEqual(padded.shape, (24,6001))
        np.testing.assert_array_equal(padded[:,central], x)
        self.assertEqual(np.count_nonzero(padded), x.size)
        np.testing.assert_allclose(classical(x,"SVD","24"), x, atol=1e-12)

    def test_synthesis_basis_matches_full_inverse(self):
        rng = np.random.default_rng(4)
        local = (rng.standard_normal((2,100,3)) + 1j*rng.standard_normal((2,100,3))).astype("complex64")
        z=np.zeros((2,100,121),dtype="complex64")
        z[:,:,59:62] = local
        real,imag=synthesis_basis()
        x=local.reshape(2,-1)
        np.testing.assert_allclose(x.real @ real + x.imag @ imag, neural.decode(z,1.,2965,70), rtol=3e-5, atol=2e-5)

    def test_outer_folds_and_strict_development(self):
        events = pd.read_csv(ROOT / "data/prepared/events.csv")
        held=[]
        for fit,score in fold_ids(events):
            self.assertFalse(set(fit)&set(score))
            self.assertEqual(set(fit)|set(score),set(events.index))
            held.extend(score)
        self.assertCountEqual(held,events.index)
        dev=events[events.split.isin(["train","validation"])]
        for tr,va,te in folds(dev):
            self.assertEqual(set(tr)|set(va)|set(te),set(dev.index))
            self.assertEqual(len(tr)+len(va)+len(te),len(dev))

    def test_clusters_require_site_time_and_distance(self):
        meta=pd.DataFrame(dict(event_time=["2020-01-01","2020-01-02","2020-01-02","2020-01-20","2020-01-02"],
                               latitude=[35,35,35,35,36],longitude=[-118]*5,array=["A","A","B","A","A"]))
        labels=clusters(meta)
        self.assertEqual(labels[0],labels[1])
        self.assertEqual(len(set(labels)),4)

    def test_stretch_is_selected_from_development_shape(self):
        cv=pd.read_csv(ROOT/"results_v3/stretch_cv_events.csv")
        e=pd.read_csv(ROOT/"data/prepared/events.csv")
        self.assertEqual(set(cv.event_id), set(e[e.split.isin(["train","validation"])].event_id))
        self.assertEqual(set(cv.factor), set(CFG["stretch"]["factors"]))
        self.assertFalse(cv.duplicated(["event_id","level","factor","method"]).any())
        grid=cv.groupby(["method","factor"],as_index=False)[["gain_db","si_gain_db"]].mean()
        chosen=pd.read_csv(ROOT/"results_v3/stretch_selection.csv")
        pd.testing.assert_frame_equal(select(grid),chosen,check_exact=False)
        test=pd.read_csv(ROOT/"results_v3/stretch_events.csv")
        self.assertEqual(set(test.event_id),set(e[e.split=="test"].event_id))
        self.assertEqual(set(zip(test.method,test.factor)),set(zip(chosen.method,chosen.factor)))

    def test_independent_noise_counts_and_ranges(self):
        e=pd.read_csv(ROOT/"data/prepared/events.csv")
        rows=pd.read_csv(ROOT/"results_v3/external_events.csv")
        self.assertEqual(set(rows.event_id),set(e[(e.array=="ridgecrest")&(e.split=="test")].event_id))
        keys=["event_id","recording","draw","level","method"]
        self.assertFalse(rows.duplicated(keys).any())
        self.assertEqual(len(rows),12*3*3*len(LEVELS)*2)
        for path in [ROOT/"data/noise_v2",*(ROOT/"data/noise_v3").glob("20*")]:
            m=json.loads((path/"continuous.json").read_text())
            self.assertEqual(len(m["segments"]),24)
            for segment in m["segments"]:
                self.assertEqual(segment["end_byte"]-segment["start_byte"]+1,30*250*4)
            with np.load(path/"continuous.npz") as data:
                self.assertEqual(data["background"].shape,(24,3000))

    def test_two_recorded_apertures_are_disjoint(self):
        e,p=load(ROOT/"data/prepared")
        m=pd.read_csv(ROOT/"data/multi_v3/manifest.csv")
        np.testing.assert_array_equal(m.event_id,e.event_id)
        self.assertTrue((abs(m.second_row-m.first_row)>=24).all())
        with np.load(ROOT/"data/multi_v3/aperture2.npz") as two:
            self.assertEqual(two["reference"].shape,p["reference"].shape)
            self.assertTrue(np.all(np.any(two["reference"]!=p["reference"],axis=(1,2))))
