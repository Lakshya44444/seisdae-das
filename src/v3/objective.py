"""Compare two decoder capacities and two losses under identical development-fold budgets."""
import numpy as np
import pandas as pd

from .. import neural
from ..data import LEVELS, ROOT, SEED, load
from ..v2.stats import aggregate
from ..v2.training import CACHE, arrays, cache, folds, score, tail
from .experiments import CFG, OUT


def synthesis_basis():
    """Real/imaginary inverse-STFT basis for the three frames covering the reference."""
    re, im = [], []
    for start in range(0, 300, 20):
        ids = np.arange(start, start+20)
        z = np.zeros((len(ids), 100, 121), dtype=np.complex64)
        z[np.arange(len(ids)), ids//3, 59 + ids%3] = 1
        re.extend(neural.decode(z, 1., 2965, 70))
        im.extend(neural.decode(z * 1j, 1., 2965, 70))
    return np.asarray(re, dtype="float32"), np.asarray(im, dtype="float32")


def fit(net, bank, pairs, fit_ids, stop_ids, capacity, objective, basis, seed):
    import tensorflow as tf
    import tf_keras as k

    k.utils.set_random_seed(seed)
    model = tail(net)
    if capacity == "Head":
        for layer in model.layers:
            layer.trainable = layer.name == "conv2d_17"
    cfg = CFG["objective"]
    opt = k.optimizers.Adam(cfg["learning_rate"])
    def unpack(ids):
        reference = np.concatenate([np.tile(pairs["reference"][i,::4], (len(LEVELS),1)) for i in ids]).astype("float32")
        return arrays(bank,ids,"decoder"), arrays(bank,ids,"target"), arrays(bank,ids,"z"), reference
    xt, yt, zt, rt = unpack(fit_ids)
    xv, yv, zv, rv = unpack(stop_ids)

    def loss(x, y, z, reference):
        prob = model(x, training=False)
        if objective == "mask CE":
            valid = tf.reduce_sum(y, axis=-1) > 0
            return -tf.reduce_mean(tf.boolean_mask(tf.reduce_sum(y * tf.math.log(tf.maximum(prob,1e-8)),axis=-1),valid))
        masked = tf.reshape(tf.cast(prob[...,0],tf.complex64) * z, (-1, 300))
        waveform = tf.math.real(masked) @ basis[0] + tf.math.imag(masked) @ basis[1]
        return tf.reduce_mean(tf.reduce_sum((waveform-reference)**2,axis=1) /
                              tf.maximum(tf.reduce_sum(reference**2,axis=1),1e-20))

    @tf.function
    def step(x, y, z, r):
        with tf.GradientTape() as tape:
            value = loss(x, y, z, r)
        opt.apply_gradients(zip(tape.gradient(value,model.trainable_weights),model.trainable_weights))

    rng = np.random.default_rng(seed)
    best = model.get_weights()
    minimum = float(loss(xv,yv,zv,rv))
    for i in range(1,cfg["steps"]+1):
        ix = rng.integers(len(xt), size=cfg["batch_size"])
        step(xt[ix],yt[ix],zt[ix],rt[ix])
        if i % cfg["check_every"] == 0:
            value = float(loss(xv,yv,zv,rv))
            if value < minimum:
                best, minimum = model.get_weights(), value
    model.set_weights(best)
    return model


def run():
    events,pairs = load(ROOT / "data/prepared")
    dev = events[events.split.isin(["train","validation"])]
    net = cache(dev,pairs)
    bank = {}
    for i,e in dev.iterrows():
        with np.load(CACHE / f"{e.event_id}.npz") as z:
            bank[i] = dict(z)
    basis = synthesis_basis()
    rows, fits = [], []
    for f,(train,stop,held) in enumerate(folds(dev),1):
        for capacity in ("Head","Decoder"):
            for obj in ("mask CE","waveform NMSE"):
                name = f"{capacity} / {obj}"
                model = fit(net,bank,pairs,train,stop,capacity,obj,basis,SEED+f)
                new = score(events,pairs,bank,held,None,name,decoder=model)
                rows.extend(dict(r,fold=f) for r in new)
                fits.append(dict(fold=f,method=name,params=sum(np.prod(w.shape) for w in model.trainable_weights),steps=CFG["objective"]["steps"]))
        print(f"Objective fold {f} complete",flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "objective_events.csv",index=False)
    pd.DataFrame(fits).to_csv(OUT / "objective_fits.csv",index=False)
    aggregate(d,["method","level"],["gain_db","si_gain_db","cc","pick_mae_ms"]).to_csv(OUT / "objective_summary.csv",index=False)


if __name__ == "__main__":
    run()
