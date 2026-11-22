"""xgbgrok: Stage 1a runs for "Delayed Generalization and Double Descent in Gradient Boosting".

Do grokking and double descent over boosting rounds happen in XGBoost?

Design rules that make every run reusable later:
  * A run is fully defined by its config dict. The run ID is a hash of that config, so a run is
    never trained twice and an interrupted batch resumes where it stopped.
  * Data depend only on (data key, seed), never on XGBoost settings, so runs that differ only in
    settings see identical data (paired comparisons). Label flips are nested: the rows flipped at
    10% are the first 10% of the rows flipped at 20%, so later noise levels pair with these runs.
  * No early stopping. Every run trains to the round cap (learning rate x rounds = 500) and stores
    every round's error and log loss on 7 evaluation sets, the per-round tree statistics, the
    saved model, and the split / flip indices. Anything else (SESGB replay, hessians, margins,
    new detectors) can be computed later from these files without retraining.
  * Detectors read only the stored curves, so definitions can be changed later without reruns.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

CODE_VERSION = "stage1a-v1 (2026-11-22)"
LEARNING_TIME = 500  # rounds = LEARNING_TIME / eta

SETTINGS = {
    "default": dict(min_child_weight=1.0, reg_lambda=1.0, weight_scale=1.0),
    "brakes_off": dict(min_child_weight=0.0, reg_lambda=0.0, weight_scale=1.0),
    # Both brakes at 0 and all sample weights x1000: lowers XGBoost's fixed split-gain floor
    # (kRtEps = 1e-6) to 1e-9 without changing leaf values.
    "brakes_off_w1e3": dict(min_child_weight=0.0, reg_lambda=0.0, weight_scale=1000.0),
}

# TabArena-v0.1 datasets (OpenML IDs and targets from TabArena's curated metadata) and MNIST.
REAL_DATASETS = {
    "heloc": dict(openml_id=46932, target="RiskPerformance"),
    "E-CommereShippingData": dict(openml_id=46924, target="ArrivedLate"),
    "in_vehicle_coupon_recommendation": dict(openml_id=46937, target="AcceptCoupon"),
    "NATICUSdroid": dict(openml_id=46969, target="Malware"),
    "SDSS17": dict(openml_id=46955, target="ObjectType"),
    # MNIST: rows 0-59999 are the original training images, 60000-69999 the original test images.
    "mnist_784": dict(openml_id=554, target="class", test_pool_start=60000),
}


# ----------------------------------------------------------------------------------------------
# Seeds
# ----------------------------------------------------------------------------------------------
def stable_int(s: str) -> int:
    """Seed integer from a string. Python's hash() changes between sessions, so it is not used."""
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")


def make_rngs(data_key: str, seed: int):
    """Independent generators for the data and for the label noise of one (data key, seed)."""
    data_ss, noise_ss = np.random.SeedSequence([stable_int(data_key), seed]).spawn(2)
    return np.random.default_rng(data_ss), np.random.default_rng(noise_ss)


def flip_plan(n: int, num_class: int, rng):
    """Order in which rows get flipped, and the new label offset for each row.

    Always draws the same numbers, whatever the noise rate, so flips are nested across rates."""
    order = rng.permutation(n)
    offsets = rng.integers(1, max(num_class, 2), size=n)  # binary: offset is always 1
    return order, offsets


def apply_flips(y, rate, num_class, plan):
    order, offsets = plan
    k = int(round(rate * len(y)))
    idx = np.sort(order[:k])
    y2 = y.copy()
    y2[idx] = (y2[idx] + offsets[idx]) % num_class
    return y2, idx


def _array_sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, pd.DataFrame):
            a = pd.util.hash_pandas_object(a, index=False).to_numpy()
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------------------------
def make_parity(n, d, k, seed, n_val=2000, n_test=10000):
    """Sparse parity. Bits uniform on {0,1}; label = XOR of k hidden bits; hidden positions drawn
    per seed (XGBoost breaks exact gain ties by column order). Test rows never appear in training."""
    rng, _ = make_rngs(f"parity|d={d}|k={k}|n={n}", seed)
    hidden = np.sort(rng.choice(d, size=k, replace=False))
    X_tr = rng.integers(0, 2, size=(n, d), dtype=np.uint8)
    X_va = rng.integers(0, 2, size=(n_val, d), dtype=np.uint8)
    seen = {r.tobytes() for r in X_tr}
    rows = []
    while len(rows) < n_test:
        for r in rng.integers(0, 2, size=(2 * n_test, d), dtype=np.uint8):
            if r.tobytes() not in seen:
                rows.append(r)
                if len(rows) == n_test:
                    break
    X_te = np.stack(rows)
    lab = lambda X: (X[:, hidden].sum(1) % 2).astype(np.int64)
    return dict(X_train=X_tr.astype(np.float32), y_train=lab(X_tr), X_val=X_va.astype(np.float32),
                y_val=lab(X_va), X_test=X_te.astype(np.float32), y_test=lab(X_te), num_class=2,
                info=dict(hidden=hidden.tolist(), chance=0.5, ceiling=1.0))


_FRIEDMAN_THRESHOLD = None


def friedman_threshold():
    """Median of the Friedman #1 function under uniform inputs, from 1,000,000 fixed draws."""
    global _FRIEDMAN_THRESHOLD
    if _FRIEDMAN_THRESHOLD is None:
        x = np.random.default_rng(20261122).random((1_000_000, 5))
        _FRIEDMAN_THRESHOLD = float(np.median(_friedman(x)))
    return _FRIEDMAN_THRESHOLD


def _friedman(x):
    return (10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3] + 5 * x[:, 4])


def make_friedman(n, d, seed, n_val=2000, n_test=10000):
    """Friedman #1 thresholded at its median: zero Bayes error, 5 informative of d columns,
    column order shuffled per seed."""
    rng, _ = make_rngs(f"friedman|d={d}|n={n}", seed)
    thr = friedman_threshold()
    X = rng.random((n + n_val + n_test, d))
    y = (_friedman(X[:, :5]) > thr).astype(np.int64)
    perm = rng.permutation(d)
    X = X[:, perm].astype(np.float32)
    informative = sorted(int(j) for j in np.flatnonzero(perm < 5))
    a, b = n, n + n_val
    return dict(X_train=X[:a], y_train=y[:a], X_val=X[a:b], y_val=y[a:b], X_test=X[b:],
                y_test=y[b:], num_class=2,
                info=dict(threshold=thr, informative_columns=informative, chance=0.5, ceiling=1.0))


def make_sklearn_classification(n, d, seed, n_val=2000, n_test=10000):
    """scikit-learn make_classification, with flip_y=0 so there is no hidden label noise (default is 0.01)."""
    from sklearn.datasets import make_classification
    rng, _ = make_rngs(f"skl_classification|d={d}|n={n}", seed)
    X, y = make_classification(n_samples=n + n_val + n_test, n_features=d, n_informative=10,
                               n_redundant=2, flip_y=0.0, class_sep=1.0, shuffle=True,
                               random_state=int(rng.integers(0, 2**31 - 1)))
    X = X.astype(np.float32)
    a, b = n, n + n_val
    return dict(X_train=X[:a], y_train=y[:a], X_val=X[a:b], y_val=y[a:b], X_test=X[b:],
                y_test=y[b:], num_class=2, info=dict(chance=0.5))


def encode_modadd(a, b, p, encoding):
    if encoding == "onehot":
        eye = np.eye(p, dtype=np.float32)
        return np.concatenate([eye[a], eye[b]], axis=1)
    if encoding == "integers":
        return np.stack([a, b], axis=1).astype(np.float32)
    if encoding == "categorical":
        cats = list(range(p))
        return pd.DataFrame({"a": pd.Categorical(a, categories=cats),
                             "b": pd.Categorical(b, categories=cats)})
    if encoding == "sincos":  # all frequencies k = 1 .. (p-1)/2
        cols = []
        for k in range(1, (p - 1) // 2 + 1):
            for v in (a, b):
                cols += [np.cos(2 * np.pi * k * v / p), np.sin(2 * np.pi * k * v / p)]
        return np.stack(cols, axis=1).astype(np.float32)
    if encoding == "aplusb":  # positive control: a + b given (without the mod)
        return np.stack([a, b, a + b], axis=1).astype(np.float32)
    raise ValueError(encoding)


def make_modadd(p, train_frac, encoding, seed):
    """(a + b) mod p over all p^2 pairs. The split depends only on (p, train_frac, seed), so all
    encodings of one seed see the same pairs. Every value of a and b appears in training."""
    rng, _ = make_rngs(f"modadd|p={p}|frac={train_frac}", seed)
    a, b = (g.ravel() for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
    c = (a + b) % p
    n_tr = int(train_frac * p * p)
    while True:
        perm = rng.permutation(p * p)
        tr, rest = perm[:n_tr], perm[n_tr:]
        if len(np.unique(a[tr])) == p and len(np.unique(b[tr])) == p:
            break
    va, te = rest[: len(rest) // 2], rest[len(rest) // 2:]
    # Baseline that only memorizes which labels go with a and with b (proposal M6).
    Ma, Mb = np.zeros((p, p)), np.zeros((p, p))
    np.add.at(Ma, (a[tr], c[tr]), 1)
    np.add.at(Mb, (b[tr], c[tr]), 1)
    tie = np.random.default_rng(stable_int(f"memorizer|{p}|{seed}")).random((len(te), p)) * 1e-6
    memo_acc = float(np.mean(np.argmax(Ma[a[te]] + Mb[b[te]] + tie, axis=1) == c[te]))
    X = encode_modadd(a, b, p, encoding)
    take = (lambda idx: X.iloc[idx].reset_index(drop=True)) if isinstance(X, pd.DataFrame) else (lambda idx: X[idx])
    return dict(X_train=take(tr), y_train=c[tr], X_val=take(va), y_val=c[va], X_test=take(te),
                y_test=c[te], num_class=p,
                info=dict(chance=1 / p, ceiling=1.0, memorizer_test_acc=memo_acc,
                          train_pairs=tr.tolist(), val_pairs=va.tolist(), test_pairs=te.tolist()))


# ----------------------------------------------------------------------------------------------
# Real datasets (OpenML), cached as parquet on Drive
# ----------------------------------------------------------------------------------------------
def prepare_openml(data_dir, names=None):
    """Download each dataset once from OpenML and store it as parquet plus a metadata JSON with
    the OpenML version and a checksum. Runs only read the parquet files."""
    import openml
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        openml.config.set_root_cache_directory(str(data_dir / "openml_cache"))
    except AttributeError:
        openml.config.cache_directory = str(data_dir / "openml_cache")
    out = []
    for name in names or REAL_DATASETS:
        spec = REAL_DATASETS[name]
        f = data_dir / f"openml_{spec['openml_id']}.parquet"
        mf = f.with_suffix(".json")
        if f.exists() and mf.exists():
            out.append(json.loads(mf.read_text()))
            continue
        try:
            ds = openml.datasets.get_dataset(spec["openml_id"], download_data=True,
                                             download_qualities=False,
                                             download_features_meta_data=False)
        except TypeError:
            ds = openml.datasets.get_dataset(spec["openml_id"])
        target = spec["target"] or ds.default_target_attribute
        X, y, _, _ = ds.get_data(target=target)
        X = pd.DataFrame(X).reset_index(drop=True)
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "string":
                X[col] = X[col].astype("category")
            elif X[col].dtype == bool:
                X[col] = X[col].astype(np.int8)
        y = pd.Series(y).reset_index(drop=True)
        classes = sorted(pd.unique(y.astype(str)))
        X["__target__"] = y.astype(str).values
        X.to_parquet(f, index=False)
        meta = dict(name=name, openml_id=spec["openml_id"], openml_version=int(ds.version),
                    target=target, rows=int(len(X)), features=int(X.shape[1] - 1),
                    categorical_features=int(sum(str(X[c].dtype) == "category" for c in X.columns[:-1])),
                    classes=classes, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        mf.write_text(json.dumps(meta, indent=2))
        out.append(meta)
        reg = data_dir / "registry.json"
        specs = json.loads(reg.read_text()) if reg.exists() else {}
        specs[name] = spec
        reg.write_text(json.dumps(specs, indent=1))
        print(f"downloaded {name}: {meta['rows']} rows, {meta['features']} features, {len(classes)} classes")
    return pd.DataFrame(out)


def _spec(name, data_dir):
    """Dataset spec from REAL_DATASETS, or from data_dir/registry.json (written by prepare_openml),
    so datasets added later need no code change in worker processes."""
    if name in REAL_DATASETS:
        return REAL_DATASETS[name]
    reg = Path(data_dir) / "registry.json"
    if reg.exists():
        specs = json.loads(reg.read_text())
        if name in specs:
            return specs[name]
    raise KeyError(f"unknown dataset {name!r}: add it to REAL_DATASETS or {reg}")


def load_real(name, data_dir):
    spec = _spec(name, data_dir)
    f = Path(data_dir) / f"openml_{spec['openml_id']}.parquet"
    meta = json.loads(f.with_suffix(".json").read_text())
    df = pd.read_parquet(f)
    y_raw = df.pop("__target__").astype(str)
    y = pd.Categorical(y_raw, categories=meta["classes"]).codes.astype(np.int64)
    return df, y, len(meta["classes"]), meta


def make_real(name, seed, data_dir, n_train=2000, n_val=2000, n_test=5000):
    """Stratified random split: n_train and n_val rows, test = min(n_test, remaining rows).
    MNIST: train and validation from the original training images, test from the original test images."""
    from sklearn.model_selection import train_test_split
    X, y, K, meta = load_real(name, data_dir)
    rng, _ = make_rngs(f"openml|{meta['openml_id']}|ntr={n_train}|nval={n_val}|nte={n_test}", seed)
    rs = lambda: int(rng.integers(0, 2**31 - 1))
    idx = np.arange(len(y))

    def take(pool, size):  # stratified sample of `size` rows from `pool`; returns (sample, rest)
        if size >= len(pool):
            return pool, pool[:0]
        return train_test_split(pool, train_size=size, stratify=y[pool], random_state=rs())

    start = _spec(name, data_dir).get("test_pool_start")
    if start is not None:  # MNIST: original train images for train/val, original test images for test
        trva, _ = take(idx[:start], n_train + n_val)
        te, _ = take(idx[start:], n_test)
    else:  # train and validation first; test = min(n_test, remaining rows)
        trva, rest = take(idx, n_train + n_val)
        te, _ = take(rest, n_test)
    tr, va = take(trva, n_train)
    tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    sub = lambda ii: X.iloc[ii].reset_index(drop=True)
    return dict(X_train=sub(tr), y_train=y[tr], X_val=sub(va), y_val=y[va], X_test=sub(te),
                y_test=y[te], num_class=K,
                info=dict(openml_id=meta["openml_id"], openml_version=meta["openml_version"],
                          data_sha256=meta["sha256"], train_rows=tr.tolist(), val_rows=va.tolist(),
                          test_rows=te.tolist(), chance=float(np.max(np.bincount(y)) / len(y))))


# ----------------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------------
def make_config(task, data, seed, setting, depth, eta, noise=0.0, rounds=None, extra_xgb=None):
    return dict(stage="stage1a", task=task, data=data, noise=float(noise), seed=int(seed),
                setting=setting, max_depth=int(depth), eta=float(eta),
                rounds=int(rounds or round(LEARNING_TIME / eta)), extra_xgb=extra_xgb or {})


def run_id(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def stage1a_configs():
    """The meeting run (23 Nov 2026). Every config is also a cell of the full Stage 1 grid."""
    C = []
    # 1. Sparse parity (grokking): d=20, k=3, depth 3, eta 0.1, 20 seeds.
    for n in (100, 300, 1000):
        for s in range(20):
            for st in ("default", "brakes_off", "brakes_off_w1e3"):
                C.append(make_config("parity", dict(n=n, d=20, k=3), s, st, depth=3, eta=0.1))
    # 2. Thresholded Friedman (double descent): 0% and 20% flipped, depth 6, eta 0.3, 20 seeds.
    for noise in (0.0, 0.2):
        for s in range(20):
            for st in ("default", "brakes_off"):
                C.append(make_config("friedman", dict(n=2000, d=20), s, st, 6, 0.3, noise))
    # 2b. make_classification(flip_y=0): 20% flipped, 10 seeds.
    for s in range(10):
        for st in ("default", "brakes_off"):
            C.append(make_config("skl_classification", dict(n=2000, d=20), s, st, 6, 0.3, 0.2))
    # 3. Modular addition (grokking / feature learning): p=31, half the pairs, 10 seeds.
    for enc in ("onehot", "integers", "categorical", "sincos", "aplusb"):
        for s in range(10):
            for st in ("default", "brakes_off"):
                extra = dict(max_cat_to_onehot=1) if enc == "categorical" else None
                C.append(make_config("modadd", dict(p=31, train_frac=0.5, encoding=enc), s, st,
                                     6, 0.3, extra_xgb=extra))
    # 4. Real tabular datasets (double descent; late improvement): 0% and 20%, 10 seeds.
    for name in ("heloc", "E-CommereShippingData", "in_vehicle_coupon_recommendation",
                 "NATICUSdroid", "SDSS17"):
        for noise in (0.0, 0.2):
            for s in range(10):
                for st in ("default", "brakes_off"):
                    C.append(make_config("openml", dict(name=name, n_train=2000, n_val=2000,
                                                        n_test=5000), s, st, 6, 0.3, noise))
    # 5. MNIST: 20% flipped only tonight (0% control later), 5 seeds. Heaviest; runs last.
    for s in range(5):
        for st in ("default", "brakes_off"):
            C.append(make_config("openml", dict(name="mnist_784", n_train=2000, n_val=2000,
                                                n_test=5000), s, st, 6, 0.3, 0.2))
    return C


def is_heavy(cfg):
    return cfg["task"] == "openml" and cfg["data"]["name"] == "mnist_784"


# ----------------------------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------------------------
def build_data(cfg, data_dir=None):
    t, d, s = cfg["task"], cfg["data"], cfg["seed"]
    if t == "parity":
        D = make_parity(d["n"], d["d"], d["k"], s)
        key = f"parity|d={d['d']}|k={d['k']}|n={d['n']}"
    elif t == "friedman":
        D = make_friedman(d["n"], d["d"], s)
        key = f"friedman|d={d['d']}|n={d['n']}"
    elif t == "skl_classification":
        D = make_sklearn_classification(d["n"], d["d"], s)
        key = f"skl_classification|d={d['d']}|n={d['n']}"
    elif t == "modadd":
        D = make_modadd(d["p"], d["train_frac"], d["encoding"], s)
        key = f"modadd|p={d['p']}|frac={d['train_frac']}"
    elif t == "openml":
        D = make_real(d["name"], s, data_dir, d["n_train"], d["n_val"], d["n_test"])
        key = f"openml|{D['info']['openml_id']}|ntr={d['n_train']}|nval={d['n_val']}|nte={d['n_test']}"
    else:
        raise ValueError(t)
    # Label noise on training and validation labels; test labels are never flipped.
    _, noise_rng = make_rngs(key, s)
    K = D["num_class"]
    tr_plan = flip_plan(len(D["y_train"]), K, noise_rng)
    va_plan = flip_plan(len(D["y_val"]), K, noise_rng)
    D["y_train_obs"], D["flip_train"] = apply_flips(D["y_train"], cfg["noise"], K, tr_plan)
    D["y_val_obs"], D["flip_val"] = apply_flips(D["y_val"], cfg["noise"], K, va_plan)
    return D


def _dmatrix(X, y, weight=None, nthread=1):
    import xgboost as xgb
    cat = isinstance(X, pd.DataFrame) and any(str(t) == "category" for t in X.dtypes)
    return xgb.DMatrix(X, label=y, weight=weight, enable_categorical=cat, nthread=nthread)


def _subset(X, idx):
    return X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[idx]


def xgb_params(cfg, num_class, nthread):
    st = SETTINGS[cfg["setting"]]
    p = dict(tree_method="hist", max_bin=256, grow_policy="depthwise", max_depth=cfg["max_depth"],
             eta=cfg["eta"], min_child_weight=st["min_child_weight"], reg_lambda=st["reg_lambda"],
             reg_alpha=0.0, gamma=0.0, max_delta_step=0.0, subsample=1.0, colsample_bytree=1.0,
             colsample_bylevel=1.0, colsample_bynode=1.0, base_score=0.5, nthread=nthread, seed=0)
    if num_class == 2:
        p.update(objective="binary:logistic", eval_metric=["error", "logloss"])
    else:
        p.update(objective="multi:softprob", num_class=num_class, eval_metric=["merror", "mlogloss"])
    p.update(cfg.get("extra_xgb") or {})
    return p


def tree_stats(booster, num_class, hidden=None):
    """Per-round tree statistics. Freeze = rounds whose trees have no split at all.
    For parity, found = some root-to-leaf path splits on every hidden bit."""
    df = booster.trees_to_dataframe()
    n_rounds = booster.num_boosted_rounds()
    n_trees = n_rounds * (num_class if num_class > 2 else 1)
    split = df[df.Feature != "Leaf"]
    leaf = df[df.Feature == "Leaf"]
    tree_ids = np.arange(n_trees)
    n_splits = split.groupby("Tree").size().reindex(tree_ids, fill_value=0).to_numpy()
    root = df[(df.Node == 0) & (df.Feature != "Leaf")].set_index("Tree").Gain
    root_gain = root.reindex(tree_ids, fill_value=0.0).to_numpy()
    max_gain = split.groupby("Tree").Gain.max().reindex(tree_ids, fill_value=0.0).to_numpy()
    max_leaf = leaf.assign(a=leaf.Gain.abs()).groupby("Tree").a.max().reindex(tree_ids, fill_value=0.0).to_numpy()
    found = np.zeros(n_trees, dtype=bool)
    if hidden is not None:
        need = {f"f{h}" for h in hidden}
        cand = split.groupby("Tree").Feature.unique()
        for t, feats in cand.items():
            if not need <= set(feats):
                continue
            nodes = df[df.Tree == t].set_index("ID")
            stack = [(f"{t}-0", frozenset())]
            while stack:
                nid, path = stack.pop()
                row = nodes.loc[nid]
                if row.Feature == "Leaf":
                    if need <= path:
                        found[t] = True
                        break
                    continue
                path = path | {row.Feature}
                stack += [(row.Yes, path), (row.No, path)]
    per = max(num_class, 1) if num_class > 2 else 1
    r = tree_ids // per
    out = pd.DataFrame(dict(round=r + 1, n_splits=n_splits, root_gain=root_gain,
                            max_gain=max_gain, max_abs_leaf=max_leaf, found=found))
    return out.groupby("round").agg(n_splits=("n_splits", "sum"), root_gain=("root_gain", "max"),
                                    max_gain=("max_gain", "max"), max_abs_leaf=("max_abs_leaf", "max"),
                                    found=("found", "any")).reset_index()


def run_one(cfg, out_dir, data_dir=None, nthread=1):
    """Train one config to the round cap and save everything. Safe to call twice: a finished run
    (DONE file present) is skipped; an unfinished one is redone from scratch."""
    import xgboost as xgb
    rid = run_id(cfg)
    rdir = Path(out_dir) / "runs" / rid
    if (rdir / "DONE").exists():
        return rid, "skipped"
    rdir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        D = build_data(cfg, data_dir)
        K = D["num_class"]
        w = np.full(len(D["y_train"]), SETTINGS[cfg["setting"]]["weight_scale"], dtype=np.float32)
        dtrain = _dmatrix(D["X_train"], D["y_train_obs"], w, nthread)
        flipped = D["flip_train"]
        kept = np.setdiff1d(np.arange(len(D["y_train"])), flipped)
        evals = [(dtrain, "train"),
                 (_dmatrix(D["X_train"], D["y_train"], nthread=nthread), "train_clean"),
                 (_dmatrix(_subset(D["X_train"], kept), D["y_train_obs"][kept], nthread=nthread), "train_kept")]
        if len(flipped):
            evals.append((_dmatrix(_subset(D["X_train"], flipped), D["y_train_obs"][flipped], nthread=nthread),
                          "train_flipped"))
        evals += [(_dmatrix(D["X_val"], D["y_val_obs"], nthread=nthread), "val_noisy"),
                  (_dmatrix(D["X_val"], D["y_val"], nthread=nthread), "val_clean"),
                  (_dmatrix(D["X_test"], D["y_test"], nthread=nthread), "test")]
        res = {}
        booster = xgb.train(xgb_params(cfg, K, nthread), dtrain, num_boost_round=cfg["rounds"],
                            evals=evals, evals_result=res, verbose_eval=False)
        train_time = time.time() - t0
        curves = {}
        for set_name, metrics in res.items():
            for m, vals in metrics.items():
                curves[f"{set_name}_{m.replace('merror', 'error').replace('mlogloss', 'logloss')}"] = np.asarray(vals, dtype=np.float32)
        curves = pd.DataFrame(curves)
        curves.insert(0, "round", np.arange(1, len(curves) + 1))
        curves.to_parquet(rdir / "curves.parquet", index=False)
        tree_stats(booster, K, D["info"].get("hidden") if cfg["task"] == "parity" else None).to_parquet(
            rdir / "trees.parquet", index=False)
        with gzip.open(rdir / "model.json.gz", "wb") as fh:
            fh.write(booster.save_raw(raw_format="json"))
        info = dict(D["info"])
        idx = {k: np.asarray(info.pop(k)) for k in list(info) if k.endswith("_rows") or k.endswith("_pairs")}
        np.savez_compressed(rdir / "indices.npz", flip_train=flipped, flip_val=D["flip_val"], **idx)
        meta = dict(run_id=rid, config=cfg, info=info, num_class=K,
                    n_train=int(len(D["y_train"])), n_val=int(len(D["y_val"])), n_test=int(len(D["y_test"])),
                    n_flipped_train=int(len(flipped)),
                    data_sha=_array_sha(D["X_train"], D["y_train_obs"], D["X_test"], D["y_test"]),
                    xgb_params=xgb_params(cfg, K, nthread), xgboost=xgb.__version__,
                    numpy=np.__version__, pandas=pd.__version__, python=platform.python_version(),
                    code_version=CODE_VERSION, host=platform.node(), nthread=nthread,
                    train_seconds=round(train_time, 2), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        (rdir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        (rdir / "DONE").write_text(meta["finished"])
        if (rdir / "ERROR.txt").exists():
            (rdir / "ERROR.txt").unlink()
        return rid, "done"
    except Exception:
        (rdir / "ERROR.txt").write_text(traceback.format_exc() + "\n" + json.dumps(cfg))
        return rid, "error"


def run_all(configs, out_dir, data_dir=None, n_jobs=None):
    """Light runs in parallel (one thread each, deterministic); heavy runs (MNIST) one at a time
    with all cores. Re-running this after a disconnect skips finished runs."""
    from joblib import Parallel, delayed
    n_jobs = n_jobs or os.cpu_count()
    todo = [c for c in configs if not (Path(out_dir) / "runs" / run_id(c) / "DONE").exists()]
    light = [c for c in todo if not is_heavy(c)]
    heavy = [c for c in todo if is_heavy(c)]
    print(f"{len(configs) - len(todo)} runs already done; {len(light)} light and {len(heavy)} heavy to run "
          f"with {n_jobs} parallel jobs")
    status = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_one)(c, out_dir, data_dir, 1) for c in light)
    for i, c in enumerate(heavy):
        status.append(run_one(c, out_dir, data_dir, nthread=os.cpu_count()))
        print(f"heavy run {i + 1}/{len(heavy)}: {status[-1]}")
    errors = [r for r, s in status if s == "error"]
    print(f"finished: {sum(s == 'done' for _, s in status)} done, {len(errors)} errors")
    for r in errors[:5]:
        print(f"--- {r}\n" + (Path(out_dir) / "runs" / r / "ERROR.txt").read_text()[-1500:])
    return status


# ----------------------------------------------------------------------------------------------
# Detectors (first look; they read only stored curves, so they can be changed later)
# ----------------------------------------------------------------------------------------------
def _first(mask):
    i = np.flatnonzero(mask)
    return int(i[0]) + 1 if len(i) else None


def detect_grokking(train_acc, test_acc, a_star, window=50):
    """Grokking: test accuracy reaches a* (and its median over the next 50 rounds stays >= a*)
    at least 2x as late as, and >= 100 rounds after, training accuracy reaches 99.9%."""
    t_mem = _first(train_acc >= 0.999)
    t_reach = None
    for t in np.flatnonzero(test_acc >= a_star):
        if np.median(test_acc[t:t + window]) >= a_star:
            t_reach = int(t) + 1
            break
    grok = bool(t_mem and t_reach and t_reach >= 2 * t_mem and t_reach - t_mem >= 100)
    return dict(t_mem=t_mem, t_reach=t_reach, grok=grok)


def detect_double_descent(err, n_test=None, floor=0.01, z=3.0, window=11):
    """Double descent over rounds on a test curve (smoothed by an 11-round moving median):
    a rise of at least delta above the best value so far, then a fall of at least delta from
    the peak. Strong: the second minimum is below the first. delta = max(floor, z * SE of the
    error at the best value so far); for log loss (n_test=None) delta = floor."""
    e = pd.Series(err).rolling(window, center=True, min_periods=1).median().to_numpy()
    m = np.minimum.accumulate(e)
    if n_test:
        delta = np.maximum(floor, z * np.sqrt(np.clip(m * (1 - m), 0, None) / n_test))
    else:
        delta = np.full(len(e), floor)
    rises = np.flatnonzero(e - m >= delta)
    out = dict(dd=False, strong_dd=False, round_first_min=int(np.argmin(e)) + 1, round_peak=None,
               round_second_min=None, rise=None, drop=None)
    if not len(rises):
        return out
    t_rise = rises[0]
    d = delta[t_rise]
    right_min = np.minimum.accumulate(e[::-1])[::-1]
    cand = np.flatnonzero(e[t_rise:] - right_min[t_rise:] >= d) + t_rise
    t_first = int(np.argmin(e[:t_rise + 1]))
    out.update(round_first_min=t_first + 1, rise=float(e[t_rise] - m[t_rise]))
    if not len(cand):
        out.update(round_peak=int(t_rise + np.argmax(e[t_rise:])) + 1)
        return out
    t2 = int(cand[np.argmax(e[cand])])
    t3 = int(t2 + np.argmin(e[t2:]))
    out.update(dd=True, strong_dd=bool(e[t3] < m[t2]), round_peak=t2 + 1, round_second_min=t3 + 1,
               drop=float(e[t2] - e[t3]))
    return out


GROK_TASKS = ("parity", "modadd")


def analyze_run(rdir):
    rdir = Path(rdir)
    meta = json.loads((rdir / "meta.json").read_text())
    cfg = meta["config"]
    c = pd.read_parquet(rdir / "curves.parquet")
    tr = pd.read_parquet(rdir / "trees.parquet")
    train_acc, test_acc = 1 - c.train_error.to_numpy(), 1 - c.test_error.to_numpy()
    row = dict(run_id=meta["run_id"], task=cfg["task"], seed=cfg["seed"], setting=cfg["setting"],
               noise=cfg["noise"], max_depth=cfg["max_depth"], eta=cfg["eta"], rounds=cfg["rounds"],
               **{f"data_{k}": v for k, v in cfg["data"].items()},
               train_seconds=meta["train_seconds"])
    row["group"] = group_label(cfg)
    row["final_test_acc"] = float(test_acc[-1])
    row["best_test_acc"] = float(test_acc.max())
    row["round_best_test_acc"] = int(np.argmax(test_acc)) + 1
    row["best_test_logloss"] = float(c.test_logloss.min())
    row["round_best_test_logloss"] = int(np.argmin(c.test_logloss.to_numpy())) + 1
    has_split = tr.n_splits.to_numpy() > 0
    row["last_round_with_split"] = int(tr["round"].to_numpy()[has_split].max()) if has_split.any() else 0
    t_mem = _first(train_acc >= 0.999)
    row["t_mem"] = t_mem
    if "train_flipped_error" in c:
        row["t_mem_flipped"] = _first(c.train_flipped_error.to_numpy() <= 0.001)
    if t_mem:  # late improvement after memorization (used on real data instead of strict grokking)
        row["test_acc_at_mem"] = float(test_acc[t_mem - 1])
        row["late_gain"] = float(test_acc[t_mem - 1:].max() - test_acc[t_mem - 1])
    if cfg["task"] in GROK_TASKS:
        chance, ceiling = meta["info"]["chance"], meta["info"]["ceiling"]
        a_star = chance + 0.8 * (ceiling - chance)
        row["a_star"] = a_star
        row.update(detect_grokking(train_acc, test_acc, a_star))
    if cfg["task"] == "parity":
        f = tr.found.to_numpy()
        row["t_find"] = int(tr["round"].to_numpy()[f][0]) if f.any() else None
    if cfg["task"] == "modadd":
        row["memorizer_test_acc"] = meta["info"]["memorizer_test_acc"]
    dd_e = detect_double_descent(c.test_error.to_numpy(), n_test=meta["n_test"])
    row.update({f"err_{k}": v for k, v in dd_e.items()})
    dd_l = detect_double_descent(c.test_logloss.to_numpy(), n_test=None)
    row.update({f"ll_{k}": v for k, v in dd_l.items()})
    return row


def group_label(cfg):
    d = cfg["data"]
    if cfg["task"] == "parity":
        return f"parity n={d['n']}"
    if cfg["task"] == "modadd":
        return f"modadd {d['encoding']}"
    if cfg["task"] == "openml":
        return f"{d['name']} noise={int(cfg['noise'] * 100)}%"
    return f"{cfg['task']} noise={int(cfg['noise'] * 100)}%"


def summarize(out_dir, save_dir=None):
    """One row per finished run, plus a table per group and setting."""
    rows = [analyze_run(p.parent) for p in sorted((Path(out_dir) / "runs").glob("*/DONE"))]
    runs = pd.DataFrame(rows)
    agg = dict(runs=("run_id", "size"), median_best_test_acc=("best_test_acc", "median"),
               median_final_test_acc=("final_test_acc", "median"),
               median_round_best=("round_best_test_acc", "median"), median_t_mem=("t_mem", "median"),
               err_double_descent=("err_dd", "sum"), err_strong_dd=("err_strong_dd", "sum"),
               logloss_double_descent=("ll_dd", "sum"),
               median_last_split_round=("last_round_with_split", "median"))
    if "grok" in runs:
        agg["grokking"] = ("grok", lambda s: int(np.nansum(s.astype(float))))
    if "late_gain" in runs:
        agg["median_late_gain"] = ("late_gain", "median")
    table = runs.groupby(["task", "group", "setting"]).agg(**agg).reset_index()
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        runs.to_csv(save_dir / "stage1a_runs.csv", index=False)
        table.to_csv(save_dir / "stage1a_table.csv", index=False)
    return runs, table


def plot_groups(runs, out_dir, fig_dir, metric="acc"):
    """One figure per task: one panel per group; median (line) and 25-75% band over seeds of
    training accuracy (dashed) and test accuracy (solid), one colour per setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    colours = {"default": "#d95f02", "brakes_off": "#1b9e77", "brakes_off_w1e3": "#7570b3"}
    paths = []
    for task, tdf in runs.groupby("task"):
        groups = sorted(tdf.group.unique())
        ncol = min(3, len(groups))
        nrow = int(np.ceil(len(groups) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
        for ax, g in zip(axes.ravel(), groups):
            for st, sdf in tdf[tdf.group == g].groupby("setting"):
                curves = [pd.read_parquet(Path(out_dir) / "runs" / r / "curves.parquet") for r in sdf.run_id]
                x = curves[0]["round"].to_numpy()
                for col, ls in (("train_error", "--"), ("test_error", "-")):
                    Y = 1 - np.stack([cv[col].to_numpy() for cv in curves])
                    q25, q50, q75 = np.percentile(Y, [25, 50, 75], axis=0)
                    ax.plot(x, q50, ls, color=colours.get(st, "k"), lw=1.4,
                            label=f"{st} {'train' if 'train' in col else 'test'}")
                    ax.fill_between(x, q25, q75, color=colours.get(st, "k"), alpha=0.15, lw=0)
            ax.set_xscale("log")
            ax.set_title(f"{g}  ({len(tdf[tdf.group == g]) // max(1, tdf[tdf.group == g].setting.nunique())} seeds)", fontsize=10)
            ax.set_xlabel("boosting round")
            ax.set_ylabel("accuracy")
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{task}: median and 25-75% band over seeds (dashed: training labels used, solid: test)", fontsize=11)
        fig.tight_layout()
        p = fig_dir / f"stage1a_{task}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths
