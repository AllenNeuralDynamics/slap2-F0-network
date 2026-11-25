"""slap2-f0 view on exported trials (the training data): listing, the trial
filter and session roles, where the loss looks, frames and traces, and
trials fetched on demand from Code Ocean data assets (with a fake API)."""
import json
import threading
import urllib.parse
import urllib.request
import warnings

import h5py
import numpy as np

from slap2_f0.io import TrialFile, _scalar
from slap2_f0.stats import load_stats
from slap2_f0.view import CodeOcean, ExportTrial, Store, make_server

T, R, C, HZ = 60, 12, 14, 10.0  # 10 Hz: the 2 s edges are 20 frames


def _trial(root, session, path, trial, peak_r=0.5, sources=True, seed=0, summary_label=True):
    """An exported trial with an unimaged corner, a discarded frame, a source
    footprint, labeled pixels outside it, and (with sources) a GIAnT fit
    artifact: activity far above the movie at one dim pixel."""
    rng = np.random.default_rng(seed + trial)
    d = root / session / f"Path{path}"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"trial_{trial:03d}.h5"
    movie = rng.poisson(8.0, (1, T, R, C)).astype(np.float32)
    act = np.zeros_like(movie)
    if sources:
        act[0, 30:36, 4:7, 4:7] = 6.0
        act[0, 40, 4:7, 4:7] = 9.0  # in the discarded frame, where the movie has no data
        act[0, :, 9, 11] = 5000.0  # fit artifact
    movie += np.where(act < 100, act, 0)
    valid = np.ones((R, C), np.uint8)
    valid[:3, :3] = 0
    discard = np.zeros(T, np.uint8)
    discard[40] = 1
    sup = np.zeros((R, C), np.uint8)
    sup[3:8, 3:8] = 1
    lab = sup.copy()
    lab[1, 8:12] = 1  # labeled, outside every footprint
    with h5py.File(fn, "w") as f:
        f["movie"] = movie
        f["activity"] = act
        f["mean_image"] = movie[0].mean(0)
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = valid
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = sup
        f["frame/discard"] = discard
        f.attrs["nSources"] = np.array([float(sources)])
        if not sources:  # as source-free exports write them
            f.attrs["eventResidualFraction"] = np.array([np.nan])
            f.attrs["eventRatioCore"] = np.array([np.nan])
        else:  # the QC of exports without a gain
            f.attrs["activityGain"] = np.array([1.0])
            f.attrs["eventRatioCore"] = np.array([1.25])
            f.attrs["eventRatio2px"] = np.array([0.9])
            f.attrs["eventRatioSources"] = np.array([3.0])
        f.attrs["alignHz"] = np.array([HZ])
        f.attrs["path"] = np.array([float(path)])
        f.attrs["trial"] = np.array([float(trial)])
        f.attrs["frames"] = np.array([float(T)])
        f.attrs["rows"] = np.array([float(R)])
        f.attrs["cols"] = np.array([float(C)])
        if summary_label:
            f.attrs["label"] = session
        if peak_r is not None:
            f.attrs["peakR"] = np.array([peak_r])
    return fn


def at_index(rows, cols, r, c):
    return int(np.nonzero((rows == r) & (cols == c))[0][0])


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read(), r.headers


def test_export_trials(tmp_path):
    data = tmp_path / "data"
    for t in range(1, 6):  # path 1: one failed calibration, 4 usable left
        _trial(data, "S1", 1, t, peak_r=0.1 if t == 2 else 0.5)
    for t in range(1, 4):  # path 2: 3 trials, too few
        _trial(data, "S1", 2, t)
    _trial(data, "S2", 1, 1, peak_r=None, sources=False)  # no calibration, no sources
    server, store = make_server(data, port=0, roles={"S1": "train", "S2": "test"})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = _get(base + "/api/meta")[0]
        assert b"NaN" not in body  # the browser's JSON.parse rejects it
        meta = {t["key"]: t for t in json.loads(body)["trials"]}
        assert meta["S2/Path1/trial_001"]["qc"]["eventResidualFraction"] is None
        assert meta["S2/Path1/trial_001"]["qc"]["eventRatioCore"] is None
        q1 = meta["S1/Path1/trial_001"]["qc"]
        assert (q1["activityGain"], q1["eventRatioCore"], q1["eventRatio2px"], q1["eventRatioSources"]) == (1.0, 1.25, 0.9, 3.0)
        assert sorted(meta) == [f"S1/Path1/trial_00{t}" for t in range(1, 6)] + \
            [f"S1/Path2/trial_00{t}" for t in range(1, 4)] + ["S2/Path1/trial_001"]
        assert all(m["kind"] == "export" and m["models"] == [] for m in meta.values())
        assert meta["S1/Path1/trial_002"]["left_out"].startswith("calibration failed")
        assert meta["S1/Path1/trial_001"]["left_out"] is None and meta["S1/Path1/trial_001"]["role"] == "train"
        assert meta["S1/Path2/trial_001"]["left_out"].startswith("path left out whole: 3 usable")
        assert meta["S2/Path1/trial_001"]["left_out"] is None and meta["S2/Path1/trial_001"]["role"] == "test"
        assert meta["S1/Path1/trial_001"]["qc"]["peakR"] == 0.5

        key = "S1/Path1/trial_001"
        fn = data / "S1" / "Path1" / "trial_001.h5"
        st = json.loads(_get(f"{base}/api/static?key={key}")[0])
        assert st["kind"] == "export" and st["layers"] == ["movie", "activity", "target"]
        assert st["shape"] == [T, R, C]
        assert st["edge_frames"] == 20 and st["discard"] == [40]
        rows, cols, loss = np.array(st["rows"]), np.array(st["cols"]), np.array(st["loss"])
        at = {(r, c): v for r, c, v in zip(rows, cols, loss)}
        # in the footprint, and labeled outside every footprint: both in the loss (training's default)
        assert at[(5, 5)] == 0 and at[(1, 8)] == 0 and at[(1, 11)] == 0
        assert st["labeled_outside"] == 4 and st["labeled_outside_in_loss"] is True
        assert at[(9, 11)] == 2 and at[(10, 12)] == 2  # artifact and its neighbours
        old = ExportTrial(fn, w_unfitted=0.0).static()  # as training up to S4: no loss there
        at0 = dict(zip(zip(old["rows"], old["cols"]), old["loss"]))
        assert at0[(5, 5)] == 0 and at0[(1, 8)] == 1 and at0[(1, 11)] == 1 and at0[(9, 11)] == 2
        assert old["labeled_outside"] == 4 and old["labeled_outside_in_loss"] is False
        unfit = load_stats([fn])[str(fn)]["unfit"]  # training's rule, from its own stats
        assert np.array_equal(loss == 2, unfit[rows, cols])
        with TrialFile(fn) as tf:
            p = tf.read(0, T, 0, R, 0, C)
        body, h = _get(f"{base}/api/frame?key={key}&t=10&avg=1")
        fr = np.frombuffer(body, np.float32).reshape(3, len(rows))
        assert json.loads(h["X-Layers"]) == ["movie", "activity", "target"]
        assert np.array_equal(fr[0], p.movie[10, rows, cols]) and np.array_equal(fr[1], p.activity[10, rows, cols])
        assert np.array_equal(fr[2], fr[0] - fr[1])
        body, _ = _get(f"{base}/api/frame?key={key}&t=40&avg=1")  # discarded: no movie, the fit stays
        fr = np.frombuffer(body, np.float32).reshape(3, len(rows))
        assert np.isnan(fr[0]).all() and np.isnan(fr[2]).all() and fr[1][at_index(rows, cols, 5, 5)] == 9
        # frames 39-41: the target averages frames 39 and 41 (40 has no movie), not mean(movie) - mean(activity)
        body, _ = _get(f"{base}/api/frame?key={key}&t=40&avg=3")
        fr = np.frombuffer(body, np.float32).reshape(3, len(rows))
        tgt = p.movie[[39, 41]][:, rows, cols] - p.activity[[39, 41]][:, rows, cols]
        assert np.allclose(fr[2], tgt.mean(0), rtol=1e-6)
        assert not np.allclose(fr[2], fr[0] - fr[1])
        # loss frames only: 8 frames at t=20 are 17-24, of which 20-24 are past the 2 s edge
        body, _ = _get(f"{base}/api/frame?key={key}&t=20&avg=8&loss=1")
        fr = np.frombuffer(body, np.float32).reshape(3, len(rows))
        tgt = p.movie[20:25][:, rows, cols] - p.activity[20:25][:, rows, cols]
        assert np.allclose(fr[2], tgt.mean(0), rtol=1e-6)
        body, _ = _get(f"{base}/api/frame?key={key}&t=5&avg=8&loss=1")  # all within the first 2 s
        fr = np.frombuffer(body, np.float32).reshape(3, len(rows))
        assert np.isnan(fr[2]).all() and np.isfinite(fr[0]).all()
        mt = np.array([np.nan if v is None else v for v in st["mean_target"]])
        tgt = p.movie[:, rows, cols] - p.activity[:, rows, cols]
        assert np.allclose(mt, np.nanmean(tgt, 0), rtol=1e-4)
        body, h = _get(f"{base}/api/trace?key={key}&r=5&c=5&rad=1")
        tr = np.frombuffer(body, np.float32).reshape(3, T)
        assert h["X-Pixels"] == "9"
        with warnings.catch_warnings():  # the discarded frame has no data
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmean(p.movie[:, 4:7, 4:7].reshape(T, -1), 1)
            ref_t = np.nanmean((p.movie - p.activity)[:, 4:7, 4:7].reshape(T, -1), 1)
        assert np.allclose(tr[0], ref, equal_nan=True) and np.allclose(tr[2], ref_t, equal_nan=True)
        assert np.isnan(tr[2][40]) and tr[1][40] == 9
        assert json.loads(_get(f"{base}/api/progress?key={key}")[0]) == {"msg": ""}

        st0 = json.loads(_get(f"{base}/api/static?key=S2/Path1/trial_001")[0])  # no sources: all of it counts
        assert set(st0["loss"]) == {0}
    finally:
        server.shutdown()
        server.server_close()


def test_path_rule_uses_the_export_summary(tmp_path):
    """Only 2 trials of a path copied locally, but the session's
    export_summary.json lists 5: the path is kept."""
    data = tmp_path / "data"
    for t in (1, 2):
        _trial(data, "S3", 1, t)
    summ = {"label": "S3", "trials": [{"label": "S3", "path": 1, "trial": t, "peakR": 0.5} for t in range(1, 6)]}
    (data / "S3" / "export_summary.json").write_text(json.dumps(summ))
    store = Store(data, roles={})
    assert [m["left_out"] for m in store.meta] == [None, None]
    # 3 usable trials plus a failed export (no file, no peak r): too few, as training sees them
    summ["trials"] = summ["trials"][:3] + [{"path": 1, "trial": 4, "error": "export failed"}]
    (data / "S3" / "export_summary.json").write_text(json.dumps(summ))
    assert Store(data, roles={}).meta[0]["left_out"].startswith("path left out whole: 3 usable")
    (data / "S3" / "export_summary.json").unlink()
    assert Store(data, roles={}).meta[0]["left_out"].startswith("path left out whole: 2 usable")


def test_predict_output_wins_in_auto_mode(tmp_path):
    """A folder with predict output shows it (not the exported trials next
    to it), unless --data export."""
    data = tmp_path / "data"
    _trial(data, "S1", 1, 1)
    with h5py.File(tmp_path / "pred.h5", "w") as f:
        f.attrs.update({"slap2_f0_predictions": 1, "session": "S1", "path": "Path1", "trial": "trial_001",
                        "n_frames": T, "n_rows": R, "n_cols": C})
        f["rows"] = np.zeros(1, np.int32)
    assert [m["kind"] for m in Store(tmp_path, roles={}).meta] == ["predict"]
    assert [m["kind"] for m in Store(tmp_path, data="export", roles={}).meta] == ["export"]


class _FakeCodeOcean(CodeOcean):
    """The Code Ocean API over files in memory: two assets of exported trials
    (one not an export, left out by name), downloads in many small ranges."""

    PART = 4096

    def __init__(self, cache_dir, blobs, monkeypatch, **kw):
        monkeypatch.setenv("CODEOCEAN_API_TOKEN", "test-token")
        super().__init__(cache_dir, **kw)
        self.blobs = blobs  # remote path -> bytes
        self.calls = []

    def _api(self, path, data=None):
        self.calls.append(path)
        if path == "/data_assets/search":
            return {"results": [{"id": "A1", "name": "slap2-F0 export phase 1 S1", "mount": "f0p1_S1"},
                                {"id": "B9", "name": "something else", "mount": "x"}], "has_more": False}
        if path.endswith("/files"):  # a session folder per summary, and the logs
            return {"items": [{"path": k.split("/")[0], "type": "folder"} for k in self.blobs
                              if k.endswith("/export_summary.json")] + [{"path": "logs", "type": "folder"}]}
        if "download_url" in path:
            return {"url": "mem://" + urllib.parse.unquote(path.split("path=", 1)[1])}
        raise AssertionError(path)

    def _get_json(self, url):
        return json.loads(self.blobs[url[len("mem://"):]])

    fail_first = 0  # this many first-range requests fail, as a timeout would

    def _range(self, url, lo, hi):
        if lo == 0 and self.fail_first > 0:
            self.fail_first -= 1
            raise TimeoutError("first range timed out")
        b = self.blobs[url[len("mem://"):]]
        return b[lo:hi + 1], len(b)


def test_codeocean_listing_download_and_cache(tmp_path, monkeypatch):
    src = tmp_path / "src"
    files = [_trial(src, "S1", 1, t, peak_r=0.1 if t == 3 else 0.5) for t in range(1, 6)]
    blobs = {f"S1/Path1/{f.name}": f.read_bytes() for f in files}
    summ = {"label": "S1", "trials": []}
    for f in files:
        with h5py.File(f, "r") as h:
            a = {k: _scalar(v) for k, v in h.attrs.items()}
        summ["trials"].append({**a, "file": f"/root/capsule/results/S1/Path1/{f.name}"})
    summ["trials"].append({"path": 1, "trial": 9, "error": "export failed"})  # no file: not listed
    summ["trials"].append({**summ["trials"][0], "trial": 7.0,  # would leave the cache: not listed
                           "file": "/root/capsule/results/../../../outside/trial_007.h5"})
    blobs["S1/export_summary.json"] = json.dumps(summ).encode()
    size = len(blobs["S1/Path1/trial_001.h5"])
    cache = tmp_path / "cache"
    co = _FakeCodeOcean(cache, blobs, monkeypatch, cache_gb=2.5 * size / 1e9)
    co.fail_first = 1  # the first download's first range fails once
    recs = co.trials()
    assert [r["key"] for r in recs] == [f"S1/Path1/trial_00{t}" for t in range(1, 6)]
    assert recs[0]["remote"] == {"asset": "A1", "rel": "S1/Path1/trial_001.h5", "mount": "f0p1_S1"}
    assert (cache / "_summaries" / "A1_all.json").exists()
    n = len(co.calls)
    co.trials()  # the summary comes from the cache now
    assert not any("download_url" in c for c in co.calls[n:])

    store = Store(None, remote=co, roles={"S1": "train"}, cache_trials=1)
    m = {x["key"]: x for x in store.meta}
    assert m["S1/Path1/trial_003"]["left_out"].startswith("calibration failed")
    assert m["S1/Path1/trial_001"]["remote"] and not m["S1/Path1/trial_001"]["cached"]
    msgs = []
    store.progress = type("P", (dict,), {"__setitem__": lambda s, k, v: (msgs.append(v), dict.__setitem__(s, k, v))})()
    tr = store.trial("S1/Path1/trial_001")
    got = cache / "slap2_f0_downloads" / "f0p1_S1" / "S1" / "Path1" / "trial_001.h5"
    assert got.read_bytes() == blobs["S1/Path1/trial_001.h5"]  # assembled from many ranges
    assert any(s.startswith("downloading") for s in msgs) and any(s.startswith("reading") for s in msgs)
    assert m["S1/Path1/trial_001"]["cached"] and isinstance(tr, ExportTrial) and tr.T == T
    assert not list(cache.rglob("*.part")) and co.fail_first == 0
    assert not (tmp_path / "outside").exists()
    # the cache holds 2.5 trials: fetching two more evicts the least recently used one not in use
    co.fetch(store.exports["S1/Path1/trial_002"]["rec"])
    co.fetch(store.exports["S1/Path1/trial_004"]["rec"])
    left = sorted(p.name for p in cache.rglob("trial_*.h5"))
    assert left == ["trial_001.h5", "trial_004.h5"]  # trial_001 is shown (in use), trial_002 went


def test_codeocean_small_cache_and_key_collisions(tmp_path, monkeypatch):
    """A trial larger than the whole cache is kept while it loads and while
    it is shown; an exported trial whose key a predict trial has stays
    reachable under its own key."""
    src = tmp_path / "src"
    f = _trial(src, "S1", 1, 1)
    blobs = {"S1/Path1/trial_001.h5": f.read_bytes()}
    with h5py.File(f, "r") as h:
        a = {k: _scalar(v) for k, v in h.attrs.items()}
    blobs["S1/export_summary.json"] = json.dumps({"label": "S1", "trials": [
        {**a, "file": "/root/capsule/results/S1/Path1/trial_001.h5"}]}).encode()
    pred = tmp_path / "pred"
    pred.mkdir()
    with h5py.File(pred / "p.h5", "w") as h:
        h.attrs.update({"slap2_f0_predictions": 1, "session": "S1", "path": "Path1", "trial": "trial_001",
                        "n_frames": T, "n_rows": R, "n_cols": C})
        h["rows"] = np.zeros(1, np.int32)
    mine = _trial(tmp_path / "cache", "S9", 1, 1)  # a trial of the user's own in the cache folder
    blobs["S2/export_summary.json"] = json.dumps({"label": "S2", "trials": [  # a second session in the asset
        {**a, "label": "S2", "file": "/root/capsule/results/S2/Path1/trial_001.h5"}]}).encode()
    co = _FakeCodeOcean(tmp_path / "cache", blobs, monkeypatch, cache_gb=1e-9)
    store = Store(pred, remote=co, roles={}, cache_trials=1)
    keys = [m["key"] for m in store.meta]
    assert keys == ["S1/Path1/trial_001", "S1/Path1/trial_001 (f0p1_S1)", "S2/Path1/trial_001"]
    assert [m["kind"] for m in store.meta] == ["predict", "export", "export"]
    tr = store.trial("S1/Path1/trial_001 (f0p1_S1)")
    assert isinstance(tr, ExportTrial) and tr.file.exists()  # not evicted before or after loading
    assert co._held == {tr.file: 1}
    co.release(tr.file)
    co._evict()
    assert not tr.file.exists()  # released: the over-full cache lets it go
    assert mine.exists()  # never a download: never deleted
