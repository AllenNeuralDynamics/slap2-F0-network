"""Browser viewer of SLAP2 F0 training data and of ``slap2-f0 predict`` output.

``slap2-f0 view DIR`` serves a page on http://127.0.0.1:8765 that shows one
trial at a time, frame by frame (scroll or play through time, average a few
frames, zoom and pan, all panels in step), with traces of a chosen pixel or
small region over the whole trial below. It shows one of two kinds of trials:

Predict output (files with the ``slap2_f0_predictions`` attribute): the
movie, the movie minus GIAnT's activity, and each model's baseline and
residual. Baselines are blanked (NaN) on discarded frames, where the movie
has no data and no network was ever trained, so frame averages and smoothed
traces near them use real frames only. Files for the same trial are merged
(models in file order, a later file's copy of a model replacing an earlier
one; ``--models`` puts the named ones first, in that order).

Exported training trials (``io.TrialFile`` layout), when DIR holds no
predict output (or with ``--data export``), and with ``--codeocean`` the
export data assets on Code Ocean (found by search, downloaded on demand into
``--cache_dir``): the movie, GIAnT's fit (its activity), the training target
movie - activity, and the target minus its mean over the trial (what
changes over time: noise, slow drift, and any activity GIAnT left in). What
the training loss ignores (``data.WindowDataset``, real windows) can be
blanked in the target: labeled pixels outside every footprint (weight
``--w_unfitted``, 0 as in training, except in trials without sources), GIAnT
fit artifacts (``stats.unfit_pixels``, computed from the trial while it
loads), frames within ``--edge_s`` (2 s) of the trial's first and last
frame, and discarded frames. Each trial shows its calibration QC and whether
training uses it: the rules of ``io.filter_trials`` (over the session's
``export_summary.json`` when there is one, else over the trials found), and
the session's role (train or test) in ``sessions/export_phase*.csv``.

A trial's arrays are held in memory while it is shown (``--cache_trials``
trials at once, counting the session's next exported trial, which loads in
the background): predict output as float16, about 7 x frames x valid pixels x 2
bytes (1.1 GB for 2254 frames x 36k pixels), an exported trial as float32, 2
x frames x valid pixels x 4 bytes (0.2-0.7 GB).

API (all GET, used by the page):

    /api/meta                         trials, their models or QC (JSON)
    /api/static?key=K                 valid pixels (rows, cols), their mean
                                      image, labeled and support flags,
                                      discarded frames; for exported trials
                                      also why each pixel has no loss
                                      (``loss``: 0 in the loss, 1 labeled
                                      outside footprints, 2 fit artifact),
                                      the edge frames without loss, and the
                                      target's trial mean (JSON)
    /api/frame?key=K&t=T&avg=A        float32 (layers, N): the mean over A
                                      frames from T - (A - 1) // 2 of each
                                      layer in ``layers`` (movie, activity,
                                      then the models; for exported trials
                                      movie, activity, target), NaN-aware
    /api/trace?key=K&r=R&c=C&rad=D    float32 (layers, frames): the mean over
                                      valid pixels within D rows and columns
                                      of (R, C); header X-Pixels the count
    /api/progress?key=K               what loading the trial is doing (JSON)
"""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np

from .io import _path_id, _scalar, filter_records, read_mean_image
from .stats import unfit_pixels

PAGE = Path(__file__).with_name("viewer.html")


def _names(v) -> list[str]:
    """A string-list attribute as a list (h5py returns one string for one item)."""
    v = _attr(v) if v is not None else []
    return [v] if isinstance(v, str) else list(v or [])


def _ordered(names: list[str], order: list[str] | None) -> list[str]:
    """The names in ``order`` first (those present), then the rest as they were."""
    order = [m for m in (order or []) if m in names]
    return order + [m for m in names if m not in order]


def _attr(v):
    a = np.asarray(v)
    if a.dtype.kind in "SO":
        out = a.astype(str)
        return out.item() if out.ndim == 0 else out.tolist()
    return a.item() if a.size == 1 else a.tolist()


class _Frames:
    """Frame means and region traces over ``_arrays()`` ((frame, pixel) each)."""

    def _arrays(self) -> list[np.ndarray]:
        raise NotImplementedError

    def _blocks(self, idx) -> list[np.ndarray]:
        """Each layer's values at ``idx`` (frames, pixels)."""
        return [a[idx] for a in self._arrays()]

    def _index(self):
        self.index = np.full((self.R, self.C), -1, np.int64)
        self.index[self.rows, self.cols] = np.arange(self.N)

    def frame(self, t: int, avg: int) -> np.ndarray:
        lo = min(max(t - (avg - 1) // 2, 0), self.T - 1)
        hi = min(max(lo + avg, lo + 1), self.T)
        out = np.empty((len(self.layers), self.N), np.float32)
        for i, blk in enumerate(self._blocks(np.s_[lo:hi])):
            blk = blk.astype(np.float32)
            with np.errstate(invalid="ignore"), _quiet():
                out[i] = blk[0] if hi - lo == 1 else np.nanmean(blk, 0)
        return out

    def trace(self, r: int, c: int, rad: int) -> tuple[np.ndarray, int]:
        sub = self.index[max(r - rad, 0): r + rad + 1, max(c - rad, 0): c + rad + 1]
        k = sub[sub >= 0]
        out = np.full((len(self.layers), self.T), np.nan, np.float32)
        if len(k):
            for i, blk in enumerate(self._blocks(np.s_[:, k])):
                with _quiet():
                    out[i] = np.nanmean(blk.astype(np.float32), 1)
        return out, int(len(k))

    def _static(self) -> dict:
        return {"rows": self.rows.tolist(), "cols": self.cols.tolist(),
                "mean": np.round(self.mean_image[self.rows, self.cols].astype(float), 3).tolist(),
                "labeled": self.labeled[self.rows, self.cols].astype(int).tolist(),
                "support": self.support[self.rows, self.cols].astype(int).tolist(),
                "discard": np.nonzero(self.discard)[0].tolist(), "layers": self.layers,
                "shape": [int(self.T), int(self.R), int(self.C)]}


class Trial(_Frames):
    """One trial's predict output (valid pixels only), merged from its files."""

    def __init__(self, files: list[Path], order: list[str] | None = None):
        self.baselines: dict[str, np.ndarray] = {}
        self.movie = self.activity = None
        for fn in files:
            with h5py.File(fn, "r") as f:
                if self.movie is None and "movie" in f:
                    self.rows, self.cols = f["rows"][:], f["cols"][:]
                    self.R, self.C = int(f.attrs["n_rows"]), int(f.attrs["n_cols"])
                    self.mean_image = f["mean_image"][:]
                    self.labeled = f["mask/labeled"][:].astype(bool)
                    self.support = f["mask/footprint_support"][:].astype(bool)
                    self.discard = f["frame/discard"][:].astype(bool)
                    self.movie = f["movie"][:]
                    self.activity = f["activity"][:]
                for name in _names(f.attrs.get("models")):
                    if f"baseline/{name}" in f:
                        self.baselines.pop(name, None)
                        self.baselines[name] = f[f"baseline/{name}"][:]
        if self.movie is None:
            raise ValueError(f"no movie in {[str(f) for f in files]} (run predict without --no_movie once)")
        self.T, self.N = self.movie.shape
        # activity only where the movie has data, so frame means and traces of data - activity
        # average the same frames
        self.activity[np.isnan(self.movie)] = np.nan
        self.baselines = {m: self.baselines[m] for m in _ordered(list(self.baselines), order)}
        for b in self.baselines.values():  # no data, never trained on: a baseline there means nothing
            b[self.discard] = np.nan
        self._index()
        self.layers = ["movie", "activity"] + list(self.baselines)

    def _arrays(self):
        return [self.movie, self.activity] + list(self.baselines.values())

    def static(self) -> dict:
        return {**self._static(), "kind": "predict"}


LOSS_REASONS = ("in the loss", "labeled, outside every footprint", "GIAnT fit artifact")


class ExportTrial(_Frames):
    """One exported training trial: its movie and GIAnT's activity on the
    valid pixels, as ``io.TrialFile.read`` gives them to training (the
    movie NaN outside the valid mask and on discarded frames), and where the
    training loss of real windows looks (``data.WindowDataset``): ``loss``
    per pixel (0 in the loss; 1 labeled outside every footprint, weight
    ``w_unfitted``, unless the trial has no sources; 2 GIAnT fit artifact,
    ``stats.unfit_pixels`` on maps computed here as ``stats.trial_stats``
    computes them) and ``edge`` frames at each end of the trial."""

    # the target, movie - activity, is a layer of its own, averaged over the frames that have a
    # movie; the activity layer keeps GIAnT's fit everywhere, for its panel
    layers = ["movie", "activity", "target"]
    # the exporter's chunks are 256 frames x 64 x 64 pixels: each read decompresses whole chunks once,
    # in small enough pieces that the server keeps answering while a trial loads
    T_BLOCK, C_BLOCK = 256, 64

    def __init__(self, fn: str | Path, edge_s: float = 2.0, w_unfitted: float = 0.0, channel: int = 0,
                 progress=None):
        with h5py.File(fn, "r") as f:
            _, T, R, C = f["movie"].shape
            self.T, self.R, self.C = T, R, C
            self.attrs = {k: _scalar(v) for k, v in f.attrs.items()}
            self.mean_image = read_mean_image(f, channel).astype(np.float32)
            valid = f["mask/valid"][:].astype(bool)
            self.labeled = f["mask/labeled"][:].astype(bool) & valid
            self.support = f["mask/footprint_support"][:].astype(bool) & valid
            self.discard = f["frame/discard"][:].astype(bool)
            self.rows, self.cols = np.nonzero(valid)
            self.N = N = len(self.rows)
            self._index()
            sv = np.isfinite(self.mean_image[self.rows, self.cols])  # stats' valid pixels (of these)
            # float32, as training reads them: GIAnT's fit artifacts reach past float16's range (65504)
            self.movie = np.full((T, N), np.nan, np.float32)
            self.activity = np.zeros((T, N), np.float32)
            n, sm, sa, sa2 = (np.zeros(N) for _ in range(4))
            mmax = np.full(N, -np.inf, np.float32)
            anan = np.zeros(N, np.float32)
            for t0 in range(0, T, self.T_BLOCK):
                t1 = min(T, t0 + self.T_BLOCK)
                disc = self.discard[t0:t1]
                if progress:
                    progress(f"reading frames {t0}-{t1 - 1} of {T}")
                for c0 in range(0, C, self.C_BLOCK):
                    c1 = min(C, c0 + self.C_BLOCK)
                    sel = valid[:, c0:c1]
                    if not sel.any():
                        continue
                    k = self.index[:, c0:c1][sel]
                    m = f["movie"][channel, t0:t1, :, c0:c1][:, sel].astype(np.float32)
                    a = f["activity"][channel, t0:t1, :, c0:c1][:, sel].astype(np.float32)
                    m[disc] = np.nan
                    self.movie[t0:t1, k] = m
                    self.activity[t0:t1, k] = a
                    m[:, ~sv[k]] = np.nan  # the maps of stats.trial_stats
                    fin = np.isfinite(m)
                    m0, a0 = np.where(fin, m, 0), np.where(fin, a, 0)
                    n[k] += fin.sum(0)
                    sm[k] += m0.sum(0, dtype=np.float64)
                    sa[k] += a0.sum(0, dtype=np.float64)
                    sa2[k] += (a0 ** 2).sum(0, dtype=np.float64)
                    mmax[k] = np.maximum(mmax[k], np.where(fin, m, -np.inf).max(0))
                    miss = ~fin & sv[k][None] & ~disc[:, None]
                    anan[k] = np.maximum(anan[k], np.where(miss, np.nan_to_num(a), 0).max(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            maps = {"mean_movie": sm / n, "mean_act": sa / n, "mean_act2": sa2 / n, "n_valid": n,
                    "movie_max": np.where(n > 0, mmax, np.nan), "act_nan": anan}
            self.mean_target = ((sm - sa) / n).astype(np.float32)
        st = {}
        for name, v in maps.items():
            full = (np.zeros((R, C), np.float32) if name in ("n_valid", "act_nan")
                    else np.full((R, C), np.nan, np.float32))
            full[self.rows, self.cols] = v
            st[name] = full
        self.unfit = unfit_pixels(st)
        self.loss = np.zeros(N, np.uint8)
        if self.attrs.get("nSources", 1) and w_unfitted <= 0:
            self.loss[self.labeled[self.rows, self.cols] & ~self.support[self.rows, self.cols]] = 1
        self.loss[self.unfit[self.rows, self.cols]] = 2
        self.edge = int(round(edge_s * float(self.attrs.get("alignHz", 80.0)))) if edge_s > 0 else 0

    def _arrays(self):
        return [self.movie, self.activity]

    def _blocks(self, idx):
        m, a = self.movie[idx], self.activity[idx]
        return [m, a, m - a]

    def static(self) -> dict:
        mt = [None if not np.isfinite(v) else round(float(v), 3) for v in self.mean_target]
        return {**self._static(), "kind": "export", "loss": self.loss.tolist(), "edge_frames": self.edge,
                "loss_reasons": list(LOSS_REASONS), "mean_target": mt}


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


QC_FIELDS = ("peakR", "peakSlope", "activityGain", "blurSigma", "eventResidualFraction", "diskR", "diskSlope",
             "nEvents", "photonScale", "giantRef")
SESSION_LISTS = ("export_phase1.csv", "export_phase2.csv")


def session_roles(folder: str | Path | None = None) -> dict[str, str]:
    """Each exported session's role (train or test) in the repository's
    session lists (``sessions/export_phase*.csv``); empty without them."""
    folder = Path(folder) if folder else Path(__file__).resolve().parents[2] / "sessions"
    out: dict[str, str] = {}
    for name in SESSION_LISTS:
        p = folder / name
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    out[row["prefix"]] = row.get("role", "")
    return out


def _export_record(a: dict, session_folder: str, path_folder: str, stem: str) -> dict:
    """An exported trial's listing entry from its attributes (or its row of
    ``export_summary.json``, which holds the same fields)."""
    label = str(a.get("label") or session_folder)
    path = _path_id(a.get("path"), path_folder)
    digits = stem[len("trial_"):]
    trial = int(float(a["trial"])) if a.get("trial") is not None else (int(digits) if digits.isdigit() else 0)

    def num(k, d=None):
        return float(a[k]) if a.get(k) is not None else d

    return {"key": f"{label}/Path{path}/trial_{trial:03d}", "kind": "export", "session": label, "path": path,
            "trial": trial, "n_frames": int(num("frames", 0)), "n_rows": int(num("rows", 0)),
            "n_cols": int(num("cols", 0)), "n_valid": int(num("validPixels", 0)), "frame_rate": num("alignHz", 80.0),
            "scale": None, "gain": None, "n_sources": int(num("nSources", 0)), "has_movie": True, "models": [],
            "qc": {k: a[k] for k in QC_FIELDS if a.get(k) is not None}, "peak_r": num("peakR", float("nan"))}


class CodeOcean:
    """Exported trials in Code Ocean data assets, downloaded on demand.

    The assets are found by a search (``query``, names starting with
    ``name_prefix``), so no asset IDs live in this public code; each asset's
    ``export_summary.json`` lists its trials (cached in
    ``cache_dir``/_summaries, as assets do not change). Trials download into
    ``cache_dir``/<asset mount>/<session>/Path<N>/ in parallel ranges; the
    least recently used ones are deleted when the cache passes ``cache_gb``.
    The API token comes from the CODEOCEAN_API_TOKEN environment variable and
    the server from CODEOCEAN_DOMAIN (default the Allen Institute's)."""

    PART = 16 << 20

    def __init__(self, cache_dir: str | Path, query: str = "tag:slap2-F0", name_prefix: str = "slap2-F0 export phase",
                 cache_gb: float = 50.0, workers: int = 4):
        tok = os.environ.get("CODEOCEAN_API_TOKEN")
        if not tok:
            raise SystemExit("--codeocean needs the CODEOCEAN_API_TOKEN environment variable")
        self._auth = "Basic " + base64.b64encode((tok + ":").encode()).decode()
        self.base = os.environ.get("CODEOCEAN_DOMAIN", "https://codeocean.allenneuraldynamics.org").rstrip("/") + "/api/v1"
        self.cache_dir = Path(cache_dir)
        self.query, self.name_prefix = query, name_prefix
        self.cache_gb, self.workers = cache_gb, max(1, workers)
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._held: dict[Path, int] = {}  # files eviction must keep, with their number of holders

    def _api(self, path: str, data=None):
        req = urllib.request.Request(self.base + path, data=None if data is None else json.dumps(data).encode(),
                                     headers={"Authorization": self._auth, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def assets(self) -> list[dict]:
        out, offset = [], 0
        while True:
            r = self._api("/data_assets/search", {"offset": offset, "limit": 100, "query": self.query,
                                                   "archived": False})
            res = r.get("results", [])
            out += [a for a in res if str(a.get("name", "")).startswith(self.name_prefix)]
            offset += len(res)
            if not r.get("has_more") or not res:
                return sorted(out, key=lambda a: a["name"])

    def _summary(self, asset: dict) -> dict:
        p = self.cache_dir / "_summaries" / f"{asset['id']}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        folders = [i for i in self._api(f"/data_assets/{asset['id']}/files", {"path": ""}).get("items", [])
                   if i.get("type") == "folder" and i.get("path") != "logs"]
        for fo in folders:
            rel = f"{fo['path']}/export_summary.json"
            try:
                url = self._api(f"/data_assets/{asset['id']}/files/download_url?path={urllib.parse.quote(rel)}")["url"]
                summ = self._get_json(url)
            except urllib.error.HTTPError:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(summ), encoding="utf-8")
            return summ
        return {}

    def trials(self) -> list[dict]:
        """One record per trial of every asset: its listing entry (see
        ``_export_record``) plus where it lives remotely."""
        assets = self.assets()
        with ThreadPoolExecutor(8) as ex:
            summaries = list(ex.map(self._summary, assets))
        out = []
        for asset, summ in zip(assets, summaries):
            for a in summ.get("trials", []):
                if a.get("error"):  # a failed export: no file
                    continue
                f = str(a.get("file", ""))
                rec = _export_record(a, str(summ.get("label", "")), f"Path{a.get('path')}",
                                     Path(f).stem if f else f"trial_{int(a.get('trial', 0)):03d}")
                rel = (f.split("/results/", 1)[1] if "/results/" in f
                       else f"{rec['session']}/Path{rec['path']}/trial_{rec['trial']:03d}.h5")
                rec["remote"] = {"asset": asset["id"], "rel": rel, "mount": asset.get("mount") or asset["id"]}
                out.append(rec)
        return out

    def local_path(self, rec: dict) -> Path:
        return self.cache_dir / rec["remote"]["mount"] / rec["remote"]["rel"]

    def cached(self, rec: dict) -> bool:
        return self.local_path(rec).exists()

    def _lock(self, p: Path) -> threading.Lock:
        with self._locks_lock:
            return self._locks.setdefault(p, threading.Lock())

    def _get_json(self, url: str):
        with urllib.request.urlopen(url, timeout=120) as r:
            return json.load(r)

    def _range(self, url: str, lo: int, hi: int) -> tuple[bytes, int]:
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
            total = int(str(r.headers.get("Content-Range", "/0")).rsplit("/", 1)[-1] or 0)
        return data, total or len(data)

    def hold(self, p: Path):
        with self._locks_lock:
            self._held[p] = self._held.get(p, 0) + 1

    def release(self, p: Path):
        with self._locks_lock:
            n = self._held.get(p, 0) - 1
            if n > 0:
                self._held[p] = n
            else:
                self._held.pop(p, None)

    def fetch(self, rec: dict, progress=None, hold: bool = False) -> Path:
        """The trial's file in the cache, downloaded first if missing. With
        ``hold``, the file is kept from eviction until ``release`` (the cache
        may pass its size meanwhile), so it can be opened."""
        dst = self.local_path(rec)
        if hold:
            self.hold(dst)
        try:
            self._fetch(rec, dst, progress)
            self._evict(keep={dst})
        except BaseException:
            if hold:
                self.release(dst)
            raise
        return dst

    def _fetch(self, rec: dict, dst: Path, progress=None):
        with self._lock(dst):
            if dst.exists():
                os.utime(dst)  # recently used: evicted last
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            r = rec["remote"]
            url_path = f"/data_assets/{r['asset']}/files/download_url?path={urllib.parse.quote(r['rel'])}"
            url = [self._api(url_path)["url"]]
            t0 = time.time()
            first, total = self._range(url[0], 0, self.PART - 1)
            done = len(first)
            with open(part, "wb") as f:
                f.truncate(total)
                f.write(first)

            def get(o):
                for attempt in range(4):
                    try:
                        return o, self._range(url[0], o, min(o + self.PART, total) - 1)[0]
                    except (urllib.error.URLError, OSError):
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                        url[0] = self._api(url_path)["url"]  # presigned URLs expire

            with ThreadPoolExecutor(self.workers) as ex, open(part, "r+b") as f:
                for fut in as_completed([ex.submit(get, o) for o in range(self.PART, total, self.PART)]):
                    o, data = fut.result()
                    f.seek(o)
                    f.write(data)
                    done += len(data)
                    if progress:
                        progress(f"downloading {done / 1e6:.0f} of {total / 1e6:.0f} MB "
                                 f"({done / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s)")
            if done != total or part.stat().st_size != total:
                raise OSError(f"incomplete download of {r['rel']}: {done} of {total} bytes")
            os.replace(part, dst)

    def _evict(self, keep=()):
        with self._locks_lock:
            protected = set(self._held) | set(keep)
        files = [p for p in self.cache_dir.rglob("trial_*.h5") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        for p in sorted(files, key=lambda p: p.stat().st_mtime):
            if total <= self.cache_gb * 1e9:
                break
            if p in protected:
                continue
            total -= p.stat().st_size
            p.unlink()


class Store:
    """The trials to show, loaded on demand: predict output under a folder,
    or exported trials under it and/or in Code Ocean data assets."""

    def __init__(self, root: str | Path | None, cache_trials: int = 2, order: list[str] | None = None,
                 data: str = "auto", remote: CodeOcean | None = None, edge_s: float = 2.0,
                 w_unfitted: float = 0.0, roles: dict[str, str] | None = None,
                 min_peak_r: float = 0.3, min_path_trials: int = 4):
        self.root = Path(root) if root else None
        self.order = order
        self.cache_trials = max(1, cache_trials)
        self.remote, self.edge_s, self.w_unfitted = remote, edge_s, w_unfitted
        self.files: dict[str, list[Path]] = {}  # predict output
        self.exports: dict[str, dict] = {}  # exported trials: {"file": Path} or {"rec": remote record}
        self.meta: list[dict] = []
        self.progress: dict[str, str] = {}
        predict, export = self._scan() if self.root is not None else ({}, [])
        if data in ("auto", "predict"):
            self._add_predict(predict)
        recs = []
        if data == "export" or (data == "auto" and not predict):
            recs += export
        if remote is not None:
            have = {r["key"] for r in recs}
            for r in remote.trials():
                if r["key"] in have:  # a local copy wins
                    continue
                r["cached"] = remote.cached(r)
                recs.append(r)
        if recs:
            self._add_exports(recs, session_roles() if roles is None else roles, min_peak_r, min_path_trials)
        self.meta.sort(key=lambda m: m["key"])
        self._cache: OrderedDict[str, _Frames] = OrderedDict()
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Event] = {}

    def _scan(self):
        groups: dict[str, list[tuple[float, Path, dict]]] = {}
        export = []
        for fn in sorted(self.root.rglob("*.h5")):
            try:
                with h5py.File(fn, "r") as f:
                    if "slap2_f0_predictions" in f.attrs:
                        a = {k: _attr(v) for k, v in f.attrs.items()}
                        a["has_movie"] = "movie" in f
                        a["n_valid"] = int(f["rows"].shape[0])
                        key = "/".join(str(x) for x in (a.get("session", ""), a.get("path", ""), a.get("trial", ""))
                                       if x != "")
                        groups.setdefault(key, []).append((fn.stat().st_mtime, fn, a))
                    elif (re.fullmatch(r"trial_\d+", fn.stem) and "movie" in f and "activity" in f
                          and f["movie"].ndim == 4):
                        rec = _export_record({k: _scalar(v) for k, v in f.attrs.items()}, fn.parent.parent.name,
                                             fn.parent.name, fn.stem)
                        if not rec["n_frames"]:
                            _, rec["n_frames"], rec["n_rows"], rec["n_cols"] = f["movie"].shape
                        if not rec["n_valid"]:
                            rec["n_valid"] = int(np.count_nonzero(f["mask/valid"][:]))
                        rec["file"] = fn
                        export.append(rec)
            except OSError:
                continue
        return groups, export

    def _add_predict(self, groups):
        for key in sorted(groups):
            g = sorted(groups[key], key=lambda x: x[0])
            self.files[key] = [fn for _, fn, _ in g]
            models: list[str] = []
            for _, _, a in g:
                for m in _names(a.get("models")):
                    if m in models:
                        models.remove(m)
                    models.append(m)
            a0 = next((a for _, _, a in g if a["has_movie"]), g[0][2])
            self.meta.append({"key": key, "kind": "predict", "session": a0.get("session", ""),
                              "path": a0.get("path", ""), "trial": a0.get("trial", ""), "n_frames": a0["n_frames"],
                              "n_rows": a0["n_rows"], "n_cols": a0["n_cols"], "n_valid": a0["n_valid"],
                              "frame_rate": a0.get("frame_rate", 80.0), "scale": a0.get("scale"),
                              "gain": a0.get("gain"), "n_sources": a0.get("n_sources", 0),
                              "has_movie": any(a["has_movie"] for _, _, a in g),
                              "models": _ordered(models, self.order)})

    def _add_exports(self, recs, roles, min_peak_r, min_path_trials):
        # the filter's path rule counts a path's trials: over the session's export_summary.json when a
        # local session folder has one (the trials found may be a subset), else over the trials found
        rule_recs, owner, summarized = [], [], set()
        for r in recs:
            if "file" in r and r["session"] not in summarized:
                sp = Path(r["file"]).parent.parent / "export_summary.json"
                if sp.exists():
                    summarized.add(r["session"])
                    for a in json.loads(sp.read_text(encoding="utf-8")).get("trials", []):
                        if a.get("error"):  # a failed export: no file, not a trial training has
                            continue
                        x = _export_record(a, r["session"], f"Path{a.get('path')}",
                                           f"trial_{int(a.get('trial', 0)):03d}")
                        rule_recs.append((x["session"], x["path"], x["trial"], x["peak_r"]))
                        owner.append(x["key"])
        for r in recs:
            if r["session"] not in summarized:
                rule_recs.append((r["session"], r["path"], r["trial"], r["peak_r"]))
                owner.append(r["key"])
        reasons, _ = filter_records(rule_recs, min_peak_r, min_path_trials)
        why = dict(zip(owner, reasons))
        for r in recs:
            key = r["key"]
            if key in self.exports or key in self.files:  # the same trial twice: keep both, apart
                key = f"{key} ({r.get('remote', {}).get('mount', 'export')})"
            self.exports[key] = {"file": r["file"]} if "file" in r else {"rec": r}
            m = {k: v for k, v in r.items() if k not in ("file", "remote", "peak_r")}
            m.update(key=key, role=roles.get(r["session"], ""), left_out=why.get(r["key"]), remote="remote" in r)
            self.meta.append(m)

    def _meta(self, key: str) -> dict | None:
        return next((m for m in self.meta if m["key"] == key), None)

    def trial(self, key: str):
        """The trial's arrays, loaded if needed. A trial loads once: a second
        request for it (the page's, or a preload) waits for the first."""
        if key not in self.files and key not in self.exports:
            raise KeyError(key)
        while True:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                ev = self._loading.get(key)
                mine = ev is None
                if mine:
                    ev = self._loading[key] = threading.Event()
            if not mine:
                ev.wait()  # loaded (or failed, then tried again here)
                continue
            try:
                tr = self._load(key)
                with self._lock:
                    self._cache[key] = tr
                    while len(self._cache) > self.cache_trials:
                        _, old = self._cache.popitem(last=False)
                        if getattr(old, "held", False):
                            self.remote.release(old.file)
            finally:
                with self._lock:
                    self._loading.pop(key, None)
                ev.set()
            return tr

    def _load(self, key: str):
        t0 = time.time()

        def say(msg):
            self.progress[key] = msg

        try:
            if key in self.files:
                tr = Trial(self.files[key], self.order)
            else:
                e = self.exports[key]
                if "rec" in e:  # held in the cache while it loads and while it is shown
                    fn = self.remote.fetch(e["rec"], say, hold=True)
                    self._meta(key)["cached"] = True
                    try:
                        tr = ExportTrial(fn, self.edge_s, self.w_unfitted, progress=say)
                    except BaseException:
                        self.remote.release(fn)
                        raise
                    tr.held = True
                else:
                    fn = e["file"]
                    tr = ExportTrial(fn, self.edge_s, self.w_unfitted, progress=say)
                tr.file = fn
        finally:
            self.progress.pop(key, None)
        print(f"loaded {key}: {tr.T} frames x {tr.N} px in {time.time() - t0:.1f} s", flush=True)
        return tr

    def preload_after(self, key: str):
        """Load the session's next exported trial in the background (with
        room in the cache for it), so stepping through a session waits less."""
        keys = [m["key"] for m in self.meta]
        i = keys.index(key) if key in keys else -1
        if self.cache_trials < 2 or i < 0 or i + 1 >= len(keys):
            return
        nxt = self.meta[i + 1]["key"]
        if nxt not in self.exports or self.meta[i + 1]["session"] != self.meta[i]["session"]:
            return
        with self._lock:
            if nxt in self._cache or nxt in self._loading:
                return

        def run():
            try:
                self.trial(nxt)
            except Exception as ex:  # noqa: BLE001  (tried again when the trial is opened)
                print(f"preloading {nxt} failed: {ex}", flush=True)

        threading.Thread(target=run, daemon=True).start()


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/meta":
                    self._json({"root": str(store.root), "trials": store.meta})
                elif u.path == "/api/static":
                    self._json(store.trial(q["key"]).static())
                    store.preload_after(q["key"])
                elif u.path == "/api/frame":
                    tr = store.trial(q["key"])
                    t = min(max(int(q.get("t", 0)), 0), tr.T - 1)
                    a = tr.frame(t, max(1, int(q.get("avg", 1))))
                    self._send(200, a.tobytes(), "application/octet-stream",
                               {"X-Layers": json.dumps(tr.layers), "X-Frame": str(t)})
                elif u.path == "/api/progress":
                    self._json({"msg": store.progress.get(q.get("key", ""), "")})
                elif u.path == "/api/trace":
                    tr = store.trial(q["key"])
                    a, n = tr.trace(int(q["r"]), int(q["c"]), max(0, int(q.get("rad", 0))))
                    self._send(200, a.tobytes(), "application/octet-stream",
                               {"X-Layers": json.dumps(tr.layers), "X-Pixels": str(n)})
                else:
                    self._json({"error": "not found"}, 404)
            except KeyError as e:
                self._json({"error": f"missing or unknown {e}"}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # report to the page instead of dropping the connection
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def make_server(root: str | Path | None, host: str = "127.0.0.1", port: int = 8765, cache_trials: int = 2,
                order: list[str] | None = None, **store_args):
    store = Store(root, cache_trials, order, **store_args)
    return ThreadingHTTPServer((host, port), make_handler(store)), store


def add_arguments(pv) -> None:
    pv.add_argument("root", nargs="?", default=None,
                    help="folder with predict output or exported trials (any depth); optional with --codeocean")
    pv.add_argument("--data", choices=("auto", "predict", "export"), default="auto",
                    help="what to show from the folder: predict output, exported trials (the training data), or "
                         "auto: predict output if there is any, else exported trials")
    pv.add_argument("--codeocean", action="store_true",
                    help="also list the exported trials in Code Ocean data assets and download them on demand "
                         "(needs CODEOCEAN_API_TOKEN)")
    pv.add_argument("--co_query", default="tag:slap2-F0", help="Code Ocean data asset search")
    pv.add_argument("--co_name", default="slap2-F0 export phase",
                    help="keep the assets whose names start with this")
    pv.add_argument("--cache_dir", default=os.environ.get("SLAP2_F0_TRIAL_CACHE", "D:/f0_trial_cache"),
                    help="where downloaded trials are kept (default $SLAP2_F0_TRIAL_CACHE or D:/f0_trial_cache)")
    pv.add_argument("--cache_gb", type=float, default=50.0,
                    help="delete the least recently used downloaded trials beyond this size")
    pv.add_argument("--edge_s", type=float, default=2.0,
                    help="exported trials: seconds at each end of a trial without loss (training's --real_edge_s)")
    pv.add_argument("--w_unfitted", type=float, default=0.0,
                    help="exported trials: loss weight of labeled pixels outside every footprint (training's)")
    pv.add_argument("--min_peak_r", type=float, default=0.3, help="trial filter (training's --min_peak_r)")
    pv.add_argument("--min_path_trials", type=int, default=4, help="trial filter (training's --min_path_trials)")
    pv.add_argument("--port", type=int, default=8765)
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--no_browser", action="store_true", help="do not open a browser tab")
    pv.add_argument("--cache_trials", type=int, default=2, help="trials kept in memory")
    pv.add_argument("--models", default="", help="comma-separated models to list first, in this order (best first)")


def serve(args) -> None:
    order = [m.strip() for m in (getattr(args, "models", "") or "").split(",") if m.strip()]
    if args.root is None and not args.codeocean:
        raise SystemExit("give a folder, --codeocean, or both")
    remote = None
    if args.codeocean:
        remote = CodeOcean(args.cache_dir, args.co_query, args.co_name, args.cache_gb)
        print(f"listing exported trials on Code Ocean ({args.co_query}, names starting {args.co_name!r}); "
              f"downloads go to {args.cache_dir}", flush=True)
    server, store = make_server(args.root, args.host, args.port, args.cache_trials, order, data=args.data,
                                remote=remote, edge_s=args.edge_s, w_unfitted=args.w_unfitted,
                                min_peak_r=args.min_peak_r, min_path_trials=args.min_path_trials)
    if not store.meta:
        raise SystemExit(f"no predict output or exported trials under {args.root}"
                         + (" or on Code Ocean" if remote else ""))
    url = f"http://{args.host}:{server.server_address[1]}/"
    kinds = sorted({m["kind"] for m in store.meta})
    print(f"{len(store.meta)} trials ({', '.join(kinds)}); serving {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
