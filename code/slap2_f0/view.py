"""Browser viewer of ``slap2-f0 predict`` output.

``slap2-f0 view DIR`` serves a page on http://127.0.0.1:8765 that shows, for
one trial at a time, the movie, the movie minus GIAnT's activity, and each
model's baseline and residual, frame by frame (scroll or play through time,
average a few frames, zoom and pan, all panels in step), with traces of a
chosen pixel or small region over the whole trial below.

Trials are read from every predict file under DIR; files for the same trial
are merged (models in file order, a later file's copy of a model replacing an
earlier one). A trial's arrays are held in memory while it is shown
(``--cache_trials`` trials at once), as float16: about 7 x frames x valid
pixels x 2 bytes, 1.1 GB for 2254 frames x 36k pixels.

API (all GET, used by the page):

    /api/meta                         trials and their models (JSON)
    /api/static?key=K                 valid pixels (rows, cols), their mean
                                      image, labeled and support flags,
                                      discarded frames (JSON)
    /api/frame?key=K&t=T&avg=A        float32 (layers, N): the mean over A
                                      frames from T - (A - 1) // 2 of each
                                      layer in ``layers`` (movie, activity,
                                      then the models), NaN-aware
    /api/trace?key=K&r=R&c=C&rad=D    float32 (layers, frames): the mean over
                                      valid pixels within D rows and columns
                                      of (R, C); header X-Pixels the count
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np

PAGE = Path(__file__).with_name("viewer.html")


def _names(v) -> list[str]:
    """A string-list attribute as a list (h5py returns one string for one item)."""
    v = _attr(v) if v is not None else []
    return [v] if isinstance(v, str) else list(v or [])


def _attr(v):
    a = np.asarray(v)
    if a.dtype.kind in "SO":
        out = a.astype(str)
        return out.item() if out.ndim == 0 else out.tolist()
    return a.item() if a.size == 1 else a.tolist()


class Trial:
    """One trial's arrays (valid pixels only), merged from its files."""

    def __init__(self, files: list[Path]):
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
        self.index = np.full((self.R, self.C), -1, np.int64)
        self.index[self.rows, self.cols] = np.arange(self.N)
        self.layers = ["movie", "activity"] + list(self.baselines)

    def _arrays(self):
        return [self.movie, self.activity] + list(self.baselines.values())

    def frame(self, t: int, avg: int) -> np.ndarray:
        lo = min(max(t - (avg - 1) // 2, 0), self.T - 1)
        hi = min(max(lo + avg, lo + 1), self.T)
        out = np.empty((len(self.layers), self.N), np.float32)
        for i, a in enumerate(self._arrays()):
            blk = a[lo:hi].astype(np.float32)
            with np.errstate(invalid="ignore"), _quiet():
                out[i] = blk[0] if hi - lo == 1 else np.nanmean(blk, 0)
        return out

    def trace(self, r: int, c: int, rad: int) -> tuple[np.ndarray, int]:
        sub = self.index[max(r - rad, 0): r + rad + 1, max(c - rad, 0): c + rad + 1]
        k = sub[sub >= 0]
        out = np.full((len(self.layers), self.T), np.nan, np.float32)
        if len(k):
            for i, a in enumerate(self._arrays()):
                with _quiet():
                    out[i] = np.nanmean(a[:, k].astype(np.float32), 1)
        return out, int(len(k))

    def static(self) -> dict:
        return {"rows": self.rows.tolist(), "cols": self.cols.tolist(),
                "mean": np.round(self.mean_image[self.rows, self.cols].astype(float), 3).tolist(),
                "labeled": self.labeled[self.rows, self.cols].astype(int).tolist(),
                "support": self.support[self.rows, self.cols].astype(int).tolist(),
                "discard": np.nonzero(self.discard)[0].tolist(), "layers": self.layers}


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


class Store:
    """The predict files under a folder, grouped by trial; trials loaded on demand."""

    def __init__(self, root: str | Path, cache_trials: int = 2):
        self.root = Path(root)
        self.cache_trials = max(1, cache_trials)
        groups: dict[str, list[tuple[float, Path, dict]]] = {}
        for fn in sorted(self.root.rglob("*.h5")):
            try:
                with h5py.File(fn, "r") as f:
                    if "slap2_f0_predictions" not in f.attrs:
                        continue
                    a = {k: _attr(v) for k, v in f.attrs.items()}
                    a["has_movie"] = "movie" in f
                    a["n_valid"] = int(f["rows"].shape[0])
            except OSError:
                continue
            key = "/".join(x for x in (a.get("session", ""), a.get("path", ""), a.get("trial", "")) if x)
            groups.setdefault(key, []).append((fn.stat().st_mtime, fn, a))
        self.files: dict[str, list[Path]] = {}
        self.meta: list[dict] = []
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
            self.meta.append({"key": key, "session": a0.get("session", ""), "path": a0.get("path", ""),
                              "trial": a0.get("trial", ""), "n_frames": a0["n_frames"], "n_rows": a0["n_rows"],
                              "n_cols": a0["n_cols"], "n_valid": a0["n_valid"], "frame_rate": a0.get("frame_rate", 80.0),
                              "scale": a0.get("scale"), "gain": a0.get("gain"), "n_sources": a0.get("n_sources", 0),
                              "has_movie": any(a["has_movie"] for _, _, a in g), "models": models})
        self._cache: OrderedDict[str, Trial] = OrderedDict()
        self._lock = threading.Lock()

    def trial(self, key: str) -> Trial:
        with self._lock:  # one load at a time; others wait for it
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if key not in self.files:
                raise KeyError(key)
            t0 = time.time()
            tr = Trial(self.files[key])
            print(f"loaded {key}: {tr.T} frames x {tr.N} px, {len(tr.baselines)} baselines in {time.time() - t0:.1f} s",
                  flush=True)
            self._cache[key] = tr
            while len(self._cache) > self.cache_trials:
                self._cache.popitem(last=False)
            return tr


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
                elif u.path == "/api/frame":
                    tr = store.trial(q["key"])
                    t = min(max(int(q.get("t", 0)), 0), tr.T - 1)
                    a = tr.frame(t, max(1, int(q.get("avg", 1))))
                    self._send(200, a.tobytes(), "application/octet-stream",
                               {"X-Layers": json.dumps(tr.layers), "X-Frame": str(t)})
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


def make_server(root: str | Path, host: str = "127.0.0.1", port: int = 8765, cache_trials: int = 2):
    store = Store(root, cache_trials)
    return ThreadingHTTPServer((host, port), make_handler(store)), store


def serve(args) -> None:
    server, store = make_server(args.root, args.host, args.port, args.cache_trials)
    if not store.meta:
        raise SystemExit(f"no predict output under {args.root} (see slap2-f0 predict)")
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"{len(store.meta)} trials under {args.root}; serving {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
