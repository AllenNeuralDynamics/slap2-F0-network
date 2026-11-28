"""A/B of the prediction engines on one cached trial-path (prep_cache.py).

usage: bench_engines.py CACHE_DIR CKPT OUT_JSON VARIANT [VARIANT ...]
  VARIANT: tiles:<out_frames>[:bench]  |  stream:<chunk>[:<max_px>][:cl][:bench]
The first variant is the reference: the others' baselines are compared with
it (largest and 99.9th-percentile absolute difference in counts, and
relative), and with the stored float16 baseline file if --stored is given.
GPU utilisation is sampled with nvidia-smi every 0.5 s during each variant.
Times: network (GPU, synchronised) wall seconds per variant.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from slap2_f0 import predict as P
from slap2_f0.registered import ArrayTrial
from slap2_f0.stream import StreamPredictor


class GpuSampler:
    """nvidia-smi utilisation and memory every `dt` seconds in a thread."""

    def __init__(self, dt=0.5):
        self.dt, self.samples, self._stop = dt, [], threading.Event()

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    u, m = o.stdout.strip().split("\n")[0].split(",")
                    self.samples.append((time.time(), float(u), float(m)))
                except Exception:
                    pass
                self._stop.wait(self.dt)
        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()

    def summary(self):
        """Mean utilisation, share of samples at 0%, peak memory (MiB)."""
        if not self.samples:
            return {}
        u = np.array([s[1] for s in self.samples])
        return {"util_mean": float(u.mean()), "util_zero_share": float((u == 0).mean()),
                "util_below50_share": float((u < 50).mean()), "mem_peak_mib": float(max(s[2] for s in self.samples)),
                "n": len(u)}


def load_cache(d: Path):
    """ArrayTrial and stats from prep_cache.py's files."""
    movie = np.load(d / "movie.npy")
    valid, discard, varfac = (np.load(d / f"{k}.npy") for k in ("valid", "discard", "varfac"))
    meta = json.loads((d / "meta.json").read_text())
    tr = ArrayTrial(movie, valid, discard, varfac, valid_only=True)
    st = {"mean_movie": np.load(d / "mean_movie.npy"), "var_movie": np.load(d / "var_movie.npy"), "gain": meta["gain"]}
    return tr, st, movie, meta


def diff(a, b):
    """Largest and 99.9th-percentile absolute difference (counts) and relative."""
    d = np.abs(a - b)
    r = d / np.maximum(np.abs(b), 1e-6)
    return {"max_abs": float(np.nanmax(d)), "p999_abs": float(np.nanpercentile(d, 99.9)),
            "p99_abs": float(np.nanpercentile(d, 99)), "max_rel": float(np.nanmax(r)),
            "p999_rel": float(np.nanpercentile(r, 99.9)), "mean_signed": float(np.nanmean(a - b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--stored", default="", help="stored baseline file (float16) of this trial-path")
    ap.add_argument("--save_ref", default="", help="save / reuse the first variant's mu here (.npy)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tr, st, movie, meta = load_cache(Path(a.cache))
    rows, cols = P.valid_index(tr)
    model, wcfg, ck = P.load_checkpoint(a.ckpt, dev)
    amp = P.prepare(model, dev, "auto")
    rec = {"cache": a.cache, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "shape": list(movie.shape),
           "valid": int(len(rows)), "meta": meta, "runs": []}
    stored = None
    if a.stored:
        with h5py.File(a.stored, "r") as f:
            stored = f["baseline"][:].transpose(0, 2, 1)[:, rows, cols].astype(np.float32)
    ref = None
    if a.save_ref and Path(a.save_ref).exists():
        ref = np.load(a.save_ref)
    for v in a.variants:
        parts = v.split(":")
        kind = parts[0]
        bench = "bench" in parts
        torch.backends.cudnn.benchmark = bench
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        info = {"variant": v}
        with GpuSampler() as gs:
            torch.cuda.synchronize()
            t0 = time.time()
            if kind == "tiles":
                mu, var = P.predict_trial(model, wcfg, tr, st, dev, amp, int(parts[1]), None, 1, rows, cols)
            else:
                mp = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                sp = StreamPredictor(model, wcfg, dev, amp, chunk=int(parts[1]), max_px=mp,
                                     layout="cl" if "cl" in parts else "ncdhw")
                mu, var = sp.predict(tr, st, rows, cols, movie=movie, masked=True)
                info["stream"] = sp.timings
            torch.cuda.synchronize()
            info["network_s"] = time.time() - t0
        info["gpu"] = gs.summary()
        info["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        info["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        if ref is None:
            ref = mu
            if a.save_ref:
                np.save(a.save_ref, mu)
        else:
            info["vs_ref"] = diff(mu, ref)
            f16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
            info["vs_ref_f16"] = diff(f16(mu), f16(ref))
        if stored is not None:
            info["vs_stored_f16"] = diff(mu.astype(np.float16).astype(np.float32), stored)
        rec["runs"].append(info)
        print(json.dumps(info), flush=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
