"""Load and stats of predict-registered, default against --fast_read /
--valid_stats, as interleaved A/B pairs on one HDF5 trial-path (GIAnT-Python
registration, as the stage capsule reads it), wall and CPU seconds.

usage: time_load.py MOCO STEM [PAIRS] [OUT.jsonl]  -> prints JSON lines and appends them to OUT
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from slap2_f0 import registered as R


def once(movie_path, adata, fast):
    """One load + ArrayTrial + movie_stats; seconds (wall, CPU) per step."""
    t0, c0 = time.perf_counter(), time.process_time()
    movie, valid, discard, varfac, hz = R.load_registered(movie_path, adata, fast_read=fast)
    t1, c1 = time.perf_counter(), time.process_time()
    tr = R.ArrayTrial(movie, valid, discard, varfac, valid_only=fast)
    st = R.movie_stats(movie, valid, discard, valid_only=fast)
    t2, c2 = time.perf_counter(), time.process_time()
    return {"fast": fast, "load_s": t1 - t0, "load_cpu_s": c1 - c0, "stats_s": t2 - t1, "stats_cpu_s": c2 - c1,
            "mean_sum": float(np.nansum(tr.mean_image)), "gain": float(st["gain"])}


def main():
    moco, stem = Path(sys.argv[1]), sys.argv[2]
    pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    mp = [m for m in R.registered_movies(moco) if m.name.startswith(stem + "_")][0]
    adata = moco / f"{stem}_ALIGNMENTDATA.h5"
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("time_load.jsonl")
    for k in range(pairs):
        for fast in ((False, True) if k % 2 == 0 else (True, False)):
            r = once(mp, adata, fast)
            r.update(movie=mp.name, pair=k)
            print(json.dumps(r), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
