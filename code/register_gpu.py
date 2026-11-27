"""Register SLAP2 sessions with GIAnT-Python on the GPU, then predict baselines.

No MATLAB: for every attached SLAP2 session (``<asset>/slap2/dynamic_data``)

1. build its trial table (``giant_python.pipeline.build_trial_table_slap2``,
   GIAnT-MATLAB's buildTrialTableSLAP2), trimmed to ``--max_duration_s``;
2. register it (``register_multiroi``, GIAnT-MATLAB's MultiRoiRegistration)
   on the GPU when there is one;
3. if ``--ckpt`` is given, compute the network's localization baselines for
   the registered movies (``slap2-f0 predict-registered``).

Outputs per session in OUT/<session>: trial_table.h5 (with the
motion_correction group), trial_table_untrimmed.h5, motion_correction/
(alignment data, the summary-statistics sidecars ``<stem>_REGSTATS-80Hz.h5``
of GIAnT-Python versions that write them; the registered movies are deleted
unless ``--keep_registered 1``), baselines/ and run_info.json (GIAnT-Python
commit, trim, registration settings, timings, GPU, peak memory, sidecars).
OUT/summary.json lists every session; OUT/stage_asset.json says what the
folder is when it is made a data asset (``kind`` "registration").

With ``--keep_registered 1`` and no ``--ckpt``, OUT is the registration
asset of the staged Code Ocean path: ``code/run --mode predict_registered``
predicts the network's baselines from it, and a SILo run (GIAnT-Python's
``giant silo``) takes both. Make the asset from the computation's
``register`` folder, so that its sessions are at its top level.

Run by ``code/run --mode register`` with GIAnT-Python, SLAP2_Utils and
GIAnT-Python's pure-Python dependencies on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

STAGE_FORMAT = 1
"""Version of the registration asset's layout (stage_asset.json)."""


def _sessions(data: Path, datasets: str):
    """Attached SLAP2 sessions: (name, dynamic_data folder)."""
    wanted = None if datasets in ("", "all") else set(datasets.split(","))
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        dyn = d / "slap2" / "dynamic_data"
        if wanted is not None and d.name not in wanted:
            continue
        if dyn.is_dir() and any(dyn.glob("*.dat")):
            yield d.name, dyn


def _peak_rss_gb():
    """Peak resident memory of this process so far (Linux), or None."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _gpu():
    """The CUDA device's name and memory, or None."""
    import torch

    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "memory_gb": p.total_memory / 2**30}


def _dir_gb(path: Path, pattern: str) -> float:
    return sum(f.stat().st_size for f in path.glob(pattern)) / 2**30


def _stem(name: str) -> str:
    """A motion-correction file's trial stem (``E1T1DMD1``)."""
    return name.split("_", 1)[0]


def _n_trials(table: Path):
    """Trials (columns of ``filename``) of a trial table, or None."""
    import h5py

    try:
        with h5py.File(table, "r") as f:
            shape = f["filename"].shape
    except (OSError, KeyError):
        return None
    return int(max(shape)) if len(shape) else 1


def sidecars(moco: Path, fields: dict) -> dict:
    """The summary-statistics sidecars (``<stem>_REGSTATS-80Hz.h5``) of the
    registered trial-paths: count, size, and the trial-paths without one."""
    import numpy as np

    have = {_stem(f.name) for f in moco.glob("*_REGSTATS-*Hz.h5")}
    failed = np.asarray(fields["registration_failed"], bool)
    names = np.asarray(fields["fn_adata"], dtype=object).reshape(failed.shape)
    want = sorted(_stem(Path(str(n).replace("\\", "/")).name) for n, bad in zip(names.ravel(), failed.ravel())
                  if str(n) and not bad)
    return {"n": len(have), "gb": _dir_gb(moco, "*_REGSTATS-*Hz.h5"),
            "missing": [s for s in want if s not in have]}


def run_session(name: str, dyn: Path, out: Path, args, info: dict) -> None:
    """Trial table, registration and (optionally) baselines of one session.

    Fills ``info`` as it goes, so a failure keeps what was measured.
    """
    import torch

    from giant_python.models.params import (
        ExecutionOptions,
        MultiRoiRegistrationParams,
    )
    from giant_python.pipeline import build_trial_table_slap2, register_multiroi

    info.update(dat_dir=str(dyn), status="failed", max_duration_s=args.max_duration_s,
                compensate_online=bool(args.compensate_online), schedule=args.schedule,
                keep_registered=bool(args.keep_registered))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    build_trial_table_slap2(
        dyn, out, max_duration_s=args.max_duration_s or None
    )
    info["trial_table_s"] = time.time() - t0
    info["n_trials"] = _n_trials(out / "trial_table.h5")
    info["n_trials_untrimmed"] = _n_trials(out / "trial_table_untrimmed.h5")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # GIAnT-Python writes the statistics sidecars by default since #37
    # (80ba65a); older refs have no such option
    stats = {"stats": True} if "stats" in inspect.signature(register_multiroi).parameters else {}
    info["stats_requested"] = bool(stats)
    t0 = time.time()
    fields = register_multiroi(
        out / "trial_table.h5",
        MultiRoiRegistrationParams(
            compensate_online=bool(args.compensate_online),
            schedule=args.schedule,
        ),
        execution=ExecutionOptions(
            device=device, verbose=True, max_workers=os.cpu_count() or 1
        ),
        **stats,
    )
    info["registration_s"] = time.time() - t0
    info["device"] = device
    if torch.cuda.is_available():
        info["registration_peak_gpu_gb"] = (
            torch.cuda.max_memory_allocated() / 2**30
        )
        torch.cuda.empty_cache()
    failed = fields["registration_failed"]
    info["n_trial_paths"] = int(failed.size)
    info["registration_failed"] = failed.astype(bool).tolist()
    moco = out / "motion_correction"
    info["registered_movies_gb"] = _dir_gb(moco, "*_REGISTERED_*")
    info["alignment_data_gb"] = _dir_gb(moco, "*_ALIGNMENTDATA.h5")
    info["regstats"] = sidecars(moco, fields)
    if stats and info["regstats"]["missing"]:
        print("no statistics sidecar for", ", ".join(info["regstats"]["missing"]), flush=True)

    if args.ckpt:
        t0 = time.time()
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "slap2_f0.cli",
            "predict-registered",
            str(moco),
            "--ckpt",
            args.ckpt,
            "--out",
            str(out / "baselines"),
        ] + args.predict_args.split()
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        proc = subprocess.run(cmd, env=env)
        info["predict_s"] = time.time() - t0
        info["predict_exit"] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError(f"predict-registered exited {proc.returncode}")
        info["n_baselines"] = len(list((out / "baselines").glob("*.h5")))
    if not args.keep_registered:
        for f in moco.glob("*_REGISTERED_DOWNSAMPLED-*"):
            f.unlink()
    info["kept_movies"] = sorted(f.name for f in moco.glob("*_REGISTERED_DOWNSAMPLED-*"))
    info["status"] = "ok"


def stage_record(env: dict, args, results: list) -> dict:
    """OUT/stage_asset.json: what the folder holds as a data asset."""
    return {
        "kind": "registration",
        "format": STAGE_FORMAT,
        "layout": "<session>/trial_table.h5, trial_table_untrimmed.h5, run_info.json, motion_correction/ "
                  "(<stem>_ALIGNMENTDATA.h5, <stem>_REGISTERED_DOWNSAMPLED-80Hz.h5 when kept, "
                  "<stem>_REGSTATS-80Hz.h5)",
        "giant_python_commit": env["giant_python_commit"],
        "giant_python_ref": env["giant_python_ref"],
        "max_duration_s": args.max_duration_s,
        "compensate_online": bool(args.compensate_online),
        "schedule": args.schedule,
        "keep_registered": bool(args.keep_registered),
        "gpu": env["gpu"],
        "sessions": {r["session"]: r["status"] for r in results},
    }


def main(argv=None) -> int:
    """Entry point; returns 1 if any session failed."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--datasets", default="all")
    p.add_argument("--max_duration_s", type=float, default=180.0)
    p.add_argument("--compensate_online", type=int, default=0)
    p.add_argument("--schedule", default="sequential")
    p.add_argument("--ckpt", default="")
    p.add_argument("--predict_args", default="")
    p.add_argument("--keep_registered", type=int, default=0)
    args = p.parse_args(argv)

    import torch

    import giant_python

    env = {
        "giant_python_version": giant_python.__version__,
        "giant_python_commit": os.environ.get("GIANT_PYTHON_COMMIT", ""),
        "giant_python_ref": os.environ.get("GIANT_PYTHON_REF", ""),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": _gpu(),
        "cpu_count": os.cpu_count(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    print(json.dumps(env, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, dyn in _sessions(args.data, args.datasets):
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        info = {"session": name}
        try:
            run_session(name, dyn, args.out / name, args, info)
        except Exception:  # keep going with the other sessions
            traceback.print_exc()
            info.update(status="failed", error=traceback.format_exc())
        info["total_s"] = time.time() - t0
        info["peak_rss_gb_so_far"] = _peak_rss_gb()
        info.update({k: env[k] for k in ("giant_python_commit", "giant_python_ref", "torch", "gpu")})
        (args.out / name).mkdir(parents=True, exist_ok=True)
        (args.out / name / "run_info.json").write_text(json.dumps(info, indent=1))
        print(json.dumps({k: v for k, v in info.items() if k != "error"}, indent=1))
        results.append(info)
    summary = {"environment": env, "sessions": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "stage_asset.json").write_text(json.dumps(stage_record(env, args, results), indent=1))
    if not results:
        print("No SLAP2 sessions found under", args.data)
        return 1
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    sys.exit(main())
