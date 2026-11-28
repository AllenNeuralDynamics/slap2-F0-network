"""The network's baselines of registered sessions attached as data assets.

Run by ``code/run --mode predict_registered`` (GIAnT-Python on PYTHONPATH).
For every registered session found under DATA, a folder with
``trial_table.h5`` and ``motion_correction/*_ALIGNMENTDATA.h5`` (the results
of ``code/run --mode register --keep_registered 1``, attached as a data
asset), the baselines of its kept trial-paths are predicted by
GIAnT-Python's ``predict_baselines``: the step ``giant silo --f0-ckpt``
runs, with the same ``slap2-f0 predict-registered`` command line and SILo's
trial preparation (activity channel, nan_thresh, motion_thresh,
discard_initial_s: GIAnT-Python's ``SiloParams`` defaults unless given).
Neither GIAnT-MATLAB nor the raw recording is needed. Outputs::

    OUT/<session>/<stem>_BASELINE-80Hz.h5    slap2-f0's baseline file
    OUT/<session>/<stem>_BASELINE-80Hz.json  giant silo's record of it, with
                                             the SHA-256 of the baseline,
                                             movie and alignment files, so
                                             that it still identifies them
                                             once both are data assets
    OUT/<session>/predict_report.jsonl       slap2-f0's report per trial-path
    OUT/<session>/run_info.json              times per trial-path, GPU, peak
                                             memory, the registration used
    OUT/stage_asset.json                     what the folder is as an asset
                                             (kind "f0_baselines")
    OUT/summary.json                         every session

A SILo run takes them with ``giant silo TABLE --baseline-dir OUT/<session>``
and the same four settings (the record refuses others). Make the data
asset from the computation's ``baselines`` folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

STAGE_FORMAT = 1
"""Version of the baseline asset's layout (stage_asset.json)."""

MAX_DEPTH = 4
"""How deep below DATA sessions are looked for (mount/register/session)."""


def find_sessions(data: Path, max_depth: int = MAX_DEPTH) -> list:
    """Registered sessions under ``data``: ``(name, folder)`` pairs, a folder
    holding ``trial_table.h5`` and a ``motion_correction`` folder with
    alignment files; the name is the folder's.

    Raises
    ------
    ValueError
        If two sessions have the same name.
    """
    found = {}

    def walk(folder: Path, depth: int) -> None:
        table = folder / "trial_table.h5"
        moco = folder / "motion_correction"
        if table.is_file() and moco.is_dir() and any(moco.glob("*_ALIGNMENTDATA.h5")):
            if folder.name in found:
                raise ValueError(f"two registered sessions named {folder.name}: {found[folder.name]} and {folder}")
            found[folder.name] = folder
            return
        if depth >= max_depth:
            return
        try:
            subs = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for sub in subs:
            if sub.name != "motion_correction":
                walk(sub, depth + 1)

    walk(Path(data), 0)
    return sorted(found.items())


def parse_asset_ids(text: str) -> dict:
    """``mount=id,mount=id`` (as the launch scripts pass them) as a dict."""
    out = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--asset_ids takes mount=id pairs, not {item!r}")
        mount, aid = item.split("=", 1)
        out[mount.strip()] = aid.strip()
    return out


def mount_of(folder: Path, data: Path) -> str:
    """The data asset (first folder below ``data``) a session folder is in."""
    try:
        rel = Path(folder).relative_to(Path(data))
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def stage_records(folder: Path, data: Path) -> dict:
    """The registration stage's records of a session: its run_info.json and
    the stage_asset.json of the folder it was written in (if they exist)."""
    out = {}
    run_info = folder / "run_info.json"
    if run_info.is_file():
        out["run_info"] = json.loads(run_info.read_text())
    for parent in [folder.parent, *folder.parent.parents]:
        marker = parent / "stage_asset.json"
        if marker.is_file():
            out["stage_asset"] = json.loads(marker.read_text())
            break
        if parent.resolve() == Path(data).resolve():
            break
    return out


# predict-registered options that --f0_args may pass: speed only, so that the records (settings,
# checkpoint) still describe the prediction
SPEED_FLAGS = {"--fast", "--async_write", "--valid_stats", "--fast_read", "--cudnn_benchmark", "--compile"}
SPEED_VALUES = {"--engine", "--stream_frames", "--stream_px", "--stream_layout", "--prefetch", "--batch"}


def speed_args(extra: str) -> list:
    """The words of --f0_args, checked to be speed options only (flags, and
    options with one value, as ``--name value`` or ``--name=value``)."""
    words, out, i = shlex.split(extra), [], 0
    while i < len(words):
        w = words[i]
        name = w.split("=", 1)[0]
        if name in SPEED_FLAGS and "=" not in w:
            out.append(w)
        elif name in SPEED_VALUES:
            if "=" in w:
                out.append(w)
            elif i + 1 < len(words):
                out += [w, words[i + 1]]
                i += 1
            else:
                raise SystemExit(f"--f0_args: {w} needs a value")
        else:
            raise SystemExit(f"--f0_args takes speed options only ({', '.join(sorted(SPEED_FLAGS | SPEED_VALUES))}), "
                             f"not {w}: settings and checkpoint go through predict_registered's own options")
        i += 1
    return out


def write_wrapper(folder: Path, report: Path, extra: str = "") -> Path:
    """An executable that runs this repository's ``slap2-f0`` with this
    interpreter and adds ``--report`` and the options in ``extra`` (say,
    ``--engine stream --prefetch 1``; predict_baselines runs it)."""
    folder.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    words = speed_args(extra)
    if os.name == "nt":
        exe = folder / "slap2-f0.cmd"
        more = "".join(f' "{w}"' for w in words)
        exe.write_text(f'@set "PYTHONPATH={code};%PYTHONPATH%"\r\n'
                       f'@"{sys.executable}" -u -m slap2_f0.cli %* --report "{report}"{more}\r\n')
    else:
        exe = folder / "slap2-f0"
        more = "".join(f" {shlex.quote(w)}" for w in words)
        exe.write_text("#!/bin/sh\n"
                       f'export PYTHONPATH="{code}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
                       f'exec "{sys.executable}" -u -m slap2_f0.cli "$@" --report "{report}"{more}\n')
        exe.chmod(0o755)
    return exe


def gpu_info() -> dict | None:
    """The first GPU's name, memory and driver (nvidia-smi; no CUDA context
    in this process, so that the prediction finds the memory free)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, mem, driver = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
    return {"name": name, "memory_gb": float(mem) / 1024, "driver": driver}


def _children_peak_rss_gb() -> float | None:
    """Peak resident memory of the largest finished child (Linux)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20


def _git_commit(folder: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


PREPARATION = ("activity_channel", "nan_thresh", "motion_thresh", "discard_initial_s")
"""SILo's settings the prediction prepares a trial with (giant silo's
``prediction_settings``)."""


def silo_params(args) -> SimpleNamespace:
    """The trial preparation settings: those given, else ``SiloParams``'
    defaults (read from the class: building one would import the whole
    SILo stack). The prediction uses no other setting."""
    from giant_python.models.params import SiloParams

    given = {"activity_channel": args.channel, "nan_thresh": args.nan_thresh,
             "motion_thresh": args.motion_thresh, "discard_initial_s": args.discard_initial_s}
    fields = SiloParams.__dataclass_fields__
    return SimpleNamespace(**{n: fields[n].default if given[n] is None else given[n] for n in PREPARATION})


def select_units(session, stems: list) -> list:
    """Kept trial-paths ``(d, t)`` (zero-based) of the session, or those of
    the given stems (``E1T1DMD1``); each needs its registered movie.

    Raises
    ------
    ValueError
        For a stem that is not a kept trial-path, or a trial-path without
        its movie (a registration asset made with --keep_registered 0).
    """
    units, by_stem = [], {}
    for d in range(session.n_paths):
        for t in range(session.n_trials):
            if not session.keep[d, t]:
                continue
            name = Path(str(session.fn_adata[d, t]).replace("\\", "/")).name
            by_stem[name.split("_", 1)[0]] = (d, t)
    wanted = stems or sorted(by_stem, key=lambda s: by_stem[s])
    missing = [s for s in wanted if s not in by_stem]
    if missing:
        raise ValueError(f"not kept trial-paths of this session: {', '.join(missing)} (kept: {', '.join(by_stem)})")
    for s in wanted:
        d, t = by_stem[s]
        if session.movie(d, t) is None:
            raise ValueError(f"{s} has no registered movie in {session.moco_dir} (register with --keep_registered 1)")
        units.append(by_stem[s])
    return units


def _read_record(path: Path) -> dict | None:
    """A baseline record, or None when it is missing or unreadable (a write
    interrupted part-way leaves empty or truncated JSON)."""
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (ValueError, OSError):
        return None


def _complete(record: dict | None) -> bool:
    """Whether a record has the baseline, movie and alignment identities
    with their SHA-256 (what a SILo run on the assets checks)."""
    return bool(record) and all((record.get(k) or {}).get("sha256") for k in ("baseline", "movie", "alignment"))


def clear_incomplete(session, out: Path, units: list) -> list:
    """Remove the baselines of ``units`` whose record is missing or
    incomplete (an interrupted run leaves the file with a record without its
    identity; a file made elsewhere has none), so that they are predicted
    again: hashing such a file now could not show where it came from."""
    from giant_python.pipeline.silo_baseline import baseline_path, record_path

    removed = []
    for d, t in units:
        base = baseline_path(out, session.movie(d, t))
        rec = record_path(base)
        if not base.exists():
            continue
        record = _read_record(rec)
        if not _complete(record):
            base.unlink()
            if rec.exists():
                rec.unlink()
            removed.append(base.name)
    return removed


def check_records(session, out: Path, units: list) -> None:
    """Every baseline of ``units`` has a complete record that describes it.

    Raises
    ------
    RuntimeError
        Otherwise.
    """
    from giant_python.pipeline.silo_baseline import baseline_path, record_path, same_file

    bad = []
    for d, t in units:
        base = baseline_path(out, session.movie(d, t))
        rec = record_path(base)
        record = _read_record(rec)
        if not base.exists() or not _complete(record) or not same_file(record["baseline"], base):
            bad.append(base.name)
    if bad:
        raise RuntimeError("baselines without a complete record: " + ", ".join(bad))


def run_session(name: str, folder: Path, args, env: dict, info: dict) -> None:
    """Predict one session's baselines into OUT/<name>; fills ``info``."""
    from giant_python.pipeline.silo_baseline import predict_baselines, prediction_settings
    from giant_python.pipeline.silo_session import SiloSession

    out = args.out / name
    scratch = args.scratch / name
    params = silo_params(args)
    info.update(status="failed", folder=str(folder), settings=prediction_settings(params))
    mount = mount_of(folder, args.data)
    info["registration"] = {"mount": mount, "asset_id": env["asset_ids"].get(mount, ""),
                            **stage_records(folder, args.data)}
    session = SiloSession.open(folder / "trial_table.h5", moco_dir=folder / "motion_correction",
                               data_dir=folder, out_dir=scratch, work_dir=scratch / "silo", inputs="collect")
    stems = [s.strip() for s in args.trial_paths.split(",") if s.strip()]
    units = select_units(session, stems)
    info["trial_paths"] = [{"path": d + 1, "trial": t + 1, "movie": session.movie(d, t).name} for d, t in units]
    out.mkdir(parents=True, exist_ok=True)
    info["removed_incomplete"] = clear_incomplete(session, out, units)
    report = out / "predict_report.jsonl"
    if args.f0_command and args.f0_args:
        raise SystemExit("--f0_args goes with this repository's slap2-f0, not with --f0_command")
    exe = args.f0_command or str(write_wrapper(scratch / "bin", report, args.f0_args))
    t0 = time.time()
    rec = predict_baselines(session, params, out, args.ckpt, units=units, device=args.device, command=exe,
                            verbose=True, out_frames=args.out_frames or None, digests=True)
    check_records(session, out, units)
    info["total_s"] = time.time() - t0
    info["predict_s"] = rec["seconds"]
    info["records_s"] = info["total_s"] - rec["seconds"]  # checks and SHA-256 of the files
    info["checkpoint"] = dict(rec["checkpoint"], path=str(args.ckpt), label=args.ckpt_label)
    info["predicted"] = rec["predicted"]
    lines = [json.loads(x) for x in report.read_text().splitlines() if x.strip()] if report.exists() else []
    by_stem = {x["stem"]: x for x in lines if not x.get("skipped")}
    for tp in info["trial_paths"]:
        stem = tp["movie"].split("_", 1)[0]
        tp.update({k: v for k, v in by_stem.get(stem, {}).items() if k not in ("stem", "movie")})
        base = out / f"{stem}_BASELINE-80Hz.h5"
        tp["baseline_gb"] = base.stat().st_size / 2**30 if base.exists() else None
    info["baselines_gb"] = sum(tp["baseline_gb"] or 0 for tp in info["trial_paths"])
    peaks = [tp.get("peak_gpu_gb") for tp in info["trial_paths"] if tp.get("peak_gpu_gb") is not None]
    info["peak_gpu_gb"] = max(peaks) if peaks else None
    info["status"] = "ok"


def stage_record(env: dict, args, results: list) -> dict:
    """OUT/stage_asset.json: what the folder holds as a data asset."""
    ok = [r for r in results if r.get("checkpoint")]
    return {
        "kind": "f0_baselines",
        "format": STAGE_FORMAT,
        "layout": "<session>/<stem>_BASELINE-80Hz.h5 with its record <stem>_BASELINE-80Hz.json (giant silo's, "
                  "with SHA-256 digests), predict_report.jsonl, run_info.json",
        "checkpoint": ok[0]["checkpoint"] if ok else {"path": str(args.ckpt), "label": args.ckpt_label},
        "settings": results[0].get("settings") if results else None,
        "out_frames": args.out_frames,
        "giant_python_commit": env["giant_python_commit"],
        "slap2_f0_commit": env["slap2_f0_commit"],
        "gpu": env["gpu"],
        "registration_assets": env["asset_ids"],
        "sessions": {r["session"]: r["status"] for r in results},
    }


def main(argv=None) -> int:
    """Entry point; returns 1 if a session failed or none was found."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path, help="folder with the attached registration assets")
    p.add_argument("out", type=Path, help="where the baselines go (OUT/<session>/...)")
    p.add_argument("--ckpt", type=Path, required=True, help="the slap2-f0 checkpoint")
    p.add_argument("--ckpt_label", default="", help="the checkpoint's name in the asset names (e.g. s6_ft)")
    p.add_argument("--sessions", default="all", help="'all' or comma-separated session names")
    p.add_argument("--trial_paths", default="", help="comma-separated stems (E1T1DMD1,...); default all kept")
    p.add_argument("--out_frames", type=int, default=0, help="network tile size in frames (0: as fits the GPU)")
    p.add_argument("--channel", type=int, default=None, help="activity channel (default: SiloParams')")
    p.add_argument("--nan_thresh", type=float, default=None)
    p.add_argument("--motion_thresh", type=float, default=None)
    p.add_argument("--discard_initial_s", type=float, default=None)
    p.add_argument("--device", default="cuda", help="predict-registered's device (cuda, cpu, auto)")
    p.add_argument("--asset_ids", default="", help="mount=id,... of the attached registration assets (recorded)")
    p.add_argument("--f0_command", default="", help="a slap2-f0 executable (default: this repository's)")
    p.add_argument("--f0_args", default="", help="speed options for this repository's slap2-f0 predict-registered "
                                                 "slap2-f0, e.g. '--engine stream --prefetch 1 --async_write'")
    p.add_argument("--scratch", type=Path, default=None, help="working folder (default OUT/../scratch_predict)")
    args = p.parse_args(argv)
    args.scratch = args.scratch or args.out.parent / "scratch_predict"

    import giant_python

    try:  # the records' digests (GIAnT-Python #41, dev 346c9dc)
        from giant_python.pipeline.silo_baseline import same_file  # noqa: F401
    except ImportError as e:
        print(f"GIAnT-Python {os.environ.get('GIANT_PYTHON_REF', '')} lacks the baseline records with digests "
              f"(GIAnT-Python #41; dev 346c9dc or later): {e}")
        return 1
    env = {
        "giant_python_version": giant_python.__version__,
        "giant_python_commit": os.environ.get("GIANT_PYTHON_COMMIT", "") or _git_commit(
            Path(giant_python.__file__).resolve().parents[2]),
        "giant_python_ref": os.environ.get("GIANT_PYTHON_REF", ""),
        "slap2_f0_commit": os.environ.get("SLAP2_F0_COMMIT", "") or _git_commit(Path(__file__).resolve().parent),
        "gpu": gpu_info(),
        "cpu_count": os.cpu_count(),
        "asset_ids": parse_asset_ids(args.asset_ids),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    print(json.dumps(env, indent=1), flush=True)
    if not args.ckpt.is_file():
        print(f"no checkpoint {args.ckpt}")
        return 1
    wanted = None if args.sessions in ("", "all") else {s.strip() for s in args.sessions.split(",")}
    sessions = [(n, f) for n, f in find_sessions(args.data) if wanted is None or n in wanted]
    if wanted is not None and wanted - {n for n, _ in sessions}:
        print("sessions not found:", ", ".join(sorted(wanted - {n for n, _ in sessions})))
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, folder in sessions:
        print(f"\n===== {name} ({folder}) =====", flush=True)
        t0 = time.time()
        info = {"session": name}
        try:
            run_session(name, folder, args, env, info)
        except Exception:  # keep going with the other sessions
            traceback.print_exc()
            info.update(status="failed", error=traceback.format_exc())
        info["wall_s"] = time.time() - t0
        info["peak_child_rss_gb_so_far"] = _children_peak_rss_gb()
        info.update({k: env[k] for k in ("giant_python_commit", "giant_python_ref", "slap2_f0_commit", "gpu")})
        (args.out / name).mkdir(parents=True, exist_ok=True)
        (args.out / name / "run_info.json").write_text(json.dumps(info, indent=1))
        print(json.dumps({k: v for k, v in info.items() if k not in ("error", "registration")}, indent=1), flush=True)
        results.append(info)
    (args.out / "summary.json").write_text(json.dumps({"environment": env, "sessions": results}, indent=1))
    (args.out / "stage_asset.json").write_text(json.dumps(stage_record(env, args, results), indent=1))
    if not results:
        print("No registered sessions found under", args.data)
        return 1
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    sys.exit(main())
