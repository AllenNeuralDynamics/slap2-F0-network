"""The register mode's records for the registration asset: the statistics
sidecars of the registered trial-paths and stage_asset.json."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/, for register_gpu.py
import register_gpu as rg  # noqa: E402


def test_sidecars_lists_trial_paths_without_one(tmp_path):
    moco = tmp_path / "motion_correction"
    moco.mkdir()
    (moco / "E1T1DMD1_REGSTATS-80Hz.h5").write_bytes(b"x" * 10)
    (moco / "E1T2DMD1_REGSTATS-80Hz.h5").write_bytes(b"x" * 10)
    fields = {
        "registration_failed": np.array([[0, 0, 0], [0, 1, 0]], bool),
        "fn_adata": np.array([["E1T1DMD1_ALIGNMENTDATA.h5", "E1T2DMD1_ALIGNMENTDATA.h5", ""],
                              ["E1T1DMD2_ALIGNMENTDATA.h5", "E1T2DMD2_ALIGNMENTDATA.h5", ""]], object),
    }
    rec = rg.sidecars(moco, fields)
    assert rec["n"] == 2 and rec["gb"] > 0
    # path 2 trial 2 failed registration and empty cells have no recording
    assert rec["missing"] == ["E1T1DMD2"]


def test_stage_record():
    env = {"giant_python_commit": "abc", "giant_python_ref": "dev", "gpu": {"name": "L40S"}}
    args = SimpleNamespace(max_duration_s=180.0, compensate_online=0, schedule="sequential", keep_registered=1)
    rec = rg.stage_record(env, args, [{"session": "s1", "status": "ok"}, {"session": "s2", "status": "failed"}])
    assert rec["kind"] == "registration" and rec["format"] == rg.STAGE_FORMAT
    assert rec["max_duration_s"] == 180.0 and rec["compensate_online"] is False and rec["keep_registered"] is True
    assert rec["sessions"] == {"s1": "ok", "s2": "failed"} and rec["giant_python_commit"] == "abc"
