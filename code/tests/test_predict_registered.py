"""The predict_registered mode (code/predict_registered.py): sessions found
under the data folder, slap2-f0 run through GIAnT-Python's predict_baselines
(the step of ``giant silo --f0-ckpt``), giant silo's records with digests
that still identify the files after both folders are copied, and refusal of
a SILo run with other trial preparation settings."""

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("giant_python")  # GIAnT-Python on the path (code/run fetches it)

from giant_python.io.hdf5 import save_struct_h5  # noqa: E402
from giant_python.pipeline import silo_baseline  # noqa: E402
from giant_python.pipeline.silo_session import SiloSession  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/, for predict_registered.py
import predict_registered as pr  # noqa: E402
from test_predict import _model  # noqa: E402
from test_registered import _write_registered  # noqa: E402

T, R, C = 200, 12, 14


def prep(**given):
    """The prediction's trial preparation settings (SILo's defaults unless given)."""
    keys = {"activity_channel": "channel"}
    ns = {keys.get(n, n): given.get(n) for n in pr.PREPARATION}
    return pr.silo_params(type("Args", (), ns))


def make_registered(folder: Path, n_trials: int = 2, failed=()) -> Path:
    """A registered session as the register mode leaves it: trial table
    (savedr of another machine), alignment files and HDF5 movies."""
    rng = np.random.default_rng(len(folder.name))
    moco = folder / "motion_correction"
    moco.mkdir(parents=True)
    fn = np.empty((2, n_trials), object)
    reg = np.empty_like(fn)
    adata = np.empty_like(fn)
    for d in range(2):
        for t in range(n_trials):
            stem = f"E1T{t + 1}DMD{d + 1}"
            fn[d, t] = f"x_DMD{d + 1}-TRIAL{t + 1:06d}.dat"
            reg[d, t] = f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.h5"
            adata[d, t] = f"{stem}_ALIGNMENTDATA.h5"
            act = rng.poisson(5.0 + 5 * rng.random((R, C)), (T, R, C)).astype(np.float32)
            _write_registered(moco, stem, act, np.full_like(act, 50), "h5")
            with h5py.File(moco / adata[d, t], "w") as f:
                f["numChannels"] = np.array([[2.0]])
                f["alignHz"] = np.array([[80.0]])
                f["recNegErr"] = (-0.5 + 0.001 * rng.standard_normal(T))[None]
                f["DSframes"] = np.arange(1.0, T + 1)[None]
                f["slap2/varFacDS"] = rng.uniform(1, 2, (T, C, R)).astype(np.float32)
    bad = np.zeros((2, n_trials))
    for d, t in failed:
        bad[d, t] = 1
    save_struct_h5(folder / "trial_table.h5", {
        "datadr": "/root/capsule/data/raw/slap2/dynamic_data",
        "savedr": f"/root/capsule/results/register/{folder.name}",
        "filename": fn,
        "epoch": np.ones((2, n_trials)),
        "slap2_info": {"first_line": np.ones((2, n_trials)), "last_line": np.full((2, n_trials), 1000.0)},
        "motion_correction": {"fn_reg_ds": reg, "fn_adata": adata, "registration_failed": bad},
    })
    (folder / "run_info.json").write_text(json.dumps({"session": folder.name, "max_duration_s": 180.0}))
    return folder


@pytest.fixture
def ckpt(tmp_path):
    model, wcfg = _model(box=True)
    path = tmp_path / "ck" / "step_001700.pt"
    path.parent.mkdir()
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "window": asdict(wcfg), "step": 1700}, path)
    return path


def test_preparation_settings_are_giant_silos():
    """The settings passed are exactly those giant silo's records hold, with
    SiloParams' defaults."""
    from giant_python.models.params import SiloParams

    settings = silo_baseline.prediction_settings(prep())
    assert sorted(settings) == sorted(pr.PREPARATION)
    assert settings == {n: SiloParams.__dataclass_fields__[n].default for n in pr.PREPARATION}
    assert prep(motion_thresh=1.5).motion_thresh == 1.5


def test_find_sessions(tmp_path):
    data = tmp_path / "data"
    a = make_registered(data / "reg_a" / "780125_2025-03-14", n_trials=1)  # asset made from results/register
    b = make_registered(data / "reg_b" / "register" / "878030_2026-08-28", n_trials=1)  # from the whole results
    (data / "ckpt").mkdir()
    (data / "ckpt" / "x.pt").write_bytes(b"")
    (data / "other" / "s").mkdir(parents=True)
    (data / "other" / "s" / "trial_table.h5").write_bytes(b"")  # no motion_correction: not registered
    assert pr.find_sessions(data) == [("780125_2025-03-14", a), ("878030_2026-08-28", b)]
    assert pr.mount_of(b, data) == "reg_b"
    make_registered(data / "reg_c" / "780125_2025-03-14", n_trials=1)
    with pytest.raises(ValueError, match="two registered sessions"):
        pr.find_sessions(data)
    assert pr.parse_asset_ids(" reg_a=abc, reg_b=def ") == {"reg_a": "abc", "reg_b": "def"}
    with pytest.raises(ValueError, match="mount=id"):
        pr.parse_asset_ids("abc")


def test_select_units(tmp_path):
    s = SiloSession.open(make_registered(tmp_path / "s", failed=[(1, 0)]) / "trial_table.h5",
                         moco_dir=tmp_path / "s" / "motion_correction", data_dir=tmp_path, inputs="collect")
    assert pr.select_units(s, []) == [(0, 0), (0, 1), (1, 1)]  # path 2 trial 1 failed registration
    assert pr.select_units(s, ["E1T2DMD2"]) == [(1, 1)]
    with pytest.raises(ValueError, match="not kept"):
        pr.select_units(s, ["E1T1DMD2"])
    (tmp_path / "s" / "motion_correction" / "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5").unlink()
    with pytest.raises(ValueError, match="keep_registered 1"):
        pr.select_units(s, [])


def test_predict_registered_end_to_end(tmp_path, ckpt):
    """Baselines and records of every kept trial-path; a SILo run reads them
    from copies of both folders (other modification times), and refuses
    them with other trial preparation settings."""
    data = tmp_path / "data"
    folder = make_registered(data / "giantreg_x_180s" / "sess_1", failed=[(1, 1)])
    (data / "giantreg_x_180s" / "stage_asset.json").write_text(json.dumps({"kind": "registration"}))
    out = tmp_path / "results" / "baselines"
    rc = pr.main([str(data), str(out), "--ckpt", str(ckpt), "--ckpt_label", "s6_ft", "--device", "cpu",
                  "--out_frames", "64", "--asset_ids", "giantreg_x_180s=aid1", "--scratch", str(tmp_path / "scratch")])
    assert rc == 0
    sdir = out / "sess_1"
    names = sorted(p.name for p in sdir.glob("*_BASELINE-80Hz.h5"))
    assert names == ["E1T1DMD1_BASELINE-80Hz.h5", "E1T1DMD2_BASELINE-80Hz.h5", "E1T2DMD1_BASELINE-80Hz.h5"]
    info = json.loads((sdir / "run_info.json").read_text())
    assert info["status"] == "ok" and info["registration"]["asset_id"] == "aid1"
    assert info["registration"]["stage_asset"]["kind"] == "registration"
    assert info["registration"]["run_info"]["max_duration_s"] == 180.0
    assert info["settings"] == {"activity_channel": 1, "nan_thresh": 0.33, "motion_thresh": 2.0,
                                "discard_initial_s": 0.0}
    assert info["checkpoint"]["label"] == "s6_ft" and len(info["checkpoint"]["sha256"]) == 64
    assert info["predicted"] == [[1, 1], [1, 2], [2, 1]]
    for tp in info["trial_paths"]:  # slap2-f0's report, per trial-path
        assert tp["frames"] == T and tp["out_frames"] == 64 and tp["network_s"] >= 0 and tp["write_s"] >= 0
        assert tp["baseline_gb"] > 0
    marker = json.loads((out / "stage_asset.json").read_text())
    assert marker["kind"] == "f0_baselines" and marker["sessions"] == {"sess_1": "ok"}
    assert marker["registration_assets"] == {"giantreg_x_180s": "aid1"}
    with h5py.File(sdir / names[0], "r") as f:
        assert f["baseline"].shape == (T, C, R) and f.attrs["step"] == 1700

    # giant silo's record, with digests of the baseline, movie and alignment
    base = sdir / names[0]
    rec = json.loads(silo_baseline.record_path(base).read_text())
    moco = folder / "motion_correction"
    assert rec["checkpoint"]["sha256"] == silo_baseline.file_sha256(ckpt)
    assert rec["movie"]["sha256"] == silo_baseline.file_sha256(moco / "E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5")
    assert rec["baseline"]["sha256"] == silo_baseline.file_sha256(base)

    # the SILo stage: both folders copied (data assets carry upload times)
    def copy(src, dest):
        shutil.copytree(src, dest, copy_function=shutil.copyfile)
        for f in dest.rglob("*"):
            if f.is_file():
                st = f.stat()
                os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        return dest

    reg2 = copy(folder, tmp_path / "silo" / "reg")
    base2 = copy(sdir, tmp_path / "silo" / "base")
    s2 = SiloSession.open(reg2 / "trial_table.h5", moco_dir=reg2 / "motion_correction", data_dir=reg2,
                          inputs="collect")
    args = (base2 / names[0], s2.movie(0, 0), s2.alignment(0, 0))
    assert silo_baseline.check_baseline_record(*args, prep()) == rec
    with pytest.raises(ValueError, match="was predicted with"):
        silo_baseline.check_baseline_record(*args, prep(motion_thresh=1.5))
    with pytest.raises(ValueError, match="was predicted with"):
        silo_baseline.check_baseline_record(*args, prep(activity_channel=2))

    # a second run into the same folder predicts nothing and changes nothing
    before = {p.name: p.stat().st_mtime_ns for p in sdir.glob("*_BASELINE-80Hz.h5")}
    assert pr.main([str(data), str(out), "--ckpt", str(ckpt), "--device", "cpu", "--out_frames", "64",
                    "--scratch", str(tmp_path / "scratch")]) == 0
    assert json.loads((sdir / "run_info.json").read_text())["predicted"] == []
    assert {p.name: p.stat().st_mtime_ns for p in sdir.glob("*_BASELINE-80Hz.h5")} == before

    # an interrupted run (record without the file's identity, or a truncated record) is
    # predicted again, not taken as it is
    rec2 = silo_baseline.record_path(sdir / names[1])
    stale = json.loads(rec2.read_text())
    stale["baseline"] = None
    rec2.write_text(json.dumps(stale))
    silo_baseline.record_path(sdir / names[2]).write_text('{"format": "giant-silo-baseline-rec')  # truncated
    assert pr.main([str(data), str(out), "--ckpt", str(ckpt), "--device", "cpu", "--out_frames", "64",
                    "--scratch", str(tmp_path / "scratch")]) == 0
    info = json.loads((sdir / "run_info.json").read_text())
    assert sorted(info["removed_incomplete"]) == sorted(names[1:])
    assert info["predicted"] == [[1, 2], [2, 1]]
    for n in names:
        assert pr._complete(json.loads(silo_baseline.record_path(sdir / n).read_text()))
    assert (sdir / names[0]).stat().st_mtime_ns == before[names[0]]


def test_predict_registered_options(tmp_path, ckpt):
    """A subset of sessions and trial-paths; settings other than SILo's
    defaults go into the records; unknown sessions and a missing checkpoint
    fail before anything runs."""
    data = tmp_path / "data"
    make_registered(data / "a" / "s1", n_trials=1)
    make_registered(data / "b" / "s2", n_trials=1)
    out = tmp_path / "out"
    common = ["--device", "cpu", "--out_frames", "64", "--scratch", str(tmp_path / "scratch")]
    rc = pr.main([str(data), str(out), "--ckpt", str(ckpt), "--sessions", "s2", "--trial_paths", "E1T1DMD2",
                  "--motion_thresh", "1.5", *common])
    assert rc == 0 and not (out / "s1").exists()
    assert [p.name for p in (out / "s2").glob("*.h5")] == ["E1T1DMD2_BASELINE-80Hz.h5"]
    rec = json.loads((out / "s2" / "E1T1DMD2_BASELINE-80Hz.json").read_text())
    assert rec["settings"]["motion_thresh"] == 1.5
    assert pr.main([str(data), str(out), "--ckpt", str(ckpt), "--sessions", "s9", *common]) == 1
    assert pr.main([str(data), str(out), "--ckpt", str(tmp_path / "none.pt"), *common]) == 1
    # a trial-path that is not in the session fails that session
    assert pr.main([str(data), str(out), "--ckpt", str(ckpt), "--sessions", "s1", "--trial_paths", "E1T5DMD1",
                    *common]) == 1
    assert "not kept" in json.loads((out / "s1" / "run_info.json").read_text())["error"]


def test_unreadable_record_counts_as_missing(tmp_path):
    rec = tmp_path / "x_BASELINE-80Hz.json"
    assert pr._read_record(rec) is None
    rec.write_text("")
    assert pr._read_record(rec) is None
    rec.write_text('{"baseline": {"sha256": "a"}')
    assert pr._read_record(rec) is None
    rec.write_text('{"baseline": {"sha256": "a"}}')
    assert pr._read_record(rec) == {"baseline": {"sha256": "a"}}
    assert not pr._complete(pr._read_record(rec))


def test_f0_args_take_speed_options_only(tmp_path):
    """--f0_args passes speed options to the wrapper and refuses settings or a
    checkpoint, which would make the records describe another prediction."""
    want = ["--fast", "--prefetch", "2", "--stream_frames=128"]
    assert pr.speed_args("--fast --prefetch 2 --stream_frames=128") == want
    assert pr.speed_args("") == []
    for bad in ("--channel 2", "--motion_thresh 1.5", "--ckpt other.pt", "--fast=1", "--prefetch"):
        with pytest.raises(SystemExit):
            pr.speed_args(bad)
    exe = pr.write_wrapper(tmp_path / "bin", tmp_path / "r.jsonl", "--engine stream --async_write")
    assert "--engine stream --async_write" in exe.read_text().replace('"', "")
