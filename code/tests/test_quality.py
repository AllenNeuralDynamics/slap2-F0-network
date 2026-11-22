"""io.filter_trials: calibration failures and paths left with too few trials."""
import argparse

import h5py
import numpy as np

from slap2_f0.io import filter_trials
from slap2_f0.train import add_quality_arguments


def _trial(root, session, path, trial, peak_r=None, label=True, path_attr=True):
    d = root / session / f"Path{path}"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"trial_{trial:03d}.h5"
    with h5py.File(fn, "w") as f:
        if path_attr:
            f.attrs["path"] = np.array([float(path)])
        f.attrs["trial"] = np.array([float(trial)])
        if label:
            f.attrs["label"] = session
        if peak_r is not None:
            f.attrs["peakR"] = np.array([peak_r])
    return fn


def _of(files, session, path):
    return {f for f in files if f.parent.parent.name == session and f.parent.name == f"Path{path}"}


def test_calibration_and_path_rules(tmp_path):
    files = []
    # A: path 1 good; path 2 has 7 failed calibrations of 10 -> 3 usable, left out whole
    files += [_trial(tmp_path, "A", 1, t, 0.45) for t in range(1, 11)]
    files += [_trial(tmp_path, "A", 2, t, 0.05 if t <= 7 else 0.44) for t in range(1, 11)]
    # B: path 2 lost most trials in GIAnT and has one failure: 4 usable, kept (only the failure goes)
    files += [_trial(tmp_path, "B", 1, t, 0.40) for t in range(1, 11)]
    files += [_trial(tmp_path, "B", 2, t, 0.25 if t == 5 else 0.40) for t in range(5, 10)]
    # C: path 2 has 3 trials, none failed -> left out whole
    files += [_trial(tmp_path, "C", 1, t, 0.50) for t in range(1, 11)]
    files += [_trial(tmp_path, "C", 2, t, 0.50) for t in (2, 3, 4)]
    kept, report = filter_trials(files, 0.3, 4)
    kept = set(kept)
    assert _of(files, "A", 1) <= kept and not (_of(files, "A", 2) & kept)
    assert _of(files, "B", 1) <= kept
    assert {f.stem for f in _of(files, "B", 2) & kept} == {f"trial_{t:03d}" for t in (6, 7, 8, 9)}
    assert _of(files, "C", 1) <= kept and not (_of(files, "C", 2) & kept)
    assert sum("calibration failed" in r for r in report) == 8
    assert sum("left out whole" in r for r in report) == 2


def test_most_failed_but_enough_left_is_kept(tmp_path):
    # like 780125_2025-03-18 path 1: 20 of 32 failed, 12 usable -> only the failures go
    files = [_trial(tmp_path, "S", 1, t, 0.05 if t <= 20 else 0.45) for t in range(1, 33)]
    kept, report = filter_trials(files, 0.3, 4)
    assert kept == files[20:] and not any("left out whole" in r for r in report)


def test_order_kept_and_rules_off(tmp_path):
    files = [_trial(tmp_path, "S", 1, t, r) for t, r in ((1, 0.4), (2, 0.1), (3, 0.4), (4, 0.4), (5, 0.4))]
    kept, _ = filter_trials(files, 0.3, 4)
    assert kept == [files[0], files[2], files[3], files[4]]
    assert filter_trials(files, 0, 0) == (files, [])
    assert filter_trials(files, 0, 4)[0] == files


def test_path_attribute_and_folder_name_are_one_path(tmp_path):
    # four trials lack the path attribute; they still belong to Path1 and do not form a group of 4
    files = [_trial(tmp_path, "S", 1, t, 0.45, path_attr=t > 4) for t in range(1, 11)]
    kept, report = filter_trials(files, 0.3, 5)
    assert kept == files and report == []


def test_files_without_calibration_are_never_left_out(tmp_path):
    # older or synthetic exports: no peakR, no label (session from the folder name), short paths
    files = [_trial(tmp_path, "old", 1, t, None, label=False) for t in range(1, 11)]
    files += [_trial(tmp_path, "old", 2, t, None, label=False) for t in (1, 2)]
    kept, report = filter_trials(files)
    assert kept == files and report == []


def test_arguments_default_on():
    p = argparse.ArgumentParser()
    add_quality_arguments(p)
    a = p.parse_args([])
    assert a.min_peak_r == 0.3 and a.min_path_trials == 4
