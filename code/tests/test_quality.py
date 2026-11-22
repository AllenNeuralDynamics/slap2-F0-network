"""io.filter_trials: calibration failures and paths with too few usable trials."""
import argparse

import h5py
import numpy as np

from slap2_f0.io import filter_trials
from slap2_f0.train import add_quality_arguments


def _trial(root, session, path, trial, peak_r=None, label=True):
    d = root / session / f"Path{path}"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"trial_{trial:03d}.h5"
    with h5py.File(fn, "w") as f:
        f.attrs["path"] = np.array([float(path)])
        f.attrs["trial"] = np.array([float(trial)])
        if label:
            f.attrs["label"] = session
        if peak_r is not None:
            f.attrs["peakR"] = np.array([peak_r])
    return fn


def test_calibration_and_path_rules(tmp_path):
    files = []
    # session A: path 1 complete and good; path 2 has 6 failed calibrations of 10 -> 4/10 usable, dropped whole
    files += [_trial(tmp_path, "A", 1, t, 0.45) for t in range(1, 11)]
    files += [_trial(tmp_path, "A", 2, t, 0.05 if t <= 6 else 0.44) for t in range(1, 11)]
    # session B: path 1 good; path 2 lost trials 1-4 in GIAnT (6/10 usable, kept) and has one failure (5/10, kept)
    files += [_trial(tmp_path, "B", 1, t, 0.40) for t in range(1, 11)]
    files += [_trial(tmp_path, "B", 2, t, 0.25 if t == 5 else 0.40) for t in range(5, 11)]
    # session C: path 2 has only 3 of 10 trials (missing in GIAnT), no failures -> dropped whole
    files += [_trial(tmp_path, "C", 1, t, 0.50) for t in range(1, 11)]
    files += [_trial(tmp_path, "C", 2, t, 0.50) for t in (2, 3, 4)]
    kept, report = filter_trials(files, 0.3, 0.5)
    kept = set(kept)
    by = lambda s, p: {f for f in files if f.parent.parent.name == s and f.parent.name == f"Path{p}"}  # noqa: E731
    assert by("A", 1) <= kept and not (by("A", 2) & kept)
    assert by("B", 1) <= kept
    assert {f.stem for f in by("B", 2) & kept} == {f"trial_{t:03d}" for t in range(6, 11)}
    assert by("C", 1) <= kept and not (by("C", 2) & kept)
    assert sum("calibration failed" in r for r in report) == 7
    assert sum("left out whole" in r for r in report) == 2


def test_order_kept_and_rules_off(tmp_path):
    files = [_trial(tmp_path, "S", 1, t, r) for t, r in ((1, 0.4), (2, 0.1), (3, 0.4), (4, 0.4))]
    kept, _ = filter_trials(files, 0.3, 0.5)
    assert kept == [files[0], files[2], files[3]]
    assert filter_trials(files, 0, 0) == (files, [])
    assert filter_trials(files, 0, 0.5)[0] == files


def test_missing_attributes_are_kept(tmp_path):
    # older exports: no peakR, no label (session from the folder name)
    files = [_trial(tmp_path, "old", 1, t, None, label=False) for t in range(1, 5)]
    kept, report = filter_trials(files)
    assert kept == files and report == []


def test_arguments_default_on():
    p = argparse.ArgumentParser()
    add_quality_arguments(p)
    a = p.parse_args([])
    assert a.min_peak_r == 0.3 and a.min_path_trials == 0.5
