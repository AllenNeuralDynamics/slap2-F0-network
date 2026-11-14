# Localization evaluation harness

These scripts run SiLo's localization and source selection on their own, without extraction, for several localization baselines (`params.localizationBaseline`). They then score the sources:
- against simulation ground truth, at 2 px, using the integration-test capsule's `giant_eval.metrics`;
- against a finished GIAnT run, at 1.5 px with a one-to-one Hungarian match.

Phase 3 of `docs/plan.md` will compare the F0 network against the moving median with them.

The baseline option lives on the GIAnT-MATLAB branch `localize-splitfreq-baseline` (commit f8c24ef; local, not pushed). It adds `localizationBaseline` = `movmedian` (default) or `splitFreq` to `localizeSources_vIM`.

| Script | Does |
|---|---|
| `locbase_trials.m` | per-trial stage (a copy of `loadAndProcessTrialAsync`), once per setting on each loaded movie |
| `locbase_select.m` | cross-trial stage (a copy of SILo.m lines ~172–290 at d2fb85c): template, alignment, averaged activity image, `getActImPeaks`, source filtering |
| `locbase_sweep.m` | re-picks sources at other `peakth` values |
| `locbase_prepsim.m` | trims and registers a simulation the way the integration-test capsule does |
| `locbase_bdiag.m`, `locbase_zdiag.m` | diagnostics: baseline ratio by pixel brightness; Z-score offsets |
| `score_sims.py`, `score_sweep.py` | simulation scores, at the default threshold and across thresholds |
| `analyze_real.py`, `real_sweep.py` | real-data comparison with a finished GIAnT run |
| `validate_harness.py` | checks the harness reproduces a SILo run's `act_im_peaks` and `act_im` exactly (it did: 40 of 40 positions, identical image) |

## Result (2026-11-14): `splitFreq` is worse than the moving median

### Simulations

Sims 1 and 2 (the only ones with movies on this machine), each in two 180 s windows:

| | precision | recall | F1 | localization error |
|---|---|---|---|---|
| movmedian | 0.607 | 0.867 | **0.712** | 0.49 px |
| splitFreq | 0.470 | 0.883 | **0.620** | 0.48 px |

The table gives medians over the 4 cases at the default `peakth` of 8. Over `peakth` 4–20, `splitFreq` has lower precision at every recall level, and movmedian's best F1 is higher in all 4 cases.

### 780125

15 trials × 2 paths:

| | sources | matched to the final GIAnT run's |
|---|---|---|
| movmedian, Path 1 | 130 | 122 of 140 |
| splitFreq, Path 1 | 125 | 112 of 140 |
| movmedian, Path 2 | 113 | 95 of 129 |
| splitFreq, Path 2 | 93 | 66 of 129 |

The final GIAnT run used patchwise inter-trial alignment, which isn't on main, so this comparison is partly confounded.

### Why

`splitFreq` keeps the lowest ~15% of 0.2 s page means. At a few photons per page those are mostly noise, so its baseline is far too low on dim pixels: 0.31× the moving median in the dimmest quintile, vs 0.69× in the brightest (780125 Path 2). That raises Z-scores unevenly across the image.

The moving median's +42% at source peaks (`docs/plan.md`) is real. But a lower-envelope fix trades it for a larger, brightness-dependent bias. Whatever replaces the baseline has to be unbiased at low counts too, which a likelihood-trained network with count-like noise should be.
