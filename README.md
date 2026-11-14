# slap2-F0-network

A blind-spot network that estimates the baseline (F0) and the noise of SLAP2 glutamate (iGluSnFR4) movies. It is meant to replace the moving-median baseline that GIAnT's SiLo uses to localize sources. The baseline it predicts at a pixel and frame never sees that pixel's own recent frames, so an event cannot push the baseline up. Its noise estimate Z-scores events.

The network is trained on residuals: the 80 Hz registered movie minus the activity GIAnT fitted. It is integrated into GIAnT as a Python step after registration that writes baseline and noise files next to the registered movies.

The design, measurements and phases are in [docs/plan.md](docs/plan.md).

## Status

| Piece | State |
|---|---|
| Training-data export (`code/matlab/exportF0TrainingData.m`) | Working. Tested on two 780125 trials, which give slope 0.97–1.00 of the movie on the activity at source peaks |
| Code Ocean data capsule (`code/run`, `code/run_export.m`, `environment/`) | Written, not yet run on Code Ocean |
| Session list (`sessions/slap2_iglusnfr4_sessions.csv`) | 142 iGluSnFR4 multiROI sessions on `aind-open-data`; start with the 4 integration-test sessions |
| Python package (`code/slap2_f0`) | Reader, slab blind-spot model (v0/v1), likelihoods, training loop, tests; not trained yet |
| Integration into GIAnT | Not started |

## Repository layout

The repository is laid out as a Code Ocean capsule, so it can be linked as one.

```
code/
  run                  capsule entry point: GIAnT on each attached session, then the export
  run_export.m         per session: trial table -> trim -> MultiRoiRegistration -> SILo -> export
  matlab/              exportF0TrainingData.m, trimTrialTableSLAP2.m
  slap2_f0/            Python package: io, data, model, losses, cli
  tests/               pytest (blind-spot exactness, likelihoods, reader)
environment/           Dockerfile and postInstall (MATLAB R2023b, Slap2DataReader, GIAnT-MATLAB)
metadata/              Code Ocean capsule metadata
sessions/              list of qualifying sessions and the survey scripts that made it
tools/measure/         one-off MATLAB scripts behind the measurements in docs/plan.md
docs/plan.md           design and plan
```

## Training data

### Running the export

- **On Code Ocean**, the `run` parameters are:
  - `giant_ref` (GIAnT commit; default `main`);
  - `max_duration_s` (seconds kept from the start of each session; default 180, the length the integration-test capsule runs on these sessions);
  - `datasets` (`all`, or comma-separated mount names);
  - `n_workers`, `channels` (default `1`);
  - `keep_giant_outputs` (`summary`, `all` or `none`).
- **Locally**, after SiLo has run on a session:

```matlab
addpath('code/matlab')   % with GIAnT-MATLAB (minus its tests folder) and the SLAP2 reader on the path
exportF0TrainingData('<session>/trial_table.h5', '<out>', 'rawDir', '<raw data folder>')
```

### Output format

One HDF5 file per trial and imaging path, `<session>/Path<k>/trial_<NNN>.h5`. Shapes are as h5py reads them:

| Dataset | Shape | |
|---|---|---|
| `movie` | (channel, frame, row, col) | 80 Hz registered movie, NaN where a pixel is not imaged or a frame is discarded |
| `activity` | (channel, frame, row, col) | GIAnT's fitted activity on the same grid and timing; **target = movie − activity** |
| `mean_image` | (channel, row, col) | |
| `varfac` | (row, col) | SLAP2 variance factor, per-pixel mean (or (frame, row, col) with `varFac='full'`) |
| `mask/valid`, `mask/labeled`, `mask/footprint_support` | (row, col) | |
| `frame/discard`, `frame/motion_row`, `frame/motion_col`, `frame/rec_neg_err`, `frame/onlineMotion*shift`, `frame/valid_mean` | (frame,) | |
| `sources/peak_row`, `sources/peak_col` | (source,) | 0-based, in the crop |

Root attributes hold provenance (session, path, trial, GIAnT commit, crop origin), the mapping that was used, and QC:
- `shiftRC`, `blurSigma`, `activityGain`: the footprint offset, blur and gain applied to the activity;
- `peakSlope`, `peakR`: movie on activity at source peak pixels, high-passed, after the gain (near 1 when the activity lines up);
- `diskSlope`, `diskR`: the same for the flux in a 4 px disk;
- `eventResidualFraction`: the residual's event-triggered peak relative to the movie's (near 0 when the activity is removed).

### How the activity is put on the movie's grid

GIAnT fits footprints H and traces at 200 Hz on its consensus canvas. The export maps H·X onto each trial's 80 Hz movie in four steps:

- **Time.** It uses the same kernel that made the movie: SLAP2 `getImages` weights the samples within max(3·dt, linesPerCycle) lines of a frame by exp(−|Δline|/dt), with dt = ceil(line rate / alignHz).
- **Space.** It applies the sub-pixel trial offset that best aligns the movie with the activity.
- **Gain.** It scales the activity by the movie-on-activity slope of the flux in a 4 px disk around each source, which barely depends on blur or offset.
- **Blur.** It applies the small Gaussian blur (typically 0.5–1 px) that makes the slope at source peak pixels match that gain. The 80 Hz frames are slightly blurrier than the 200 Hz data the footprints were fitted to.

## Python

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"     # CPU torch: add --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pytest                 # Windows; .venv/bin/python elsewhere
slap2-f0 inspect <export folder>
slap2-f0 train <export folder> --out results/train
```
