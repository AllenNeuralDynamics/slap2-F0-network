# slap2-F0-network

A blind-spot network that estimates the baseline (F0) and the noise of SLAP2 glutamate (iGluSnFR4) movies. It is meant to replace the moving-median baseline that GIAnT's SiLo uses to localize sources. The baseline it predicts at a pixel and frame never sees that pixel's own recent frames, so an event cannot push the baseline up. Its noise estimate Z-scores events.

The network is trained on residuals: the 80 Hz registered movie minus the activity GIAnT fitted. It is integrated into GIAnT as a Python step after registration that writes baseline and noise files next to the registered movies.

The design, measurements and phases are in [docs/plan.md](docs/plan.md).

## Status

| Piece | State |
|---|---|
| Training-data export (`code/matlab/exportF0TrainingData.m`) | Working. Tested on four 780125 trials: after the per-trial gain (0.83–0.97), the movie-on-activity slope is 0.98–1.03 |
| Code Ocean data capsule ("SLAP2 F0 network - training data", `run --mode export`) | First 180 s runs in progress (780125, 885233) |
| Code Ocean training capsule (`run --mode train`) | Code ready, tested on CPU; the GPU capsule is not created yet (see below) |
| Session list (`sessions/slap2_iglusnfr4_sessions.csv`) | 142 iGluSnFR4 multiROI sessions on `aind-open-data`; start with the 4 integration-test sessions |
| Python package (`code/slap2_f0`) | Reader, slab blind-spot model (v0/v1), likelihoods, training loop with validation and calibration, tests; not trained yet |
| Integration into GIAnT | Not started |

## Repository layout

The repository is laid out as a Code Ocean capsule. Two capsules use it, with the same code and environment and different machines: a large-memory CPU machine for the export, and a GPU machine for training.

```
code/
  run                  capsule entry point: --mode export (default) or --mode train
  run_export.m         per session: trial table -> trim -> MultiRoiRegistration -> SILo -> export
  startup.m            MATLAB path for the capsule (SLAP2 reader, NoRMCorre, Fast_Tiff_Write)
  matlab/              exportF0TrainingData.m, trimTrialTableSLAP2.m
  slap2_f0/            Python package: io, data, model, losses, train, cli
  tests/               pytest (blind-spot exactness, likelihoods, reader, training smoke test)
environment/           Dockerfile and postInstall (MATLAB R2023b, Slap2DataReader, GIAnT-MATLAB, PyTorch)
.codeocean/            the capsule's machine (each capsule keeps its own; see below)
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

## Training

`slap2-f0 train <folder>` trains the slab blind-spot network on every exported trial under `<folder>`.

- **Validation split:** whole sessions are held out when there are at least 3, otherwise whole trials.
- **Every `--val_every` steps** it reports the validation loss and the calibration of the predicted noise. z = (target − μ)/σ should have mean 0 and variance 1, with about 95% of values within ±2.
- **Outputs** in `--out`: `config.json`, `metrics.csv`, `best.pt` and `last.pt`.
- **Hardware:** on a GPU it uses mixed precision (bf16 where supported). A CPU manages about one step a minute even with a narrow model, so real training needs the GPU.

### On Code Ocean

1. **Create the GPU capsule.** New Capsule → Clone from Git → `AllenNeuralDynamics/slap2-F0-network`. In its Environment, choose a GPU machine. g5.4xlarge (one A10G with 24 GB, 16 vCPUs for data loading, 64 GB of memory) is a good start.
2. **Keep its machine setting out of GitHub.** Changing the machine writes `.codeocean/resources.json`, and that file in GitHub belongs to the data capsule. Commit the change in the GPU capsule, but don't sync that commit to GitHub. Updates from GitHub `main` merge cleanly as long as `main` doesn't change this file.
3. **Make the training data a data asset.** Save each export run's results as a data asset, then attach those assets to the GPU capsule.
4. **Run.** Use `--mode train --train_args "--steps 20000 --batch 8 --max_hours 10"`. Checkpoints and metrics go to `/results/train`, and the log to `/results/logs/train.log`.

## Python

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"     # CPU torch: add --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pytest                 # Windows; .venv/bin/python elsewhere
slap2-f0 inspect <export folder>
slap2-f0 train <export folder> --out results/train
```
