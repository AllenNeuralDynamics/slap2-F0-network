# slap2-F0-network

A blind-spot network that estimates the baseline (F0) and the noise of SLAP2 glutamate (iGluSnFR4) movies. It is meant to replace the moving-median baseline that GIAnT's SiLo uses to localize sources. The baseline it predicts at a pixel and frame never sees that pixel's own recent frames, so an event cannot push the baseline up. Its noise estimate Z-scores events.

The network is pretrained on simulated trials with a known baseline, built on real trials as templates, then fine-tuned on residuals: the 80 Hz registered movie minus the activity GIAnT fitted. It is integrated into GIAnT as a Python step after registration that writes baseline and noise files next to the registered movies.

The design, measurements and phases are in [docs/plan.md](docs/plan.md).

## Status

| Piece | State |
|---|---|
| Training-data export (`code/matlab/exportF0TrainingData.m`) | Working. Tested on four 780125 trials: after the per-trial gain (0.83–0.97), the movie-on-activity slope is 0.98–1.03 |
| Code Ocean data capsule ("SLAP2 F0 network - training data", `run --mode export`) | First 180 s runs in progress (780125, 885233) |
| Code Ocean training capsule (`run --mode train`) | GPU capsule running. The first network (v1, Student-t loss on real residuals) lost to the plain mean image on a held-out session; see docs/plan.md "Results" |
| Session list (`sessions/slap2_iglusnfr4_sessions.csv`) | 142 iGluSnFR4 multiROI sessions on `aind-open-data`; start with the 4 integration-test sessions |
| Python package (`code/slap2_f0`) | v2: whole-trial stats, simulator, model with blind window statistics and a mean-image start, unbiased loss, sim/real training, ceiling analysis, tests |
| Integration into GIAnT | Not started |

## Repository layout

The repository is laid out as a Code Ocean capsule. Two capsules run its code: the data capsule (linked to this repository; MATLAB environment, large-memory CPU machine) runs the export, and a GPU capsule (PyTorch starter environment, GPU machine; code pushed to it, see Training) runs training.

```
code/
  run                  capsule entry point: --mode export (default) or --mode train
  run_export.m         per session: trial table -> trim -> MultiRoiRegistration -> SILo -> export
  startup.m            MATLAB path for the capsule (SLAP2 reader, NoRMCorre, Fast_Tiff_Write)
  matlab/              exportF0TrainingData.m, trimTrialTableSLAP2.m
  slap2_f0/            Python package: io, stats, simulate, data, model, losses, train, ceiling, cli
  tests/               pytest (blind-spot exactness, simulator, likelihoods, reader, ceiling, training smoke tests)
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

`slap2-f0 train <folder>` trains the slab blind-spot network on the exported trials under `<folder>`.

- **Data** (`--data`):
  - `sim` (default): simulated windows with the true baseline as the target (`slap2_f0/simulate.py`), for pretraining.
  - `real`: real windows with movie − activity as the target, for fine-tuning a pretrained checkpoint (`--init best.pt`).
- **Simulation.** Each training window is a new virtual trial built on a random real trial:
  - **Baseline:** its whole-trial mean minus its mean activity, times a slow drift (bleaching up to 5%, fluctuations up to 2%).
  - **Events:** its own fitted activity, shifted in time and rescaled.
  - **Noise:** photon noise matched to the trial's measured gain, temporal and spatial correlations, long-average inflation and skew (`slap2_f0/stats.py`).
  - **Excess:** bright pixels of real trials vary more than photon noise explains, frame to frame and over seconds. The simulator matches each pixel's residual variance of single frames and of 64-frame means: a fast part scales the photon noise, and a slow part is added to the baseline as smooth fluctuations over 1–20 s (`SimConfig.excess = 0` turns it off). Without it, the pretrained network over-subtracted on the brightest sessions.
  - **Check:** `slap2-f0 simcheck <folder>` compares the simulated noise statistics with the real ones.
- **Model.** The branches see the movie and three whole-trial maps:
  - the mean image;
  - SiLo's variance factor;
  - the trial's variance relative to photon noise, which marks pixels with events.

  The fusion layers also get the blind window mean and variance at the output frame. The baseline is the mean image, plus a learned mix toward the window mean (bounded to ±1) and a learned correction (a factor of 0.5–2). Both start at zero, so the untrained network predicts the mean image.
- **Loss.** For the baseline: squared error weighted by 1 / photon variance, which is minimized by the mean, so it is unbiased at any brightness. For the noise: a Gaussian likelihood around the detached baseline, weighted by `--w_var`.
- **Validation split:** whole sessions are held out when there are at least 3, otherwise whole trials.
- **Every `--val_every` steps** it evaluates fixed windows: real (`real_*`) and simulated (`sim_*`) from the held-out files, and simulated or real from the training files (`tr_*`).
  - **Baselines compared:** the network, the mean image, the window mean and SiLo's moving median.
  - **Oracles,** which know the activity: the mean image minus the trial's mean activity (`oracle`, the best constant) and the blind window mean of movie − activity (`oracle_blind`).
  - **Scales:** per frame and on 8-frame averages (`mse8_*`, what SiLo Z-scores).
  - **Pixel classes:** all, footprint support (`sup_`), background (`bg_`) and labeled pixels outside every footprint (`lab_`). The last only counts where it has loss weight: with `--w_unfitted` > 0, or in a session without sources.
  - **Noise calibration:** `z_*`, and `r8`, which converts per-frame σ to SiLo's.
- **best.pt** is picked on held-out windows of the training data kind (`--select`). The default, `balanced`, is the network's 8-frame MSE relative to the mean image's, averaged over support and background pixels. The per-frame loss is too noisy to pick checkpoints on real data. The untrained network counts (score 1), so best.pt is never worse than the mean image.
- **Outputs** in `--out`: `config.json`, `metrics.csv`, `best.pt`, `last.pt` and `stats/` (cached whole-trial statistics).
- **Hardware:** on a GPU it uses mixed precision (bf16 where supported). A CPU manages about one step a minute even with a narrow model, so real training needs the GPU.
- **Per-session scores:** `slap2-f0 evaluate <folder> --ckpt name=best.pt ...` scores checkpoints on fixed real and simulated windows of every session (`eval.csv`, same metrics as validation). It also reports each session's residual variance over photon noise (`excess.csv`). With the training seed, a held-out session gets the windows its training validation used.
- **Controls without activity:** a session where SiLo finds no sources, such as a GFP mouse without a glutamate sensor, is exported with zero activity. All its valid pixels are exact targets with loss weight 1, so any bias on its labeled pixels (`lab_bias_*`) is spurious ΔF.
- **Ceiling:** `slap2-f0 ceiling <folder> --smooth 8` measures how much any baseline could gain over the mean image, using oracle references.

### On Code Ocean

The GPU capsule can't share the data capsule's environment: Code Ocean has no starter environment with both MATLAB and GPU support, and training doesn't need MATLAB.

1. **Create the capsule from a starter, not from Git.** Use the newest PyTorch starter with GPU (CUDA 12) support. Add the pip package `h5py`, and `numpy` if the starter lacks it. Choose a GPU machine: g5.4xlarge (one A10G with 24 GB, 16 vCPUs for data loading, 64 GB of memory) is a good start.
2. **Push the code into it.** The capsule's own git remote takes this repository's `main`, merged so that the capsule keeps its own `environment/` and `.codeocean/`. The capsule isn't linked to GitHub, so nothing flows back.
3. **Attach the training data.** Each export run's results are saved as a data asset (`f0train_<session>`); attach those to the capsule.
4. **Run.** For example, `--mode train --train_args "--data sim --steps 4000 --batch 2 --out_frames 192 --out_px 48 --max_hours 2.5"`. Then fine-tune with `--data real --init <best.pt>`, with the pretraining results attached as a data asset. Checkpoints and metrics go to `/results/train`, and the log to `/results/logs/train.log`. With `--finetune_args`, fine-tuning follows in the same run (`/results/finetune`); with `--eval_args "--windows 192"`, the best checkpoints are then scored on every session (`/results/eval`). `--mode eval --eval_args "--ckpt name=<path> ..."` scores checkpoints from earlier runs attached as data assets.

## Python

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"     # CPU torch: add --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pytest                 # Windows; .venv/bin/python elsewhere
slap2-f0 inspect <export folder>
slap2-f0 train <export folder> --out results/train
```
