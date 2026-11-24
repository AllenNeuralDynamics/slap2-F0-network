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
  run                  capsule entry point: --mode export (default), train, eval, predict or probe
  run_export.m         per session: trial table -> trim -> MultiRoiRegistration -> SILo -> export
  startup.m            MATLAB path for the capsule (SLAP2 reader, NoRMCorre, Fast_Tiff_Write)
  matlab/              exportF0TrainingData.m, trimTrialTableSLAP2.m
  slap2_f0/            Python package: io, stats, simulate, data, model, losses, train, ceiling, predict, view, cli
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
  - `max_duration_s` (seconds kept from the start of each session; default 900; 0 keeps everything);
  - `datasets` (`all`, or comma-separated mount names);
  - `n_workers` (GIAnT's parallel workers; 12 or fewer for long 2026 sessions, which ran out of memory with 32), `channels` (default `1`);
  - `export_workers` (trials exported at once, default 8, at most `n_workers`; about 15 GB each);
  - `compensate_online`, `peak_method`, `wmf_peakth` (GIAnT's `compensateOnline`, `peakMethod`, `wmfPeakth`; defaults 1, `wmf`, 3), `save_tif` (0: registered movies as deflated HDF5), `stats` (1: each trial's stats next to it);
  - `keep_giant_outputs` (`summary`, `all` or `none`).
- **Locally**, after SiLo has run on a session:

```matlab
addpath('code/matlab')   % with GIAnT-MATLAB (minus its tests folder) and the SLAP2 reader on the path
exportF0TrainingData('<session>/trial_table.h5', '<out>', 'rawDir', '<raw data folder>', 'nWorkers', 8)
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
- `shiftRC`, `blurSigma`, `activityGain`: the footprint offset, blur and gain applied to the activity. It is 1 in exports that have `activitySlope`. Older exports multiplied the activity by a fitted slope; divide by `activityGain` to get GIAnT's activity back;
- `blurShapeRatio`: at the chosen blur, the movie-on-activity slope of the flux in each source's core (≤ 1.5 px) over that of the flux 1.5–3.5 px out (median over sources; near 1);
- `activitySlope`: movie on activity for the flux in a 4 px disk, high-passed, at blur 0 (median over sources). This is the slope that older exports applied as `activityGain`, clamped to 0.5–1.5. It is a QC number, not a scale: a least-squares slope sits near the movie–activity correlation;
- `peakSlope`, `peakR`: movie on activity at source peak pixels, high-passed, for the activity as written (after `activityGain`);
- `diskSlope`, `diskR`: the same for the flux in a 4 px disk;
- `eventResidualFraction`: the residual's event-triggered peak relative to the movie's, with events picked on the activity (near 0 when the activity is removed);
- `eventRatioCore`, `eventRatio2px`: activity / movie event amplitude at ≤ 1.5 px and 1.5–2.5 px from each source. Events are picked on the movie 3–5 px away, so neither the core's noise nor the activity's selects them; sources touching GIAnT fit artifacts are left out. Both are near 1 when the activity has the movie's size. The selection favours spatially broad events. `eventRatioSources`, `eventRatioEvents` and `eventRatioArtifactSources` count what went in. The exporter computes them with `code/matlab/ringEventRatios.m`. Its tested Python reference is `slap2_f0/ringqc.py`; `python -m slap2_f0.ringqc <folder>` prints the same ratios for any export, old ones included (the activity divided by `activityGain`).

### How the activity is put on the movie's grid

GIAnT fits footprints H and traces at 200 Hz on its consensus canvas. The export maps H·X onto each trial's 80 Hz movie in three steps, and does not change its scale:

- **Time.** It uses the same kernel that made the movie: SLAP2 `getImages` weights the samples within max(3·dt, linesPerCycle) lines of a frame by exp(−|Δline|/dt), with dt = ceil(line rate / alignHz).
- **Space.** It applies the sub-pixel trial offset that best aligns the movie with the activity.
- **Blur.** It applies the Gaussian blur (from `blurSigmas`) for which the movie-on-activity slope of the flux in each source's core (≤ 1.5 px) equals that of the flux 1.5–3.5 px out. Scaling the activity scales both slopes, so the choice depends only on the shape of the activity, not on its size. The 80 Hz frames are slightly blurrier than the 200 Hz data the footprints were fitted to.

Older exports (without `activitySlope`) also scaled the activity by the disk-flux slope (`activityGain`, median 0.63), and chose the blur that made the peak-pixel slope match it. A least-squares slope of the movie on a noisy activity sits near their correlation, so that gain removed about half of GIAnT's activity (`docs/plan.md`, "Activity in the training targets").

## Training

`slap2-f0 train <folder>` trains the slab blind-spot network on the exported trials under `<folder>`.

- **Architecture** (`docs/figures/network_architecture.svg`): the deep architecture by default. Pointwise residual blocks follow every dilated block (`--branch_mlp 1`) and the fusion (`--fuse_blocks 4`), starting as the identity (`--pointwise_init zero_out`); `--fuse_blocks 0 --branch_mlp 0` gives the shallow model (v9, the S4 control). A shallow `--init` checkpoint gets the new blocks as identities, so it starts out computing what it did.

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

### Viewing the training data and baselines

`slap2-f0 view` shows exported trials (the training data) or `slap2-f0 predict` output in a browser, frame by frame.

```bash
slap2-f0 view --codeocean                     # the export data assets on Code Ocean, downloaded on demand (CODEOCEAN_API_TOKEN)
slap2-f0 view <export folder>                 # exported trials on disk
```

- **Training data** (a folder of exported trials without predict output, `--data export`, or `--codeocean`).
  - **Panels:** the movie; GIAnT's fit (its activity); the training target (movie − activity); and the target minus its trial mean.
  - **Loss mask:** "Blank what the loss ignores" blanks, in the target panels, the pixels and frames that real windows give no loss. These are labeled pixels outside every footprint (`--w_unfitted`), GIAnT fit artifacts (`stats.unfit_pixels`, computed while the trial loads), the first and last `--edge_s` seconds, and discarded frames. Traces shade those frames.
  - **QC line:** calibration QC, and whether training uses the trial. The trial filter (`--min_peak_r`, `--min_path_trials`) is applied over the session's `export_summary.json` when there is one; the session's role comes from `sessions/export_phase*.csv`.
  - **Code Ocean:** `--codeocean` finds the assets by search (`--co_query`, `--co_name`), so no asset IDs are needed. Trials download in parallel ranges into `--cache_dir`/slap2_f0_downloads (default `$SLAP2_F0_TRIAL_CACHE` or `D:/f0_trial_cache`). Beyond `--cache_gb` the least recently used are deleted, but only from that folder.
  - **Navigation:** pick a session and a trial; Page Up and Page Down step through trials. The session's next trial loads in the background.

`slap2-f0 predict` runs saved checkpoints over whole trials and adds SiLo's baseline; `slap2-f0 view` shows the result.

```bash
slap2-f0 predict <export folder> --ckpt v6=<best.pt> v5b=<best.pt> --files 878030_2026-08-28_14-41-35/Path1/trial_002 --out <predictions>
slap2-f0 view <predictions> --models v6,v5b   # opens http://127.0.0.1:8765; --models lists those first
```

- **predict** tiles each trial in the checkpoint's training width (`out_px` pixels plus the model's reach), so every pixel sees the same kind of window as in training, and by default each tile spans the whole trial (or as many frames as fit in GPU memory), placed to cover the valid pixels with few tiles. On a GPU it runs the model in fp16 (`--precision`; within 3e-4 of fp32). It writes one file per trial, `<predictions>/<session>/<path>/<trial>.h5`, with the movie, GIAnT's activity and each baseline at the trial's valid pixels (float16). It needs a GPU: on an A10G v6 takes 45–110 s per trial of the five viewer trials, on a CPU about 30× longer. `--gain movie` takes the trial's photon gain from the movie alone, for trials without GIAnT activity. On Code Ocean: `--mode predict --predict_args "--ckpt name=<path> ... --files ..."` (results in `/results/predict`). To add a checkpoint later, run it alone with `--no_movie --no_silo`; the viewer merges files for the same trial.
- **view** shows, per frame, the data, the data minus GIAnT's activity, and each model's baseline and residual (data − activity − F0, data − F0, ΔF/F, or data − activity − F0 in units of photon noise, √(gain · F0 / frames averaged)). Images can be shown linear or square-root scaled. Baselines are blanked on discarded frames, where no network was trained and outputs mean nothing. Scroll over the images to step through time, average up to 64 frames, or play. Ctrl + scroll zooms all panels together, and drag pans. Clicking a pixel plots its traces (or a small ROI's) over the whole trial: data, baselines and residuals, with smoothing, zoom and hover values.

### Baselines for SiLo's localization

`slap2-f0 predict-registered <motion_correction folder> --ckpt <best.pt> --out <dir>` runs a checkpoint over GIAnT's registered 80 Hz movies (`*_REGISTERED_DOWNSAMPLED-80Hz.tif` or `.h5`, the HDF5 movies GIAnT-Python writes by default, with `*_ALIGNMENTDATA.h5`), prepared as SiLo prepares them for localization (motion-discarded frames, valid pixels), and writes `<dir>/<trial stem>_BASELINE-80Hz.h5` (`/baseline`, frame × A × B float16; MATLAB's `h5read` gives SiLo's orientation). The GIAnT-MATLAB branch `localization-network-baseline` (local) reads these when `params.localizationBaselineDir` is set and uses them in `localizeSources_vIM` instead of the 4 s moving median. `tools/localization_eval/locnet_*` compare baselines in localization on single trials (simulated with known sources, or real). To check whether a SILo run's sources are real, `source_patches.py` cuts movie patches around two runs' sources (plus matched control pixels) and `source_flash_check.py` tests each for a synapse-sized flash with sensor decay, measured outside the window used to pick its events; `activity_viewer.py` writes a self-contained HTML page comparing runs' activity images and sources, with those test results.

### Registration and baselines on the GPU, without MATLAB

`code/run --mode register` runs on the GPU capsule. For every attached raw SLAP2 session, GIAnT-Python (branch `giant_python_ref`, fetched at the start with SLAP2_Utils and its pure-Python dependencies) builds the trial table (its port of `buildTrialTableSLAP2`, trimmed to `--max_duration_s`) and registers it (its port of `MultiRoiRegistration`, `--compensate_online`); with `--ckpt`, `predict-registered` then writes the baselines. Results per session are in `/results/register/<session>`: `trial_table.h5` (with the `motion_correction` group) and `trial_table_untrimmed.h5`, `motion_correction/` (alignment data; the registered movies are deleted unless `--keep_registered 1`), `baselines/` and `run_info.json` (times, GPU, peak memory); `/results/register/summary.json` lists every session and `/results/logs/usage.txt` follows memory, disk and GPU use. SILo still needs GIAnT-MATLAB on a CPU machine.

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
