# Blind-spot baseline network for SiLo: plan

Written 2026-11-14 and updated the same day with the decisions at the end and the resolved training target.

Goal: replace the baseline that SiLo uses to localize sources with a network that predicts each pixel's baseline from pixels distant in space and time. The baseline should not rise when there is an event, and the network should also give a calibrated, time-varying noise estimate for Z-scoring. Later, the same baseline could seed GIAnT's fit. The project lives in a new repository, and training data are produced on Code Ocean.

## What SiLo does now

`localizeSources_vIM` (called by `loadAndProcessTrialAsync` for every trial) works on the 80 Hz registered movie:

1. **Smooth:** a moving mean over 8 frames (half of `denoiseWindow_s` = 0.2 s), weighted by SLAP2's per-frame variance factor `varFacDS`.
2. **Baseline:** a moving median over `baselineWindow_Glu_s` = 4 s (320 frames).
3. **Noise model:** σ = √(Vk·baseline·varFac + Vb). Vb and Vk are fitted on the first 500 frames, and Vb is inflated by the fudge factor `VIF` = 1.67.
4. **Detect:** the Z-score is passed through an exponential matched filter (τ = 24 ms), then a difference-of-Gaussians filter (σ = 1.33 px), then non-maximum suppression. That builds the activity image whose peaks become sources.

The network would replace steps 1–3: its two outputs, the baseline μ and the noise σ, take the place of `IMb` and `stdIM`.

## Measurements behind the numbers below

All from one session: 780125 Neuron1 2025-03-14, path 1, trial 10 (80 Hz, 320 × 845 px, 2253 frames, 36,498 valid pixels), plus that session's GIAnT outputs. The training target was built as the 80 Hz movie minus GIAnT's activity: per-trial footprints × `dF_denoised`, resampled from 200 to 80 Hz. Scripts: `tools/measure/`.

| Quantity | Value | What it sets |
|---|---|---|
| Footprint radius holding 90% / 95% of the mass | 3.1 / 3.5 px | spatial blind spot |
| Event time course (channel 1, event-triggered average) | peak at +5 ms, below 10% of peak by +90 ms (7 frames at 80 Hz) | temporal blind spot |
| Activity at source peak pixels | above 10% of baseline in 72% of 80 Hz frames; mean 45% of baseline; 11 events/s per source (median) | events are dense, so the context always contains them |
| Noise correlation, space (fast residual) | 0.41 one column away, 0.27 one row away, 0.12 diagonal, ≤ 0.08 at 2 px | the blind spot must also hide correlated noise |
| Noise correlation, time | 0.59 at lag 1, 0.13 at lag 2 | the 80 Hz frames are formed with a kernel wider than one frame |
| Noise level | variance ≈ 6.6 × mean; a few photons per pixel per frame | count statistics, not Gaussian, on dim pixels |
| Event bias of today's baseline at source peak pixels | +42% (10th–90th percentile +23% to +72%) higher than the same filter on the residual; r = 0.91 with activity level | the problem to fix |
| Same, with GIAnT's `splitFreq` lower envelope instead | −2% (−12% to +8%) at source peaks, but 0.31× the moving median on the dimmest pixels | not a fix: localization gets worse (see Phase 0) |
| Residual 0.2–4 s component on footprint pixels | std 32% of mean, two thirds of it photon noise | how much the baseline moves beyond noise |
| Its spatial correlation | 0.29 at 1–2 px (mostly correlated noise), 0.027 at 2–4 px, ≤ 0.013 beyond 4 px | there is little shared fast background |
| Residual > 4 s component, spatial correlation | 0.51 (1–2 px), 0.23 (2–4), 0.07 (4–8), 0.03–0.05 (8–64), 0.02 (64–128) | slow drifts are shared over tens of pixels |
| Linear blind-spot predictors of the 0.2 s-smoothed residual (R², ceiling ≈ 0.33 from noise) | same pixel, 0.2–2 s away: 0.14; mean of pixels 4–128 px away, same time: 0.003; all combined: 0.17 | most of the predictable baseline is temporal |
| Global (whole-field) component of the 0.2–4 s band | R² ≈ 0 | no need for a big same-time spatial context on these data |
| Channel 2 | event-locked (peak dF/F 1.6 at +15 ms); band correlation with channel 1 at the same pixel 0.01 | not a clean background reference; leave it out of v1 |
| Activity → 80 Hz alignment | resolved: a boxcar window gave slope 0.69. The movie's own `getImages` kernel (weights exp(−&#124;Δline&#124;/dt), dt = 134 lines) gives 0.80; adding the trial's sub-pixel offset gives 0.90; adding a 0.5–0.75 px blur of the footprints gives 0.97–1.00. Flux in a 4 px disk has slope 0.95–0.99 (r 0.87–0.89), so the units agree | the export applies all three per trial and records them with QC |

Caveat: these are one trial of one session. Phase 0 repeats them on more sessions.

## Answers to the four questions

### Blind spot and context

| | Hidden (blind spot) | Seen (context) |
|---|---|---|
| Space | 9 × 9 px box (radius 4) | 65 × 65 px patch (±32), plus per-frame whole-field features |
| Time | ±8 frames (±100 ms) | ±256 frames (±3.2 s); the far part can be at reduced time resolution |

- **Radius 4 px** covers 95% of a footprint (3.5 px) and all of the noise correlation (≤ 2 px).
- **±8 frames** follows from the event's extent. An event that affects frame t starts between t−7 and t+1, and is visible from 1 frame before its onset to 7 after. Hiding all of it takes ±8 frames, which also covers the 2-frame noise correlation.
- **The blind spot is a box in space × time, not a slab.** Pixels inside the 9 × 9 box are visible more than 8 frames away. That matters, because the most informative pixels for the slow baseline are the pixel itself and its immediate neighbours (slow-component correlation 0.5 at 1–2 px).
- **±3.2 s of time** covers the 1–4 s fluctuations that temporal context predicts. Predictive value peaked with context out to 2 s, and today's baseline window is 4 s.
- **±32 px of space** covers the spatially shared slow drifts (correlation 0.03–0.07 out to 64 px). Global changes such as z-drift, laser power or bleaching go in as cheap per-frame features: field mean, motion offsets, `recNegErr`, mean `varFacDS`.
- **Ablate:** radius 2–5, gap ±3/5/8/12, time context ±64/128/256/512, space context ±8/16/32/64.

### Architecture

Build the blind spot into the network, so that one forward pass predicts every pixel. Masking-based training (Noise2Void-style) would need hundreds of inference passes for a blind spot this large.

The context is the complement of the box: everything outside it. That is the union of six half-spaces:

- **Past branch:** frames ≤ t−9, all pixels in the patch.
- **Future branch:** frames ≥ t+9, all pixels.
- **Four spatial branches:** pixels more than 4 px above, below, left or right, at any frame. These use half-plane convolutions as in Laine et al. 2019, with a 4 px offset, and share weights through 90° rotations.

Each branch is built from convolutions whose receptive field stays inside its half-space:

- **Time:** causal, dilated temporal convolutions (kernel 3, dilations 1…128, about 510 frames of reach), with the output shifted by 9 frames.
- **Space:** a per-frame 2-D encoder that downsamples by 4×, so the temporal stacks run on small feature maps.

The branches meet only in 1 × 1 × 1 layers, which keeps the blind spot exact. A unit test checks that the gradient of each output with respect to every input inside its box is exactly zero.

Build it in stages, because the measurements say most of the gain is temporal:

| Version | Context | Expected value on 780125 |
|---|---|---|
| v0 | one pixel's own trace, ±8-frame gap | the bulk of the temporal gain; very cheap |
| v1 | past + future branches over the whole patch (a slab: all pixels hidden within ±8 frames) | adds the shared slow drifts; still cheap |
| v2 | + the four spatial branches (the full box) | uses pixels farther away at the same time; worth it only if some datasets have a shared fast background (neuropil, hemodynamics, motion artefacts) |

**Inputs:**
- the activity channel, in counts divided by a per-trial scale;
- the trial mean image, as a static channel;
- `varFacDS`;
- the valid / NaN mask and discarded-frame mask;
- per-frame scalars (motion offsets, `recNegErr`, field mean).

**Outputs:**
- the baseline μ, as a positive multiplier on the mean image;
- the noise σ², parametrized as (g·μ + v₀)·exp(s). The network learns g, v₀ and the per-pixel, per-frame correction s, so the Poisson-like default is built in.

### Loss

- **Target:** the residual at the centre pixel, R = Y₈₀ − a·H·X₈₀, at every valid pixel and frame. The architecture provides the blind spot, so no random masking is needed.
- **Likelihood:** heteroscedastic, either Gaussian NLL with β = 0.5 (β-NLL, so the variance head doesn't stall learning of the mean) or Student-t NLL with learned ν. Unfitted events give the residual a heavy positive tail, and Student-t is robust to it. Under symmetric zero-mean noise the mean head converges to E[baseline | context], which is the quantity wanted.
- **Loss mask:**
  - Exclude NaN and discarded frames.
  - Include footprint-support pixels and unlabelled (off-dendrite) pixels.
  - Down-weight or exclude labelled pixels outside every footprint. On 780125 that is 60% of labelled pixels, and their residual still contains activity that GIAnT never fitted.
  - Keep trial edges, with a padding mask: trial ends are where today's baseline fails worst (issue #17).
- **Ablations:** Gaussian vs Student-t; a mixture (noise + positive exponential event component), which would also give an event posterior; and the mask choices above.

### Time-varying uncertainty

Yes: the σ² head trained by NLL is exactly that.

- **Expected drivers:** μ itself (count statistics); frames near motion or discards; partially sampled pixels; and baseline uncertainty near trial edges and gaps. `varFacDS` barely varies within a trial on 780125 (CV 0.04 per pixel), so expect μ to explain most of it. A free correction s has to earn its place with better held-out NLL and calibration.
- **Correlated noise in the matched filter:** the 80 Hz noise is correlated in time (0.59 at lag 1), so filtered Z-scores don't have unit variance. Today's `VIF` = 1.67 absorbs this ad hoc. With a calibrated per-frame σ and a per-session noise autocorrelation (cheap to measure), the variance after the τ = 24 ms filter can be computed exactly, and `VIF` dropped.
- **Calibration checks:** on frames GIAnT marks inactive, Z should have mean 0 and variance 1. Coverage should be right at ±2σ and ±3σ. The activity-image peak threshold (`peakth` = 8) should then mean the same thing on every dataset.

## Phases

**Phase 0: headroom and target construction (in GIAnT and scratch; no new repo yet)**
1. **Cheap comparator. Done; it lost.** I added a `splitFreq` option to `localizeSources_vIM` (GIAnT branch `localize-splitfreq-baseline`, f8c24ef, local) and scored localization with `tools/localization_eval/`.
   - **Simulations:** median F1 drops from 0.712 to 0.620 (sims 1 and 2, two 180 s windows each; the other 10 simulations' movies aren't on this machine). It is worse at every threshold.
   - **780125:** fewer sources match the final GIAnT run.
   - **Why:** the lower envelope is far too low on dim pixels (0.31× the moving median), which skews the Z-scores.
   - **Consequence:** the moving median is the comparator. Any replacement must be unbiased at low counts as well as robust to events.
2. **Fix the target. Done.** The 0.69 slope came from three mismatches between my resampling and the way the 80 Hz movie is made (see the alignment row above). Two 780125 trials now give peak slopes of 0.97 and 1.00, with the event-triggered residual at 6% and 17% of the movie's.
3. **Repeat the scale measurements** on 3–5 more sessions (other mice, depths and brightness) to confirm the blind spot and context sizes.

**Phase 1: repository and data on Code Ocean**
- **Repository layout** (as built; the repository is itself a Code Ocean capsule, see README.md):
  - `code/matlab/exportF0TrainingData.m` runs right after SiLo, where the footprints, offsets and frame lines are all at hand. Per trial it writes chunked HDF5 with: the 80 Hz movie (channel 1 by default), the activity on the trial's grid, masks, the variance factor and per-frame scalars.
  - `code/slap2_f0/` holds the reader, window sampler, models, losses and training loop. Prediction and evaluation are still to come.
  - `code/tests/` holds the blind-spot exactness test, likelihood tests and a reader test.
  - `code/run`, `code/run_export.m`, `environment/` and `metadata/` make up the data capsule.
- **Data capsule:** reuses the integration-test capsule's SLAP2 path (`buildTrialTableSLAP2` → `MultiRoiRegistration` → `SILo`) at a pinned GIAnT commit, then the export.
- **Pin the GIAnT version first.** The target inherits GIAnT's split between baseline and activity, so pin the version first: after #40 (penalized-spline baseline) is merged.
- **Volume:** about 36k valid pixels × 2253 frames per trial → about 330 MB per trial in float16 for input + target. That is about 10 GB per session (15 trials × 2 paths), or about 300 GB for 30 sessions.
- **Diversity over volume:** aim for at least 10 mice, both paths, and a range of depths, brightness and motion. Hold out whole mice for testing.

**Phase 2: training (PyTorch, GPU capsule)**
- Train v0 → v1 (→ v2 if warranted) on 64 × 64 × 513-frame patches, with mixed precision.
- Augment with flips and gain scaling. No 90° rotations of the input for v0/v1: SLAP2 noise is anisotropic (0.41 along a row, 0.27 along a column).
- Run the ablations listed above.

**Phase 3: evaluation**
- **Simulations, which have a true baseline:** baseline error near and away from events, and Z-score calibration.
- **Localization:** today's moving median vs the network, with `tools/localization_eval/`, on the simulations (ground-truth F1, position error) and on real data (agreement with final sources, cross-trial stability, SNR of the resulting sources). All 12 simulations need their movies, from `s3://aind-scratch-data/iGluSnFR_simulations_default/` (about 35 GB) or on Code Ocean.
- **Runtime:** per trial, on GPU and CPU.

**Phase 4: integration**
- **v1 of the integration:** a Python `predict` step after registration writes `*_BASELINE-80Hz.h5` (μ, σ) next to the registered movie. SiLo uses it when present, behind a parameter. That avoids a Deep Learning Toolbox dependency; an ONNX import into MATLAB can come later.
- **Optionally:** upsample μ to 200 Hz to initialize `fitB`.
- **Retrain once** on targets regenerated with the improved localization, since the targets depend on GIAnT's fits.

## Risks

- **Headroom.** The moving median's event bias (+42% at source peaks) is not simply harmful: the obvious fix (`splitFreq`) made localization worse, because it traded that bias for a brightness-dependent one. The network has to remove the event bias while staying unbiased on dim pixels, and track 1–4 s fluctuations (R² ≈ 0.14–0.17 of a ≈ 0.33 ceiling with linear predictors), with calibrated σ. Phase 3 must show this improves localization over the moving median.
- **The target is GIAnT's definition of baseline.** With activity above 10% of baseline in 72% of frames at source peaks, the split between baseline and activity is model-dependent. The 0.69 gain is resolved. The per-trial QC (peak and disk slopes, event residual) flags trials where the activity doesn't line up with the movie.
- **Contaminated targets.** Activity outside fitted footprints stays in the residual. The loss mask and the robust likelihood handle it.
- **Domain shift** across indicators, microscopes and imaging settings. Track it with held-out mice and sessions.

## Decisions (2026-11-14)

1. **Repository:** AllenNeuralDynamics/slap2-F0-network.
2. **Sessions:** all iGluSnFR4 SLAP2 sessions in multiROI mode on `aind-open-data`, starting with the 4 integration-test sessions (780125_2025-03-14, 803121_2025-10-29, 878030_2026-08-28, 885233_2026-10-02).
   - A survey found 142 qualifying sessions from 33 mice, about 9 TB of raw data.
   - The list and the criteria are in `sessions/`.
3. **GIAnT version:** #40 (penalized-spline baseline) is merged, so targets come from GIAnT main at or after d2fb85c.
4. **Integration:** a Python step that writes baseline and noise files next to the registered movies.

## Results: first training run and the ceiling (2026-11-14)

**Training run c9f8664e** (4 sessions, 878030 held out, 4000 steps, Student-t loss): negative.
- On the held-out session the network's baseline is worse than the mean image it starts from: MSE 6.40 vs 5.50. SiLo's moving median scores 5.65.
- Validation NLL rose from 1.85 (untrained) to 2.01. The network predicts too low, z mean +0.3.
- Training loss was flat at about 1.8 the whole time; the early "rise" is the running average warming up from the first batch. `train.py` now also evaluates fixed training windows (`tr_*` columns), to tell "learns nothing" from "doesn't generalize".

**How much is there to learn?** `slap2-f0 ceiling` (`code/slap2_f0/ceiling.py`) compares reference baselines against the target on one trial per session (1493–2254 frames each). The numbers are MSE change vs the mean image, on 8-frame averages of the target (what SiLo Z-scores):

| session (scale) | pixels | oracle constant | oracle blind temporal | blind movie average | SiLo |
|---|---|---|---|---|---|
| 780125 (5.5) | support | −9.4% | −12.8% | −2.2% | −12.0% |
| | background | −0.4% | +1.6% | +2.0% | +7.7% |
| 803121 (9.1) | support | −7.4% | −8.2% | +0.2% | −7.6% |
| | background | −0.1% | +2.2% | +2.3% | +1.7% |
| 878030 (2.2) | support | −8.1% | −6.7% | +2.2% | −1.9% |
| | background | −0.1% | +2.5% | +2.5% | +12.1% |
| 885233 (3.9) | support | −11.0% | −9.7% | +1.5% | −6.1% |
| | background | −0.5% | +1.9% | +2.5% | +6.1% |

- **Oracle constant:** mean image minus mean activity. **Oracle blind temporal:** a blind-spot average of the residual itself. **Blind movie average:** the same average of the movie, events included.
- **Per frame** (the training loss), all of these shrink 3–4×: the oracle constant gains only 0.4–3.2% on loss pixels. Photon noise (under 1 photon per pixel per frame) dominates.
- **What can be learned:** excluding events on footprint pixels, worth 7–11% at the localization scale. Temporal drift adds little on these short trials. On background pixels the mean image is already optimal.
- **What SiLo loses:** the moving median reads low at low counts (by 0.09–0.49 of the trial scale), costing 2–12% on background, worst on the dimmest session.
- **Why the run failed:** a per-frame loss sees only a 1–3% effect. The Student-t likelihood pulls the location below the mean on skewed low-count data, the same failure as `splitFreq` and the median.
- **Naive event clipping fails the same way.** Leaving out frames 2σ above a whole-trial mean is biased low on background pixels by 0.15–0.45 (+11–16% MSE): it removes noise tails along with events. A replacement for the median has to be mean-unbiased at low counts.

## v2: simulator first (2026-11-15)

Decided after the ceiling analysis:
1. **Unbiased loss.** The baseline is fitted by squared error weighted by 1 / photon variance, which is minimized by the conditional mean. The noise head gets a Gaussian likelihood around the detached baseline.
2. **Start no worse than the best simple alternative.**
   - **Never worse at the start:** the baseline is the mean image plus a learned mix toward the blind window mean, times a learned correction. Mix and correction start at 0, so the untrained network predicts the mean image, and step 0 is a candidate for best.pt.
   - **Window statistics enter only at the output frame.** The blind window mean and variance (gap < |Δt| ≤ gap + 254) are computed inside the model and fed to the fusion layers there. A context frame's window would reach into the hidden frames, so they never go into the branches.
   - **Variance input:** the branches also see the trial's variance relative to photon noise. Pixels with events have more variance than photon noise explains.
3. **Pretrain on simulations, then fine-tune on real residuals.** A noise-free target removes the photon noise from the loss, which buried the 1–3% that can be learned in v1.
   - **Built on real trials:** the simulator keeps their geometry, brightness and events.
   - **Matched noise statistics:** gain, correlations at 1–2 frames and 1 pixel, long-average inflation and skew.
   - **Unlimited data:** each training window is a new virtual trial.
   - **Calibration:** simulations give exact bias and z calibration at every brightness.
   - **Drift:** kept modest (bleaching up to 5%, fluctuations up to 2%) because real trials show little.
4. **Per-trial gain.** The noise model uses each trial's measured variance/mean ratio. It is constant across trials but differs between sessions and paths (7.9–14.6 in the 80 Hz movie), which [GIAnT-MATLAB#45](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB/issues/45) tracks.

The context window is noisy as a baseline but fine for Z-scores. Background pixels get 0.3–0.9 photons per frame, and correlated noise makes N frames worth about N/3 independent ones. Per-pixel relative SD: 8–16% for the 508-frame window (5–9% pooled over 3×3), 6–11% for the whole trial. In SiLo's 8-frame Z-scores this adds about 2% variance. Trials are 19–28 s, so longer windows can't help much; pooling along structure can.

**Simulator check** (`slap2-f0 simcheck`, one whole simulated trial per session, real / simulated):

| session | gain | ρ 1 frame | ρ 2 frames | ρ next row | ρ next column | F64 | skew |
|---|---|---|---|---|---|---|---|
| 780125 | 7.97 / 7.95 | 0.680 / 0.673 | 0.331 / 0.332 | 0.269 / 0.271 | 0.358 / 0.361 | 3.35 / 3.30 | 3.41 / 3.38 |
| 803121 | 10.74 / 10.65 | 0.503 / 0.506 | 0.180 / 0.171 | 0.236 / 0.238 | 0.377 / 0.380 | 2.49 / 2.38 | 2.69 / 2.71 |
| 878030 | 12.36 / 12.27 | 0.602 / 0.603 | 0.245 / 0.239 | 0.233 / 0.236 | 0.386 / 0.390 | 2.67 / 2.48 | 4.80 / 4.79 |
| 885233 | 9.79 / 9.73 | 0.667 / 0.671 | 0.315 / 0.315 | 0.243 / 0.246 | 0.377 / 0.380 | 3.06 / 2.83 | 3.60 / 3.57 |

The simulated F64 runs 2–8% low; the real background still holds some slow structure.

## v2 results (2026-11-15)

The run did 2557 steps of simulation pretraining (1.6 h) on 52 training trials, then 1442 steps of fine-tuning on real residuals (0.8 h, learning rate 5e-5). Session 878030 (19 trials) was held out. Numbers are 8-frame MSE change vs the mean image on 48 fixed validation windows.

| | support | background | all |
|---|---|---|---|
| held-out simulations, after pretraining | −19% (blind window: −31%) | −21% | −8% |
| real 878030, after pretraining only | −0.6% | +0.2% | +0.1% |
| real 878030, after fine-tuning | −1.4% | +0.2% | 0.0% |
| real 878030, SiLo | −5.6% | +11.1% | +8.7% |

- **Calibration:** z variance 0.98 on simulations and 0.98–0.99 on real data; z mean under 0.01.
- **Bias moved about the same in absolute terms.** On support pixels the bias went from −0.28 to −0.23 (trial-scale units) on simulations and from −0.32 to −0.28 on real data. The percentages differ because the real 8-frame target carries photon noise, which dominates its MSE.
- **The net is not near a limit.** On held-out simulations the plain blind window beats it on support pixels (−31% vs −19%). Fine-tuning was still improving steadily when it stopped (real support bias −0.28 → −0.24).
- **Per-frame loss can't pick checkpoints.** Real validation loss was 0.9731 at the start of fine-tuning and 0.9730 at the end, while support MSE improved by 1%.
- **Training is unstable.** Pretraining loss spiked to about 6·10⁴ at step 250. Real validation MSE jumped to 2.4–13 (vs 1.8) at 4 of 13 checkpoints and then recovered. Three outputs have too much room:
  - the mix toward the window mean is unbounded;
  - the multiplicative correction is clamped only at e^±5;
  - the variance-factor input is used raw.

## v3 results (2026-11-15)

v3 bounds the heads (mix in ±1, baseline correction 0.5–2×, noise correction about 20×, log variance factor clamped to ±3) and picks best.pt on 8-frame MSE vs the mean image, averaged over support and background. Two oracles that know the activity are now reported: the mean image minus the trial's mean activity (the best constant) and the blind window mean of movie − activity. The run did 4000 steps of simulation pretraining (2.4 h, all planned steps), then 1700 steps of fine-tuning on real residuals (0.9 h, learning rate 1e-4). Same split and validation windows as v2: session 878030 held out, 48 fixed windows, 8-frame MSE change vs the mean image.

| | support | background | all |
|---|---|---|---|
| held-out simulations, after pretraining | −83% | −51% | −40% |
| simulations, oracle (best constant) | −95% | −9% | −43% |
| simulations, oracle (blind window) | −84% | +168% | +43% |
| real 878030, after pretraining only | −6.4% | +0.5% | −0.6% |
| real 878030, after fine-tuning (best.pt, step 600) | −6.0% | 0.0% | −0.9% |
| real 878030, SiLo | −5.6% | +11.1% | +8.7% |
| real 878030, oracle (best constant) | −9.7% | −0.1% | −1.5% |
| real 878030, oracle (blind window) | −6.8% | +2.4% | +1.0% |

- **Stable.** No loss spikes in either phase, and validation moved smoothly at every checkpoint.
- **Longer pretraining did the work.** On held-out simulations the net now matches the blind-window oracle on support and beats the best constant on background, because it follows drift without the window's noise. Real support improved alongside simulated support up to step 2500–3000 (−6.6%), then held.
- **Pretraining alone beats SiLo on real support pixels** (−6.4% vs −5.6%) without SiLo's background penalty (+0.5% vs +11.1%). That is two thirds of the way to the best constant (−9.7%). It rests on one held-out session.
- **The simulator overshoots the bias correction for real data.** Real support bias went from −0.32 to +0.12 (crossing 0 at step 2500), while simulated bias settled at −0.04.
- **Fine-tuning on real data added nothing.** The selection score went from 0.9703 (pretrained start) to 0.9698 at best, a tie. Fine-tuning pulled the mix from 0.26 to about 0, brought real support bias back to −0.07, and lost much of the simulation skill (support −83% → −68%, background −51% → −1%).
- **Calibration:** z variance 0.95–1.0 and z mean under 0.02 on both data kinds.
- **The simulated "all" column is erratic** (+31% at step 1500, −40% at step 4000) because it includes labeled pixels outside every footprint. Those carry weight 1 in simulations and 0 on real data. After fine-tuning the net is 27% worse than the mean image on them in simulations.

## v3 on all four sessions (2026-11-15)

`slap2-f0 evaluate` scored both v3 checkpoints on 192 fixed windows per session. The pretrained network never saw real residuals, so every session is new to it (the trials of 780125, 803121 and 885233 served as simulation templates). The fine-tuned network trained on real residuals of 780125, 803121 and 885233; only 878030 is held out for it. Numbers are 8-frame MSE change vs the mean image, support / background.

| session | pretrained only | fine-tuned | SiLo | oracle (best constant) |
|---|---|---|---|---|
| 780125 | +15.9% / +3.7% | −6.9% / 0.0% | −9.0% / +8.6% | −11.7% / −0.4% |
| 803121 | +6.2% / +11.3% | −4.6% / +0.1% | −8.0% / +2.1% | −6.6% / 0.0% |
| 878030 | −5.0% / +0.6% | −5.2% / 0.0% | −4.6% / +10.9% | −8.2% / −0.1% |
| 885233 | −5.8% / +0.4% | −6.0% / −0.2% | −5.8% / +5.5% | −10.7% / −0.5% |

- **Pretraining alone does not carry over.** In 780125 and 803121 the pretrained network over-subtracts (support bias +0.25 and +0.12) and is worse than the mean image. Its noise is too low there too: z variance 1.24 and 1.23, against about 1.0 in the other two sessions and on simulations. The v3 conclusion that pretraining alone beats SiLo rested on 878030, one of the two sessions where it works.
- **Fine-tuning fixes it.** The fine-tuned network is never worse than the mean image on background and is 4.6–6.9% better on support. SiLo does better on support in 780125 and 803121, by 2–3 points, but costs 2–11% on background.
- **Cause: real residuals vary more than the simulator's photon noise, on bright pixels and at two time scales** (one trial per session, real vs simulated residual on bright labeled pixels):
  - fast: 10–30% more variance per frame, with the photon noise's time structure;
  - slow: means over 256 frames (3.2 s) vary 2–15× more than in simulations (15× in 780125), so the baseline itself moves over seconds.

  The excess is largest in the two brightest sessions, 780125 and 803121. It is not motion: it hardly depends on the mean image's gradient. Nor is it one shared factor: removing the frame-wide mean fluctuation leaves it unchanged. A network trained on photon-only simulations reads the extra spread as activity.
- **Confirmed by adding the measured excess to simulated windows.** The test used windows centred on support pixels of one trial per session, with the excess per-frame variance added to the movie as AR(1) noise. The pretrained network then over-subtracts on simulations too:

  | session | simulated | simulated + excess | real |
  |---|---|---|---|
  | 780125 | −0.02 | +0.19 | +0.10 |
  | 803121 | −0.02 | +0.35 | +0.30 |
  | 878030 | −0.02 | +0.37 | +0.26 |
  | 885233 | −0.08 | +0.13 | +0.00 |

  Values are support bias. Its correction below the mean image grows from 0.14–0.31 to 0.35–0.71, against 0.26–0.59 on real data.

## v4: the simulator adds the measured excess (2026-11-15)

- **Stats** add per-pixel residual variance of single frames and of 8- and 64-frame means.
- **The simulator splits each pixel's excess into two parts** (3×3-smoothed): a fast part that scales the photon noise, and a slow part added to the baseline as smooth fluctuations with time constants of 1–20 s.
- **Check on one trial per session:** real/simulated variance of 1-, 8- and 64-frame means on the brightest 10% of support pixels:

  | session | photon-only simulator | with the excess |
  |---|---|---|
  | 780125 | 1.25 / 1.37 / 2.23 | 1.07 / 1.10 / 1.25 |
  | 885233 | 1.13 / 1.24 / 1.44 | 1.04 / 1.11 / 1.17 |

  It is within about 10–20% elsewhere. Some variance at the 8-frame scale is still missing.
- **The run** repeats v3 (same split, steps and seed) with the new simulator, fine-tunes, and scores both checkpoints on all four sessions.

## v4 results (2026-11-15)

- **Pretraining on the excess-noise simulator now works in every session.** 8-frame MSE relative to the mean image on real windows, footprint support / background. 878030 is the held-out session; the fine-tuned scores for the other three are in-sample (fine-tuning trained on their real windows):

  | session | v3 pretrained | v4 pretrained | v3 fine-tuned | v4 fine-tuned | SiLo | oracle (support) |
  |---|---|---|---|---|---|---|
  | 780125 | +15.9 / +3.7 | −7.1 / +0.3 | −6.9 / 0.0 | −9.0 / −0.1 | −9.0 / +8.6 | −11.7 |
  | 803121 | +6.2 / +11.3 | −5.0 / +0.4 | −4.6 / +0.1 | −6.0 / 0.0 | −8.0 / +2.1 | −6.6 |
  | 878030 (held out) | −5.0 / +0.6 | −3.2 / +0.7 | −5.2 / 0.0 | −5.3 / +0.1 | −4.6 / +10.9 | −8.2 |
  | 885233 | −5.8 / +0.4 | −3.3 / +0.4 | −6.0 / −0.2 | −6.4 / −0.3 | −5.8 / +5.5 | −10.7 |

- **The excess removed the over-subtraction in the bright sessions** (780125, 803121). The pretrained network now beats the mean image everywhere and keeps the background flat.
- **It costs about 2 points on support in the dimmer sessions** (878030, 885233) compared with v3 pretrained.
- **v4 fine-tuned beats v3 fine-tuned on support in every session** (by 0.1–2.1 points) and ties on background. On the held-out session it beats SiLo on support (−5.3 vs −4.6), and on background by 11 points.
- **Calibration:** z variance 0.96–0.99 pretrained and 0.99–1.01 fine-tuned.
- **Bias on support** (target − baseline, in the trial's scale) on real windows:

  | | 780125 | 803121 | 878030 | 885233 |
  |---|---|---|---|---|
  | v4 pretrained | −0.18 | −0.08 | −0.15 | −0.15 |
  | v4 fine-tuned | −0.04 | +0.04 | −0.02 | −0.06 |
  | mean image | −0.33 | −0.12 | −0.29 | −0.20 |

  The pretrained baseline sits between the mean image and the true baseline, so it removes only about half of the mean activity. Fine-tuning removes most of the rest.

## GFP control: is the excess spillover? (2026-11-15)

- **Session:** 847983_2026-02-24_08-47-45, an EGFP mouse with no activity sensor (whether the label is CAAX-GFP is not confirmed). SiLo finds no sources, so the export writes zero activity and the residual is the movie itself. Settings: 180 s, 7 trials per path.
- **Code changes:**
  - In a trial with `nSources` 0, every valid pixel now has loss weight 1, since its activity is known to be zero.
  - `evaluate` adds a `lab_` class: labeled pixels outside every footprint.
  - The excess report adds a labeled class.
- **Fast excess is not from the sensor.** On dendrite pixels, GFP's single-frame residual variance exceeds photon noise by about as much as on the sensor sessions' footprint support at the same brightness, 5–30%. The excess grows with brightness, so it is multiplicative.
- **Slow excess is smaller in GFP at matched brightness.** At 64 frames GFP is at the low end, close to 803121 and 885233. 780125 and 878030 are well above it.
- **Regional modes remove most of GFP's slow excess.** The model fits brightness (laser, hemodynamics), xy shift and defocus blur (z motion) per frame and per 48-px block. The same fit removes all of 803121's slow excess but only part of 780125's.
- **780125's leftover excess is largest outside the footprints.** Labeled pixels outside every footprint carry more than footprint support at the same brightness. That fits activity GIAnT did not fit there, whether spillover or undetected synapses, which GFP cannot tell apart.

  Real residual variance over photon noise, at 1 / 8 / 64 / 256-frame means. One trial per session; motion model per 48-px block, fitted on half the pixels and tested on the other half:

  | session, pixels (brightness 12–30) | raw | after the motion model |
  |---|---|---|
  | GFP, labeled | 1.08 / 1.12 / 1.21 / 1.23 | 1.07 / 1.10 / 1.10 / 1.01 |
  | 780125, support | 1.02 / 1.11 / 1.47 / 1.90 | 1.01 / 1.07 / 1.19 / 1.23 |
  | 780125, labeled outside footprints | 1.18 / 1.36 / 1.86 / 2.05 | 1.15 / 1.29 / 1.51 / 1.41 |
  | 803121, labeled outside footprints | 1.05 / 1.10 / 1.26 / 1.22 | 1.03 / 1.02 / 1.00 / 0.90 |
  | 878030, labeled outside footprints | 1.16 / 1.27 / 1.55 / 1.23 | 1.15 / 1.25 / 1.39 / 1.16 |
  | 885233, support | 1.08 / 1.14 / 1.15 / 0.99 | 1.08 / 1.13 / 1.17 / 0.97 |

- **Session-wide excess** (`evaluate`'s report, session medians over trials; labeled pixels, 1 / 8 / 64 frames):

  | session | labeled | brightest 10%: fast, slow SD/mean, brightness |
  |---|---|---|
  | GFP 847983 | 1.09 / 1.14 / 1.22 | +0.13, 0.036, 7.7× scale |
  | 780125 | 1.23 / 1.37 / 1.76 | +0.78, 0.068, 26× |
  | 803121 | 1.01 / 1.10 / 1.37 | −0.07, 0.036, 18× |
  | 878030 | 1.11 / 1.20 / 1.24 | +0.13, 0.074, 5.2× |
  | 885233 | 1.07 / 1.13 / 1.18 | +0.07, 0.060, 5.1× |

- **Networks on GFP dendrites.** Bias (target − baseline) in the trial's scale, and 8-frame MSE relative to the mean image. With no activity, any bias is spurious ΔF:

  | baseline | bias, real | bias, simulated | 8-frame MSE, real | 8-frame MSE, simulated |
  |---|---|---|---|---|
  | v3 pretrained | +0.30 | +0.31 | +12.9% | +484% |
  | v3 fine-tuned | +0.26 | +0.25 | +7.5% | +223% |
  | v4 pretrained | +0.05 | +0.04 | +0.2% | −28% |
  | v4 fine-tuned | +0.19 | +0.18 | +3.7% | +94% |
  | SiLo (moving median) | +0.12 | +0.14 | −2.4% | +82% |
  | mean image | 0.00 | 0.00 | 0 | 0 |

  Background bias is about 0 for every network. The simulated GFP windows come from the v4 simulator, so they carry GFP's measured excess. Dendrites are 2.0× scale (median) to 2.8× (mean) here, so a bias of 0.19 is about 7–10% ΔF/F on a sample with no activity.
  - **v3 reads excess variance as activity.** It was trained without the excess, and its bias appears in simulated windows only because those now include the excess.
  - **v4 pretrained is nearly unbiased.** It has the smallest bias of all, and on simulated windows it beats the mean image by 28% while SiLo is 82% worse.
  - **Fine-tuning on the sensor sessions brings the bias back** (+0.19 for v4). On those sessions bright footprint pixels always carry activity, so the network learns to subtract a typical amount even where none is visible. That prior helps the average sensor-session score but biases quiet synapses.
  - **Remedies:** fine-tune with activity-free data in the mix (GFP sessions, where every pixel is an exact target), or with sims whose activity is randomly silenced per region. Keep a second GFP session held out to test it.
  - **On real GFP dendrites SiLo beats the mean image by 2.4% at 8 frames**, while v4 pretrained ties. Real dendrites carry slow fluctuations that a running median follows (motion and brightness modes, above), and the network does not yet track them on real data.

## v5: activity-free data in fine-tuning (2026-11-16)

**Where v4's GFP bias comes from.** Real fine-tuning windows weigh labeled pixels outside every footprint 0 (their residual holds activity GIAnT did not fit), so every bright pixel in the loss carries activity. The v4 evaluation on simulated windows of the sensor sessions shows the trade-off this sets up:

| v4 checkpoint | support bias, simulated | labeled outside footprints (no activity), simulated | GFP dendrites, real |
|---|---|---|---|
| pretrained | −0.11 to −0.17 (baseline too high) | +0.06 to +0.19 | +0.05 |
| fine-tuned | −0.05 to +0.00 | +0.19 to +0.37 (one session −0.17) | +0.19 |

Fine-tuning fixes the support pixels and moves every bright pixel's baseline down, including the ones without events.

**No second GFP session.** DocDB has one GFP session among its 310 SLAP2 records (847983, 2026-02-24). On the scratch share, 531 session folders from 2025 onward have metadata files, and only the same session mentions GFP; the `EGFP` folders there are from 2024. So GFP stays out of training, and is the test.

**Three fine-tuning variants**, each from the v4 pretrained checkpoint with v4's fine-tuning settings (1700 steps, lr 1e-4), GFP excluded (`--exclude`):

- **v5a, mix with silent regions:** half the windows simulated. 20% of virtual trials have no activity, and the rest lose it in 0–80% of the field, in smooth patches of 20–60 px (`--silent_trial 0.2 --silent_frac 0,0.8`). The footprint geometry then no longer says which pixels are active; only the events do.
- **v5b, mix without silent regions:** the control for v5a. Does simply keeping simulated windows in the mix (where unfitted labeled pixels weigh 1) undo the prior?
- **v5c, activity-subtracted real windows:** half the real windows get movie − activity as input, with the residual's mean and variance maps (`--subtract_frac 0.5`). This is real noise and excess on bright dendrites with no fitted events, and needs no simulator.

Validation adds held-out activity-free simulations (`--val_free`, `free_*`). best.pt is chosen on real support, real background and activity-free labeled pixels (`--select balanced_free`). The evaluation scores real, simulated and activity-free simulated windows of all five sessions (`--kinds real,sim,free`).

**Spatial context.** v1–v5 use the same model. Each output sees ±23 px (a 47 × 47 square, an upper bound; the effective field is concentrated near the centre) in frames more than 8 frames away. It sees ±254 frames beyond that gap in time, plus each pixel's own blind window mean and variance. Nothing from the 17 hidden frames reaches it, at any distance.

**Excitation-line profile check.** A change in the excitation line's intensity profile scales every pixel at the same position along the line by the same factor. It should therefore show up as residual correlation that stays flat along the scan direction and decays across it. Test: the spatial correlation of the standardized residual (movie − activity) after removing each frame's global mean, on labeled pixels (brightest 75%) of one local trial per session, along image rows and along image columns:

| session | 8-frame means, along rows: 16 / 32 / 64 px | along columns | 64-frame means, along rows | along columns |
|---|---|---|---|---|
| 780125 | 0.025 / 0.015 / −0.003 | 0.031 / 0.018 / 0.012 | 0.065 / 0.039 / 0.005 | 0.081 / 0.047 / 0.025 |
| 803121 | 0.065 / 0.058 / 0.002 | 0.046 / −0.004 / 0.034 | 0.095 / 0.070 / −0.110 | 0.080 / −0.020 / 0.124 |
| GFP 847983 | 0.000 / −0.001 / −0.009 | 0.001 / −0.008 / 0.005 | −0.004 / 0.002 / −0.021 | 0.005 / −0.016 / 0.019 |
| 878030 | 0.003 / −0.004 / 0.002 | −0.006 / 0.004 / −0.089 | −0.005 / −0.012 / −0.011 | 0.012 / 0.065 / −0.250 |
| 885233 | 0.024 / 0.002 / 0.009 | −0.010 / 0.057 / 0.000 | 0.056 / 0.035 / 0.004 | −0.058 / 0.091 / 0.000 |

- **No axis keeps its correlation.** Along both axes it falls to about 0 by 64 px; in GFP it is about 0 beyond 2 px.
- **Size limit.** A line-profile mode is at most about 1–2% of the residual variance of single frames, and of 64-frame means, along rows (where there are many pixel pairs).
- **The column values at 64 px are noisy.** The trials are only 223–573 rows tall, so few labeled pairs are 64 rows apart.
- **What remains is isotropic.** It is the shared slow component of the sensor sessions, about 0.05–0.1 at 16–32 px for 64-frame means, and it is smaller in GFP.
- **Noise anisotropy at 1 px.** Neighbouring pixels in the same row correlate more than neighbours in the same column (0.40–0.45 vs 0.24–0.31, background included). That is noise structure, which the simulator already models with separate row and column taps.
- **So the context stays isotropic.** If a line-profile mode appears in other sessions, it would argue for a long, thin context along the scan direction. With the slab blind spot, only its slow part (beyond 8 frames) could be predicted at all. Using same-frame pixels far along the line needs the box blind spot (v2 design above).

## v5 results (2026-11-16)

Evaluated on held-out windows of all five sessions (192 per session and kind). Support: mse8 against the true baseline, relative to the mean image: the mean over the four sensor sessions for real support, their range otherwise. Bias in scale units, positive = baseline too low.

| checkpoint | real support | real support bias | real background | GFP dendrites (real), bias | GFP dendrites, mse8 | simulated background |
|---|---|---|---|---|---|---|
| v4 pretrained | −4.7% | −0.08 to −0.18 | +0.3 to +0.7% | +0.049 | +0.3% | −23 to −26% |
| v4 fine-tuned | **−6.7%** | −0.06 to +0.04 | −0.3 to +0.1% | +0.191 | +3.8% | +16 to +36% |
| v5a, mix + silent regions | −5.5% | −0.09 to −0.18 | +0.1 to +0.3% | +0.034 | +0.1% | −18 to −22% |
| v5b, mix | −5.9% | −0.07 to −0.15 | +0.1 to +0.4% | +0.041 | +0.2% | −16 to −21% |
| v5c, activity-subtracted real | −4.7% | −0.11 to −0.23 | −0.2 to +0.0% | +0.029 | −0.0% | −2 to +15% |

- **All three remove the GFP bias.** It drops from +0.19 to +0.03–0.04, below the pretrained network's +0.05.
- **They keep only part of the fine-tuning gain on real support.** v5b keeps about 60% of it (−5.9% vs −4.7% pretrained and −6.7% fine-tuned), v5a about 40%, v5c none.
- **Support bias goes back to the pretrained level.** It is −0.07 to −0.15 again, where fine-tuning had brought it to about 0. With pixels hidden for 17 frames, the network cannot tell from its inputs which bright pixels have events under the gap. Lowering every bright pixel's baseline fixes support and biases event-free pixels. Simulated windows in the mix make it stop doing that.
- **Silent regions add nothing.** v5a matches v5b on GFP (+0.034 vs +0.041) and is slightly worse on support.
- **Activity-subtracted real windows break simulated windows.** v5c does best on real background and GFP, but on simulated background it falls to around the mean image (−2 to +15%), and in 780125 simulated labeled pixels get worse than the mean image (+65%, bias −0.51).
- **v6 fine-tunes with v5b's recipe:** half the windows simulated, no silent regions.

## v6: box blind spot (2026-11-16)

**Why.** Real frames carry fast brightness changes shared across the field, and a slab blind spot cannot see them: it hides every pixel of the 17 frames around the output. Measured on one local trial per session, X = movie − activity:

- **Shared fractional brightness** g(t) = Σ(X − M) / ΣM over bright pixels (M the trial mean). Its SD is noise-free: the covariance of left- and right-half pools cancels photon noise.
- **"Unseen by the slab"** = the 8-frame mean minus the blind window mean (8 < |dt| ≤ 254), which is what a pixel's own trace beyond the gap can predict.

| session | field-wide, per frame | field-wide, unseen by the slab | regional (48-px block minus field), per frame | regional, unseen | field-wide autocorrelation at 8 frames |
|---|---|---|---|---|---|
| 780125 | 4.7% | 2.7% | 3.8% | 2.2% | 0.73 |
| 803121 | 3.6% | 3.1% | 3.2% | 2.5% | 0.16 |
| GFP 847983 | 2.0% | 1.4% | 2.4% | 1.8% | 0.23 |
| 878030 | 4.1% | 3.7% | 2.7% | 2.8% | 0.54 |
| 885233 | 3.5% | 3.2% | 3.7% | 2.8% | 0.45 |

- **It is not activity.** The GFP control has it too.
- **Most of it is faster than the slab's reach.** In 803121 and GFP, autocorrelation is about 0.2 by 8 frames.
- **It matters for ΔF/F.** A 2–4% change shared by every synapse in a frame is a coherent ΔF/F error. A pixel's own photon noise is larger, but independent.
- **What same-frame context could gain.** Take the 8-frame residual that a pixel's own blind window cannot predict. The brightness change of the surrounding pixels at the same frames (a square minus the 9 × 9 box, any size from 8 px to the whole frame, coefficient about 1, fitted on half the pixels and scored on the other half) explains 6–7% of it in 780125 and 13–17% in 803121. In 803121 that is about all of its excess over photon noise. In the dimmer sessions it is small: 1–2% in 885233 and under 0.5% in 878030 and GFP, where photon noise dominates each pixel's residual.

**Model** (`box_radius`, see `model.py`). Hidden: |dt| ≤ 8, |dr| ≤ 4, |dc| ≤ 4, the 9 × 9 box of the original design (footprint radius 3.5 px, noise correlation ≤ 2 px). Visible: everything else, including the output frame beyond the box. Two parts are added to the slab model:

- **Half-plane branches** (Laine et al. 2019). Convolutions see rows ≤ r only: kernel 2 × 3 with top padding, and an aligned half-resolution path. The output is shifted down 5 rows. Four 90° rotations of the input share one trunk (24 channels, 5 blocks, spatial reach 22 px toward the half-plane), and the fusion layer gets each direction separately. In time they are symmetric with ±31 frames of reach, and run only on the output frames plus that reach. Slow changes stay the slab branches' job.
- **Pooled same-frame features.** For squares of half-width 8, 16, 32 and the whole window, minus the box: Σ valid·(movie − mean image) / Σ valid·mean image, in the output frame and over the 8 frames around it. They are computed with integral images, are exact, and cost little.

Checks:

- **Blindness:** a gradient test at every position of a small window confirms nothing inside the box reaches the output, and that the nearest pixels outside it, in the output frame, do.
- **Region crop:** cropping to the loss region gives the same outputs as the full window.
- **Warm start:** `load_compatible` loads a slab checkpoint with zero weights on the new fusion inputs, so the box model starts out computing exactly what the slab model did.
- **Window size:** the window stays 94 × 94 px (reach 23).

**Simulator.** `fast_global_sd` and `fast_field_sd` add fast multiplicative brightness changes to the drift factor: one field-wide process and 3 regional fields (15–60 px), with SDs drawn up to 6% and 4% and time constants of 1.5–20 frames. `evaluate --kinds fast` scores simulated windows with them.

**Training.** The GPU has 22 GB, and v4 already peaked at 18.3 GB with batch 2, so `--accum` runs 2 micro-batches of 1 window per optimizer step. On the CPU, one step of 1 window takes 65 s with the box, against 45 s without.

**Runs.** Pretraining starts from v4 pretrained, on simulations with the fast brightness changes (3000 steps, best.pt chosen on fast simulations). Fine-tuning then follows v5b (mix, half simulated with the fast changes, GFP excluded, 1700 steps).

## v6 pretraining results (2026-11-16)

Same evaluation as v5 (192 held-out windows per session and kind). Support: mean over the four sensor sessions; other columns: range over them.

| checkpoint | real support | real support bias | real background | GFP dendrites (real), bias | simulated support | fast-brightness simulations, support | fast, background |
|---|---|---|---|---|---|---|---|
| v4 pretrained | −4.7% | −0.08 to −0.18 | +0.3 to +0.7% | +0.049 | −33 to −45% | −30 to −42% | −21 to −24% |
| v4 fine-tuned | −6.7% | −0.06 to +0.04 | −0.3 to +0.1% | +0.191 | −47 to −54% | | |
| v5b | −5.9% | −0.07 to −0.15 | +0.1 to +0.4% | +0.041 | −42 to −52% | | |
| **v6 pretrained** | **−7.0%** | −0.06 to −0.11 | +0.2 to +0.7% | **+0.015** | −51 to −59% | −52 to −62% | −31 to −37% |

- **The box model beats every slab model on real support, with no real data in training.** Per session: −9.6% (780125), −7.1% (803121), −4.8% (878030), −6.3% (885233), against −7.1 / −5.0 / −3.2 / −3.3% for v4 pretrained. The gain is largest in the two sessions where same-frame context explained the most residual (780125, 803121), but it is there in all four.
- **Support bias shrinks** without the cost fine-tuning had. It drops to −0.06 to −0.11, and the GFP bias falls to +0.015, the lowest of any checkpoint. Same-frame neighbours tell the network when events are under the gap, so it no longer has to lower every bright pixel's baseline.
- **Simulated windows improve too, also without fast changes.** On activity-free labeled pixels it is about the same as v4 pretrained, slightly worse in 780125 and 878030.
- **Training:** 2.8 s per step (2 windows), peak 12.5 GB. best.pt is the last checkpoint (step 3000). Validation on fast simulations was still improving slowly; real support had been flat for the last 1000 steps.
- **A check for fine-tuning.** The real target is movie − GIAnT activity, and a source's footprint can reach past the box when the output pixel sits at its edge, so on real windows the visible pixels carry a little of the target's estimation noise. The pretrained network never saw real windows, so it cannot use this. After fine-tuning, the GFP session (no sources) and simulated windows are the clean comparison.
