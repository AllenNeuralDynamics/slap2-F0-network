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

## v6 fine-tuning results (2026-11-16)

v6 pretrained, fine-tuned with v5b's recipe: half the windows real, half simulated with the fast brightness changes, GFP excluded, 1700 steps (best.pt at step 1200). Same evaluation as above.

| checkpoint | real support | real support bias | real background | GFP dendrites (real), bias | fast-brightness simulations, support | simulated background |
|---|---|---|---|---|---|---|
| v4 fine-tuned | −6.7% | −0.06 to +0.04 | −0.3 to +0.1% | +0.191 | | +16 to +36% |
| v5b | −5.9% | −0.07 to −0.15 | +0.1 to +0.4% | +0.041 | −39 to −50% | −16 to −21% |
| v6 pretrained | −7.0% | −0.06 to −0.11 | +0.2 to +0.7% | +0.015 | −52 to −62% | −28 to −32% |
| **v6 fine-tuned** | **−7.8%** | −0.05 to −0.11 | −0.2 to +0.4% | +0.033 | −57 to −69% | −23 to −28% |

- **Best on real support in every session:** −9.9% (780125), −8.9% (803121), −5.3% (878030, tied with v4 fine-tuned) and −7.2% (885233). The largest step from fine-tuning is in 803121 (−7.1% → −8.9%), the session where same-frame context explained the most residual.
- **Background now matches v4 fine-tuned** (−0.2 to +0.4%), and the GFP bias stays small (+0.033, against +0.19 for v4 fine-tuned).
- **No sign of a target leak.** On real windows the target is movie − GIAnT activity, and a footprint can reach past the box (see v6 pretraining). The GFP session has no sources, so its target is the movie itself: there v6 fine-tuned scores −0.2% against v6 pretrained's +0.0%, no gain from a leak. On simulated support, where the target is exact, fine-tuning also improves (−57 to −61% against −51 to −59%).
- **Costs:** simulated background loses a few points against v6 pretrained (still better than v5b). Activity-free labeled pixels of 780125 simulations get worse (−13% against −45%, bias −0.24), as they did for v5b.
- **v6 fine-tuned is the best network so far.**

## Viewer, and a flaw in an input (2026-11-16)

`slap2-f0 predict` runs checkpoints over whole trials in their training geometry, and `slap2-f0 view` shows data, data − activity, and each model's baseline and residual frame by frame, with zoom and pixel traces (README, "Viewing baselines"). What it showed on one trial per session:

- **Discarded frames.** 780125's trial loses long stretches to motion (up to a third of the frames in places). On discarded frames the movie has no data and the networks' outputs mean nothing (v6 swings by ±500 counts); no loss ever covered them. The viewer blanks them.
- **Fast tracking.** On a soma pixel in 780125, between the gaps, v6 follows part of the fast (0.1–0.3 s) swings of data − activity, by ±100–150 counts, where v5b and SiLo stay flat.

**The variance-factor input saturates.** `make_inputs` divides SiLo's variance factor by its median over the window, and unimaged pixels are 0 there. When less than half of the window is imaged, the median is 0, and the input clamps at its maximum on every imaged pixel. That happens in 34% (780125), 94% (803121), 78% (878030) and 100% (885233, GFP) of 94 × 94 training windows, so for most windows this input carries nothing. Every checkpoint so far was trained this way, so predictions with the training geometry reproduce it. The fix is a median over imaged pixels only, which needs retraining.

## Variance-factor fix (2026-11-16)

`TrialFile.varfac` is now relative to its median over the trial's imaged pixels, and the network takes its log as it is (`ModelConfig.varfac_norm = "trial"`; pixels without a factor get 0). The input is the same in every window, so a slab model's full-trial output no longer depends on the tiling at all. On the five viewer trials the new input spans −0.5 to +1.1 (780125, 803121) and stays near 0 elsewhere, and it never saturates.

- **Old checkpoints:** those saved before the field load as `"window"` and get exactly the inputs they were trained on (2e-7 on real windows).
- **Warm starts:** `train --init` from such a checkpoint folds the old variance-factor weights into the validity flag and biases of the pointwise input layers (`adapt_varfac_input`), so the warm start computes what it did in saturated windows; tested on a box model.

Six runs, all with v5b's fine-tuning recipe after the simulation stage (half real windows, half simulated with fast brightness changes, GFP held out), evaluated on real, simulated, activity-free and fast-brightness windows:

| run | starts from | input | simulation stage | spatial reach |
|---|---|---|---|---|
| v6r | v6 pretrained (folded) | trial median | 3000 steps | 23 px |
| v6c (control) | v6 pretrained | window median (old) | 3000 steps | 23 px |
| v6s | scratch | trial median | 5530 steps (the 4.8 h cap) | 23 px |
| v6f | v6 pretrained (folded) | trial median | 3000 steps | 14 px (all spatial dilations 1) |
| v6d | v6 pretrained (folded) | trial median | 3000 steps | 23 px; baseline correction cap 3 instead of 0.7 (see the localization results) |
| v6k | v6r pretrained | trial median | 2500 steps, with dark sources (`--dark_frac 0,0.3`, also in fine-tuning) | 23 px |

v6c gets the same extra training as v6r with the old input, so v6r − v6c is the effect of the fix; v6f is the faster architecture, and v6d and v6k address the dark-pixel sources found in localization (below).

**Results.** 8-frame MSE against the mean image on real support pixels (four sensor sessions, evaluated as before), background, the GFP session's bias, and simulations with fast brightness changes:

| checkpoint | real support, mean (780125, 803121, 878030, 885233) | real background | GFP bias | fast sims: support / background / activity-free labeled |
|---|---|---|---|---|
| v6 fine-tuned (before) | −7.8% (−9.9, −8.9, −5.3, −7.2) | −0.2 to +0.4% | +0.033 | −57 to −69% / −26 to −32% / −21 to −71% |
| **v6r pretrained** | **−7.9%** (−9.8, −9.2, −5.0, −7.5) | +0.1 to +0.7% | +0.020 | −63 to −70% / −32 to −41% / −30 to −78% |
| v6r fine-tuned | −7.2% (−10.1, −5.6, −5.5, −7.7) | −0.2 to +0.3% | +0.080 | −60 to −63% / −32 to −37% / +15 to −33% |
| v6c pretrained (old input) | −7.6% (−10.0, −8.1, −4.9, −7.5) | +0.1 to +0.8% | +0.029 | −64 to −68% / −31 to −42% / −26 to −77% |
| v6c fine-tuned (old input) | −7.8% (−10.1, −7.7, −5.5, −7.8) | −0.2 to +0.4% | +0.091 | −61 to −69% / −31 to −37% / −11 to −55% |
| v6f (reach 14) | −7.3% (−9.4, −7.6, −4.8, −7.5) | +0.1 to +0.9% | −0.003 | −61 to −69% / −27 to −41% / −24 to −77% |
| v6s pretrained (scratch, 5530 steps) | −7.2% (−10.1, −5.4, −5.4, −7.7) | +0.1 to +0.6% | +0.021 | −55 to −68% / −31 to −40% / −28 to −59% |
| **v6s fine-tuned** | **−8.3%** (−11.2, −8.1, −5.5, −8.3) | −0.2 to +0.3% | +0.090 | −60 to −68% / −23 to −37% / −28 to −65% |
| v6d pretrained (cap 3) | −7.9% (−9.8, −9.1, −5.0, −7.6) | +0.1 to +0.7% | +0.014 | −63 to −69% / −32 to −41% / −28 to −78% |
| v6d fine-tuned (cap 3) | −7.5% (−10.2, −6.4, −5.5, −7.7) | −0.2 to +0.4% | +0.078 | −60 to −66% / −31 to −37% / +10 to −39% |
| v6k (dark sources) | both stages kept step 0, so its best checkpoints are v6r pretrained (same numbers); see the localization results for its last checkpoint | | | |

- **The fix is worth little on its own:** v6r against v6c, same training, −7.9% against −7.6% pretrained. The variance-ratio input already carried most of what the variance factor says (pixels with more variance than photon noise).
- **More simulation training is what helped:** 3000 more steps took v6 pretrained from −7.0% to −7.9% on real support with no real data, as good as v6 fine-tuned, and better on every simulated class.
- **After a warm-started simulation stage, fine-tuning adds nothing reliable.** Its validation is flat after 300 steps (real support −6.9 to −7.1% on the 48 validation windows), and the selection metric moves with noise in the activity-free labeled-pixel bias, so the chosen step is arbitrary. In v6r it picked a step that is worse on 803121 (−5.6%) and on activity-free labeled pixels (bias up to +0.7 in simulations; GFP +0.08), so v6r's best checkpoint is the pretrained one (data asset mount `f0v6r_run`, `train/best.pt`).
- **From scratch (v6s)**, 5530 simulation steps in the 4.8 h cap reach −7.2% (less than v6r's warm-started −7.9%), but here fine-tuning helps, to **−8.3%, the best real support so far** (best in 780125 and 885233), with background as good as v6 fine-tuned. Its cost is the activity prior on activity-free dendrites: GFP bias +0.090 (v6r pretrained +0.020, v6 fine-tuned +0.033, v4 fine-tuned +0.19). **Choice:** v6s fine-tuned for the best baseline under activity, v6r pretrained where activity-free pixels matter more; both are in the viewer.
- **Reach 14 (v6f) costs about half a point** (−7.3% against −7.9%) for 1.7× faster inference (780125 trials: 107 and 98 s against 185 and 168 s on the RTX 4060 Ti), and has the smallest GFP bias. Its fine-tuning never beat step 0.

## Inference time against GIAnT main (2026-11-16)

**GIAnT main** (4cef08c) SILo on 780125, 15 trials × 2 paths, on a 16-core Threadripper PRO 5955WX with 12 workers: **2571 s** (43 min) in all. Per path: localization 410 and 296 s (loading each trial's 80 Hz movie and `localizeSources_vIM`, 12 trials at a time), alignment and source selection 19 and 27 s, extraction 748 and 993 s.

One trial's `localizeSources_vIM` (export of 780125 trial 2, 2254 frames, 36k valid pixels): 63 s on one thread (as on a SILo worker), of which **the moving median the network would replace is 20 s**; 27 s and 1.6 s on 16 threads. 878030 trial 2 (1493 frames, 18k pixels): 32 s and 11 s on one thread.

**The network** on the same five viewer trials (780125, 803121, 847983, 878030, 885233), seconds per trial in the network (A10G on Code Ocean; stats and reading excluded):

| | 780125 | 803121 | 847983 | 878030 | 885233 |
|---|---|---|---|---|---|
| v6 box, before (bf16 autocast, 512-frame tiles on a grid) | 185 | 191 | 135 | 70 | 75 |
| same precision and tiles, new code (greedy tiles, no padded copies) | 146 | 145 | 90 | 57 | 64 |
| **new default: fp16 model, whole-trial tiles** | **109** | **112** | **71** | **45** | **48** |
| new default + `torch.compile` (first trial includes compiling) | 163 | 91 | 55 | 40 | 42 |
| v6r pretrained, new default | 113 | 112 | 72 | 44 | 48 |
| **v6f (reach 14), new default** | **68** | **81** | **46** | **32** | **35** |
| v5b slab, before | 108 | 119 | 83 | 43 | 45 |
| v5b slab, new default | 55 | 73 | 41 | 28 | 29 |

On the local RTX 4060 Ti (8 GB; tiles of at most ~700 frames fit) a 780125 trial takes 184 s with the new code, and the registered-movie path (`predict-registered`) adds 8 s to load the TIFF and 10 s for stats. On the CPU (16 threads of the Threadripper, fp32) the box model needs about 1.8 µs per window voxel, 20–30× the GPU's: close to an hour per 780125 trial, so the network needs a GPU.

What made it faster:
- **Precision.** The ReZero scale is a float32 parameter, so under bf16 autocast the residual stream became float32 after the first block, and some convolutions ran as TF32. Casting the whole model to fp16 is 1.4× faster than bf16 autocast and 10× closer to fp32 (99th-percentile relative difference 3e-4 against 3e-3).
- **Tiles.** The model reaches 23 px and 263 frames beyond each output, and the imaged pixels are thin dendrites (6–20% of the frame), so 512-frame tiles on a 48-px grid processed 24–45× more voxels than they output. Whole-trial tiles halve that (13–26×), and placing tiles greedily (bands from the first uncovered valid row, tiles from the first uncovered column) removes another 10–25%. Smaller tiles are worse: the margin dominates.
- **Memory**, so whole trials fit: half-plane rotations one at a time in inference, branch outputs cropped as soon as they are done, in-place residual adds, and convolution padding instead of padded copies.
- The precision and tile changes move the old checkpoints' outputs by up to a few percent at some pixels (99th percentile 2–5% in 780125, 803121 and 878030, 0.2% in 847983 and 885233), because their variance-factor input and the box model's pooled features depend on where the window sits. Retrained checkpoints keep only the pooled features' dependence.

**Per session**, 780125's 30 trial-paths would take about 55 min in the network on one A10G with v6 (45 min compiled), or about 34 min with the reach-14 v6f, against 43 min for all of SILo; the moving median it replaces costs about 10 min of one core per session. Remaining levers:
- **Smaller spatial reach:** reach 14 instead of 23 (all spatial dilations 1): 1.4–1.6× faster per trial on the A10G (1.7× on the RTX 4060 Ti) for half a point of real support MSE, and the same localization on the simulations (v6f, above). Spatial context explained little of the baseline (R² 0.003), and the pooled same-frame features keep the field-wide information at no cost.
- **The box model's half-plane branches cost as much as both causal branches:** a slab model is 1.9× faster per voxel, but loses the same-frame tracking.
- **`torch.compile`** fuses the elementwise work (the model is memory-bound): 1.2× on Linux, plus a compile per trial length.
- **Several GPUs:** trials are independent.

## Network baselines in SiLo's localization (2026-11-16)

**Harness** (`tools/localization_eval/locnet_*`): `localizeSources_ext.m` is GIAnT main's `localizeSources_vIM` with an optional external baseline, which replaces the 4 s moving median in the high-pass and in the variance model (Vk and Vb fitted as in main). `locnet_run.m` localizes one trial with several baselines and sweeps the peak threshold on the summary image; `locnet_score.py` matches the sources to reference sources at 2 px. On real trials the harness's moving median and the Python copy of SiLo's baseline (`silo`) find exactly the same sources.

**Simulations with known sources.** `locnet_sims.py` simulates one trial from each sensor session's viewer trial (the default simulator, and the one with fast brightness changes), so the true sources are the template's 152 GIAnT sources (106, 16, 5, 25) and the true baseline is known. Single-trial localization, sums over the four sessions:

| baseline | default threshold (8): precision / recall / F1, fast-brightness sims | same, default sims | best single threshold: F1 (threshold), fast | same, default |
|---|---|---|---|---|
| moving median (main) | 0.96 / 0.53 / **0.68** | 0.94 / 0.57 / **0.71** | 0.79 (4) | 0.78 (4) |
| v6 fine-tuned | 0.84 / 0.78 / **0.81** | 0.90 / 0.78 / **0.83** | 0.83 (6) | 0.84 (7) |
| true baseline | 0.71 / 0.93 / 0.81 | 0.79 / 0.95 / 0.87 | 0.87 (15) | 0.88 (15) |
| v6s fine-tuned | 0.85 / 0.82 / 0.84 | 0.87 / 0.80 / 0.83 | 0.84 (8) | 0.84 (7) |
| v6r pretrained | 0.82 / 0.81 / 0.81 | 0.87 / 0.78 / 0.82 | 0.82 (6) | 0.82 (8) |
| v6f (reach 14) | 0.80 / 0.76 / 0.78 | 0.88 / 0.78 / 0.83 | 0.82 (6) | 0.86 (7) |
| min(v6 fine-tuned, moving median) | 0.92 / 0.68 / 0.79 | 0.93 / 0.66 / 0.77 | 0.81 (5) | 0.80 (4) |

- **At SiLo's threshold the network baseline finds half again as many true sources** (recall 0.53 → 0.78) for a small loss of precision. The moving median sits high at event pixels (+42% at source peaks), which shrinks every event's Z-score; the true baseline shows how much there is to gain.
- **With each baseline's best threshold** (one threshold for all trials; thresholds 2.5–20 swept), the network still beats the moving median (0.83–0.84 against 0.78–0.79) and closes about half of the gap to the true baseline. The best threshold moves (moving median 4, network 6–7, truth 15), so part of the gain at the default threshold is calibration.
- **Per session:** the gain is in 780125 (F1 0.68 → 0.86 at the default threshold) and 803121 (0.30 → 0.52–0.58); the 5-source 878030 and 885233 are about even (one session up, one down).
- **The networks localize alike:** v6 fine-tuned, v6s, v6r and the reach-14 v6f all reach 0.82–0.86 at their best threshold (v6s is best at the default threshold, 0.83–0.84); the differences between them are within what eight trials can resolve.
- Single trials, so SiLo's averaging of activity images over trials is not in this test; the network was trained on simulations of the same kind (the fast-brightness ones are its own pretraining distribution).

**Real trials** (one viewer trial per session), against the session's final GIAnT sources (from all trials, localized with the moving median, so this reference favors it):

| session (final sources) | moving median: found, F1 at 8, best F1 | v5b | v6 pretrained | v6 fine-tuned |
|---|---|---|---|---|
| 780125 (106) | 52, 0.57, 0.61 | 72, 0.64, 0.65 | 78, 0.62, 0.65 | 91, 0.63, 0.65 |
| 803121 (16) | 13, 0.48, 0.55 | 21, 0.38, 0.48 | 22, 0.37, 0.46 | 23, 0.46, 0.49 |
| 878030 (5) | 3, 0.25, 0.50 | 8, 0.62, 0.62 | 11, 0.50, 0.60 | 14, 0.53, 0.60 |
| 885233 (25) | 14, 0.51, 0.62 | 18, 0.56, 0.63 | 21, 0.61, 0.64 | 24, 0.61, 0.62 |
| 847983, GFP (0) | 0 | 0 | 0 | 0 |

The networks find more of the final sources in three of four sessions, none in the GFP session (no false positives where there is no activity), and fewer in 803121. Pooled over the four sessions: F1 0.54 (moving median) against 0.58–0.60 (networks) at the default threshold, and 0.58 against 0.59–0.60 at each one's best threshold.

**Full SILo on 780125** with network baselines (`slap2-f0 predict-registered` for all 30 registered movies, then SILo of the GIAnT branch `localization-network-baseline` with `localizationBaselineDir`), against the main run above:

| | moving median (main) | v6 fine-tuned |
|---|---|---|
| Path 1 sources | 130 | 192: 126 of main's 130, and 66 more (median SNR 4.5) |
| Path 2 sources | 113 | 146: 77 of main's 113, and 69 more (median SNR 5.7) |
| median SNR of all sources | 5.9 / 6.9 | 4.7 / 5.4 |
| SILo time (with other jobs on the machine) | 2571 s | 3597 s |
| v6r pretrained, same test | | Path 1: 198 (125 of 130, 73 more); Path 2: 150 (77 of 113, 73 more); the same 36 dark-pixel sources not found (artifacts, see below) |

- **More sources, as in the simulations:** 135 sources main did not find, with lower but useful SNR (4.5–5.7); main's sources that both runs found keep 91–97% of their SNR.
- **36 of main's Path 2 sources are not found.** By GIAnT's SNR they looked like good ones (median 7.7, higher than main's average), but they are not synapses (see the next section; this bullet and the next record how it looked before that test). They sit on nearly dark pixels: the movie is exactly 0 in 45–82% of frames, the mean is 1–6 counts, and events reach 30–60 counts. The moving median there is about 0, so every event stands out; the network's baseline equals the pixel's mean, events included (1.00× the mean, against 0.91× at kept sources). Such event-dominated pixels are rare in training (GIAnT's activity is more than half the mean on 0.2–2.7% of footprint pixels). The baseline is also the mean image times a correction soft-capped at a factor of 0.5–2 (`max_log_correction` 0.7), so v6d retrained with the cap at 3 (a factor of 0.05–20): it scores like v6r on windows, and **its baseline at the lost sources is still 0.99× the mean** (kept sources: 0.91×). The cap was not what held it back; the network has not learned what such pixels look like. **v6k** therefore simulates them: `--dark_frac 0,0.3` scales the baseline of up to 30% of the footprint pixels (in patches of 2–6 px) by 0.02–0.3 while their activity stays, and `evaluate --kinds dark` scores fast simulations with 30% of the support dark.
- **v6k did not fix it, and shows why.** Warm-started from v6r pretrained, 2500 steps with dark sources never beat step 0 on the selection metric (fast simulations), in either stage, so its best checkpoints are v6r pretrained. Its last checkpoint (step 2500) moves the baseline at the lost sources only from 1.00× to 0.97× the mean (kept sources 0.88× → 0.83×). The reason is in the dark simulations: **v6r already handles them**, support MSE against the mean image −61, −78, −54, −56% (dark simulations of the four sensor sessions' viewer trials, 8 windows each), against −61, −88, −61, −72% for the true baseline (v6k last: −64, −82, −59, −57%, with background in 878030 worse than the mean image, +24%). The real lost sources are a different case. The simulated dark pixels sit inside bright footprints, but the real ones are isolated: in an 11×11 neighborhood none of their pixels is in the brightest quarter (kept sources: 55%), the neighborhood's median mean is 3.1 counts (all valid pixels 4.2), and the pixel's median is 0 (events are its whole mean). To the network such a pixel looks like dim background, where the baseline is the mean. (The planned next test, simulating sources on dim background, is moot: these are not sources.)
- The extra time is mostly extraction of the extra sources (and other jobs running at the same time); reading the baseline files costs a few seconds per trial.
- **Diagnostic: the smaller of the two baselines** (`localizationBaselineCombine = 'min'`, same network baselines). Path 1: 154 sources, 125 of main's 130 and 29 more; Path 2: 125, 106 of main's 113 (32 of the 36 dark-pixel sources come back) and 19 more; matched sources keep their SNR. It adds fewer sources than the network alone, because the moving median's low bias on dim background raises the spread of the summary image, and the peak threshold is relative to that spread. On the simulations it lands between the two (best single-threshold F1 0.81 / 0.80). Since the 32 sources it brings back are artifacts (next section), combining baselines this way only adds false sources.

## Are the lost sources synapses? (2026-11-17)

Earlier GIAnT versions made artifacts in dim areas of the activity image (lines along the scan, series of fine dots), so the 36 Path 2 sources the network runs do not find were tested on the movies themselves (`tools/localization_eval/source_patches.py`, `source_flash_check.py`):

- **Patches:** 17×17 px around every source of main and every network-only source, all 15 trials of the registered 80 Hz movies, plus **control pixels** more than 6 px from any source and matched to the lost sources' brightness, share of zero frames and distance to the ROI edge.
- **Flash:** the 30 largest 3-frame events of the centre 3×3, their mean image (minus frames t−25..t−10) with a 2-D Gaussian fit, and the event-triggered centre trace.
- **The selection-free test.** Picking any pixel's largest frames makes a "flash" at that pixel (centre z ≈ 31–35 for every category, control pixels included). So the signal is measured only outside the window used to pick the events: in a ring 3–4 px out in the same frames (a synapse-sized disk reaches it) and in the decay tail t+3..t+8 (the sensor's decay), each as a z-score against the same statistic at random times. A source passes with both above 3.

Path 2:

| | n | flash sigma, major / minor (px) | kinetics FWHM (frames) | zero frames | ring z | tail z | **pass** |
|---|---|---|---|---|---|---|---|
| found by both runs | 77 | 1.92 / 1.56 | 4 | 0.00 | 16.8 | 9.0 | **75 (97%)** |
| only main (moving median) | 36 | 1.05 / 0.84 | 3 | 0.61 | 0.4 | 0.6 | **0** |
| only the network (v6 fine-tuned) | 69 | 2.01 / 1.55 | 4 | 0.00 | 15.1 | 6.7 | **66 (96%)** |
| control pixels | 36 | 1.09 / 0.93 | 3 | 0.53 | 0.9 | 0.4 | 1 (3%) |

- **The lost sources are noise picked at single pixels.** Their flashes are one pixel and three frames (the size of the picking window), with no ring and no decay, exactly like the control pixels; none of the 36 is a disk with a tail. They are not tied to motion (registration shift at their events at the 47th percentile of all frames), and around several of them main's activity image shows horizontal streaks rather than a blob.
- **GIAnT's own event times agree.** At the 30 largest GIAnT events of each source (activity channel), the decay tail is z 0.8 for the lost sources (3% above 3) against 7.7 for kept ones (91%). Their ring signal (z 3.9) is what picking on GIAnT's footprint makes (13% of its mass lies in the ring), and control pixels at the same times show nothing (z −0.3), so the events are local, not frame-wide.
- **Where they come from:** 14% of Path 2's imaged pixels are exactly 0 in more than half the frames. Such pixels are 26% of those within 2 px of an ROI edge, 56% of the two small ROIs (under 2000 px) and 6% of the interior. The lost sources sit there (median 4.7 px from the edge against 11.7 px for kept sources; a third in the small ROIs, against 1%). On such pixels the moving median of the 8-frame-smoothed movie is half the mean (0.51× within 2 px of the edge, 0.87× in the interior), so photon noise stands out. The network's baseline there is the mean, which is right.
- **Path 1** has fewer zero-dominated pixels (10%, no small ROIs). All 126 sources found by both runs pass, 62 of the 66 network-only ones pass, and 1 of 20 control pixels. Main's 4 Path 1 sources without a match pass too, but each has a network source 1.4–1.7 px away and a second main source 1.8–2.1 px away; they are casualties of the one-to-one matching at 1.5 px, not losses.

**On 780125 the network baseline finds every validated source and adds very few false ones:**

| | Path 1: sources, passing | Path 2: sources, passing |
|---|---|---|
| moving median (main) | 130, 130 (100%) | 113, 75 (66%) |
| v6 fine-tuned | 192, 188 (98%) | 146, 141 (97%) |

The network adds 128 validated sources (62 + 66), loses none, and removes 36 false ones. The real-trial table above scored against the final moving-median sources, which include these false ones, so it favors the moving median.

**Viewer:** `tools/localization_eval/activity_viewer.py` writes one self-contained HTML file (`C:\GIAnT_testdata\bench_f0net\source_check\activity_viewer_780125.html`, 9 MB). It shows two panels side by side: any run's activity image (all trials or one trial) or the mean image, with synchronized zoom and pan, contrast and gamma, and hold-to-flicker. Sources are marked by which run has them, dashed where they fail the flash test. Clicking a source shows its flash image, kinetics and test values, and a sortable table lists them all. It covers four runs: main, v6 fine-tuned, v6r pretrained, and the min combination.

**Decisions:**
- **No more dark-source simulation.** v6k's direction is closed (`--dark_frac` stays as an option); the network's mean-level baseline on zero-dominated pixels is correct.
- **The simulator's low counts should match the real data.** Real movies are never negative, and dim pixels are exactly 0 in many frames (37% at 0.5–2 counts, 27% at 2–4, 9% at 4–8), most of all at ROI edges and in small ROIs. The simulated movies are negative in 6.7% of values and never exactly 0. That is why the localization simulations gave the moving median a precision of 0.94–0.96 while on real Path 2 it is 0.66: they lack its main false-positive mode. Next simulator change: non-negative photon-count noise with the measured zero fractions, plus zero-dominated ROI edges and small ROIs. Check first that the moving median then makes edge false positives in the localization simulations, then retrain on it.
- **Real-data localization is scored with the flash test from now on** (precision = share of a run's sources that pass; recall against the union of passing sources), not against one run's final sources.
- **For GIAnT:** with network baselines these false sources disappear. If the moving median stays as an option, SiLo needs a guard at zero-dominated pixels: the flash test as a source-selection check, or a noise model that does not take the median there.

## Realistic low counts, flips, and activity in the baseline (2026-11-17)

**Realistic low counts** (`SimConfig.photon_zeros`, `train --photon_zeros`). Real movies are photon counts: never negative, and dim pixels are exactly 0 in many frames. The simulated movies were negative in 7.7% of voxels and never exactly 0. On pixels where zeros are likely, the simulator now maps its noise, voxel by voxel, onto a zero-inflated distribution with the same mean and variance: 0, or else lognormal. It uses a Gaussian copula through a correlated field with the template's noise kernels, so the noise keeps its structure in time and space. The share of zeros is exp(−mean/c), with c the template pixel's mean count per photon event, taken from its measured share of zero frames (stats v5: `zero_movie`). So ROI edges and small ROIs come out as zero-dominated as in the template. 780125 (Path 1 viewer trial):

| | negative voxels | zero frames at 0.5–2 / 2–4 / 4–8 counts | moving median of 8-frame means / mean: within 2 px of the edge, 2–4 px, interior |
|---|---|---|---|
| real | 0 | 0.42 / 0.31 / 0.11 | 0.66 / 0.73 / 0.92 |
| simulated, old noise | 7.7% | 0 / 0 / 0 | 0.59 / 0.63 / 0.89 |
| simulated, photon_zeros | 0.1% (bright pixels, unchanged) | 0.45 / 0.31 / 0.11 | 0.74 / 0.75 / 0.89 |

The old noise already pulled the moving median of dim pixels below the mean, as much as the real data do. So the moving median's false sources on real Path 2 come mostly from the template geometry, the zero-dominated small ROIs that only Path 2 has. The localization simulations so far used Path 1 templates only.

**Augmentation.** Training windows were flipped at random along rows and columns (no rotations). Flips are now up/down only by default (`--flip rows`; rows are the DMD's short axis), because the microscope scans and only that flip leaves the image alike. The box model's half-plane branch, though, ran one trunk over all four 90° rotations of the window, which assumed the image looks the same rotated. It now has one trunk per direction (`ModelConfig.hp_shared` False). A shared v6 trunk warm-starts all four, so the start is unchanged (tested).

**Activity in the baseline.** Measured three ways (`tools/leak_check.py`; `evaluate` now reports `sup_leak_*`):

- **Simulations, against the true baseline:** the slope of (baseline − truth) on the activity at support pixels, 8-frame means. All v6 models follow the activity: v6 fine-tuned 0.14–0.27, v6r pretrained 0.10–0.24, v6s fine-tuned 0.10–0.22, v6f 0.09–0.24 (8 simulated trials, 4 sessions). v6r pretrained never saw real data, so the leak is not only learned in real fine-tuning.
- **Real trials, slope** of the baseline's fast part (8-frame mean minus its 4 s running mean) on the activity's fast part at support pixels: v6 models 0.026–0.036 (780125), 0.051–0.098 (803121), 0.037–0.047 (885233); the slab models v4 and v5b −0.016 to +0.017; SiLo's moving median about 0. Fine-tuning does not change it (v6r 0.031 → 0.032, 0.072 → 0.076).
- **Real trials, during the strongest population responses** (the 10% of frames with the largest summed activity, against the quieter half), rise of the fast part in % of the mean:

| session | movie at background pixels | movie − activity at support | v6 baselines at support | v6 baselines at background | v5b at support | SiLo at support |
|---|---|---|---|---|---|---|
| 780125 | +1.5% (SE 0.4) | +0.9% | +2.0 to +3.2% | +1.6 to +2.1% | 0.0% | −0.1% |
| 803121 | +5.2% (SE 0.9) | +2.6% | +2.3 to +2.5% | +1.8 to +2.0% | +0.1% | −0.1% |
| 885233 | +5.2% (SE 0.8) | +4.7% | +3.5 to +3.7% | +3.5 to +3.8% | +0.5% | −0.3% |
| 878030 | −0.3% (SE 0.8) | +9.0% | +2.1 to +2.6% | −0.1 to +0.2% | −0.1% | −0.4% |

- **Background pixels brighten with the responses**, by up to 5%: a real shared brightening (neuropil glutamate or stimulus light). The box models follow it at background pixels, partly; the slab models and the moving median don't.
- **At synapses the v6 baselines rise 2–4% during strong responses.** In 803121 and 885233 that is no more than the shared brightening. In 780125 and 878030 it is 1–2.5 points more: a small leak of the synapses' own activity, which matches what the viewer shows during stimuli.
- **Mechanism:** the box model watches the same frame outside its blind box (half-planes and pooled features) to track fast shared brightness, and many synapses responding together look like a frame-wide brightening. The slab models can't see the same frame and don't leak.
- **The stimulus rhythm (2 s at 2 Hz, then 1 s off) is used too, but it is the smaller part.** Simulations whose activity timeline is cut at quiet frames and reordered (`SimConfig.shuffle_prob`; `locnet_sims.py` kind "simshuf"; events stay whole, the rest of the simulation is the same) against the same simulations in the original order, leak against the true baseline:

| | v6r pretrained | v6r fine-tuned | v6s pretrained | v6s fine-tuned |
|---|---|---|---|---|
| 780125, rhythm kept / shuffled | 0.159 / 0.148 | 0.179 / 0.169 | 0.096 / 0.082 | 0.140 / 0.129 |
| 803121 (clearest rhythm: summed activity correlates +0.28 at 1.5 s, +0.19 at 3 s) | 0.142 / 0.114 | 0.195 / 0.172 | 0.067 / 0.046 | 0.102 / 0.083 |
| 885233 | 0.242 / 0.226 | 0.267 / 0.252 | 0.187 / 0.161 | 0.223 / 0.199 |
| 878030 (5 sources, no clear rhythm) | 0.100 / 0.124 | 0.148 / 0.173 | 0.118 / 0.124 | 0.170 / 0.188 |

  The rhythm accounts for 0.01 (780125) to 0.02–0.03 (803121, 885233) of the leak; most of it remains without any rhythm. In 878030, with 5 sources and no clear rhythm, shuffling makes no consistent difference (the leak there rests on few events).
- **Real fine-tuning adds leak**: +0.02 to +0.05 against the true baseline in every session (v6r and v6s pretrained → fine-tuned). Its target, movie − activity, still holds the part of the events GIAnT did not fit, which the context makes predictable. The checkpoint with the least leak is v6s pretrained (simulations only, from scratch).

**Guards against the leak** (options, used in v7): `--sim_event_weight` weights simulated voxels during events (activity above half the photon-noise SD, ±8 frames) more, so following the activity costs more where the truth is known. `--event_mask` drops real training voxels during GIAnT events, where the target still holds the part of the events GIAnT did not fit. `--shuffle_prob` breaks the stimulus rhythm in a share of simulated trials.

**Shuffling and the baseline.** Shuffling reorders only the activity: the true baseline of a shuffled simulation is bit-for-bit that of the same simulation in order (bleaching, slow and fast shared brightness and the slow excess stay continuous), and the v6 models track it as well (error against the truth within 1–4 points of the ordered simulations, 4 sessions × 4 checkpoints). But the activity is never zero, even at the quiet frames where it is cut, so every cut is a step in the activity pattern. The summed change of the support pixels' activity at a cut is 3.4× the usual frame-to-frame change (50 cuts in a 780125 trial, 2% of frames, carry 7.5% of all activity change). A network trained on that may learn that abrupt, lasting steps are activity, and real baselines do step (motion, fast shared brightness). And it buys little: the rhythm explains 0.01–0.03 of a 0.1–0.25 leak. So v7c is v7 without shuffling: v7 against v7c shows what training on shuffled timing costs or gains, v7c against v7b what the other two guards do. Unless v7 clearly beats v7c, shuffling stays a diagnostic.

## Bias at trial edges (2026-11-17)

**Method: cut trials into pieces** (`tools/edge_pieces.py`). Neither GIAnT's activity nor a residual against the movie is a sound reference near trial edges: the activity was fitted with GIAnT's own baseline, which may be biased there too, and the residual assumes activity does not change near edges. So each trial is predicted whole and again cut into 512-frame pieces, each piece predicted as a trial of its own but with the whole trial's mean image, variance maps and gain, so only the truncation in time differs. At a cut the pieces sit at a trial edge while the whole trial's prediction is in its middle: (piece − whole) / whole, by distance from the cut, is the edge bias alone, on real data as on simulated, and each cut adds two edges. It is 0 beyond the model's temporal reach (263 frames), as it must be (|piece − whole| ≤ 0.2% at 256–400 frames). On simulated trials the pieces and the whole trial are also compared with the true baseline.

**Simulations** (8 trials: the 4 sessions' viewer trials as templates, default and fast-brightness simulations; 2–3 cuts each), piece − whole in % of the baseline (± SE over trials):

| | 0–8 frames | 8–16 | 16–32 | 32–64 | 64–128 | 128–256 |
|---|---|---|---|---|---|---|
| v6r pretrained, support, end of a piece | +4.1 ± 1.2 | +2.5 ± 0.9 | +0.9 ± 0.5 | +0.7 ± 0.2 | +0.5 ± 0.1 | +0.2 |
| v6r pretrained, support, start of a piece | +1.8 ± 0.5 | +0.1 | +0.1 | +0.5 | +0.2 | +0.1 |
| v6r pretrained, background, end | +2.2 ± 0.3 | +1.5 | +1.2 | +1.0 | +0.6 | +0.2 |
| v6r pretrained, background, start | +1.5 ± 0.2 | +0.3 | +0.3 | +0.5 | +0.3 | +0.2 |
| v6s fine-tuned, support, end | +0.1 | −0.6 | −0.6 | −0.1 | +0.1 | +0.1 |
| v6s fine-tuned, support, start | +2.9 ± 0.8 | +2.9 ± 0.6 | +2.2 ± 0.5 | +1.3 ± 0.3 | +0.4 | 0.0 |
| v6s fine-tuned, background, end | +0.4 | −0.5 | −0.2 | +0.3 | +0.3 | +0.1 |
| v6s fine-tuned, background, start | +1.2 ± 0.3 | +1.2 | +1.0 | +0.7 | +0.3 | +0.1 |

Against the true baseline the pieces agree: at support pixels v6r is 5.1 points above its mid-trial bias in the last 8 frames before a cut, v6s 4.4 points above in the first 8 after one. **Both checkpoints put the baseline too high at trial edges, each mostly at one edge (v6r at the end, v6s at the start), by 2–4% of the baseline at footprint pixels in the first or last 0.1–0.2 s and ~1% out to 0.4–0.8 s.** A too-high baseline at the edges shrinks events there (SiLo's Z-scores, extracted amplitudes).

**Real trials** (the five sessions' viewer trials, 2–3 cuts each; footprint support from the export, which only labels pixels), piece − whole in % of the baseline (± SE over trials):

| | 0–8 frames | 8–16 | 16–32 | 32–64 | 64–128 | 128–256 |
|---|---|---|---|---|---|---|
| v6r pretrained, support, end of a piece | +5.2 ± 2.2 | +3.9 ± 2.0 | +1.9 ± 1.3 | +1.3 ± 0.7 | +0.7 ± 0.2 | +0.1 |
| v6r pretrained, background, end | +2.3 ± 0.6 | +1.8 ± 0.5 | +1.4 ± 0.3 | +1.2 ± 0.2 | +0.7 ± 0.1 | +0.2 |
| v6r pretrained, support / background, start | +1.6 / +1.1 | 0.0 / +0.2 | −0.1 / +0.2 | +0.4 / +0.4 | +0.2 / +0.3 | +0.1 |
| v6s fine-tuned, support, start of a piece | +2.9 ± 1.1 | +2.8 ± 0.9 | +2.2 ± 0.7 | +1.1 ± 0.3 | +0.3 ± 0.1 | +0.1 |
| v6s fine-tuned, background, start | +1.0 ± 0.4 | +1.1 ± 0.4 | +0.9 ± 0.3 | +0.6 ± 0.1 | +0.2 | 0.0 |
| v6s fine-tuned, support / background, end | +0.1 / +0.7 | −0.7 / −0.3 | −0.5 / 0.0 | 0.0 / +0.4 | +0.1 / +0.3 | +0.1 |

The same pattern and size as in the simulations, so the simulations' edge bias is what real trials get. It is largest in the dim 878030 (v6r, all pixels: +5.9% in the last 8 frames) and small, slightly negative, in the GFP control. (The earlier estimates from residuals against GIAnT's activity, or against the movie averaged over trials, are superseded: they depended on GIAnT's own baseline or assumed steady activity near edges.)

**Why, and the fix:** uniformly placed training windows put only ~7% of their outputs in the first or last 64 frames of a trial. `--edge_frac` makes that share of windows start or end their loss region at a trial edge (±64 frames). The v7 runs do not use it; the next run should, judged with `edge_pieces.py` on simulated (against the truth) and real trials.

## Machines without a GPU (2026-11-17)

**CPU inference is not an option.** Forward time per output voxel on the 16-core Threadripper (fp32, all threads) against the local RTX 4060 Ti (fp16), same tiles (128 output frames × 48² px):

| variant | parameters | CPU cost relative to v6 | CPU / GPU |
|---|---|---|---|
| v6 box | 310k | 1 | 24× |
| v6f (reach 14) | 310k | 0.68 | 25× |
| slab (no box, as v5) | 188k | 0.76 | 26× |
| box, width 16 / 12 | 78k | 0.51 | 25× |
| box, width 8 / 8 | 26k | 0.24 | 19× |
| slab, width 16 | 48k | 0.34 | 23× |

A v6 780125 trial-path takes 184 s on the RTX 4060 Ti, so about 74 min on the CPU: ~37 h for a 30-trial-path session, against 43 min for all of SILo. The smallest variant (untrained here, and certain to lose accuracy) would still need ~9 h per session. Conv3d on CPU already runs through oneDNN, and this Zen 3 CPU has no bf16 units, so runtimes such as ONNX Runtime would not close a 20× gap.

**Recommendation: make a modest NVIDIA GPU mandatory for the network baseline, and keep the moving median as an explicit option, never a silent fallback.**
- **Hardware:** any CUDA GPU supported by current PyTorch builds, 8 GB recommended. `predict` sizes its tiles to the free memory, so 4–6 GB cards work with shorter tiles, more slowly. A 780125 trial-path: 184 s on an RTX 4060 Ti 8 GB (~1.5 h per session with v6, ~55 min with the reach-14 v6f), 109 s / 68 s on an A10G. Trials are independent, so several GPUs scale linearly.
- **GIAnT:** a parameter `localizationBaseline` = 'network' (default once validated) or 'movmedian'. With 'network', GIAnT runs `slap2-f0 predict-registered` (a configured Python environment and checkpoint) for trials without a baseline file, after motion correction and before localization. Without a usable GPU it stops with a clear message: run on a GPU machine, compute the baseline files elsewhere, or choose 'movmedian'. The current branch already fails loudly when a baseline file is missing, rather than falling back. The baseline used (and the checkpoint) should be written into the outputs, so results stay traceable.
- **CPU-only users** get the baseline files from a GPU elsewhere (the Code Ocean GPU capsule, a cluster GPU node, a lab workstation): `predict-registered` needs only the motion-correction folder and writes one small HDF5 file per trial. Or they choose 'movmedian' knowingly. On Path 2 of 780125 a third of the moving median's sources failed the flash test, and it finds fewer real sources.
- **A CPU-friendly distilled model** is possible (width 8 is ~4× cheaper) but would still be hours per session and less accurate. Not worth it unless many users lack GPUs.
- Apple Silicon (PyTorch MPS) and AMD (ROCm) are untested.

## v7: guards against the leak (2026-11-18)

**Runs.** All three start from v6r pretrained and train 3000 steps on simulations, then fine-tune 1700 steps on real and simulated windows, with realistic low counts (`--photon_zeros`), up/down flips and per-direction half-plane trunks:

- **v7b:** no guards.
- **v7c:** `--sim_event_weight 4` throughout, and `--event_mask 0.5` in fine-tuning.
- **v7:** v7c plus `--shuffle_prob 0.5`.

**No guarded checkpoint beat its start.** Checkpoint selection (fast-brightness simulations in pretraining, real plus activity-free data in fine-tuning) kept step 0 in both stages of v7 and v7c, so their best checkpoints are v6r's own weights. v7b pretraining ended slightly better than v6r (best at step 2000); its fine-tuning did not, so its best checkpoint is step 0.

Validation over training, averaged over the six checkpoints after step 0. Leak is the slope at support pixels; bias is truth − baseline in units of the trial's median pixel brightness, so positive means too low:

| | leak, fast-brightness sims | bias, fast-brightness sims, support | bias, activity-free sims, support | leak, real windows |
|---|---|---|---|---|
| start (v6r pretrained) | 0.098 | −0.052 | +0.052 | 0.040 |
| v7b pretraining | 0.079 | +0.041 | +0.091 | 0.023 |
| v7c pretraining | 0.072 | +0.136 | +0.150 | 0.017 |
| v7 pretraining | 0.070 | +0.143 | +0.150 | 0.016 |
| v7b fine-tuning (from v7b at 0.094 / −0.015 / +0.065 / 0.038) | 0.106 | +0.001 | +0.090 | 0.051 |
| v7c fine-tuning (from v6r) | 0.080 | +0.047 | +0.111 | 0.018 |
| v7 fine-tuning (from v6r) | 0.081 | +0.047 | +0.111 | 0.019 |

- **Shuffling does nothing.** v7 and v7c agree at every step within noise. In the full evaluation the v6r leak is 0.136 on rhythmic and 0.131 on shuffled fast-brightness simulations. Shuffling stays a diagnostic.
- **Event weighting trades leak for a baseline that is too low everywhere**, including where there is no activity at all. In pretraining it lowers the leak by 0.007 and lowers the baseline by 0.06 in activity-free simulations and by 0.10 at activity.
- **Why, by construction:** given what the network sees, a frame-wide brightening can be either activity or a baseline change. The unweighted loss is minimized by the conditional mean, which is unbiased. Weighting event voxels w× is the same as telling the network that events are w× more likely than the simulator makes them: it moves the optimum toward "it was activity" and lowers the baseline wherever the brightening is ambiguous. A scalar toy (brightness = baseline + event + noise, events in 20% of frames) gives: weight 1, bias 0 and leak slope 0.36; weight 4, bias +0.20 overall (+0.33 on frames without an event) and leak slope 0.33. So the leak of the unweighted fit is not a training failure the loss can fix. Lowering it honestly needs more information in the inputs, or simulated activity and shared brightening whose statistics match the real data.
- **Masking GIAnT's events in real fine-tuning is the useful part.** It removes targets that still hold unfitted activity rather than reweighting correct ones. Fine-tuning without guards raises the real-window leak (v7b 0.038 → 0.051, as in v6). With the guards it stays near 0.018, at a smaller bias cost than event weighting has in pretraining. The runs did not separate the event mask from the event weight in fine-tuning.

**Full evaluation, v7b pretrained against v6r pretrained** (means over sessions):

- **Simulations:** v7b is better on fast-brightness simulations. Its support error is 8.6% above the oracle's, against 17.6% for v6r, and 68.2% below the mean image's, against 65.8%. Leak is 0.129 against 0.136.
- **Real support pixels:** about the same, −8.1% against −7.9% relative to the mean image.
- **Real background:** slightly worse, +0.5% against +0.3%, and its bias doubles (+0.024 against +0.010).

**Real trials** (`leak_check --real`, the viewer trials): slope of the baseline's fast part on the activity at support pixels.

| | 780125 | 803121 | 878030 | 885233 |
|---|---|---|---|---|
| v6r pretrained | 0.031 | 0.072 | 0.002 | 0.043 |
| v7 at the end of pretraining | 0.030 | 0.055 | 0.012 | 0.035 |
| v7 at the end of fine-tuning | 0.023 | 0.056 | 0.004 | 0.030 |
| v7b pretrained | 0.034 | 0.063 | 0.015 | 0.044 |
| v7b at the end of fine-tuning | 0.032 | 0.062 | 0.027 | 0.042 |

The guarded checkpoints follow the activity 20–30% less in 780125, 803121 and 885233. 878030 has 5 sources and is noise. During strong population responses the guarded fine-tuned baseline at support pixels rises 1.7–3.0%, against 2.1–3.5% for v6r.

**Decisions.**

- Drop `--sim_event_weight`, and keep shuffling as a diagnostic.
- Keep v6r pretrained as the default checkpoint: v7b is no clear improvement.
- **v8** adds `--edge_frac 0.25` to v7b's pretraining, and `--event_mask 0.5` with `--edge_frac 0.25` in fine-tuning. Judge it on leak and bias in the simulations, on the real-trial leak, and on the edge bias with `edge_pieces.py`.
- Whether the baseline should follow the stimulus-locked shared brightening is a question about the simulator's shared-brightness statistics (`--fast_global_sd`, `--fast_field_sd`), not about the loss.

## v8: edge windows and the event mask alone (2026-11-18)

**Run.** v7b's recipe plus `--edge_frac 0.25` in both stages and `--event_mask 0.5` in fine-tuning: from v6r pretrained, 3000 steps on simulations and 1700 steps of fine-tuning on real and simulated windows, with realistic low counts, up/down flips and per-direction half-plane trunks, and no event weighting or shuffling. Selection kept step 2500 of pretraining and step 600 of fine-tuning, the first fine-tuning in these runs to beat its start. Validation is noisy (the selection score moves by up to 0.07 between neighbouring checkpoints), so which step won is partly luck.

**Validation over training** (means of the checkpoints after step 0, as in the v7 table):

| | leak, fast-brightness sims | bias, fast-brightness sims, support | bias, activity-free sims, support | leak, real windows |
|---|---|---|---|---|
| v7b pretraining | 0.079 | +0.041 | +0.091 | 0.023 |
| v8 pretraining | 0.080 | +0.039 | +0.092 | 0.024 |
| v7b fine-tuning | 0.106 | +0.001 | +0.090 | 0.051 |
| v8 fine-tuning | 0.089 | −0.001 | +0.068 | 0.032 |
| v7c fine-tuning (event mask and event weight) | 0.080 | +0.047 | +0.111 | 0.018 |

- The edge windows leave the mid-trial validation windows unchanged, as they should.
- **The event mask on its own** cuts fine-tuning's real-window leak from 0.051 to 0.032, and it lowers the bias on activity-free simulations rather than raising it. The event weight's further cut, to 0.018, came only with the too-low baseline.

**Full evaluation** (means over sessions; bias in units of the trial's median pixel brightness, positive = baseline too low):

| | v6r pretrained | v7b pretrained | v8 pretrained | v8 fine-tuned |
|---|---|---|---|---|
| fast-brightness sims: support error above the oracle's | +17.6% | +8.6% | **+3.6%** | +15.4% |
| fast-brightness sims: leak | 0.136 | 0.129 | **0.106** | 0.136 |
| fast-brightness sims: support bias | −0.061 | −0.043 | −0.037 | −0.085 |
| activity-free sims: support / background bias | +0.050 / +0.011 | +0.055 / +0.022 | +0.049 / +0.017 | **+0.017 / +0.001** |
| real support pixels, error vs mean image | −7.9% | −8.1% | −8.2% | −8.0% |
| real background, error vs mean image | +0.3% | +0.5% | +0.5% | **0.0%** |
| real background bias | +0.010 | +0.024 | +0.019 | **0.000** |

- **v8 pretrained is the best checkpoint so far on simulations.** Its support error is within 4% of the oracle's and it leaks least. On real data it matches v7b: support pixels like v6r, background slightly worse.
- **v8 fine-tuned is the first checkpoint that matches the mean image on real background**, and it is unbiased where there is no activity. But it raises the baseline everywhere by about 0.03. At support pixels with activity it ends up higher than v6r's (−0.085 against −0.061), and its leak is back at v6r's level.

**Real trials** (`leak_check --real`, the viewer trials): slope of the baseline's fast part on the activity at support pixels.

| | 780125 | 803121 | 878030 | 885233 |
|---|---|---|---|---|
| v6r pretrained | 0.031 | 0.072 | 0.002 | 0.043 |
| v7 at the end of fine-tuning (event weight) | 0.023 | 0.056 | 0.004 | 0.030 |
| v8 pretrained | 0.030 | 0.064 | 0.011 | 0.040 |
| v8 fine-tuned | 0.033 | 0.054 | 0.013 | 0.041 |

On real trials v8 follows the activity about as much as v6r. Only the event-weighted v7 follows it less, at the cost of the too-low baseline.

**Trial edges** (`edge_pieces.py`, the same trials and cuts as before): piece − whole, in % of the baseline, by frames from the cut. SE of the first bin: about 1–1.5 at support pixels and 0.3–0.8 in background (simulations), 1.5–2.6 and 0.5–1.1 (real trials).

| simulations | 0–8 frames | 8–16 | 16–32 | 32–64 | 64–128 | 128–256 |
|---|---|---|---|---|---|---|
| support, end of a piece: v6r pretrained | +4.1 | +2.5 | +0.9 | +0.7 | +0.5 | +0.2 |
| v7b pretrained | +5.0 | +3.8 | +1.9 | +1.2 | +0.5 | +0.2 |
| v8 pretrained | +4.2 | +3.3 | +1.6 | +1.0 | +0.5 | +0.3 |
| v8 fine-tuned | +2.2 | +2.5 | +1.4 | +0.9 | +0.4 | +0.2 |
| support, start of a piece: v6r / v7b / v8 pretrained / v8 fine-tuned | +1.8 / +1.8 / +1.2 / +0.3 | +0.1 / +0.6 / +0.2 / +0.4 | +0.1 / −0.1 / −0.5 / −0.2 | +0.5 / +0.1 / −0.1 / 0.0 | +0.2 / 0.0 / −0.1 / 0.0 | +0.1 / 0.0 / 0.0 / 0.0 |
| background, end: v6r pretrained | +2.2 | +1.5 | +1.2 | +1.0 | +0.6 | +0.2 |
| v7b pretrained | +2.9 | +2.4 | +1.7 | +1.3 | +0.5 | +0.2 |
| v8 pretrained | +2.0 | +2.1 | +1.5 | +1.1 | +0.5 | +0.2 |
| v8 fine-tuned | −0.7 | +0.2 | +0.3 | +0.4 | +0.4 | +0.2 |
| background, start: v6r / v8 pretrained / v8 fine-tuned | +1.5 / +1.5 / +0.8 | +0.3 / +0.3 / +0.6 | +0.3 / −0.2 / +0.1 | +0.5 / 0.0 / +0.2 | +0.3 / 0.0 / +0.1 | +0.2 / +0.1 / +0.1 |

| real trials | 0–8 frames | 8–16 | 16–32 | 32–64 | 64–128 | 128–256 |
|---|---|---|---|---|---|---|
| support, end of a piece: v6r pretrained | +5.2 | +3.9 | +1.9 | +1.3 | +0.7 | +0.1 |
| v8 pretrained | +5.0 | +4.3 | +2.1 | +1.4 | +0.6 | +0.1 |
| v8 fine-tuned | +2.6 | +3.1 | +1.7 | +1.2 | +0.6 | +0.1 |
| background, end: v6r pretrained | +2.3 | +1.8 | +1.4 | +1.2 | +0.7 | +0.2 |
| v8 pretrained | +2.2 | +2.5 | +1.9 | +1.3 | +0.6 | +0.2 |
| v8 fine-tuned | −0.5 | +0.5 | +0.6 | +0.6 | +0.4 | +0.2 |
| support, start: v6r / v8 pretrained / v8 fine-tuned | +1.6 / +0.8 / 0.0 | 0.0 / −0.1 / +0.2 | −0.1 / −0.6 / −0.3 | +0.4 / −0.2 / 0.0 | +0.2 / −0.2 / −0.1 | +0.1 / 0.0 / 0.0 |
| background, start: v6r / v8 pretrained / v8 fine-tuned | +1.1 / +1.2 / +0.6 | +0.2 / +0.3 / +0.6 | +0.2 / −0.2 / +0.1 | +0.4 / −0.1 / +0.2 | +0.3 / −0.1 / 0.0 | +0.1 / +0.1 / +0.1 |

- **Edge windows in pretraining help little.** Against v7b, the same run without them, the excess in the last 8 frames falls from 5.0% to 4.2% at support pixels and from 2.9% to 2.0% in background. On real trials v8 pretrained is no better than v6r.
- **v8 fine-tuning halves the excess at the end of a trial at support pixels**, and it removes it at the start and in background. What remains is +2–3% at support pixels in the last 16 frames (0.2 s), and about 1% out to 64 frames.
- Averaged over the training distribution, the conditional mean is unbiased at edges as anywhere else. A remaining edge bias therefore means one of two things: the network still falls short of the conditional mean at edges, or the test trials differ there from what it was trained on. A larger edge share or longer pretraining is cheap to try and to check with `edge_pieces.py`.

**Localization on the simulations.** Same harness and the same 8 simulated trials with known sources as the v6 table above; pooled over trials:

| baseline | default threshold (8): precision / recall / F1, fast-brightness sims | same, default sims | best single threshold: F1 (threshold), fast | same, default |
|---|---|---|---|---|
| v6r pretrained | 0.82 / 0.81 / 0.81 | 0.87 / 0.78 / 0.82 | 0.82 (6) | 0.82 (8) |
| v6s fine-tuned | 0.85 / 0.82 / 0.84 | 0.87 / 0.80 / 0.83 | 0.84 (8) | 0.84 (7) |
| v8 pretrained | 0.80 / 0.84 / 0.82 | 0.79 / 0.80 / 0.80 | 0.82 (8) | 0.80 (10) |
| v8 fine-tuned | 0.79 / 0.80 / 0.80 | 0.82 / 0.78 / 0.80 | 0.81 (7) | 0.80 (9) |

- **v8 localizes no better than v6r**, and slightly worse on the default simulations. At the default threshold, v8 pretrained finds 249 of the 304 true sources (v6r 242) but makes 63 false ones (v6r 46), most of them in the dim 878030 and 885233. The true baseline makes 95 false ones at that threshold, so more false sources at a fixed threshold partly come with a more accurate baseline. But v8 does not gain at its own best threshold either.
- The differences are within what eight trials can resolve. Still, v8's better baseline error and leak in simulations do not carry over into localization. A checkpoint should be chosen on localization as well: these simulations, and the flash test on real trials.

**Decisions.**

- Keep `--event_mask 0.5` in fine-tuning and `--edge_frac 0.25` in both stages; neither costs anything measurable. `--sim_event_weight` stays dropped.
- **Keep v6r pretrained as the default.** v8 pretrained has the more accurate, less leaky baseline in simulations, and v8 fine-tuned the unbiased background and the smaller edge bias. Neither localizes better, and on real trials both follow the activity as much as v6r does.
- The edge bias at the end of trials is the one systematic error here that training can still plausibly reduce. At source pixels it is +2–3% in the last 0.2 s with v8 fine-tuned, and +4–5% with the other checkpoints. Next, try a larger `--edge_frac`, judged with `edge_pieces.py` and the localization simulations.

## Residual motion and blur (2026-11-18)

Can the network see small, fast motion left after registration and follow it in the baseline? In principle it could: pixels 5 px or more from the predicted one are visible in the output frame, so a frame's displacement can be read from the surround within the 94 × 94 window. Nothing time-varying from beyond the window enters: the whole-trial inputs are constant. The pooled features cannot carry motion either. They are fractional deviations from the mean image, averaged over rings, and a shift brightens one edge and darkens the opposite one, which cancel.

**Simulator** (all off by default; without them the output is bit-identical to before):

- `motion_sd_px`, `motion_tau_frames`: rigid displacement d(t), one OU process per axis.
  - The baseline and the activity move (cubic-spline shift of each frame). The scan does not, so the valid pixels and the noise stay where they are.
  - `truth/baseline` is the moved baseline; `truth/baseline_static` and `truth/motion` are written too.
  - The mean image and the variance map are those of the moving trial (the static mean moved by 128 of the displacements).
  - Motion is drawn last, so the same seed without motion gives the static baseline exactly.
  - Check: moved − static baseline against −d · ∇B has slope 1.01 and correlation 0.99.
- `motion_substeps`: the motion is drawn at n steps per frame and pooled through the frame-formation kernel, as the 80 Hz frames pool 200 Hz samples. The kernel is the template's temporal noise kernel, interpolated to the steps.
  - Each frame sees the kernel-weighted mean displacement. It is blurred by the displacements' spread within the kernel, applied to second order: ½ c_rr ∂²/∂row² + ½ c_cc ∂²/∂col² + c_rc ∂²/∂row∂col.
  - Against brute-force pooling of spline-shifted frames, the second-order blur leaves 15% of the blur (2% or less of the motion's effect).
  - At SD 0.25 px, in-plane blur is 4% of the shift's effect on the baseline for τ 1.5–4 frames, and 15% for jitter at τ 0.3–1 frames.
- `focus_sd_px2`: focus fluctuations, b_r ∂²/∂row² + b_c ∂²/∂col² of each frame (px²), with b = an isotropic part plus a part per axis (`focus_aniso` × its SD). Each part is a fast (τ 1.5–8 frames) plus a slow (40–320 frames) OU process. Check: drawn SDs of 0.19 / 0.21 px² read 0.16 / 0.20 with `residual_motion.py --blur`, and a rows–cols correlation of 0.71 slow and 0.60 fast (drawn 0.76). On the steepest quarter of pixels it changes the baseline by 2.3–3.0%, as in the real trials.

**Test** (`tools/motion_test.py`). One trial each of 780125, 803121, 878030 and 885233 is the template; the fast-brightness simulator with realistic low counts, as v6–v9 were trained on. Each template gives four trials from the same seed:

- none;
- fast: τ 1.5–4 frames, inside the 8 hidden frames;
- medium: τ 8–30 frames;
- slow: τ 80–320 frames, 1–4 s.

The SD is 0.25 px per axis. The baseline's motion part m = moved − static baseline is 6–10% of the baseline (RMS) on the quarter of pixels where it varies most ("moving" pixels). "Followed" is the slope of (baseline of the moving trial − baseline of the no-motion trial) on m, after removing each pixel's mean; 1 means the baseline follows the motion, 0 that it ignores it. A 17-frame running mean splits the slow part from the fast part. Moving pixels, mean ± SE over the four trials:

| motion | checkpoint | all | slow part | fast part |
|---|---|---|---|---|
| fast | v6r pretrained | 0.05 ± 0.03 | 0.12 ± 0.05 | 0.04 ± 0.03 |
| | v8 pretrained | 0.04 ± 0.03 | 0.11 ± 0.05 | 0.04 ± 0.02 |
| | v8 fine-tuned | 0.03 ± 0.02 | 0.09 ± 0.04 | 0.03 ± 0.02 |
| medium | v6r pretrained | 0.14 ± 0.05 | 0.20 ± 0.06 | 0.03 ± 0.03 |
| | v8 pretrained | 0.14 ± 0.05 | 0.20 ± 0.06 | 0.02 ± 0.02 |
| | v8 fine-tuned | 0.11 ± 0.04 | 0.16 ± 0.05 | 0.02 ± 0.02 |
| slow | v6r pretrained | 0.47 ± 0.09 | 0.49 ± 0.09 | 0.03 ± 0.03 |
| | v8 pretrained | 0.50 ± 0.09 | 0.52 ± 0.09 | 0.01 ± 0.02 |
| | v8 fine-tuned | 0.38 ± 0.07 | 0.40 ± 0.07 | −0.01 ± 0.02 |

- **No checkpoint follows fast motion.** The fast part is followed by 0.04 or less, whatever the motion's time constant.
- **Even motion over seconds is followed only halfway** (0.4–0.5), and less in background (0.21–0.30). The rest stays in data − baseline.
- The networks never saw motion in pretraining. Fine-tuning on real data, where it is small, does not teach it: v8 fine-tuned follows least.
- Without the no-motion trial, the slope of (baseline − static baseline) on m gives the same numbers within 0.03–0.08. The baseline's other errors correlate with m by chance, most in the slow part.

**Residual motion in the real trials** (`tools/residual_motion.py`, the five viewer trials).

- **Method.** Each kept frame of data − activity is fitted as the trial's mean image M, scaled and shifted: (1 + a) M − d_r ∂M/∂row − d_c ∂M/∂col, by least squares weighted by 1 / M.
- **Removing the noise.** Each half of the pixels (8 × 8 checkerboard) gets its own fit. Rigid motion is shared by the halves and noise is not, so their covariance gives the motion's variance free of noise.
- **Validation.** On the motion simulations the estimates match the true SDs to within 0.01 px; without motion they read 0.03–0.05 px.

| trial | rigid motion SD, rows / cols (px) | fast part | slow part | brightness a |
|---|---|---|---|---|
| 780125 | 0.10 / 0.04 | 0.06 / 0.03 | 0.08 / 0.03 | 4.6% |
| 803121 | 0.15 / 0.13 | 0.14 / 0.13 | 0.06 / 0.04 | 3.7% |
| 847983 (GFP) | 0.05 / 0.06 | 0.04 / 0.06 | 0.02 / 0.02 | 2.2% |
| 878030 | 0.18 / 0.20 | 0.16 / 0.17 | 0.06 / 0.09 | 4.2% |
| 885233 | 0.06 / 0.07 | 0.06 / 0.07 | 0.02 / 0.03 | 4.3% |

- **803121 and 878030 have rigid fast motion of 0.13–0.17 px SD per axis.** It is the same in the top and bottom halves of the field, and in the left and right halves. It is unchanged when the fit gets a free additive offset per frame.
- **Its power is spread over 0.5–40 Hz, most at 10–40 Hz**, up to the 80 Hz frames' limit. A flat spectrum like that is what per-frame errors of the registration would give; so would fast motion that the registration does not track. Either one shifts the whole frame.
- The other three trials have 0.05–0.1 px. 780125's slow row motion is not rigid: the top and bottom halves covary negatively, and so do the left and right halves.
- This agrees with the earlier per-frame shifts (`diag_gfp_motion`: 0.20 / 0.17 px in 803121 and 0.39 / 0.41 px in 878030). Those included the estimates' own noise, which is most of the per-frame variance.

**Local motion** (`tools/local_motion.py`, per block of 32 and 64 px: the block's displacement minus the field's; noise removed with two halves on a checkerboard within each block). The same estimator on the no-motion simulations is not zero: it reads 0.08–0.19 px fast and 0.04–0.12 px slow. Within a block, a brightness change that varies across a structure is indistinguishable from a shift of it: for a Gaussian blob, (x − x₀) G ∝ ∂G. The simulator's brightness fields therefore read as local motion. Real trials read more (64 px blocks, rows / cols):

| trial | fast: real / no-motion simulation | slow: real / simulation |
|---|---|---|
| 780125 | 0.11 / 0.12 vs 0.08 / 0.09 | 0.18 / 0.19 vs 0.04 / 0.07 |
| 803121 | 0.15 / 0.18 vs 0.10 / 0.13 | 0.19 / 0.23 vs 0.06 / 0.07 |
| 878030 | 0.27 / 0.22 vs 0.15 / 0.12 | 0.13 / 0.18 vs 0.09 / 0.05 |
| 885233 | 0.12 / 0.12 vs 0.11 / 0.12 | 0.11 / 0.09 vs 0.07 / 0.05 |

- **Slow local changes of this shape are 1.5–4× the simulator's in every session.** Over the simulator's level, they are about 0.07–0.22 px. That is more than the rigid slow motion.
- Fast local changes clearly exceed the simulator's only in 803121 and 878030 (by about 0.12 and 0.2 px), slightly in 780125. In 803121 and 878030 that is about as much as the rigid fast motion.
- Whether they are local motion, local brightness gradients or misfit of GIAnT's activity at edges, a block cannot tell. For the baseline, the first two are the same.

**Blur** (`residual_motion.py --blur`: second derivatives of M added to the per-frame fit, + b_r ∂²M/∂row² + b_c ∂²M/∂col², px²).

- **Motion within a frame should blur it along the motion,** by half the displacement's variance within the frame-formation kernel.
- **In the simulations the blur term is what it should be.** Without motion it reads 0.02 px². With rigid motion it is the shift's own second-order term, ½ d²: 0.05 px², correlation with d² 0.67.
- **The real trials have much more:**

| trial | blur SD rows / cols (px²) | isotropic / anisotropic | b_r–b_c correlation, slow / fast | effect on the steepest quarter of pixels: shift / blur (% of baseline) |
|---|---|---|---|---|
| 780125 | 0.15 / 0.11 | 0.11 / 0.08 | +0.59 / −0.03 | 3.3 / 3.6 (rows) |
| 803121 | 0.32 / 0.27 | 0.25 / 0.16 | +0.71 / +0.18 | 4.4 / 4.6 |
| 847983 (GFP) | 0.10 / 0.13 | 0.09 / 0.07 | +0.51 / +0.11 | 2.1 / 2.7 |
| 878030 | 0.19 / 0.15 | 0.11 / 0.13 | −0.21 / −0.16 | 5.1 / 3.8 |
| 885233 | 0.12 / 0.14 | 0.10 / 0.09 | +0.28 / +0.11 | 2.2 / 2.9 |

- **Blur changes the baseline as much as the displacement does,** slow and fast alike.
- **It is not in-plane motion blur.** It does not follow the frame's squared speed (correlation −0.06 to +0.11, and +0.25 in GFP) or d².
- **Sub-frame motion cannot make this much.** In a simulated trial with sub-frame jitter (τ 0.3–1 frames, seen SD 0.13–0.14 px, spread 0.21 px within the kernel), the blur fluctuates by 0.012 px² SD. That is 0.4% at the steepest pixels, against 3.9% for the shift. The real blur would need spreads of 0.5–0.8 px within frames.
- **The slow part is mostly isotropic** in four sessions, which is what focus (z) changes do. In 780125 it moves with the row displacement (+0.72) and dims the frame (correlation with brightness −0.61), as tissue moving obliquely would.
- **The fast part is not isotropic,** and its source is open.
- **It is not GIAnT's activity misfit:** its correlation with the field's total activity is weak (|r| ≤ 0.27), and GFP has it too.

**Whole-field correction afterwards** (scratch test on the motion simulations).

- **Method.** Estimate d(t) per frame from the whole field, from the raw movie; removing the true activity first changes nothing. Add the fast part of −d · ∇M to the network's baseline, shrunk by s² / (s² + n²). s² is the shared variance of the two halves' estimates and n² the field estimate's noise.
- **Precision.** The field estimate's error is 0.03 px (780125, 803121), 0.05–0.07 px (885233) and 0.09–0.11 px (878030).
- **Result.** RMS error of v8 fine-tuned against the moved truth, fast band, all pixels, in % of the baseline (v6r and v8 pretrained alike):

| motion | before | corrected | corrected, shrunk |
|---|---|---|---|
| none | 6.15 | 6.57 | 6.17 |
| fast | 10.92 | 7.68 | 7.61 |
| medium | 7.99 | 7.08 | 6.92 |
| slow | 6.43 | 6.67 | 6.36 |

- **The correction removes about three quarters of the excess error that fast motion adds.** With shrinkage it costs nothing without motion.
- **On the real trials the shrinkage factors are:** 0.97 (803121), 0.71 (878030), 0.86 / 0.54 (780125, rows / cols), 0.73 / 0.77 (GFP) and 0.52 / 0.44 (885233). That is the share of the rigid fast motion's variance that the correction would remove.
- The field-wide blur term could be corrected the same way (not tested).

**What this means.**

- Real displacement and blur each change the baseline by 1–5% (RMS) at the steepest pixels. The networks follow none of the fast part and about half of the slow part.
- Fast changes are zero-mean noise at edges, unless they are locked to the stimulus or to behaviour. Then they would show as activity in event-triggered averages; that is not checked yet.
- **Per-frame local detection is limited by photon counts.** From the field-wide fits, one frame's displacement is known to 0.07–0.1 px (780125, 803121, GFP), 0.2 px (885233) and 0.3 px (878030) from a 45 × 45 px neighbourhood (the network's reach), and 0.2–0.8 px from 20 px. The motion is 0.05–0.17 px. Even an ideal local detector would recover about two thirds of the fast motion's variance in 803121 and 10–30% elsewhere. Rigid motion is better estimated from the whole field.
- **Slow changes are detectable** (averages over 17 frames cut the noise 3–4×). The network can already see them, but it was never trained on them.

**Next.**

1. Pretraining with motion and focus in the simulations, at the measured sizes: rigid motion, smooth local motion, sub-frame pooling and focus fluctuations. Judge it with `motion_test.py`.
2. In the same retrain, try inputs that build in −d · ∇F0 and b ∇²F0:
   - pooled displacement and blur features: the deviation from the mean image projected on its gradient and second derivatives, over the existing rings, box excluded;
   - the mean image's derivatives as inputs.
3. Keep a whole-field correction with shrinkage as a separate cheap step for rigid motion.

## v9: twice the edge windows (2026-11-18)

**Run.** v8's recipe with `--edge_frac 0.5` in both stages instead of 0.25: from v6r pretrained, 3000 steps on simulations, then 1700 steps of fine-tuning with `--event_mask 0.5`. Selection kept step 2500 of pretraining and the last step of fine-tuning. Fine-tuning barely moved the selection score (0.905, against 0.913 at its start).

**Full evaluation** (as in the v8 table: means over sessions; bias in units of the trial's median pixel brightness, positive = baseline too low):

| | v6r pretrained | v8 pretrained | v8 fine-tuned | v9 pretrained | v9 fine-tuned |
|---|---|---|---|---|---|
| fast-brightness sims: support error above the oracle's | +17.6% | +3.6% | +15.4% | +2.4% | **+1.7%** |
| fast-brightness sims: leak | 0.136 | 0.106 | 0.136 | **0.098** | 0.122 |
| fast-brightness sims: support bias | −0.061 | −0.037 | −0.085 | −0.028 | −0.029 |
| activity-free sims: support / background bias | +0.050 / +0.011 | +0.049 / +0.017 | **+0.017 / +0.001** | +0.053 / +0.019 | +0.070 / +0.015 |
| real support pixels, error vs mean image | −7.9% | −8.2% | −8.0% | −8.2% | **−8.7%** |
| real background, error vs mean image | +0.3% | +0.5% | 0.0% | +0.5% | 0.0% |
| real background bias | +0.010 | +0.019 | 0.000 | +0.021 | +0.015 |

- **v9 fine-tuned is the most accurate checkpoint so far, on simulations and on real data.** Unlike v8's, its fine-tuning keeps the simulation accuracy (within 2% of the oracle). On real support pixels it beats v6r by 0.8 points against the mean image, and on real background it matches the mean image.
- **But it is biased low where nothing happens:** +0.070 at support pixels of activity-free simulations (v6r +0.050, v8 fine-tuned +0.017). v8 fine-tuned's unbiased baseline is not reproduced. It came from that run's checkpoint (step 600) or its edge share, not from fine-tuning as such.
- v9 pretrained is slightly better than v8 pretrained on fast-brightness simulations (error, leak, bias), slightly worse on activity-free ones, and the same on real data.

**Real trials** (`leak_check --real`): slope of the baseline's fast part on the activity at support pixels.

| | 780125 | 803121 | 878030 | 885233 |
|---|---|---|---|---|
| v6r pretrained | 0.031 | 0.072 | 0.002 | 0.043 |
| v8 fine-tuned | 0.033 | 0.054 | 0.013 | 0.041 |
| v9 pretrained | 0.027 | 0.063 | 0.014 | 0.039 |
| v9 fine-tuned | 0.030 | 0.056 | 0.027 | 0.043 |

As with v8: on real trials v9 follows the activity about as much as v6r.

**Trial edges** (`edge_pieces.py`, same trials and cuts): piece − whole, in % of the baseline, by frames from the cut; positive = the piece's baseline is higher. SEs as in the v8 section (first bin about 1–1.5 at support pixels in simulations, 1.5–2.6 on real trials).

| simulations | 0–8 frames | 8–16 | 16–32 | 32–64 | 64–128 | 128–256 |
|---|---|---|---|---|---|---|
| support, end of a piece: v8 pretrained | +4.2 | +3.3 | +1.6 | +1.0 | +0.5 | +0.3 |
| v9 pretrained | +3.8 | +3.0 | +1.5 | +1.0 | +0.4 | +0.2 |
| v8 fine-tuned | +2.2 | +2.5 | +1.4 | +0.9 | +0.4 | +0.2 |
| v9 fine-tuned | +4.3 | +3.5 | +2.1 | +1.4 | +0.5 | +0.2 |
| support, start: v8 / v9 pretrained, v8 / v9 fine-tuned | +1.2 / +1.0, +0.3 / +1.2 | +0.2 / +0.2, +0.4 / +0.5 | −0.5 / −0.4, −0.2 / −0.2 | −0.1 / −0.1, 0.0 / 0.0 | −0.1 / −0.1, 0.0 / 0.0 | 0.0 |
| background, end: v8 / v9 pretrained | +2.0 / +1.9 | +2.1 / +2.2 | +1.5 / +1.6 | +1.1 / +1.2 | +0.5 / +0.5 | +0.2 / +0.2 |
| background, end: v8 / v9 fine-tuned | −0.7 / +1.9 | +0.2 / +1.9 | +0.3 / +1.4 | +0.4 / +1.0 | +0.4 / +0.4 | +0.2 / +0.2 |
| background, start: v8 / v9 pretrained, v8 / v9 fine-tuned | +1.5 / +1.3, +0.8 / +1.2 | +0.3 / +0.4, +0.6 / +0.7 | −0.2 / −0.2, +0.1 / +0.2 | 0.0 / 0.0, +0.2 / +0.2 | 0.0 / 0.0, +0.1 / +0.1 | +0.1 |

| real trials | 0–8 frames | 8–16 | 16–32 | 32–64 | 64–128 | 128–256 |
|---|---|---|---|---|---|---|
| support, end of a piece: v6r pretrained | +5.2 | +3.9 | +1.9 | +1.3 | +0.7 | +0.1 |
| v8 / v9 pretrained | +5.0 / +4.6 | +4.3 / +4.0 | +2.1 / +2.1 | +1.4 / +1.4 | +0.6 / +0.6 | +0.1 / +0.1 |
| v8 / v9 fine-tuned | +2.6 / +4.6 | +3.1 / +4.0 | +1.7 / +2.4 | +1.2 / +1.6 | +0.6 / +0.6 | +0.1 / +0.1 |
| background, end: v8 / v9 pretrained | +2.2 / +2.2 | +2.5 / +2.6 | +1.9 / +2.0 | +1.3 / +1.4 | +0.6 / +0.7 | +0.2 / +0.2 |
| background, end: v8 / v9 fine-tuned | −0.5 / +2.1 | +0.5 / +2.1 | +0.6 / +1.6 | +0.6 / +1.1 | +0.4 / +0.5 | +0.2 / +0.2 |
| support, start: v9 pretrained / fine-tuned | +0.7 / +0.8 | −0.1 / +0.2 | −0.6 / −0.4 | −0.2 / −0.1 | −0.2 / 0.0 | 0.0 / 0.0 |
| background, start: v9 pretrained / fine-tuned | +1.1 / +1.1 | +0.3 / +0.7 | −0.2 / +0.2 | −0.1 / +0.2 | −0.1 / +0.1 | +0.1 / +0.1 |

Against the true baseline (simulations), the support pixels in the last 8 frames of a piece sit 4.7 (v9 pretrained) and 5.3 points (v9 fine-tuned) above the checkpoint's mid-trial bias, as v8 pretrained's did (5.1).

- **Twice the edge windows does nothing measurable.** In the last 8 frames of a piece at support pixels, simulations show 5.0 (v7b, no edge windows), 4.2 (v8, a quarter) and 3.8 (v9, half). That trend is well inside the SE. Real trials show 5.2 (v6r), 5.0 and 4.6.
- **v9 fine-tuned has the edge bias of the pretrained checkpoints**, not v8 fine-tuned's halved one, so that came with v8's checkpoint, not with fine-tuning on edge windows.
- **So the edge bias is not a shortage of edge examples.** The mean loss is minimized by the conditional mean, and these simulations come from the training simulator. Yet the baseline sits about 5 points above the truth at the ends of pieces and close to it mid-trial. The network falls short of the conditional mean at edges however often it sees them. Next places to look: how the model represents frames beyond the trial edge (padding, validity input), or else a fixed correction. Real trials show the same profile as the simulations, so a correction calibrated on simulations would transfer. Until then, the first and last ~16 frames (0.2 s) of a trial carry a baseline that is 2–5% too high at source pixels.

**Localization on the simulations** (same harness and 8 trials as the v8 table):

| baseline | default threshold (8): precision / recall / F1, fast-brightness sims | same, default sims | best single threshold: F1 (threshold), fast | same, default |
|---|---|---|---|---|
| v6r pretrained | 0.82 / 0.81 / 0.81 | 0.87 / 0.78 / 0.82 | 0.82 (6) | 0.82 (8) |
| v8 pretrained | 0.80 / 0.84 / 0.82 | 0.79 / 0.80 / 0.80 | 0.82 (8) | 0.80 (10) |
| v8 fine-tuned | 0.79 / 0.80 / 0.80 | 0.82 / 0.78 / 0.80 | 0.81 (7) | 0.80 (9) |
| v9 pretrained | 0.80 / 0.83 / 0.81 | 0.79 / 0.80 / 0.79 | 0.81 (9) | 0.80 (7) |
| v9 fine-tuned | 0.82 / 0.83 / 0.83 | 0.82 / 0.80 / 0.81 | 0.83 (7) | 0.83 (9) |

- **v9 fine-tuned localizes best among the v8 and v9 checkpoints, and about as well as v6r.** At the default threshold it finds 248 of the 304 true sources (v6r 242), with 53 false ones (v6r 46, v8 pretrained 63). As before, the differences are within what eight trials can resolve. v6s fine-tuned, an older checkpoint, still scores highest (0.84 / 0.83).

**Decisions.**

- **Keep v6r pretrained as the default for now; v9 fine-tuned is the candidate to replace it.** It is the most accurate baseline on simulations and on real data, and it localizes about as well on the simulations. Against it: the activity-free bias (+0.070 against +0.050), and an edge bias about as large as v6r's. Switch once the flash test on real trials (780125 Path 2, as for v6r) shows it finds as many real sources with no more false ones.
- Keep `--edge_frac 0.5`: it costs nothing, and v9 pretrained is marginally better than v8 pretrained. But more edge windows will not remove the edge bias.

## Motion-correction offsets as predictors (2026-11-19)

**Question.** Could the motion-correction offsets predict baseline changes, and could a network learn the relation at each location? The exports carry, per frame: the microscope's online offsets (x, y, z, from the line headers); the offline registration's shifts (rows, columns); and its error (`frame/onlineMotion{X,Y,Z}shift`, `frame/motion_row`, `frame/motion_col`, `frame/rec_neg_err`). GIAnT's newest registration (compensated registration, GIAnT-MATLAB #43) corrects each line for its own online offset; these exports predate it.

**Test** (`tools/motion_covariates.py`, the five viewer trials). Covariates: the six offsets, their frame-to-frame changes, the offline motion's squared speed and the registration error.

**A. The frame's own fit** (brightness a, residual shift d and blur b per frame, as in `residual_motion.py --blur`). Correlations are noise-free, from the two halves:

- **Fast brightness** follows the registration error (r −0.48 to −0.91). That error is computed from the image itself. Fast brightness also follows the online z velocity (+0.36 to +0.61, in four trials). The network already sees a frame's brightness in the surround.
- **Fast blur follows the squared speed of the motion the registration removed:** r +0.38 to +0.51 along rows, in four trials. The R² of fast blur on all covariates is 0.2–0.33. The earlier finding ("not in-plane motion blur") holds for the residual motion only. Motion that registration corrects frame by frame still blurs each frame.
- **Frames in which the online offset switches have more fast blur:** row blur SD 0.155 against 0.089 px² in 780125, and higher in all five trials. The residual shift is the same in both kinds of frame. A switch within a frame blurs that frame, and compensated registration removes this.
- Slow brightness, shift and blur follow the online offsets in 780125 (R² 0.6–0.85: drift), less in the others.

**B. Per pixel, out of sample.** The residual after v9 fine-tuned's baseline (8-frame means, fraction of the baseline) is regressed per pixel on the covariates. The fit uses alternate 128-frame blocks and is scored on the others. The same regression with the covariates circularly shifted gives the chance level, which is subtracted. Share of the residual's variance explained, quarter of pixels with the steepest mean image:

| trial | frame's own fit (5) | offsets (12) | both | fast band (minus a 33-frame mean): own fit / offsets / both |
|---|---|---|---|---|
| 780125 | 0.6% | 1.7% | 1.8% | 0.5 / 0.8 / 0.9% |
| 803121 | 5.1% | 7.2% | 10.5% | 2.9 / 3.3 / 5.6% |
| 847983 (GFP) | 1.1% | 0.9% | 1.4% | 0.8 / 0.5 / 1.1% |
| 878030 | 0.7% | 0.5% | 1.0% | 0.5 / 0.9 / 1.5% |
| 885233 | 0.5% | 0.9% | 1.2% | 0.2 / 1.5 / 1.5% |

- **The offsets carry some information per location** beyond the frame's own fit, but little: at most 1.5% of the fast residual's variance, except in 803121 (3–6%).
- Much of the fast part comes from the online offsets, where 878030 and 885233 switch in 62–72% of frames. That is the switch blur above, which is fixed upstream.
- The slow part is drift. The network can follow it from its temporal context once it is trained on slow motion (v10).

**How a network would learn it per location.**

- The covariates would enter as frame-level input channels, spread over the window.
- The coefficients differ at every location, so the network would also need a per-pixel regression fitted in context: blind ridge regression on the context frames outside ±8. The convolutional trunks have no multiplicative path to fit those coefficients themselves.
- The simulator would have to produce covariates consistent with its motion.
- Every input path would have to carry them: exports, and `predict-registered` reading them from the alignment files.

**Decision: dropped.** It is too unwieldy for at most 1.5% of the residual variance in four of the five trials. Most of that is either fixed upstream (online switches) or visible in the images.

- v10b gets the physical version from the images instead: the surround's displacement and blur, mapped through the pixel's own mean-image derivatives (below).
- New exports should use GIAnT's compensated registration (`params.compensateOnline`).

## v10: motion in the simulations, motion features (2026-11-19)

**Simulator: local motion** (`local_sd_px`, `local_px`, `n_local_fields`, `local_tau_frames`). On top of the rigid shift there is now a smooth displacement field per frame. It has three smooth spatial fields per axis (spacing 20–60 px), each weighted by a fast (1.5–8 frames) and a slow (40–320 frames) OU process. The fields are scaled so that the drawn SD is the displacement's SD per pixel on average. Each frame of the baseline and activity is cubic-spline warped, like the rigid shift.

- Without it the simulator's output is bit-identical to before, with or without rigid motion and focus.
- The warp is exact (unit test).
- Moved − static baseline against −D · ∇B: slope 1.00, correlation 0.97 (780125 template, 0.4 px).
- A 718 × 94 × 94 training window takes 3.6 s to simulate with rigid motion, focus and local motion (1.35 s without). The 12 data workers keep up with the GPU.

**Training options:** `--motion_sd`, `--motion_tau`, `--focus_sd`, `--local_sd`. `--val_motion` and the evaluation kind `motion` (`motionpz` with realistic low counts) use the measured sizes (`train.MOTION_SIM`):

- rigid SD 0–0.2 px per axis, τ 0.5–40 frames;
- focus 0–0.25 px²;
- local motion 0–0.2 px each for its fast and its slow part.

**Model: motion features** (`--motion_features`, box model). For each pool scale (squares minus the box, as for the pooled features), each frame's deviation from the mean image M is fitted by weighted least squares as a M − d · ∇M + b_r ∂²M/∂row² + b_c ∂²M/∂col². The feature is the change the fitted shift and blur make at the pixel itself, for the frame and for the 8 frames around it. The pixel's own ∇M / M and ∂²M / M are also inputs.

- A shift brightens one side of a structure and darkens the other, which the fractional pooled features average away. These features keep it.
- They are computed in float32 outside autocast.
- They keep the box blind spot (gradient test), and a checkpoint without them warm-starts a model with them exactly.
- On a scene of Gaussian blobs shifted by (0.25, −0.15) px, the feature matches the true change at each pixel (slope within 15%, r > 0.95).

**Runs.** Both arms start from v6r pretrained, as v9 did, with v9's recipe and steps (3000 pretraining, 1700 fine-tuning), plus motion, focus and local motion in all simulated windows. Pretraining selects on the motion simulations.

- v10a: no other change.
- v10b: v10a plus the motion features.
- Both evaluate on real, free, fast, fastpz and motionpz windows. v10a also re-scores v9 (the v9 run is now a data asset) so the motion simulations compare like with like.

**Status.** Both runs finished in the afternoon, after a morning when runs with data assets failed to start on Code Ocean. Results: "v10 results" below.

## Flash test: v9 fine-tuned on 780125 (2026-11-19)

**Run.** As for v6r: `slap2-f0 predict-registered` with v9 fine-tuned for all 30 registered movies of 780125, then SILo with those localization baselines (GIAnT branch `localization-network-baseline`, 3460 s against 3252 s for v6r). Then `source_patches.py` and `source_flash_check.py` on v6r's sources and v9 fine-tuned's (one-to-one matches within 1.5 px), both paths.

| | Path 1: sources, passing | Path 2: sources, passing |
|---|---|---|
| found by both | 186, 183 | 139, 133 |
| only v6r pretrained | 12, 12 | 11, 10 |
| only v9 fine-tuned | 19, 19 | 15, 14 |
| **v6r pretrained** | 198, 195 (98.5%) | 150, 143 (95.3%) |
| **v9 fine-tuned** | 205, 202 (98.5%) | 154, 147 (95.5%) |
| for comparison: v6 fine-tuned (earlier test) | 192, 188 | 146, 141 |

- **v9 fine-tuned finds 11 more validated sources than v6r (349 against 338), with the same number that fail (10 each).** Against the union of passing sources, its recall is 0.94 on both paths, against 0.91 for v6r.
- The sources only one of the two runs finds pass the test as often as the shared ones (93–100%). The two baselines differ at the margin, and both margins are mostly real synapses.

**Decision: v9 fine-tuned replaces v6r pretrained as the default.** It meets the condition set in the v9 section: as many validated real sources (more), and no more false ones. It is also the most accurate checkpoint on simulations and real windows. What remains against it: a low baseline at support pixels of activity-free simulations (+0.070 against +0.050), and an edge bias as large as v6r's. v10 is judged against v9 fine-tuned from here on.

## Data available: iGluSnFR4f multiROI sessions on aind-open-data (2026-11-19)

**Survey** (`sessions/survey/survey_durations.py`; list in `sessions/slap2_iglusnfr4f_multiroi.csv`).

- Every bare `<subject>_<date>_<time>` prefix on the bucket (5,199) has its `data_description.json` read: 252 are SLAP2.
- 39 of those were not in the 2026-11-14 list. They are 9 iGluSnFR4s sessions, 10 ASAP7y (OpenScope) sessions, 15 iGluSnFR4f sessions holding only structural reference stacks, 4 new iGluSnFR4f multi-roi raster sessions (2026-11-17/18, "indicator testing") and one legacy duplicate.
- Recording time comes from every experiment `.dat` file's first and last line timestamps (52,427 files). Checks: a 780125 trial reads 28.0 s (2254 frames at 80 Hz), and 803121 reads 1,740 s on both paths.

**What qualifies** (iGluSnFR4f, multiROI raster, 2025 onward):

| | sessions | mice | recording | path-time (two DMD paths at once) | raw .dat |
|---|---|---|---|---|---|
| continuous | 98 | | 60.6 h | | |
| triggered trials | 42 | | 32.5 h | | |
| **all** | **140** | **32** | **93.0 h** | **186 h (669,451 s)** | **8.9 TB** |

- Sessions last 6–77 min (median 29.5). Per mouse: 0.3–14 h, most 1–5 h.
- Processed assets on the bucket: 69 sessions (52 h) are motion-corrected, 62 source-extracted, 11 processed. 68 sessions have none yet.
- Left out: 6 iGluSnFR4s multiROI sessions (2.3 h); 5 iGluSnFR4f sessions whose scan mode is in doubt (1.9 h); the voltage (ASAP, FORCEB) and GFP sessions.

**Against what the network trains on.** v9 trained on 1,015 s of path-time, from three sessions. Those three sessions alone hold 15,100 s, and the validation session 7,600 s. Each export was capped at 180 s per path (`--max_duration_s 180`). All qualifying sessions together hold about 660 times the current training data, from 32 mice instead of 3.

## Export plan: more real data, scaling and held-out tests (2026-11-19; launched at 15 min per session, see below)

**What may be used.** iGluSnFR4f multiROI raster recordings from 2025 onward, with two exclusions:

- **No reference stacks**, whatever the metadata says. Some sessions keep `refstack_*` acquisitions inside `dynamic_data`, and the export would take them today: `run_export.m` passes every `.dat` there to `buildTrialTableSLAP2`. An acquisition is a reference stack if its name says so (refstack, refStack, structure, reference), its annotation says `type: ref_stack` or `full-field raster`, or its `.meta` is a simple raster or has `enableStack` set. One session whose metadata calls it multiROI is dropped for this reason (838410_2026-03-19, annotation `type: ref_stack`).
- **No session with an integration scan in any acquisition.** That is any Integrate ROI in any `.meta`, even alongside raster ROIs. This drops 15 sessions: 11 already listed (mice 880917, 880918, 890921, with 1–11 Integrate ROIs each), and all 4 new sessions of 2026-11-17/18 (1–9 Integrate ROIs on at least one path).

**What remains:** 125 sessions from 26 mice, 90.4 h of recording, 650,000 s of path-time (`sessions/slap2_iglusnfr4f_multiroi.csv`, minus the sessions above). The mice fall into five cohorts:

- 2025, jRGECO1a, triggered: 776266, 776270, 780125, 787725, 794237;
- 2025, RCaMP3, triggered: 796630, 801381, 828408, 828409, 829704;
- 2025, no red channel, continuous: 803496, 804730, 804733, 810196;
- 2025, RCaMP3, continuous: 803121, 809047;
- 2026, RCaMP3, continuous: 826033, 834788, 838410, 867430, 876767, 878029, 878030, 885233, 885234, 885235.

**Phase 1** (`sessions/export_phase1.csv`): 17 sessions from 17 mice, one session each.

| role | mice | cap per path | path-time |
|---|---|---|---|
| training | the 3 current training mice (780125, 803121, 885233) and 6 new ones, one per cohort (776266, 828408, 803496, 809047, 826033, 867430) | 30 min | 30,700 s (30× the current 1,015 s) |
| validation (selects checkpoints) | 878030 (as now), 794237, 804730, 834788 | 10 min | 4,800 s |
| test (reported only, never used to select) | 776270, 810196, 829704, 876767 | 10 min | 4,800 s |

- Every cohort has mice in training, in validation and in test. No mouse is in more than one role.
- The GFP session 847983 stays the separate activity-free control.
- About 12 MB of export per path-second, so 0.5 TB in total.
- Compute: the capped exports took up to 12 machine-seconds per path-second, fixed costs included. That makes at most about 130 machine-hours, run as one computation per session; the longest is about 12 h.
- The remaining 108 sessions (17 training-pool mice) wait for the scaling result.

**Export changes needed first** (none made yet):

1. **Acquisition filter in `run_export.m`.** Build a folder of links to the allowed acquisitions and pass it to `buildTrialTableSLAP2`. The rules are those above. A session with any integration ROI is skipped whole. `run_info.json` lists every acquisition dropped and why.
2. **Caps per session from the manifest**, and 0 = no cap.
3. **Current GIAnT main with compensated registration** (`params.compensateOnline`; other new options off). The online offsets switching within a frame blur that frame; compensated registration removes this. The three current training sessions and 878030 are exported again the same way, so all runs see one kind of data.
4. **Trial statistics computed in the export run and stored with it.** Training spent 935 s on stats for 71 files, and phase 1 has about 1,500 files.

**Training changes needed:**

- Explicit splits: `--val_sessions` (today one session is drawn at random), with the test sessions never loaded.
- Keep every checkpoint (`step_<N>.pt` every 250 steps; today only `last.pt` and `best.pt`).
- An evaluation of all checkpoints on every held-out session, with bootstrap intervals over mice and windows.

**Experiment 1: does more data help, and is it amount or diversity?** All runs use the same recipe (v10's winner; that is v9's, see "v10 results"), steps, validation mice and test mice:

| run | training data | path-time | question |
|---|---|---|---|
| S0 | 3 mice, first 180 s per path (the current data, from the new export) | 1,000 s | baseline |
| S1 | the same 3 mice, up to 30 min per path | 10,000 s | more data, same mice |
| S2 | all 9 training mice, about 9 min per path each | 10,000 s | same amount, 3× the mice |
| S3 | all 9 training mice, up to 30 min per path | 30,700 s | both |
| S3-long | S3 with twice the steps | 30,700 s | is the step budget the limit? |
| S0, S3 seed 2 | repeats | | run-to-run noise |

- **Readouts on the test mice:**
  - real windows: error against the mean image, at support pixels and background;
  - simulations from test-mouse templates: error against the oracle, leak, bias without activity, edge bias, motion following (`motion_test.py`), localization;
  - the flash test on two test sessions, for the final candidates only.
- **Decisions:**
  - If S1 or S3 beats S0 by more than the seed-to-seed spread, export the rest (phase 2).
  - If S2 beats S1, diversity matters more than amount. Phase 2 then takes one or two sessions from every mouse before whole sessions.
  - If S3-long beats S3, longer training comes with more data.

**Experiment 2: do more held-out sessions help?** Today one validation session (48 windows) picks the checkpoint, and neighbouring checkpoints' scores differ by up to 0.07. From every checkpoint of every Experiment 1 run, scored on all 8 held-out mice:

- **Selection regret:** pick the checkpoint with 1 (878030 alone), 2 or 4 validation mice, and with 48 or 192 windows per mouse. Measure how much worse each pick scores on the test mice than the best checkpoint.
- **Agreement:** rank correlation of checkpoints and of runs between validation mice and test mice.
- **Resolution:** how many mice it takes to separate S0–S3 beyond the seed-to-seed spread.
- **Outcome:** how many held-out mice and windows are worth using from then on.

**Order:**

1. Export filter, caps and stats, and the training splits and checkpoints. Test them on one session.
2. Export phase 1.
3. Run S0–S3 and the repeats (about 9 GPU runs of about 6 h each; at most a few run at once on Code Ocean).
4. Evaluate, then decide on phase 2.

## Export phase 1: launched (2026-11-19)

**Settings** (the user's, 2026-11-19): the filter above; GIAnT main (9695d27) with compensated registration (`compensateOnline`) and WMF seeding with `wmfPeakth` 3; **15 min per session** (first 900 s); trial stats stored next to each trial (`slap2-f0 stats`, `*_stats_v5.npz`, found first by training wherever the data are mounted). The 17 sessions of `sessions/export_phase1.csv` run as one Code Ocean computation each.

**Why 15 min, and HDF5 movies.** GIAnT's working files must all exist at once (SiLo reads every trial). On 780125 (845 × 320 px canvas) they are about 600 MB per second of recording with TIFF movies: 30 min would be about 1 TB, against a 300 GB results volume. The registered movies are 87% NaN outside the ROIs and shrink about 9× as deflated HDF5 (GIAnT's `saveTif` false, lossless), which brings it to about 225 MB/s (the alignment files' uncompressed `varFacDS`, 2.4 GB per 28 s trial and path, is most of the rest). At 15 min that is about 200 GB plus about 20 GB of export. Each run logs disk use every 5 min (`logs/disk_usage.txt`).

**Data, at 15 min:** training 16,200 s of path-time (16× the current 1,015 s), validation 7,200 s, test 7,200 s. Experiment 1 becomes:

| run | training data | path-time |
|---|---|---|
| S0 | 3 mice, first 180 s per path | 1,080 s |
| S1 | the same 3 mice, 15 min per path | 5,400 s |
| S2 | 9 mice, 5 min per path | 5,400 s |
| S3 | 9 mice, 15 min per path | 16,200 s |

**Code** (commits 2417673, 42f2be5, ec96430):

- Export: `selectAcquisitionsSLAP2` / `slap2MetaInfo` apply the filter and give GIAnT a folder of links to the usable acquisitions (`removeLinkDir` removes the links without following them); `run_info.json` records each decision; `--compensate_online`, `--peak_method`, `--wmf_peakth`, `--save_tif`, `--stats`; `--max_duration_s 0` = no cap.
- Training: `--val_sessions` / `--train_sessions`, `--max_train_s` (first trials of each session and path), `--keep_every` (`steps/step_<N>.pt`); `evaluate --ckpt run=.../steps/step_*.pt` and `--per_window` (`windows.csv`, each window's 8-frame error sums, for bootstrap intervals).

**Local test** (780125, first 120 s = 5 trials, so SiLo used WMF): every stage ran, including the filter (one activity acquisition used), links, cleanup and stats. But the export QC changed against the old export of the same session (GIAnT d2fb85c, uncompensated, getActImPeaks, 7 trials):

| | old export | new export |
|---|---|---|
| sources, path 1 / 2 | 106 / 74 | 220 / 193 |
| activity gain | 0.78–0.86 | 0.53–0.63 |
| peak slope after the gain | 0.95–1.03, blur 0–0.75 px | 1.10–1.29 at blur 0 |
| event residual | 0.08–0.18 | 0.20–0.31 |

Two to three times the sources, and GIAnT's activity now overshoots the 80 Hz movie by about 1.7× (was 1.2×), with more event signal left in movie − activity. Likely cause: many weak WMF sources at threshold 3, whose noisy traces dilute the gain regression; compensated registration (a sharper movie) may add to it. A local diagnostic separates the two (compensated + getActImPeaks; uncompensated + WMF 3): see below.

**Diagnostic: WMF 3 causes the change; compensation does not.** The same 120 s of 780125 with the same exporter:

| | sources, path 1 / 2 | gain | peak slope | event residual |
|---|---|---|---|---|
| old export (GIAnT d2fb85c: uncompensated, getActImPeaks, 7 trials) | 106 / 74 | 0.78–0.86 | 0.95–1.03 | 0.08–0.18 |
| C: compensated, getActImPeaks | 99 / 77 | 0.78–0.86 | 0.97–1.02 | 0.11–0.21 |
| B: uncompensated, WMF 3 | 214 / 194 | 0.53–0.68 | 1.05–1.30 | 0.16–0.33 |
| phase 1: compensated, WMF 3 | 220 / 193 | 0.53–0.63 | 1.10–1.29 | 0.20–0.31 |

- **It is not weak sources pulling the gain down.** The slope of the movie on the written activity at each source's peak pixel is the same in every strength quartile (1.13–1.16 on path 1, 1.25–1.28 on path 2), and does not depend on how many sources are nearby. With WMF the activity is flatter than the movie for strong and weak sources alike. The gain (from the flux in a 4 px disk) fits the flux, and the peak pixel keeps 10–30% of the activity. The blur that should make up the difference is at its lower bound (0), so the exporter cannot sharpen.
- **The two training targets at the same events.** Both compensated runs have identical movies, so both targets can be read at the same pixels and frames. Events are the peaks of each method's own activity at its sources:

| events at | sources per trial | events | event size | left in WMF target | left in getActImPeaks target |
|---|---|---|---|---|---|
| path 1: found by both | 69 | 1,852 | 78 | 0.31 | 0.19 |
| path 1: only WMF | 150 | 5,136 | 23 | 0.13 | 1.00 |
| path 1: only getActImPeaks | 10 | 282 | 73 | 0.58 | 0.15 |
| path 2: found by both | 62 | 2,165 | 57 | 0.33 | 0.15 |
| path 2: only WMF | 130 | 5,239 | 27 | 0.20 | 0.99 |
| path 2: only getActImPeaks | 5 | 190 | 49 | 0.70 | 0.22 |

- **Net effect:** summed over these events (size × count × fraction left), the WMF target keeps about half as much event signal as the getActImPeaks one (path 1: 72k against 149k; path 2: 75k against 158k). The reason is that getActImPeaks misses about 140 weaker sources per path, whose events stay whole in its target. In exchange, WMF removes the strong synapses that both methods find only half as well (31–33% left against 15–19%). It also mostly leaves the few strong sources that only getActImPeaks finds (58–70% left). That fits WMF merging close pairs: on path 1, 89 getActImPeaks sources have a WMF source within 2 px, but only 69 WMF sources have a getActImPeaks source that close.
- Some events at the WMF-only sources may be noise peaks; the WMF validation put the false-source rate at 3% or less at threshold 3.
- The phase-1 exports keep WMF 3. What would improve them: WMF footprints that are as peaked as the movie and keep close pairs apart (GIAnT), after which a re-export would be worthwhile.

**Out of memory.** Three 2026 RCaMP3 sessions (867430, 876767, 878030) were killed about 20 min into registration with 32 MATLAB workers. They were relaunched with 12 workers (17:43). Runs now log memory use with disk use every minute (8b5e577).

**Disk test (finished after the launch).** Code Ocean's results volume grows as it fills: from 300 GB to 1,800 GB, with 1,500 GB written in 10 GB files and no failure (about 8 GB/min). The 15 min cap was not needed for disk. Phase 2 can use 30 min or whole sessions, as long as the run time allows it.

**Phase 1 results (2026-11-20).** 14 of 17 sessions exported: 28 session-paths, 1,175 trials, 6.85 h of path-time (900 s per path). Each run took 3.3 to 12 h: registration 0.5 to 1.3 h, SiLo 1.4 to 4 h, export 1.1 to 7 h. Two failures and one crash:

- 803496 and 804730 (train and validation, the only one-channel recordings) exported no trials. Every trial failed in `exportF0TrainingData`: `squeeze` dropped the channel dimension of SiLo's one-channel `mean_im`. The run still reported `ok`, and data assets were made (empty). Fixed (e83933f): `mean_im` is reshaped to channels × pixels, a path with one source no longer fails the same way, and `run_export` now fails a session in which no trial exported (`trials_exported`, `trials_failed` in `run_info.json`). Both relaunched (04:32).
- 828408 (train, RCaMP3) died with 32 workers at SiLo's localization of path 2 (12 trials at once, estimated 16.4 GB each, 209 GB free), most likely out of memory. Relaunched with 10 workers (04:16). The three relaunched with 12 workers peaked at 146 to 182 GB of 246 GB with the same 12 trials at once.

QC of the 14 (medians over trials, per session-path):

| | median | range |
|---|---|---|
| activity gain | 0.60 | 0.50 to 0.84 |
| peak slope after the gain | 1.13 | 0.99 to 1.34 |
| blur | 0 px | 0 to 0.75 px |
| event residual | 0.15 | 0.06 to 0.32 |

Over 15 min the QC is better than in the 120 s diagnostic on 780125 (gain 0.66 and 0.53, peak slope 1.09 and 1.27, event residual 0.15 and 0.23, for paths 1 and 2). The pattern is the same: most paths sit at blur 0 with the peak slope above 1, so WMF's footprints stay flatter than the movie. Four paths (794237 both, 826033 path 1, 810196 path 1) match the old getActImPeaks exports (peak slope 1.0, blur 0.5 to 0.75).

**Decisions (the user's, 2026-11-20).** WMF 3 stays, and Experiment 1 trains on these exports. Phase 2 also takes 15 min per session. The two empty assets and 828408's first failed run (225 GB of working files) were deleted.

**828408 ran out of memory again.** The relaunch with 10 workers died at the same step, SiLo's localization of path 2 with 10 trials at once. Memory rose from 23 to 209 GB of 246 within 5 min: SiLo's estimate (16.4 GB per trial) is too low for this session. Relaunched with 6 workers (08:33).

**Export about 6× faster, same output (dc4d38e).** The export took 1.1 to 7 h per session on Code Ocean, more than SiLo on some sessions. One trial of the 120 s test on 780125 took 94 s; the profile showed:

- `placeFootprints`: 46 s. For each of 90 placements per trial (sub-pixel offsets, then blurs), every source's footprint was copied out of the whole canvas, searched and interpolated on the whole canvas grid. Each footprint is now interpolated on a crop around it that holds every grid neighbour of its sample points, which gives the same values: 3.4 s.
- HDF5 writes: 25 s. Deflate 1 instead of 4 writes movie and activity 40% faster, with files 5 to 7% larger (lossless).
- Footprint reads: SiLo's `per_trial_profiles` is a dense [trials sources 1 rows cols] array, uncompressed and contiguous, with trials varying fastest on disk. Reading one trial walks the whole dataset, about 10 GB per path in a 15 min session, once per trial, so the cost grows with the square of the number of trials (1 GB here, with 5 trials). All trials are now read in one pass of contiguous column blocks, as sparse matrices.
- Two full pixels × frames double arrays (about 5 GB each) were made only to read some of their rows; only those rows are computed now.
- Trials are independent and now run in parallel (`--export_workers`, default 8, at most `--n_workers`), about 15 GB each.

| 780125, 120 s, 10 trials | time |
|---|---|
| old exporter | 922 s |
| new, one trial at a time | 456 s |
| new, 5 at once, deflate 1 | 164 s |

Every dataset, attribute and summary value is identical to the old exporter's (deflate changes only the file bytes).

On Code Ocean, 794237 (15 min, 2 paths, 64 trials) re-exported with the new code, same GIAnT commit and machine: the export took 9.7 min instead of 67, and the whole run 2.12 h instead of 3.05. Reading the footprints took 5 and 1 s per path, and memory peaked at 98 GB during the export (8 trials at once). All 27 per-trial QC values of all 64 trials are identical to phase 1. 794237 was one of the quicker exports (1.1 h); the slow ones (up to 7 h) should gain more, not measured. Notes for others: AllenNeuralDynamics/slap2-dev#63.

**Experiment 1, S0 and S1.** Their training mice (780125, 803121, 885233) are exported, so they can start before the rest. Launch files use v9's recipe (from v6r pretrained; 3000 + 1700 steps), `--train_sessions`, `--val_sessions 878030,794237,834788` for best.pt in every run (804730 joins the held-out evaluation), `--keep_every 250`, `--max_train_s 180` for S0, and `--max_hours` 5 and 4 (v10 was cut short at 3). A local run checked the arguments, the split and the checkpoints. The GPU capsule refused the update until its pending IDE change (Code Ocean deleting `.codeocean/environment.json`, since the Dockerfile had been edited by hand; the Dockerfile keeps the same packages) was committed. S0 started 2026-11-20 10:52; S1's first launch found no free GPU machine and is retried every 15 min.

**S0 and S1 crashed at their first validation (3c90dca).** Both ended after 12 min in the simulator's Poisson draw (numpy: lam value too large, i.e. NaN expected counts). 878030 Path 1 trial 27 has no noise statistics: gain, the noise correlations and the skew are NaN. The exporter's labeled mask (median-filtered mean image above 3x its 25th percentile over the canvas) covered every valid pixel in this dim trial, so the stats had no unlabeled valid pixels to work from. It is the only such trial among the 506 of S0's six sessions. Training and evaluation now leave out trials without finite noise statistics and name them. The exporter takes the percentile over the valid pixels when fewer than 5% of them would be unlabeled; normal trials are unchanged (the local 120 s export on 780125 is identical). Code Ocean marks a run "succeeded" whatever its exit code: these two exited with 1, so check `exit_code`. Both relaunched.

**All 17 phase-1 exports are done (2026-11-20).** The three relaunched sessions exported 96 trials (803496), 93 (804730) and 64 (828408, with 6 workers).

**S1 ran out of memory; S1 relaunched, S2 and S3 launched (ed47628).** The relaunched S1 was killed after 1.3 h with exit code 137 while S0 kept running. Each of the 12 training workers kept every trial and simulation template it had opened, about 8 MB per trial: S1 has 253 training trials, S0 about 40. On top of that, each of the 5 validation sets kept 12 persistent workers. Steps slowed from about 140 s to 500 s per 50 before the kill. Each worker now keeps at most 32 trials open, least recently used first out (the windows are unchanged), and validation workers stop after each validation. S2 (all 9 training mice, first 300 s per path) and S3 (9 mice, 15 min) use the same recipe and validation mice as S0 and S1.

**S2 and S3 crashed at start on one-channel trials (c91b440).** 803496 and 804730 store `mean_image` without the channel axis, so every reader got its first row; the export-time stats of those two sessions covered only the columns valid in that row. Readers now handle both layouts, those stats are recomputed at training time, and the exporter keeps the channel axis.

**S0–S3 finished (2026-11-21).** All ran the full 3000 + 1700 steps (exit code 0, 4.3–5.2 h). Best validation scores (the 3 validation mice, which also pick the checkpoint, so not a fair comparison):

| run | pretraining fast_balanced (step) | fine-tuning real_balanced_free (step) |
|---|---|---|
| S0 | 0.557 (2500) | 0.815 (1500) |
| S1 | 0.545 (2000) | 0.817 (1700) |
| S2 | 0.530 (3000) | 0.807 (1700) |
| S3 | 0.534 (2500) | 0.810 (1700) |

More mice help on the simulations (0.53 vs 0.55–0.56). The fine-tuned differences (0.01) are within the checkpoint-to-checkpoint noise, so the held-out evaluation decides. S2 and S3 each had one loss spike between steps 2850 and 2900 (running mean 77 and 99 against about 0.015; gradients are clipped at 1, and validation at 3000 was unaffected): one simulated window, identified below.

**The loss spikes: a GIAnT fit artifact on 803121 path 2 (093f9ee).** They were not limited to S2 and S3. S1's pretraining had five (steps 1150–2700), and fine-tuning had one each in S0 (step 700) and S2 (step 750). In 20 of 803121 path 2's 46 trials, a few dozen dim pixels carry a mean fitted activity of 100–1300x the movie's (6–14 counts), so the residual there is hugely negative. The simulator turned that residual's variance into baseline fluctuations up to 685x the pixel's mean (median file: 0.6x), and real windows had targets far below the movie. S1 trains on all of 803121, while S0 and S2 have only its first trials (trial 1: 147x). Training now leaves out pixels whose mean fitted activity exceeds 2x the movie's mean, and their neighbours: no loss weight in real windows, and no activity or excess noise in the simulator's templates. The simulated slow fluctuation is also capped at the pixel's mean. On S0's six sessions this masks pixels in 26 of 505 trials (median 58 pixels), and only 803121 path 2 has more than a few. S0–S3 were trained before the fix; the held-out evaluation does not include 803121. The artifact itself is GIAnT's to explain (a cluster of about 5x6 pixels near row 80, column 877 of the cropped canvas).

**Held-out evaluation, launched 2026-11-21.** Every kept checkpoint of S0–S3 on the 8 held-out mice (validation 878030, 794237, 804730, 834788; test 776270, 810196, 829704, 876767), 192 windows per mouse and kind, with per-window error sums (`--per_window`). Fine-tuning checkpoints (8 per run) are scored on real, sim, free and fast windows, one GPU run per S run. Pretraining checkpoints (13 per run) are scored on fast windows only, since fast_balanced is what picks them, in two runs. That is about 26 GPU-hours. With seed 0, the first 48 windows of each validation mouse are the ones training used, so Experiment 2's 48-versus-192 comparison comes from the same runs.

**Real windows drop each trial's first and last 2 s (10802bb, the user's call).** GIAnT does not fit the baseline well at trial edges, so the activity and the target (movie − activity) are biased there. Real windows now weigh 0 within `real_edge_s` (2 s) of a trial's first or last frame, in training and in validation and evaluation (checkpoints from before the option get the default). Real training windows are no longer placed at trial edges: `edge_frac` now applies only to simulated windows, whose true baseline is known, so the network still learns the edges. S0–S3 were fine-tuned with real windows at the edges (`--edge_frac 0.5`), so the next runs differ from them in this as well. The four fine-tuning evaluations were stopped after 15 min and relaunched on this code, so their real-window scores leave out the edges and the 803121-type artifacts. The two pretraining evaluations kept running on the earlier code: they score simulated windows only, which neither change affects.

## v10 results (2026-11-19)

All checkpoints are scored on the same windows: v10a's evaluation re-scored v9, and the mean-image errors of the v10a and v10b evaluations agree exactly. The sessions are 780125, 803121, 847983, 878030 and 885233. The table gives the change in 8-frame MSE against v9 fine-tuned, as the mean over sessions:

| | real support | real background | fast support | fastpz support | motionpz support | motionpz background | free support |
|---|---|---|---|---|---|---|---|
| v10a (= v6r pretrained) | +0.8% | +0.3% | +15.0% | +15.9% | +11.3% | −5.7% | +0.7% |
| v10b fine-tuned | +0.3% | −0.1% | +20.0% | +21.2% | +11.7% | −13.3% | +10.9% |
| v9 pretrained | +0.5% | +0.5% | +1.6% | +0.2% | −0.2% | −9.3% | +3.5% |

- **v10a is v6r pretrained, bit for bit.** No pretraining step beat v6r's start on the motion simulations: motion_balanced was 0.470 at step 0 and 0.49–0.52 afterwards. Fine-tuning then kept its start too: real_balanced_free was 0.894 at step 0 and 0.90–0.94 afterwards.
- **v10b**:
  - Pretraining's best was step 1000, at 0.459, 2% better than the start.
  - Fine-tuned, it ties v9 fine-tuned on real windows and is better on the background of the motion simulations.
  - It is worse at support pixels of every simulation (+11 to +21%), with a more negative support bias on real windows (−0.073 against −0.022).
- The 3 h limit stopped both pretraining runs at 2,443 and 2,359 of 3,000 steps, because simulated motion makes windows 2.7× slower. v10a's validation never came near its start, so the missing steps are unlikely to matter.

**Decision (by the rule set above): v9 fine-tuned stays the default, and Experiment 1 uses v9's recipe, with no simulated motion and no motion features.**


## Source detection with the network baseline: queued (2026-11-20)

The question: once the network supplies μ and σ, is SiLo's detection the right statistic for photon counts? SiLo computes a Z-score, applies the exponential matched filter and the difference of Gaussians (DoG), then sums the squares of local maxima. Scripts are in `tools/detection/`.

**Real data** (780125, 10 trials, 112 isolated sources). Each source's signal pattern comes from regressing the movie on its own GIAnT trace, cross-fitted over alternating 200-frame blocks. Detection d′ relative to SiLo's spatial filter:

| change | d′ |
|---|---|
| Gaussian without the DoG surround | +10% |
| weighting by brightness (raw counts, Z, 1/σ²) | within 2% of Z |
| template weighted by the mean image | +6% |
| isotropic template whitened for the stationary noise correlation (0.36 to the next column, 0.22 to the next row) | 0% |
| the source's own footprint | +22% |
| its own footprint, whitened | +35% (stationary correlation), +45% (local covariance) |

In time, the 24 ms exponential keeps 95% of the whitened matched filter's d′. So the Z weighting and the temporal filter are fine. The surround costs 10%, and the larger gains need the footprint's shape, which exists only after a first fit.

**Photon noise** (simulated, baseline known):

- **Gaussian thresholds:** at 0.03–1 photons per pixel and frame, the filtered statistic exceeds 4 SD 6–42× as often as Gaussian noise. 780125's median pixel gets 0.5–0.67.
- **Sign-flip null:** the negative tail is nearly empty, so a sign-flip null does not see the excess.
- **Today's noise model:** SiLo's Vb floor (VIF) shrinks dim pixels' Z to a variance of 0.13–0.6, which hides the problem.
- **With the network's σ:** without that floor, dim pixels' null activity image is 1.2–1.5× higher.

**Combining frames:** the sum of squared local maxima comes within 0.01–0.03 AUC of an oracle sparse-event likelihood for weak, frequent events. For rare strong events it scores 0.84 against 0.93 (0.3 Hz, 2 SD at the peak).

**Queued, in order.** All three are in GIAnT's SiLo (`localizeSources_vIM`), each as an option that is off by default. Each is tested with the integration capsule on the simulations with ground truth and on the 4 sessions.

1. **No DoG surround once the network baseline is in SiLo** (see "Network baselines in SiLo's localization").
   - The surround makes up for the moving median's bias; with an unbiased blind baseline it only costs sensitivity (+10% d′ without it).
   - Check that the activity image's 5×5 median subtraction still removes what the surround did (residual motion, shared brightness changes).
2. **A null calibrated per pixel brightness.**
   - Per pixel, the null distribution of the activity image comes from a parametric bootstrap. Noise movies with the trial's baseline, gain, correlations and skew (`slap2_f0.simulate` matches these; with realistic low counts only since 2026-11-21, see the last section) go through the same filters.
   - Thresholds then mean the same false-positive rate on dim and bright pixels.
   - The bootstrap replaces the sign-flip FDR estimate (GIAnT-MATLAB#42).
   - A cheaper partial fix above 1 photon per frame: a skew correction computed from convolutions of the baseline.
   - Drop the VIF floor once this is in.
3. **A soft-threshold score.** Replace the sum of squared local maxima with the sum over frames of a sparse-event log-likelihood ratio, or of max(0, u − 2)², which does not add the energy of every noise maximum.

Later: a second detection pass with whitened footprint templates after GIAnT's first fit (+35–45% d′ for those templates), an extension of WMF's deblending.

**Architecture ideas from the DeepInterpolation study on Neuropixels:** nothing to adopt.

- **Two-branch blind spot with a centre-tap-zeroed spatial branch:** the v6 box blind spot already covers this.
- **Showing the neighbouring frames (omission0):** this keeps their spikes, but it would leak our events into the baseline.
- **Fold geometry, NAF blocks, L2 vs Charbonnier, spike weighting, training recipes:** probe-specific, or null or harmful in their tests.

Width was their largest replicated gain, and our model has 0.31 M parameters, so a wider run is the one cheap arm. Their lessons on evaluation carry over: reconstruction loss did not rank detection, and weak units lost the most.

## Same-frame neighbour correlations, and the low-count simulator (2026-11-21)

**Question.** Would each pixel's whole-trial zero-lag correlation with its neighbours, as an input, help the network follow fast artifacts and other spatially structured errors?

**Test** (`tools/neighbor_predictability.py`, `tools/neighbor_leak.py`; the five viewer trials). The target is the residual after v9 fine-tuned's baseline. It is predicted linearly from the same frame's deviation from that baseline at neighbouring pixels. Fits are cross-fitted on alternate 128-frame blocks, with the chance level (predictors shifted in time) subtracted, and frames within 2 s of a trial edge are not scored. Share of the residual's variance explained, single frames / fast part of 8-frame means:

| neighbours | coefficients per pixel | one filter for all pixels |
|---|---|---|
| ring 5–7 px away (visible to v9 in the output frame) | ≤ 0.4%; 1.0–1.3% at support pixels of 803121 | 0.0% |
| rings 2–4 px away (inside the box) | 0–1.8%; up to 6.6% in 803121 | 0–4.3% |
| the 8 nearest pixels (inside the box) | 31–36% / 38–54% | 31–35% / 34–48% |

- **What v9 can see holds almost nothing.** A per-pixel correlation map could add at most about 1%, and only in 803121, the trial with the most motion. This agrees with the motion covariates (at most 1.5%, 803121 3–6%) and with v10b's per-pixel motion features, which did not help.
- **The nearest pixels' share is shared photon noise from image formation.** In counts, the residual's correlation with the next column is 0.39–0.46 and with the next row 0.24–0.33 (the templates' rho_c and rho_r are 0.36–0.39 and 0.23–0.27). One filter for all pixels captures nearly all of it, and the GFP session (no activity) shows the same 31%.
- **Using the nearest pixels would put the events into the baseline.** A filter on the 8 nearest pixels that predicts the residual follows 70–73% of the pixel's own activity (support pixels, 780125 and 803121). Per-pixel coefficients still follow 28–33%, and all pixels within 4 px 11–14%. v9 follows 3–6%. This is what the box is for.

**Decision: no correlation input.** Neighbour covariance belongs in the detection stage's null (queued item 2): shared photon noise makes any spatial sum vary more than per-pixel σ implies.

**Simulator: realistic low counts lost half of the correlations.** Up to now, `photon_zeros` (training and validation of v7–v9 and S0–S3) worked as follows:

- it redrew only voxels with at least 2% expected zeros;
- the value above zero was lognormal;
- the draw went through a copula field with the template's own kernels, independent of the other voxels' filtered Poisson noise.

Its counts came out with about half the template's correlations and too much skew, which a simulated null would have inherited. Whole simulated trials, statistics as in `simcheck`:

| template | | ρ next row | ρ next column | ρ 1 frame | ρ 2 frames | F64 | skew | zeros | gain |
|---|---|---|---|---|---|---|---|---|---|
| 878030 | real | 0.233 | 0.386 | 0.602 | 0.245 | 2.67 | 4.80 | 0.424 | 12.36 |
| | before | 0.124 | 0.234 | 0.430 | 0.127 | 2.01 | 6.22 | 0.443 | 11.21 |
| | now | 0.230 | 0.384 | 0.601 | 0.243 | 2.47 | 4.40 | 0.443 | 12.35 |
| 780125 | real | 0.269 | 0.358 | 0.680 | 0.331 | 3.35 | 3.41 | 0.159 | 7.97 |
| | before | 0.169 | 0.239 | 0.548 | 0.222 | 2.87 | 4.93 | 0.167 | 7.84 |
| | now | 0.270 | 0.359 | 0.686 | 0.330 | 3.50 | 3.08 | 0.168 | 7.94 |

**Fix** (`simulate._photon_counts`, `Template.pz_kernels`):

- **Every voxel** of a low-count trial is drawn through one Gaussian copula field. Exact zeros come with probability exp(−mean/c), as before.
- **The value above zero is gamma, not lognormal.** In real dim pixels its skew is 3.1–3.5; a gamma with the same mean and variance gives 2.9–3.2, a lognormal 7–9. Gamma quantiles come from a table, since scipy's gammaincinv takes 3 s per training window.
- **The field's kernels are stronger than the template's**, so that the counts correlate as the template does. The mapping turns a latent correlation ρ into A(ρ) = Σ c_k² ρ^k / V, with Hermite coefficients pooled over 300 background pixels as the statistics pool them. The latent targets are A⁻¹ of the template's correlations, and the temporal kernel's F64 target is adjusted until the counts' inflation matches.
- **Cost:** a 718 × 94 × 94 window takes 1.06 s against 1.10 s before.

Checks:

- The default and fast simulations are bit-identical to before, so the held-out evaluation (kinds real, sim, free, fast; no low counts) is unaffected.
- `pz_legacy` (`--pz_legacy` in `train` and `evaluate`) reproduces the old draw bit for bit.
- New tests: a correlated, zero-inflated template is reproduced (correlations within 0.04, and closer than before), and the gamma quantiles agree with scipy within 0.5%.

**No retraining for this.** The next training run with `--photon_zeros` gets the new noise, in training and in its validation simulations. Runs compared like for like with S0–S3 (seed repeats) should pass `--pz_legacy`. The fastpz and motionpz evaluation kinds also change, and `--pz_legacy` re-scores them as before.

## Low-count simulator: skew, and artifacts where pixels leave the field (2026-11-21)

**Skew: the values above zero were too light-tailed.** With `--photon_zeros` the simulated skew was 8–10% below the template's.

- **Cause:** on real dim background pixels the values above zero (over their pixel's mean) follow a generalized gamma, θ·G^s with G ~ Gamma(k), at s = 1.25–1.5. Quantiles from 5% to 99.9% agree in all five viewer sessions and every zero-share bin. The gamma (s = 1) has a lighter tail and more values near zero: with the real moments it predicts a median pixel skew of 4.55, against 4.78 measured (878030).
- **Not the cause:**
  - the zero share: 1–2 points high in the simulation, which lowers the skew by under 1%;
  - the skew factor q: `photon_zeros` does not use it;
  - the copula: each voxel's marginal is exact. Correlated frames bias the measured skew down by about 5%, in real and simulated trials alike.

**Fix:** `SimConfig.pz_shape` = 1.4 (1 = the gamma).

- k comes from the positive part's CV through a table per s, θ from its mean. The gamma quantile table is raised to s.
- Mean, variance and zero share stay exact.
- `pz_kernels` derives the latent correlations from the new mapping.

Whole simulated trials (`simcheck` statistics):

| template | | skew | ρ next row | ρ next column | ρ 1 frame | ρ 2 frames | F64 | zeros | gain |
|---|---|---|---|---|---|---|---|---|---|
| 878030 | real | 4.80 | 0.233 | 0.386 | 0.602 | 0.245 | 2.67 | 0.424 | 12.36 |
| | gamma | 4.41 | 0.230 | 0.384 | 0.601 | 0.243 | 2.48 | 0.443 | 12.35 |
| | s = 1.4 | **4.79** | 0.231 | 0.385 | 0.601 | 0.242 | 2.49 | 0.443 | 12.20 |
| 780125 | real | 3.41 | 0.269 | 0.358 | 0.680 | 0.331 | 3.35 | 0.159 | 7.97 |
| | gamma | 3.09 | 0.270 | 0.359 | 0.686 | 0.330 | 3.50 | 0.168 | 7.94 |
| | s = 1.4 | **3.46** | 0.270 | 0.360 | 0.685 | 0.330 | 3.57 | 0.168 | 7.93 |
| 885233 | real | 3.60 | 0.243 | 0.377 | 0.667 | 0.315 | 3.06 | 0.220 | 9.79 |
| | gamma | 3.26 | 0.244 | 0.378 | 0.672 | 0.316 | 2.89 | 0.236 | 9.78 |
| | s = 1.4 | **3.62** | 0.244 | 0.379 | 0.672 | 0.317 | 2.96 | 0.236 | 9.71 |

- **Unchanged:** the zero share stays 1–2 points high and F64 within 7%. The gain reads up to 1.3% lower with the heavier tail.
- **Checks:** default, fast-brightness, motion and `pz_legacy` windows are bit-identical; `pz_shape=1` is bit-identical to the gamma draw.
- **Tests:**
  - the quantiles agree with scipy's `gengamma`;
  - a template whose positive part has s = 1.4 is reproduced within 2% of its skew, while the gamma is 10–11% low, as on the real templates.
- **No retraining for this:** as with 82860cb, the next `--photon_zeros` run gets it.

**Fit artifacts where pixels leave the imaged field.** The artifact mask missed four pixels on 804730 path 2, trial 38 (column 959, rows 282–285, next to the cluster).

- **Why:** they are missing from the movie in 25% of the frames, because the field's right edge shifts.
  - In exactly those frames GIAnT's activity reaches 1,000–1,890 counts, against a movie maximum of 80–185. GIAnT's fit is unconstrained there: SILo fills samples that have no data.
  - The activity decays over 4–5 frames once the movie returns: 228 counts on a movie of 20, so real targets of −200 with full weight.
  - Statistics skip frames without movie, so no mean, RMS or variance rule sees these pixels: mean activity 0.8–1.5 counts, residual variance 1.2–1.35× the movie's.
  - The simulator read the activity in every frame and put the spikes into its movie.
- **Missing frames are common otherwise, and harmless.** 8–29% of valid pixels miss some frames, mostly at the edges. Their activity there stays below the movie's maximum: at most 0.8×, and 99% below 0.35×, in nine trials of six sessions.
- **Activity far above the movie in frames that have movie** (the obvious per-frame rule) flags nothing beyond the existing mask in nine trials. The largest excess outside the mask is 5 SD: the decay above.

**Fix** (`stats.unfit_pixels`; no version bump, so no sidecar is recomputed):

- **With every sidecar:** after the mean rule and its 1-px dilation, the mask grows by up to 2 px through pixels that miss frames (`n_valid` below the trial's most complete pixel) and carry activity. Trials without a mean-rule artifact are unchanged.
- **With new stats:**
  - `trial_stats` also stores `movie_max` and `act_nan`, the largest activity in kept frames without movie.
  - Activity in a missing frame above twice the movie's maximum is an artifact wherever it is.
  - Older sidecars lack the two maps and are used as they are. A sidecar or overlay that has them comes first.

Pixels in the mask ("new stats" = recomputed with the two maps):

| trial | mask before | existing sidecars (growth) | new stats (growth + per-frame rule) |
|---|---|---|---|
| 804730 path 2, trial 27 | 105 | 127 | 136 |
| 804730 path 2, trial 33 | 135 | 156 | 162 |
| 804730 path 2, trial 38 | 106 | 125 | 125 |
| 804730 path 2, trial 42 | 113 | 119 | 119 |
| 780125, 803121, 847983, 878030, 885233 (one trial each) | 0 | 0 | 0 |

- **The flagged pixels:** column 959, rows 281–287: the four and the weaker pixels around them, in trials 38 and 27 alike.
- **No false positives:** on pixels that miss frames outside artifacts, activity in missing frames reaches at most 0.53× the movie's maximum in 803121, the trial with the most motion and 1,203 such pixels, against the threshold of 2.
- **The cause** (SILo's filled samples) belongs to GIAnT; a fix there is in progress (GIAnT-MATLAB branch `fit-artifact`).

**Decided (2026-11-22): existing data stays as it is.** The per-frame rule applies only to stats computed from now on. The growth rule already covers the 804730 case on the existing sidecars and on the `f0p1_sidecars` overlay, which has `n_valid` but not the two maps.

- If a session needs the per-frame rule later, run `slap2-f0 stats ROOT --out DIR`. It recomputes into the overlay folder every trial whose sidecar or cached stats lack the two maps, and a rerun resumes from that folder. Before the Copilot review fix (8b7f588), `--out` reused such sidecars and wrote nothing.
- A version bump would recompute all ~2,800 trials. A pass that computes only the two maps still reads the whole trial: half the cost of the full statistics (37 s against 74 s for an 878030 trial).

## Experiments 1 and 2: held-out evaluation results (2026-11-21)

**Runs.** All six evaluation runs finished: 7.3 h per fine-tuning run, 6.1 h per pretraining run. Each scored every kept checkpoint of S0–S3 on 192 windows per held-out mouse and kind.

**Scores** (`tools/heldout_eval.py exp1|exp2 RESULTS_DIR`):
- Each score is the 8-frame MSE as a % change against a reference, summed per mouse and pixel class. "Balanced" averages support and background, then mice.
- References:
  - real windows: the mean image (as in checkpoint selection) and SiLo;
  - simulations: the oracle, the window's true mean baseline. It is static, so "vs oracle" is the excess error over a perfect constant baseline.
- Intervals are 95%:
  - "mice": a hierarchical bootstrap, resampling mice and then windows within each mouse;
  - "windows": windows only, with these mice fixed.
- Differences between runs are paired on the same windows.
- Checkpoints:
  - "picked": the one training's validation chose (S0 step 1500; S1–S3 step 1700);
  - "late": the mean of the five checkpoints at steps 750–1700.

**Experiment 1, test mice** (776270, 810196, 829704, 876767). Balanced % change, with the mouse-level interval for S0 and S3:

| | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| training data | 3 mice, 180 s per path | 3 mice, 15 min | 9 mice, 5 min | 9 mice, 15 min |
| real vs mean image, picked | −3.2 [−4.0, −2.5] | −3.1 | −3.3 | −3.4 [−4.2, −2.6] |
| real: support / background | −5.9 / −0.4 | −5.8 / −0.4 | −6.1 / −0.5 | −6.3 / −0.5 |
| real vs SiLo, picked | −1.8 | −1.8 | −2.0 | −2.1 |
| sim vs oracle, late | +63 | +62 | +56 | +56 |
| fast vs oracle, late | +38 [+15, +59] | +37 | +32 | +31 [+15, +48] |
| activity-free vs mean image (the true mean there), late | −16 | −19 | −22 | −20 |
| leak at simulated support (mean image: 0.44) | 0.28 | 0.27 | 0.26 | 0.26 |

Paired differences of the late means, in percentage points of the reference's MSE (negative: the first run is better). Interval over mice, then over windows:

| | real vs mean image | fast vs oracle |
|---|---|---|
| S1 − S0 (amount, same mice) | −0.0 [−0.1, +0.1] | −1.0 [−5.2, +2.6]; [−2.1, +0.3] |
| S2 − S1 (diversity, same amount) | −0.2 [−0.4, +0.1]; [−0.2, −0.1] | −5.9 [−17.1, +4.8]; [−8.0, −4.2] |
| S3 − S0 | −0.2 [−0.5, +0.1]; [−0.3, −0.2] | −7.1 [−17.6, +3.0]; [−9.1, −5.7] |
| S3 − S2 (amount, 9 mice) | −0.1 [−0.1, −0.0] | −0.2 [−2.5, +2.1]; [−0.8, +0.3] |

- **More mice help; more time per mouse does not.**
  - S2 and S3 beat S0 and S1 on every readout:
    - on fast simulations from the test mice by 6–7 points, a sixth to a fifth of the network's excess error over the oracle;
    - on the other simulations by 6–8 points;
    - on real windows by 0.2 points (0.4 over all 8 held-out mice).
  - S1 (3× S0's data, from the same mice) is no better than S0, and S3 is no better than S2.
- **Significance.**
  - Over windows, the S2 and S3 gains are clear.
  - Over mice, with 4 test mice, every interval crosses 0.
  - Over all 8 held-out mice, every subset of 3 or more gives the same sign (see Resolution below).
- **Real windows hardly separate checkpoints.**
  - The network is 3.1–3.4% better than the mean image (about 6% at support, 0.5% in background) and 2% better than SiLo.
  - All 32 fine-tuning checkpoints lie within 0.6 points.
  - The real target is the 8-frame mean of movie − activity, so most of its error is noise that no baseline removes.
- **Fine-tuning costs simulation accuracy.**
  - On fast simulations, fine-tuning's step 0 (the pretraining pick) scores +24 to +33% against the oracle; the late fine-tuning checkpoints score +31 to +38%.
  - In exchange, real windows improve by 0.1–0.3 points.
- **Checkpoint noise.**
  - Within a run, fast vs oracle moves 3–5 points (SD) between fine-tuning checkpoints, about the size of the S2/S3 gain, so single checkpoints compare poorly.
  - The seed repeats were not run, so the run-to-run spread is unknown.
- **Activity-free labeled pixels.** In two test mice (776270, 876767) and validation mouse 878030, without activity the network sets the baseline too low at labeled pixels outside the support:
  - its bias there is +0.15 to +0.28 (target − net, scaled units);
  - its MSE is 1.1–2.1× the mean image's at the picked checkpoints, and 8–9× at S0 step 500;
  - it expects activity wherever pixels are labeled. In the other mice the MSE is 0.4–0.9× the mean image's.
- **804730's simulations break the network.**
  - One simulated window per kind (sim window 36; fast windows 138, plus 135 and 68) gives errors up to 6×10⁵ times the mean image's, and dominates 804730's simulated scores.
  - It happens at every checkpoint, including fine-tuning step 0, which holds the pretrained weights.
  - On the pretraining evaluations' code (1034c10), those windows held GIAnT fit artifacts like 803121's: there the mean image's error was 3.4×10⁹.
  - With the artifact mask (093f9ee), the target is normal but the network's output is not, so some network input still carries the artifact.
  - **Diagnosed and fixed (same day).** 804730 path 2 has a GIAnT fit artifact like 803121's: about 10×10 pixels near row 284, column 950 carry fitted activity up to 3×10⁵ counts, on a movie of about 13 (trial 33: above 1000 counts in 1388 of 1497 frames).
    - 093f9ee's mask flags them and cleans the trial statistics, so the simulated mean image and variance map say these pixels are dim.
    - But the simulator copied each frame's activity from the trial file unmasked, so the simulated movie carried the artifact, and the network extrapolated from it.
    - Now no activity is simulated at flagged pixels, and real windows with the residual as input (`subtract_frac`, unused in the current recipe) show the movie there instead.
    - Reproduced locally from the four trials involved: the network's error on these windows drops from 160–18,000× the mean image's to 0.5–1.3× (S0 step 1500).
    - Four pixels next to the cluster in trial 38 still spike to 1890 counts. Their mean and RMS activity are within the rule's bounds, so catching them would need a per-frame statistic.
  - 804730 is left out of the simulated scores below; these evaluations ran before the fix.
  - The test mice's windows are identical on both code versions.

**Experiment 2: selection.**
- Regret is the test score of the checkpoint that the validation mice pick, minus the best checkpoint's, in percentage points.
- It is averaged over every subset of k validation mice and, for 48 windows, over the four disjoint 48-window blocks.
- Fine-tuning is selected on real windows (balanced vs mean image; 878030, 794237, 834788, 804730).
- Pretraining is selected on fast windows (balanced vs mean image; without 804730).

Each cell gives 48 / 192 windows per mouse:

| pool | test readout | random pick | 1 mouse | 2 mice | 3 mice | 4 mice |
|---|---|---|---|---|---|---|
| fine-tuning, all 32 checkpoints | real vs mean image | 0.30 | 0.12 / 0.08 | 0.01 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 |
| | fast vs oracle | 9.1 | 7.4 / 7.0 | 4.0 / 3.9 | 3.5 / 3.3 | 3.3 / 3.3 |
| pretraining, all 52 checkpoints | fast vs mean image | 4.9 | 5.0 / 5.7 | 0.6 / 0.4 | 0.5 / 0.5 | |
| | fast vs oracle | 15.9 | 15.3 / 17.2 | 3.9 / 3.3 | 4.5 / 4.5 | |

- **Mice limit selection, windows do not.**
  - 48 and 192 windows per mouse pick about equally well.
  - One mouse is little better than a random pick, and in pretraining no better.
  - Two or three mice remove most of the regret on the readout that is scored.
  - Within one run (8 or 13 checkpoints that differ little), 3 mice leave anywhere from 0 to 100% of a random pick's regret.
- **Selecting on real windows does not select for simulation accuracy.** Within runs, the rank correlation between the validation mice's real score and the test mice's fast-vs-oracle score is −0.05 to +0.50. Between real scores it is +0.64 to +0.93.
- **balanced_free picks worse than chance.**
  - It is training's fine-tuning score, which adds activity-free labeled pixels.
  - In S0, S1 and S3 it picks worse than a random checkpoint on the test mice's real score.
  - Its labeled-pixel term follows the bias above, not real accuracy.
- **Resolution** (late means, all held-out mice; fast without 804730):

  | | real vs mean image | fast vs oracle | same sign across mouse subsets |
  |---|---|---|---|
  | S3 − S0 | −0.42 | −8.6 | every subset of 3+ mice; 95–96% of pairs; 71–88% of single mice |
  | S2 − S1 | −0.34 | −7.1 | 94–100% of subsets of 3+ mice; 86–96% of pairs; 71–75% of single mice |
  | S1 − S0 | −0.02 | −1.0 | not even across subsets of 4 mice |
  | S3 − S2 | −0.06 | −0.45 | not even across subsets of 4 mice |

**Against the plan's decision rules:**
- **Diversity: met.** S2 beats S1 by about as much as S3 beats S0. So phase 2 should first take one or two sessions from every available mouse, and 5 min per path is enough (S2 = S3).
- **Phase 2: open.** S3 beats S0 by more than the spread between checkpoints, and consistently across mice. The seed-to-seed spread, which the rule names, is unmeasured.
- **S3-long: not run.** At 3000 + 1700 steps the extra time per mouse adds nothing. S3-long would test whether more steps make it useful.
- **Selection: change it.**
  - Use 3–4 validation mice with 48 windows each.
  - Drop the labeled-pixel term from the fine-tuning score.
  - Since real windows barely separate checkpoints, weigh simulated windows from the validation mice into fine-tuning selection too.

## Export phase 2: launched (2026-11-21)

**The user's call:** go straight to phase 2, organised by mice, after fixing 804730. Experiment 1 says more mice help and more minutes per mouse do not. Phase 2 therefore takes one or two sessions from every training-pool mouse, at 15 min per session as decided on 2026-11-20 (training can cap shorter with `--max_train_s`).

**Sessions** (`sessions/export_phase2.csv`, 23 sessions, all training):
- **New mice (15 sessions).** The 9 mice that phase 1 left out: 787725, 796630, 801381, 804733, 828409, 838410, 878029, 885234, 885235.
  - Two sessions each, chosen far apart in time.
  - Only one session exists for 787725, 885234 and 885235.
  - 801381's two include its one continuous recording.
- **Second sessions (8).** One more session, on another day, for each phase-1 training mouse except 885233, which has only one session.
- **Excluded:**
  - held-out mice (validation and test): their sessions stay out of training;
  - 890920, and mice 880917, 880918 and 890921: integration scans or indicator testing.
- After phase 2: 18 training mice and 32 training sessions, counting phase 1's 9.

**Settings:**
- Same as phase 1: the data capsule at c91b440 (no export code has changed since); GIAnT pinned to 9695d27, phase 1's commit and still GIAnT's main; compensated registration; WMF seeding with `wmfPeakth` 3; trial stats stored with each trial.
- `--n_workers` 6 for 828408 and 828409 (828408 ran out of memory in phase 1 with 10), and 12 for the rest.
- Started 14:51–14:54, one Code Ocean computation per session. Each finished export becomes a data asset with mount `f0p2_<session>`.

**Before the next training:**
- Merge main into the GPU capsule. It needs 8bc88b6 (no simulated activity at GIAnT fit artifacts) and 82860cb (realistic low counts).
- Change selection as Experiment 2 suggests: 3–4 validation mice at 48 windows each, no labeled-pixel term, simulated windows in fine-tuning selection.

**Results (2026-11-22): 20 of 23 sessions exported, 3 rerunning.**
- **Out of memory in SILo's localization (4 sessions):**
  - 796630 and 801381 (2025-09-26) passed with `--n_workers 6`, and 828409_2025-11-20 with 2 (9.6 h).
  - Cause: SILo sized its pool from each path's first trial, which is often partial. 828409_2025-11-20's was a few frames long, so SILo estimated 1.2 GB per trial against at least 20 GB used. GIAnT-MATLAB PR #51 sizes the pool for the largest trial.
- **Out of memory in the exporter (3 sessions):** 787725_2025-02-26, 776266_2025-01-09 and 803121_2025-11-05.
  - They were killed after GIAnT had finished both paths: memory stood at 140–170 GB when the exporter opened its 8-worker pool, and peaked at 180–200 GB of 246.
  - Rerunning in full with `--export_workers 4`, since a run cannot resume from GIAnT's outputs. 787725 is a new mouse with no other session.
- **QC against phase 1** (per-path medians of the exporter's calibration):

  | | phase 1 (34 paths, 1,428 trials) | phase 2 (40 paths, 1,598 trials) |
  |---|---|---|
  | activity gain | 0.62 (0.50–0.86) | 0.66 (0.50–0.86) |
  | peak slope | 1.09 (0.99–1.34) | 1.03 (0.69–1.35) |
  | blur | 0 (0–0.75 px) | 0 (0–1.5 px) |
  | event residual | 0.13 (−0.01 to 0.32) | 0.12 (0 to 0.32) |

- **Trials where GIAnT's activity does not match the movie:** gain at its lower bound (0.50), blur at its upper bound (1.5 px), peak correlation 0.03–0.18, and event residuals down to −1.5 (activity above the events).
  - 780125_2025-03-18 path 1, 18 of 30 trials (2–18, 20, 21; mostly the first ones), and phase 1's 867430_2026-06-16 path 2, 12 trials (30 onwards). S0–S3 trained on the latter.
  - Their training targets (movie minus activity) are too low wherever activity is overestimated.
  - Trials where only the gain hits its bound and the correlation stays normal (about 0.37: 803121_2025-10-29, 804733_2025-07-25 and 885234 path 2) look fine.
  - `--exclude` works per session only, so it cannot leave out these trials alone.
- **Short paths:** 801381_2025-09-26 path 2 (12 of 32 trials), 838410_2026-03-18 path 2 (14 of 49), 801381_2025-05-29 path 2 (30 of 49). The exporter recorded no errors; it skips trials for which GIAnT produced no frames.
- **One-channel sessions:** 804733's exports keep the channel axis (mean image 1 × H × W, or 2 × H × W in its two-channel session). Their sidecars are therefore valid and are not recomputed.

**Trial quality filter (the user's decisions, 2026-11-22): training and evaluation leave out failed calibrations, and paths left with fewer than 4 trials** (`io.filter_trials`, on by default in `train` and `evaluate`).
- **Calibration:** a trial is left out if the exporter's peak correlation (attribute `peakR`) is below 0.3 (`--min_peak_r`).
  - Over all 3,026 trials of both phases, peak correlation is 0.30 or more for 2,992 (median 0.43). 30 trials lie at 0.03–0.18 and 4 at 0.20–0.27.
  - The 34 are 780125_2025-03-18 path 1 (20 trials, including two partial failures with blur 1.25 and gain 0.54), 867430_2026-06-16 path 2 (12), and two others between 0.20 and 0.30.
- **Paths:** a path with fewer than 4 usable trials is left out whole (`--min_path_trials` 4). This applies only to calibrated exports (trials with `peakR`), so older or synthetic files are never left out.
  - The user first asked to drop paths that lost many trials, then changed it: 780125 is an important test and development asset and stays.
  - No path of the current exports falls below 4. The shortest are 801381_2025-09-26 path 2 (12 trials), 780125_2025-03-18 path 1 (12 after the calibration cut) and 838410_2026-03-18 path 2 (14).
- `--min_peak_r 0 --min_path_trials 0` restores S0–S3's data.
- Runs print what was left out under "trial quality".

## GPU machine: re-evaluation (2026-11-21)

**Question (the user's).** Is g5.4xlarge (one A10G, 24 GB) still the right machine for the GPU capsule, or would a larger GPU be faster at a reasonable cost? Consider also the deeper network planned once phase 2's data are in.

**Where today's time goes** (S3's logs and the held-out evaluation runs, g5.4xlarge):
- **An S3-type run takes about 5 h:**
  - loading trial stats: 43 min;
  - 4,700 steps at 2.7 s: 3.5 h;
  - 14 validations: 0.7 h.
- **Stats loading is not the GPU.** S0 and S1 loaded their stats in 7–11 s. Runs that include a one-channel phase-1 session (803496 in S2 and S3, 804730 in the evaluations) take 16–43 min. Those sessions' export-time sidecars are stale, so every run recomputes them from the movies.
- **The GPU is the limit, not the data loaders.** Evaluation costs 0.46–0.49 s per window whether the window is real (read from disk) or simulated (about 1 s of CPU each, spread over 12 workers). An evaluation run (49,000 windows) takes 7.2 h.
- **Most of the compute is context.** The network processes 718 frames × 94² px per window, and the loss covers 192 × 48², so 7% of the processed voxels are scored.

**Profile** (`python -m slap2_f0.gpu_bench CKPT key=value ...`, or the grid `slap2-f0 bench CKPT`: the trained model's own configuration on random inputs of its window; local RTX 4060 Ti, 8 GB):
- **cuDNN picks a slow algorithm.** Half of a training step's GPU time (49%) was one cuDNN kernel, a direct algorithm for a convolution's backward pass, chosen by cuDNN's heuristics.
  - `torch.backends.cudnn.benchmark = True` lets cuDNN time the algorithms once per shape, and the window shape is fixed in training.
  - With it, a step is 2.1× faster: 1.88 → 0.88 s at 62² px, and 2.36 → 1.14 s at 70² px.
  - The forward pass alone, as in evaluation, gains only about 10%.
  - Neither the code nor S0–S3 had it on.
- **After that, memory traffic dominates.** Layout conversions to and from channels-last are 23% of GPU time, and element-wise kernels (residual adds in float32, bf16 casts, GELU) most of the rest. The convolutions themselves are a small share.
  - Bandwidth speeds this up in hardware, and `torch.compile` in software. Compile needs Triton, which is untested here because it isn't available on Windows.
  - Channels-last as is made it slower (1.04 s against 0.88 s), because the model's padding and slicing break the layout.
- **The model is memory-bound.** Extrapolated to the full window, the 4060 Ti would take 4.3 s per step against the A10G's 2.7 s at the same settings. Its memory bandwidth is half the A10G's (288 against 600 GB/s), and its compute is 0.7×.
- **Larger loss regions are cheaper per scored frame.** 448 output frames instead of 192 cost 20% less per loss frame.
- **Depth and width** (62² px, autotuning on):

  | variant | parameters | time per step | GPU memory | at the full window (scaled) |
  |---|---|---|---|---|
  | current | 0.31 M | 0.89 s | 4.5 GB | ~10 GB (A10G measured: 9.8) |
  | 2× depth | 0.59 M | 1.34 s (1.5×) | 6.3 GB | ~15 GB |
  | 1.5× width | 0.69 M | 1.53 s (1.7×) | 6.2 GB | ~14 GB |
  | 2× width | 1.23 M | out of local memory | 8.2 GB | ~19 GB |

  These sizes are with activation checkpointing. Without it, or with a larger loss region, they exceed 24 GB.

**Machines in use at AIND's Code Ocean** (capsules' `.codeocean/resources.json` on GitHub) and on-demand AWS prices, approximate:

| machine | GPU | GPU memory, bandwidth | vCPU / RAM | $/h |
|---|---|---|---|---|
| g5.4xlarge (now) | A10G | 24 GB, 600 GB/s | 16 / 64 GB | 1.62 |
| gr6.4xlarge | L4 | 24 GB, 300 GB/s | 16 / 128 GB | 1.54 |
| g6e.2xlarge | L40S | 48 GB, 864 GB/s | 8 / 64 GB | 2.24 |
| g6e.4xlarge | L40S | 48 GB, 864 GB/s | 16 / 128 GB | 3.00 |
| g5.12xlarge | 4× A10G | 4× 24 GB | 48 / 192 GB | 5.67 |

- **No A100 or H100 capsules were found.** AWS sells the A100 only in 8-GPU machines, which is far too large for a 1 M-parameter model at batch 1.
- **The L40S** has 1.44× the A10G's bandwidth and about 3× its compute, so a memory-bound model gains about 1.4–2×.
- **The L4** has half the bandwidth, so it would be slower here.

**Estimates (not yet measured on these machines):**
- Software fixes: `--stats_from` for the stale one-channel sessions, cuDNN autotuning, and on 48 GB no checkpointing (25% less work per step).
- Ranges carry the uncertainty of the A10G's autotuning gain (assumed 1.5–2.1×) and the L40S's speed relative to the A10G (assumed 1.4–2×).

| run | g5.4xlarge now | g5.4xlarge with the fixes | g6e.2xlarge with the fixes | g6e.4xlarge with the fixes |
|---|---|---|---|---|
| S3-type training, current model | 5.0 h, $8.1 | 2.4–3.0 h, $3.9–4.9 | 1.1–1.8 h, $2.4–4.0 | 1.1–1.8 h, $3.3–5.4 |
| the same, 2× depth | ~7 h, ~$11 | 3.6–4.6 h, $5.8–7.4 | 1.5–2.7 h, $3.4–6.0 | 1.5–2.7 h, $4.5–8.0 |
| held-out evaluation (8 checkpoints) | 7.2 h, $11.7 | ~6.0 h, ~$9.7 | 3.0–4.3 h, $6.7–9.6 | 3.0–4.3 h, $9.0–12.9 |

- **On g6e.2xlarge, 8 vCPUs** run about 7 data workers, about 7 simulated windows/s. A training step there consumes about 2.2–3.3 windows/s and evaluation about 3–4. Enough, but with the least margin of the options.
- **g5.12xlarge** could run an evaluation's checkpoints on 4 GPUs at once: about 1.6 h for about $9, with a small change to split checkpoints across GPUs. Training on it would need data-parallel code.

**Recommendation:**
1. **Software first.** cuDNN autotuning in training and evaluation, and `--stats_from` for 803496 and 804730. On today's machine this roughly halves a training run's time and cost. Phase 2's sidecars come from the fixed exporter, so they need no recomputing.
2. **Then move training to g6e.2xlarge (L40S, 48 GB).** At 1.4× the hourly price it should cost the same or less per run, and finish in about half the time of the fixed A10G.
   - The 48 GB matters most for the deeper network: no checkpointing, and larger loss regions.
   - g6e.4xlarge only if 8 vCPUs turn out to starve the GPU.
   - Not the L4. g5.12xlarge only if evaluation wall time matters.
3. **Confirm before switching.** One short benchmark job each on g5.4xlarge and g6e.2xlarge: the real model; autotuning; `torch.compile`; checkpointing on and off; batch 1–4; deeper and wider; larger loss regions. That is about 30 min and $2 in all.

**Benchmark results and decision (same day; these supersede the estimates above):**
- `run --mode bench` measured S0's fine-tuned model at its window (718 frames × 94² px, loss on 192 × 48²), and data loaders on 878030.
- One run on g5.4xlarge (A10G) and two on g6e.4xlarge (L40S); the two L40S runs agree within 0.5%.
- g6e.2xlarge is refused at AIND ("forbidden instance type"); g6e.4xlarge (16 vCPU, 128 GB, $3.00/h) is allowed.

| per window | A10G | L40S |
|---|---|---|
| training step, today's settings | 1.36 s | 1.37 s |
| + cuDNN autotuning | 1.32 s | 0.83 s |
| + `torch.compile` | 0.82 s | 0.49 s |
| + no activation checkpointing | 0.73 s (18.7 GB) | 0.44 s (19.2 GB) |
| forward pass (evaluation) | 0.37 s | 0.23 s |
| forward pass, compiled | 0.20 s | 0.12 s |
| 2× depth, compiled | 1.25 s | 0.73 s |
| 1.5× width / 2× width (autotuned) | 2.30 / 2.92 s | 1.47 / 1.83 s |
| 2× depth and 1.5× width | out of memory | 2.20 s (22.5 GB) |
| loss region 448 frames × 96² px | out of memory | 3.47 s (42.8 GB) |
| data loaders, 16 vCPU, 12 workers (simulated / real) | real 5.9 with 7 workers | 5.1 / 6.8 windows/s |

- **Autotuning:** it matters on the L40S (1.6×) and hardly on the A10G (3%). Without it, the L40S is no faster than the A10G. The 4060 Ti behaved like the L40S.
- **Compile:** `torch.compile` speeds up training steps 1.7× and forward passes 1.8×, on both machines.
- **Machine:** the L40S is 1.7× the A10G at equal settings, at 1.85× the price.
- **Batch size** does not help: time per window stays the same.
- **Memory:** 24 GB does not fit the uncompiled model without checkpointing, larger loss regions, or 2× depth with 1.5× width. 48 GB fits all of these.
- **Loss regions:** 448 frames × 96² px scores 9.3× the voxels of today's region in 4.2× the time, so it is 2.2× cheaper per scored voxel.
- **Data loaders:** they keep up with training. A compiled L40S training step without checkpointing uses 2.3 windows/s of the 5.1 available.
  - Compiled evaluation would want 8 windows/s on the L40S, so evaluation becomes data-bound at about 5.4 windows/s.
  - The windows are the same for every checkpoint. Generating each once and scoring all checkpoints on it would remove that limit.

**Estimates from the measurements:**
- A training run here is S3-type: 4,700 steps of 2 windows, 14 validations of 240 windows, 5 min start, no stats recomputing. An evaluation is 49,000 windows.
- Evaluation on the L40S is limited by the data loaders. Training without checkpointing is not measured for 2× depth compiled.

| | training run | evaluation |
|---|---|---|
| A10G, today's settings | 4.0 h, $6.50 | 5.1 h, $8.40 |
| A10G, compiled, no checkpointing | 2.2 h, $3.50 | 2.9 h, $4.60 |
| L40S, compiled, no checkpointing | 1.3 h, $4.00 | ~2.6 h, ~$7.80 (data-bound) |
| A10G, 2× depth, compiled | 3.6 h, $5.90 | 4.2 h, $6.90 |
| L40S, 2× depth, compiled | 2.2 h, $6.50 | ~2.7 h, ~$8.10 |

**Decision: the GPU capsule stays on g6e.4xlarge (L40S).**
- **Why:** a training run takes about 40% less time for 10–15% more per run. The deeper or wider network and larger loss regions need the 48 GB.
- **Code (4d3f593; GPU capsule d33555d):** training and evaluation now autotune and compile by default. `fast_model` falls back to eager without Triton or on a compile error; `--no_compile` and `--no_cudnn_benchmark` turn them off.
- **For the next training:**
  - add `--no_checkpoint` (19 GB of 48);
  - add `--sidecars /root/capsule/data/f0p1_sidecars` for 803496 and 804730, whose stale one-channel stats otherwise cost 16–43 min per run (see "Stats overlay" below; `--stats_from` was not suitable).
- **Evaluation:** until windows are shared across checkpoints, the A10G is the better value. It is about as fast, because the L40S waits for data, and costs 40% less. Switching back is a one-line change to `.codeocean/resources.json`.
- **Not yet confirmed:** compile has been measured in the benchmark, not in a full training run.

## Selection, evaluation and stats for the next training (2026-11-21)

**Checkpoint selection per validation mouse** (Experiment 2: the number of mice, not windows, limits selection):
- **Per-mouse windows.** Every validation set (`real`, `sim`, `free`, `fast`, `motion`) now draws `--val_windows` windows from each validation session separately. These are the first windows held-out evaluation draws for that session with the same seed.
- **`balanced`** is the mean over mice of each mouse's ratio, the mean of support and background (8-frame MSE against the mean image). The set columns of `metrics.csv` keep their names, as means over mice, and each mouse's score is logged as `<set>@<mouse>_balanced`.
- **Several sets with weights:** `--select_on` takes them, for example `real,fast` or `real:1,fast:0.5`. The score is the weighted mean of the sets' scores, and the column is named `real+fast_balanced`.
- **`--val_pooled`** restores pooled windows, as S0–S3 used.
- **Recommended for the next runs:** `--val_sessions 878030,794237,834788,804730 --val_windows 48 --select balanced`. Pretraining selects on `fast` as before, and fine-tuning on `real,fast`.
  - The weights wait for the fine-tuning analysis (session finetune-tradeoff). The fast score moves about 10× more between checkpoints than the real one, so equal weights let the simulations decide.
  - Do not use `balanced_free`: it picked worse than chance.
- **Cost:** 4 mice × 48 windows × 4 sets = 768 windows per validation.

**Evaluation scores all checkpoints on shared windows** (`evaluate_checkpoints`):
- Checkpoints with the same window configuration are scored on the same windows. Each session's windows are drawn once and scored by every checkpoint in turn, in chunks of `--share_chunk` windows (default 64, about 100 MB each).
- Checkpoints of one architecture share one compiled model and load their weights in turn. Compiling each checkpoint separately would recompile, and after torch's limit of 8 recompilations it falls back to uncompiled code.
- Compiled, the data loaders were the limit (about 5 simulated windows/s on 16 vCPUs). With 8 checkpoints the windows are now made once instead of 8 times.
- Same numbers as before: on the four local 804730 trials with two checkpoints, all 316 metrics and 264 per-window values are identical.

**Stats overlay for stale sidecars:**
- `slap2-f0 stats ROOT --out DIR`, or `run --mode stats`, writes trial-stats sidecars into an overlay folder, `DIR/<session>/<path>/<trial>_stats_v5.npz`, instead of next to the (read-only) trials.
- `train` and `evaluate --sidecars DIR[,DIR]` look there after each trial's own sidecar. Like sidecars, the overlay must match the trial's file size.
- This replaces recomputing the one-channel phase-1 sessions (803496, 804730) in every run, which took 16–43 min.
- `--stats_from` was not suitable: its cache key includes the mount path and modification time, and it copies a whole earlier run's cache (about 25 GB for S3).
- **The overlay for 803496 and 804730 exists:** data asset mount `f0p1_sidecars`, made by `run --mode stats` on the GPU capsule (189 trials in 66 min, 4.0 GB).
  - Next runs attach it and pass `--sidecars /root/capsule/data/f0p1_sidecars`. The session folders sit at the asset's root.
  - Checked: its sidecars all have `mean_image_axes`. Two 804730 trials computed locally with today's code give identical stats, and trials without a sidecar of their own load from the overlay without recomputing.
  - The export-time sidecars of these sessions differ, as expected, because they read only the mean image's first row: they lack `mean_image_axes`, and gain, noise correlations, activity trace and valid-pixel counts differ (movie and residual moments are identical).

## Low photon counts: the baseline sits low (2026-11-21)

**The finding (held-out evaluation, Experiment 1).** In activity-free simulations (`free`) of 776270, 876767 and 878030, the S0–S3 networks set the baseline too low at labeled pixels outside the footprint support:
- bias +0.15 to +0.28 (target − net, scaled units);
- 8-frame MSE 1.1–2.1× the mean image's (the mean image is the true mean there), and 8–9× at S0 fine-tuning step 500.

In the other five held-out mice the MSE is 0.4–0.9× the mean image's and the bias about 0.

**Data used:**
- one trial per path of each of the 17 phase-1 sessions (`tools/lowcount_charac.py`; stats through `load_stats`, so the stale one-channel sidecars of 803496 and 804730 are recomputed);
- the evaluation CSVs;
- network probes run locally on the RTX 4060 Ti (`tools/lowcount_probe.py`).

**It is not about labeled pixels: it is low photon counts.** Picked checkpoints of S0–S3 (mean of the four), activity-free windows, bias as a share of each class's baseline:

| mouse | photons per frame: support / background / labeled outside | zero frames at support | trial skew | bias: support / background / labeled outside | the same, in SD of an 8-frame mean | real background: net's bias minus the mean image's |
|---|---|---|---|---|---|---|
| **776270** | 0.53 / 0.28 / 1.76 | 10% | 3.6–3.7 | +10.0 / +3.6 / +5.6% | +0.13 / +0.03 / +0.13 | +0.028 |
| **876767** | 0.47 / 0.17 / 0.92 | 12% | 4.3–5.0 | +9.5 / +4.0 / +7.4% | +0.12 / +0.03 / +0.13 | +0.032 |
| **878030** | 0.24 / 0.17 / 0.87 | 36% | 4.2–5.0 | +10.7 / +5.7 / +4.6% | +0.09 / +0.04 / +0.07 | +0.062 |
| 794237 | 1.92 / 0.99 / 4.90 | 0% | 2.0–2.1 | +1.6 / +1.3 / +0.7% | +0.04 / +0.02 / +0.03 | +0.012 |
| 804730 | 1.74 / 1.34 / 3.23 | 0% | 1.6–2.0 | +1.3 / +0.9 / +1.8% | +0.04 / +0.02 / +0.06 | +0.017 |
| 810196 | 1.64 / 0.98 / 2.51 | 1% | 1.9–2.6 | +2.4 / +1.5 / +3.2% | +0.05 / +0.03 / +0.09 | +0.014 |
| 829704 | 1.07 / 0.52 / 2.18 | 1% | 2.6–3.0 | +1.8 / +0.4 / +0.9% | +0.03 / +0.01 / +0.03 | +0.007 |
| 834788 | 1.01 / 0.56 / 4.80 | 3% | 2.7–2.8 | +2.7 / +1.4 / +0.7% | +0.05 / +0.02 / +0.03 | +0.013 |

- **The three mice are the dimmest.** They have under 0.6 photons per frame at footprint support, 10–36% zero frames there, and trial skew 3.6–5.0.
  - Every other held-out mouse has at least 1.0 photons per frame at support, and skew of at most 3.0.
  - S0's training mice: 885233 0.41–0.49, 780125 0.65–0.72, 803121 0.89–1.04. S2/S3 add 867430 (0.21–0.55) and five mice at 0.5–2.4.
- **Every class is low, by 4–11% of its baseline.**
  - Labeled pixels outside the support are the brightest structures, at 2–5× the trial scale (support 1.0–1.4×).
  - So the same relative bias is largest there in scaled units, and the mean image is most accurate there. That is why the labeled class stood out.
- **Real windows show it too.** On real background of 878030, which has no activity, the network sits 0.06 (about 7%) below the mean image, against 0.01–0.02 in the bright mice.
- **In noise units it is small:**
  - at support and labeled pixels, 0.07–0.13 SD of an 8-frame mean (bright mice 0.03–0.09);
  - at background, at most 0.04 SD everywhere.
  - So it matters more for ΔF/F (a baseline 5–10% low) than for SiLo's Z-scores, where it is roughly uniform in space.

**Probes on activity-free simulations** (`tools/lowcount_probe.py`; windows of 64 frames × 24² px, so outputs are exact; the same windows for every checkpoint and variant). Bias as a share of the baseline, support / background / labeled outside:

| 878030, 48 windows | S0 fine-tuned (step 1500) | S0 pretrained (its pick = fine-tuning step 0) | v6r pretrained (S0's start) |
|---|---|---|---|
| as evaluated (`free`) | +8.1 / +5.9 / +3.9% | +10.6 / +10.1 / +5.4% | +4.6 / +3.1 / +1.8% |
| realistic low counts (legacy draw, as S0–S3 trained) | +5.6 / +2.5 / +3.1% | +7.4 / +5.4 / +4.2% | +2.7 / +0.8 / +1.1% |
| 4× the photons (gain / 4, same skew per photon) | +1.9 / +0.8 / +0.4% | +2.8 / +2.1 / +1.0% | +0.5 / +0.2 / −0.5% |
| flat field at the trial scale, no excess, variance factor 1 (32 windows) | +5.6 / +4.8 / +6.3% | | +3.4 / +2.7 / +4.0% |

876767 is the same (`free`): +5.7 / +5.1 / +6.3% for S0 fine-tuned and +3.2 / +3.2 / +3.6% for v6r.

- **Photons set it.** At 4× the photons it falls to 0.4–2%.
- **No structure is needed.** A flat field, with no dendrites, no excess and no variance-factor input, is as low as the real geometry. The network does not key on the labeled morphology.
- **The flat field's bias follows the photon count, in a training mouse too** (S0 fine-tuned, 32 windows; photons per frame at the trial scale in parentheses):
  - 878030 (0.2): +4.8 to +6.3%;
  - 776270 (0.45): +2.0 to +2.6%;
  - 885233, a training mouse (0.4): +1.5 to +2.5%;
  - 829704 (1.0): −0.6 to +0.2%.
- **The evaluation kinds overstate it a little.** `free`, `sim` and `fast` have no realistic low counts, which S0–S3 trained with. With them the bias is 10–60% smaller, but it stays.
- **It came with S0's pretraining.**
  - v6r (no low counts, the old export) has about half of it.
  - On the fast evaluation windows, the labeled-pixel bias in 776270 went from 0.01 (step 0) to 0.22 after 250 pretraining steps.
  - On the training mice's own simulated windows (`tr_lab_bias_net` in training's metrics) it rose from −0.005 at step 0 to +0.09…+0.25 (S0). In S3 it swings from −0.09 to +0.42 between checkpoints.

**Why.**
- **The network subtracts activity it cannot see.**
  - Every simulated training trial has activity at its footprints.
  - Below about 1 photon per frame an event is no larger than a single photon event, so neither the visible frames nor the variance maps tell whether a pixel is active.
  - The network then subtracts the activity it expects for pixels of that brightness, even on a flat field. Above about 1 photon per frame it can see that there is none.
- **Training barely penalizes it.**
  - The mean loss weighs each voxel by 1 / photon-noise variance, so a relative bias δ costs δ² × photons per frame.
  - At 0.2 photons a 5% bias costs as much as a 1.6% bias at 2 photons.
  - That also explains why the bias wanders from checkpoint to checkpoint.
- **878030 lies below the training range.** Its background and support (0.17–0.24 photons) are dimmer than anything S0's training mice provide, and its bias is the largest.

**Code (branch `labeled-bias`):**
- **Photon-count augmentation** (`SimConfig.photon_gain`, `train --photon_gain lo,hi`; off by default, drawn last so earlier configurations draw exactly as before):
  - Per virtual trial, the template's gain (counts per photon) is multiplied by a factor drawn log-uniform in the range. Baseline and activity in counts stay; photon noise variance, the photon event size and the counts per zero-count event grow by k.
  - Normalized by the trial scale, this is the noise of a dimmer session, from any template.
  - The window reports its gain, which goes to the network's gain input and the loss weights.
  - With `photon_zeros` the copula's kernels stay the template's, so correlations come out slightly weaker where k adds zeros.
- **"dim" evaluation kinds** (`evaluate --kinds freedim,fastdim,...`): the base kind's windows with a third of the photons. The bright held-out mice (1–2.5 photons per frame at support) then get a low-count readout at 0.3–0.8, the range of the three dim mice.
- **`mean_target` per class in `evaluate`'s output**, so the bias can be read as a share of the baseline. `tools/heldout_eval.py bias RESULTS_DIR` prints the picked checkpoints' bias per mouse, class and simulated kind (eval.csv only).
- **Tests:** a new `test_photon_gain` covers:
  - off by default, with unchanged draws;
  - noise variance k × gain × mean;
  - more exact zeros with low counts;
  - the dataset passing the gain on;
  - the dim kinds using the base kind's windows.
- The full suite passes: 77 tests before the rebase, 79 on top of 5ab43a9 (main's selection and evaluation changes).

**Local fine-tuning test.**
- **Setup:** S0's pretraining pick, fine-tuned 200 steps with S0's fine-tuning recipe; loss region 64 × 24²; one trial per path of S0's three training mice.
- **Control arm:** activity-free windows of 878030 / 876767 / 829704, 32 each:

  | | before (S0's pretraining pick) | after 200 steps (control) |
  |---|---|---|
  | `free`, support / background / labeled outside | +10.4 / +9.8 / +4.4%; +7.0 / +8.8 / +7.9%; +1.7 / +1.4 / +1.6% | +8.4 / +5.9 / +3.2%; +6.5 / +4.9 / +7.4%; +1.4 / −0.7 / +1.4% |
  | real background: net below the mean image | 12.3%; 10.0%; 2.7% | 8.2%; 5.8%; 0.4% |

  - Fine-tuning on real windows lowers it, mostly at background, but leaves the dim mice 3–8% low.
- **The two candidate arms did not finish.** Silent trials (`--silent_trial 0.3`) and photon gain (`--photon_gain 1,4`) stalled on the shared machine: all 32 cores and the 8 GB GPU were taken by other sessions' jobs, and steps took 8–20 s. So they are untested.

**Recommendation for the next batch: one arm against the planned control.**
- **The arm, "lowcount":** the planned recipe plus `--photon_gain 1,4 --silent_trial 0.2`, in both pretraining and fine-tuning. These affect only the simulated windows.
  - `--silent_trial 0.2` makes a fifth of the simulated trials activity-free. "Every pixel of this brightness carries activity" then stops being true in training, and where the network cannot see events it should subtract less.
    - This is the part aimed at the in-distribution bias (885233: +2% at 0.4 photons).
    - Expected cost: less subtraction at active pixels in dim sessions. Simulated support bias would go more negative, and the real support score shows the net effect.
    - v5a's silent regions "added nothing" on bright sessions and the GFP control, where events are visible. The test here is the dim mice.
  - `--photon_gain 1,4` takes every template down to as little as a quarter of its photons. The training mice (0.4–2 photons per frame at support) then cover 0.1–2, which includes 878030's 0.2.
- **Evaluate with** `--kinds real,sim,free,fast,freedim,fastdim` and read `tools/heldout_eval.py bias`. Decide in this order:
  1. Real balanced vs the mean image: no loss beyond about 0.1 point.
  2. `free` and `freedim` bias by class: in the dim mice, and in the dim versions of the bright mice.
  3. Real background bias of 878030 and 876767 against the mean image.
- **If only one change fits:** `--silent_trial 0.2`. The bias exists inside the training range, and phase 2's new 2026 RCaMP3 mice (878029, 885234, 885235) probably add dim templates on their own.
- **Selection:** the labeled-pixel term is already gone (`balanced_free` is not to be used; see the previous section). It followed this bias, not real accuracy.

## Edge bias: where it comes from, and an arm to train (2026-11-22)

**Question.** Every checkpoint since v6 puts the baseline 2–5% too high at footprint-support pixels in a trial's last 16 frames (sections "Bias at trial edges", v8, v9). Twice the edge windows (v9) did not help. Why, and what should the next training batch change?

**Method: truncated windows** (`tools/edge_windows.py`, new).
- **Draws.** Each draw is a new virtual trial from the training simulator: fastpz with `--pz_legacy`, as v9 and S0–S3 trained, on the four viewer trials as templates. One window per draw, with its 128-frame loss region in the trial's middle.
- **Three predictions per window:**
  - whole;
  - "end": every frame after the loss region removed (NaN, as past a trial's last frame);
  - "start": every frame before it removed.
- **Only the truncation differs:** same noise, activity and whole-trial maps. So end − whole and start − whole measure the edge effect alone, paired, with far less noise than a bias against the truth. This is `edge_pieces.py`'s cut, on many short simulated windows instead of a few whole trials.
- **The forward pass is split in two:** `model.parts` (everything the fusion layers get) and `model.head` (fusion layers and heads). The outputs are unchanged. This gives two read-outs:
  - the model's own split of the baseline: mu = base × exp(correction), with base = mean image + mix × (window mean − mean image);
  - **path swaps:** the truncated window's baseline recomputed with one input of the fusion layers taken from the whole window. The inputs swapped are the trunk branch on the missing side (the future branch at an end, the past branch at a start), the window statistics, or the same-frame context (half-planes, pooled features). Edge − swapped is that path's share.

**Result 1: the network itself puts the baseline too high at a trial's end, on the data it was trained on.** Edge − whole, in % of the true baseline, by frames from the edge (v9 fine-tuned: 16 draws; S0 fine-tuned step 1500: 8 draws; ± SE over draws):

| | 0–4 | 4–8 | 8–16 | 16–32 | 32–64 | 64–128 |
|---|---|---|---|---|---|---|
| v9 ft, support, end | +3.7 ± 1.0 | +3.6 ± 1.1 | +3.2 ± 1.0 | +2.1 ± 0.7 | +1.5 ± 0.4 | +0.6 ± 0.1 |
| v9 ft, support, start | +0.9 ± 0.4 | +1.0 ± 0.4 | +0.3 ± 0.3 | −0.3 ± 0.2 | +0.2 ± 0.1 | +0.2 ± 0.1 |
| v9 ft, background, end | +2.2 ± 0.5 | +2.2 ± 0.5 | +2.3 ± 0.4 | +1.8 ± 0.3 | +1.3 ± 0.2 | +0.7 ± 0.1 |
| v9 ft, background, start | +1.3 ± 0.3 | +1.5 ± 0.3 | +1.0 ± 0.2 | +0.5 ± 0.1 | +0.6 ± 0.1 | +0.4 ± 0.1 |
| S0 ft, support, end | +3.4 ± 1.3 | +3.2 ± 1.7 | +2.7 ± 1.4 | +1.2 ± 0.7 | +0.8 ± 0.4 | +0.5 ± 0.2 |
| S0 ft, support, start | +1.5 ± 0.5 | +2.2 ± 0.7 | +0.8 ± 0.3 | −0.2 ± 0.1 | +0.3 ± 0.1 | +0.3 ± 0.1 |
| S0 ft, background, end | +1.5 ± 0.7 | +1.6 ± 0.7 | +1.8 ± 0.5 | +1.0 ± 0.3 | +0.7 ± 0.2 | +0.5 ± 0.1 |

- **The same profile and size as `edge_pieces.py` found** on whole trials, simulated and real. For v9 fine-tuned, support, end of a piece: +4.3, +3.5, +2.1, +1.4, +0.5.
- **These windows come from the training simulator, so this is not a simulator–data mismatch.**
- **S0 (phase-1 data, trained before the real-edge cut of 10802bb) is no different from v9.**

**Result 2: it enters through the correction head, driven by the empty future branch.** v9 fine-tuned, end:

| | 0–4 | 4–8 | 8–16 | 16–32 | 32–64 | 64–128 |
|---|---|---|---|---|---|---|
| change of the log correction (× 100), support | +4.0 | +3.8 | +3.6 | +2.6 | +1.8 | +0.8 |
| change of the base (window-mean path), support | −0.2 | −0.2 | −0.2 | −0.1 | −0.1 | 0.0 |
| share of the future trunk branch, support | +2.9 ± 1.0 | +2.8 | +2.5 | +1.6 | +1.0 | +0.2 |
| share of the window statistics, support | +0.5 ± 0.3 | +0.5 | +0.6 | +0.5 | +0.5 | +0.4 |
| share of the same-frame context, support | +0.4 ± 0.1 | +0.4 | +0.2 | 0.0 | 0.0 | 0.0 |
| share of the future trunk branch, background | +1.3 ± 0.5 | +1.2 | +1.3 | +1.1 | +0.6 | +0.1 |
| share of the window statistics, background | +0.7 ± 0.1 | +0.7 | +0.8 | +0.8 | +0.7 | +0.5 |

- **The base path is not the problem.** The mix (0.25 at support, 0.19 in background) does not change at edges, and the one-sided window mean is not biased. So the candidates aimed at it would not help: padding, the window statistics' normalisation, and replicate or reflective filling of the missing frames. Reflective filling would also break the blind spot, since mirrored frames sit within the gap of other output frames.
- **Most of it comes from the future branch when its context is empty or partial:** 2.9 of 3.7 points at support, out to about 64 frames.
- **Two smaller paths:**
  - the one-sided window statistics add about 0.5 points at both ends, out to the window's 254 frames;
  - the half-planes add 0.4 points, within their 31 frames.
- **At a start, the missing past branch contributes nothing or a little negative.** What remains there (+1 point) is the statistics and the half-planes.
- **Reading.** The fusion layers are pointwise and shallow (two layers), and seem to add up the evidence each side brings.
  - Mid-trial, much of the activity evidence comes from the future branch: events seen after the output frame, which tell the correction how much of the mean image to subtract at a source pixel.
  - With the future missing, that subtraction falls short, most at source pixels.
  - Doing better needs an interaction: "if the future is empty, lean more on the past". The fusion can build that only weakly, from the empty branch's code through a GELU.

**Result 3: refits of the fusion on a frozen trunk** (`tools/edge_refit.py`, new): the current architecture cannot learn the edges, and edge features let it learn part of them.
- **Method.** The trunk of v9 fine-tuned (`model.parts`) is frozen and run once on simulated windows placed as in training: half of them at a trial's first or last frame (±64 frames). Its outputs are cached. Only the fusion layers and heads are then trained, from v9's weights, with training's loss.
- **Three arms on the same cache:**
  - ctrl: v9's architecture;
  - edge: with `--edge_features` (below);
  - gated: with `--edge_features`, but only the edge-gated head is trained (132 parameters), so a trial's middle cannot change.
- **Evaluation:** the truncated windows above, from separate draws (48 draws, 16 px), and the whole windows' error against the truth.
- **Two runs:** 128 cached windows for 1500 steps, and 384 (the same 128 plus 256 all at trial edges) for 3000 steps.

| edge − whole, % of the truth, 0–4 / 4–8 / 8–16 frames | support, end | background, end | support, start | whole windows: MSE / oracle, support bias |
|---|---|---|---|---|
| v9 fine-tuned, no refit (48 draws) | +3.6 / +3.5 / +3.3 | +2.0 / +2.0 / +2.4 | +0.7 / +0.7 / +0.1 | 1.22, +2.4% |
| 128 windows: ctrl | +2.2 / +2.5 / +2.3 | +1.4 / +1.6 / +2.0 | +1.0 / +1.2 / −0.1 | 1.27, +1.0% |
| 128 windows: edge | +1.2 / +1.6 / +1.0 | +0.2 / +0.5 / +0.3 | −0.4 / −0.3 / −2.2 | 1.34, +2.4% |
| 128 windows: gated | +1.4 / +1.3 / +1.4 | −1.1 / −1.1 / +0.7 | +3.5 / +3.7 / +2.0 | 1.24, +1.1% |
| 384 windows: ctrl | +2.8 / +2.8 / +3.4 | +1.4 / +1.4 / +2.5 | −0.8 / −0.8 / −0.7 | 1.30, +4.8% |
| 384 windows: edge | +2.3 / +2.4 / +2.6 | +1.0 / +1.1 / +1.5 | 0.0 / 0.0 / −0.5 | 1.33, +4.8% |
| 384 windows: gated | +3.9 / +3.8 / +5.8 | +1.9 / +1.9 / +5.2 | +5.2 / +5.3 / +5.2 | 1.22, +2.5% |

The 128-window rows were evaluated on the first 24 of the 48 draws. On those, v9 itself gives +2.8 / +2.7 / +2.3 at support and +2.0 / +2.0 / +2.1 in background at the end.

- **The current fusion cannot move the edge effect.** Refitting it on windows half or more at trial edges leaves the effect where it was (ctrl, both runs).
- **With edge features, the refit lowers the end effect.** Compared with ctrl on the same cache, it is lower by 0.4–1.3 points at support and 0.4–1.7 in background. That is with a frozen trunk and at most 384 windows.
- **The refits also shift the whole baseline, which confounds the gated arm.**
  - On uniformly placed training windows, weighted as the loss weighs them, v9 fine-tuned is 1.7% low in a trial's middle (the low activity-free bias of the v9 section). It is close to the truth at true trial edges: −0.1% after a start and +0.3% before an end, in the first 16 frames (about 30 windows per side, so ±1).
  - The ctrl and edge refits raise the baseline everywhere (support +2.4 points in the 384-window run) and overfit a little (MSE 1.22 → 1.30–1.33).
  - The gated head cannot touch a trial's middle. So it raises everything within 262 frames of an edge (the reach of its gates) instead, and at the truncated windows' support pixels, which were already high, the effect grows.
- **So the gated head alone is not the test.** Nor can a frozen trunk and a few hundred windows show how far a full training run would get. What the refits do show: the current fusion has no way to change edges apart from the middle, and edge features give it one.
- **The two biases interact.** In these simulations:
  - background pixels are low mid-trial, and the edge effect brings them close to the truth;
  - support pixels are slightly high mid-trial (+2.4% in these support-centred windows), and the edge effect takes them to about +6%.
  - Training weighs both. Judge an edge arm at support and background separately, on the paired effect (which the network's overall bias does not affect) as well as against the truth.

**The arm: edge features** (`--edge_features`, `ModelConfig.edge_features`; default off).
- **Per-side statistics.** The fusion layers also get the window statistics for each side separately (6 channels, all from frames outside the gap):
  - how much of the past and of the future window has data, on a log scale and within 16 frames of the gap;
  - each side's mean relative to the mean image.
- **Edge-gated corrections.** The baseline's and the noise's log corrections get g_past × e_past + g_future × e_future:
  - g is the share of that side's window without data (0 in a trial's middle, 1 when that side is empty);
  - e is a 1×1 head on the fused features.
  - This is the interaction the fusion lacks.
- **Exact warm start.** The new fusion columns and the head start at 0, so a checkpoint without them warm-starts exactly (`load_compatible`, tested).
- **Blind spot.** Tested at a trial edge and mid-window, slab and box.
- **Cost:** 6 fusion inputs and a 4-channel head, computed on the loss region only.

**Recommendation for the next training batch: an arm with `--edge_features`, otherwise the same recipe.**
- **Stages.** Run it in both stages: pretraining from v6r with `--edge_features` starts exactly as v6r, and fine-tuning continues from it. The cheaper fallback is fine-tuning only, which starts exactly as the pretraining pick.
- **No other change.** `--edge_frac 0.5` stays (simulated windows only).
- **Judge it against the same run without the flag:**
  - `tools/edge_windows.py` on the new checkpoints (fastpz, with the run's low-count draw): end and start effect at support and background. The aim is under 1% at support in the last 16 frames.
  - `edge_pieces.py` on the viewer trials (real), where the support end effect is +4.6% today.
  - The held-out evaluation, which must not lose accuracy in a trial's middle.
- **Not recommended:**
  - more edge windows (v9: no effect);
  - an edge-weighted loss (the same lever);
  - padding or filling of the missing frames: the base path is not involved, and reflective filling breaks the blind spot;
  - a fixed correction calibrated on simulations: the effect differs between checkpoints (v8 fine-tuned had half of it), pixel classes and ends.
- **What edge features cannot remove.** One-sided context carries less information on which pixels are active. So part of the support-pixel effect may be the right answer given less information. If so, it should shrink with the arm but not vanish, and background should reach zero first.
- **Until the arm is trained:** relative to the rest of the trial, the baseline in its first and last 0.2 s is 3–4% too high at source pixels and 2% too high in background, most of all at the end.

**Tools.**
- `tools/edge_windows.py OUT.npz --trials ... --ckpt name=path ... [--kind fastpz --pz_legacy]` measures any checkpoint's edge effect. It takes minutes on a free GPU (32 draws, 2 checkpoints). It saves partial results, `--summary` prints them, and `--need_gb` waits for free GPU memory.
- `tools/edge_refit.py` runs the frozen-trunk refits.
- **On this machine,** when the shared 8 GB GPU runs short, Windows pages GPU memory to the host and a forward pass takes minutes instead of 0.5 s. Both tools also run on the CPU (`--device cpu --threads N`).
- **CPU refits** use their own small AdamW: torch.optim's step probes the CUDA stream, which hangs while the full GPU opens a context.

## Fine-tuning trade-off: real-data gain against simulation accuracy (2026-11-21)

**Question.** In S0–S3, fine-tuning (v9 recipe: `--data mix --sim_frac 0.5 --lr 1e-4`, 1700 steps, `--event_mask 0.5`) gains 0.1–0.3 points on real windows and loses 5–7 points on fast simulations. Is fine-tuning worth it, and which recipe keeps both the simulation accuracy and the real-data gain?

**Data:**
- The held-out evaluation of S0–S3: every kept step, 192 windows per mouse and kind. `tools/finetune_tradeoff.py RESULTS_DIR` prints the numbers below.
- Test mice: 776270, 810196, 829704, 876767. "All held-out mice" adds the four validation mice. Simulated kinds leave out 804730.
- "Late" is the mean of the fine-tuning checkpoints at steps 750–1700. Scores are % changes of the 8-frame MSE, balanced over support and background, as in Experiment 1.

**The real-data gain is small but consistent, and almost all of it is at background pixels:**

| all 8 held-out mice | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| real vs mean image, step 0 → late | −2.55 → −2.73 | −2.54 → −2.75 | −2.77 → −3.09 | −2.85 → −3.16 |
| gain, 95% interval over mice and windows | −0.18 [−0.42, +0.02] | −0.21 [−0.41, −0.05] | −0.32 [−0.40, −0.25] | −0.31 [−0.49, −0.12] |
| real vs SiLo, step 0 → late | −1.14 → −1.31 | −1.12 → −1.32 | −1.36 → −1.68 | −1.45 → −1.75 |
| share of the gap to the oracle closed, background | 0.11 → 0.28 | 0.14 → 0.28 | 0.19 → 0.34 | 0.16 → 0.35 |
| the same at support | 0.60 → 0.60 | 0.59 → 0.61 | 0.63 → 0.68 | 0.66 → 0.69 |

- **The gap to the oracle.** The oracle (the mean image minus the trial's mean activity) is the best constant baseline. On real windows it is 8.1% better than the mean image at support and 2.0% at background.
- **Size.** Against SiLo, fine-tuning adds 15–25% to the network's advantage. The gain is significant over mice in three of the four runs, and the same against the oracle.
- **Timing.** It is learned in the first 250–500 steps, while the learning rate is still warming up, and then stays flat.
- **It corrects a level.**
  - At step 0 the network's baseline is too low at background, on real windows (bias +0.007 to +0.020, target − net) and on fast simulations (+0.011 to +0.023) alike. Fine-tuning brings both to about 0.
  - This looks like the low baseline at low photon counts ("Low photon counts: the baseline sits low", above): pretraining puts it in. If the "lowcount" arm (`--photon_gain 1,4 --silent_trial 0.2`) removes it, most of this gain may become unnecessary.
  - It is not the output heads: `head_mu`'s bias moves by 0.0002. The whole trunk moves, 2.3% of the weights' norm (pretraining moves them about 9%).

**The simulation cost is mostly checkpoint noise and lost selection luck, not drift:**

| test mice, fast vs oracle | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| fine-tuning steps 0 / 250 / 500 / 750 / 1000 / 1250 / 1500 / 1700 | 33 / 33 / 35 / **43** / 32 / **44** / 38 / 35 | 32 / 32 / 36 / **39** / 34 / **42** / 35 / 37 | 26 / 31 / 26 / **36** / 30 / **35** / 27 / 30 | 24 / 37 / 35 / 31 / 28 / **39** / 27 / 31 |
| late pretraining steps (1500–3000): mean, SD | 34.1, 1.9 | 30.9, 0.7 | 30.6, 7.7 | 31.7, 7.5 |
| support leak, step 0 → late (simulated support; mean image 0.44) | 0.28 → 0.29 | 0.27 → 0.28 | 0.26 → 0.28 | 0.24 → 0.27 |

- **Noise.** Fine-tuning checkpoints move 3–5 points (SD) between kept steps, as late pretraining checkpoints do.
- **Bad steps are shared.** Steps 750 and 1250 are bad in S0–S2, and 1250 in S3.
  - Training draws each window from (seed, index), so all runs see the same order of real and simulated windows and the same simulation draws.
  - The runs' common deviation correlates with the share of real windows in the preceding 25–50 steps (r = +0.7, over 7 steps).
- **Bad checkpoints are bad on both readouts.** Within runs, the test mice's real and fast scores rank-correlate +0.7 (S0, S2, S3; S1 0.0). The bad checkpoints also let more of the activity into the baseline (leak 0.29–0.31, against 0.24–0.28 at step 0).
- **Step 0 had the selection's luck.**
  - It is pretraining's pick among 13 checkpoints, chosen on the validation mice's fast windows. On the test mice it is 0–7 points better than the late pretraining mean (S3: 24.5 against 31.7).
  - Fine-tuning selects on real windows, which barely rank fast windows, and so gives that luck back.
- **The good fine-tuning checkpoints cost little.** Picking each run's best checkpoint on the test mice's own fast windows (step 0 included, so an upper bound) gives +28.6 against step 0's +29.0, and keeps part of the real gain (−3.16, against −3.01 at step 0 and −3.24 late).

**Selecting on both readouts** (fine-tuning checkpoints with step 0, chosen by the validation mice's real score + w × their fast score, both balanced against the mean image; read on the test mice, mean of the four runs):

| choice | steps (S0–S3) | real vs mean image | fast vs oracle |
|---|---|---|---|
| step 0 | 0, 0, 0, 0 | −3.01 | +29.0 |
| late mean | | −3.24 | +34.6 |
| real only (w = 0) | 250, 750, 1000, 1000 | −3.31 | +32.5 |
| real + fast, any w from 0.03 up, and fast only | 250, 750, 500, 1000 | −3.30 | +31.5 |

- Selection on both keeps the whole real gain and halves the simulation cost.
- Real alone already does most of it, because real and fast co-vary within runs. Fast decides where real cannot (S2).
- The weight hardly matters: every w from 0.03 to fast alone picks the same checkpoints.

**Weight averaging:** not yet scored.
- **Queued locally:** on 10 trials of each test mouse (48 fast and 96 real windows per mouse):
  - S3's fine-tuning soup (steps 750–1700) and WiSE-FT interpolations from step 0 toward it (0.25, 0.5);
  - the pretrained trunk with the fine-tuned heads, and the reverse;
  - a soup of pretraining steps 2000–3000.
- The local GPU is busy with other sessions' jobs, so the results will be added here when they finish.
- They show whether averaging alone recovers the simulation accuracy. The EMA arm below tests that in training in any case.

**Is fine-tuning worth it? Yes, as a short calibration on real data.**
- It buys 0.2–0.3 points on real windows, consistently across runs and mice: 15–25% of the network's advantage over SiLo, at background pixels.
- The 5–7 points it costs on simulations come from noisy checkpoints and from selecting on real windows only, so they are avoidable.
- If the "lowcount" arm removes the low background baseline, rerun this comparison on it: fine-tuning may then add little.

**Recommended fine-tuning for the next training batch:**
- **Baseline arm:** the v9 fine-tuning recipe with the new selection: `--val_sessions 878030,794237,834788,804730 --val_windows 48 --select balanced --select_on real:1,fast:0.1`, plus `--val_fast` and `--keep_every 100`.
  - With w = 0.1 the two sets move about equally between checkpoints. In S0–S3 any w from 0.03 up picked the same checkpoints.
- **Arm to add: `--ema 0.998`** (a 500-step horizon), otherwise the same.
  - Validation and best.pt then score the moving average, which removes most of the checkpoint noise and should keep the shared bad steps out.
  - `--keep_every 100` keeps the raw weights too (`model_raw`), so other horizons and soups can be compared afterwards with `slap2-f0 average --key model_raw`, without retraining.
  - Cost: one more fine-tuning run, about 1700 steps (roughly an hour on the L40S).
- **Not recommended yet:**
  - `--l2sp`: drift is not the main cost. `--l2sp 30` would roughly halve the random drift (a 330-step pull at lr 1e-4) at the price of a bias toward the start.
  - A lower learning rate or a higher `--sim_frac`: they shrink the noise only in proportion and slow the background correction, and no runs tested them.
  - Fewer steps: the gain is in by step 500, but the noise does not shrink with fewer steps.

**Code (branch finetune-regularize):**
- `train --ema DECAY`: validation, best.pt, last.pt and the kept steps hold an exponential moving average of the weights, started at the initial weights. The raw weights are saved as `model_raw`.
- `train --l2sp L` (needs `--init`): after each step, the weights move back toward the initial ones by lr × L of the distance. This is decoupled L2-SP, like AdamW's weight decay but toward the start.
- `slap2-f0 average OUT CKPT[=W] ...`: the weighted average of checkpoints, from paths, globs and weights, for soups of kept steps and WiSE-FT interpolation (`pre.pt=0.5 ft.pt=0.5`). The network has no normalization buffers, so the averaged weights are a complete model.
- `tools/finetune_tradeoff.py`: the tables above. With `--averages`, it gives paired summaries of model variants scored on the same windows.
- Tests: `test_ema_and_l2sp` and `test_average_checkpoints`.
