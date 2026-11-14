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
| Same, with GIAnT's `splitFreq` lower envelope instead | −2% (−12% to +8%) | a cheap fix that removes most of the bias |
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
1. **Cheap comparator.** Add a `splitFreq` lower-envelope baseline as an option in `localizeSources_vIM`, and compare localization with today's baseline: precision/recall/F1 on the 12 simulations, and source agreement and stability on real sessions. The network has to beat this, not just the moving median.
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
- **Localization:** today's baseline vs `splitFreq` vs the network, on the simulations (ground-truth F1, position error) and on real data (agreement with final sources, cross-trial stability, SNR of the resulting sources).
- **Runtime:** per trial, on GPU and CPU.

**Phase 4: integration**
- **v1 of the integration:** a Python `predict` step after registration writes `*_BASELINE-80Hz.h5` (μ, σ) next to the registered movie. SiLo uses it when present, behind a parameter. That avoids a Deep Learning Toolbox dependency; an ONNX import into MATLAB can come later.
- **Optionally:** upsample μ to 200 Hz to initialize `fitB`.
- **Retrain once** on targets regenerated with the improved localization, since the targets depend on GIAnT's fits.

## Risks

- **Headroom.** `splitFreq` already removes the mean event bias. The network's extra value is tracking 1–4 s fluctuations (R² ≈ 0.14–0.17 of a ≈ 0.33 ceiling with linear predictors) plus calibrated σ. Phase 0 and Phase 3 must show that this improves localization.
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
