function summary = exportF0TrainingData(trialTableFn, outDir, varargin)
%EXPORTF0TRAININGDATA Training data for the SLAP2 F0 network from a SiLo run.
%
%   summary = exportF0TrainingData(trialTableFn, outDir, Name, Value, ...)
%
%   For every imaging path and every trial that SiLo analysed, writes
%   <outDir>/Path<k>/trial_<NNN>.h5 holding, on the trial's 80 Hz registered
%   grid (cropped to the box around the pixels SiLo treats as valid):
%     /movie      the registered movie
%     /activity   GIAnT's fitted activity on the same grid and timing; the
%                 training target is movie - activity
%     /mask/*     valid, labeled and footprint-support pixels
%     /frame/*    per-frame discard flags, motion, recNegErr, online offsets,
%                 mean of the valid pixels
%     /varfac     SLAP2 variance factor (per-pixel mean, or every frame)
%     /sources/*  each source's peak pixel
%   plus <outDir>/export_summary.json with per-trial QC. A path where SiLo
%   found no sources (e.g. a GFP control without an activity sensor) is
%   exported with zero activity, an empty footprint support and no
%   /sources; its residual is then the movie itself. Arrays are written
%   so that h5py reads them in C order as (channel, frame, row, col),
%   (row, col) and (frame,); pixel coordinates are 0-based in the crop, and
%   the crop origin is stored as an attribute.
%
%   The activity is GIAnT's per-trial footprints times its denoised traces
%   (dF_denoised, movie units), mapped onto the 80 Hz movie the way the
%   movie was made:
%     time   an 80 Hz frame is the getImages weighted mean of the samples
%            within max(3*dt, linesPerCycle) lines of its line, with weights
%            exp(-|dline|/dt), dt = ceil(line rate / alignHz). The 200 Hz
%            traces are averaged with the same weights (discarded 200 Hz
%            frames left out).
%     space  footprints live on SiLo's consensus canvas. Each trial's
%            sub-pixel offset is the one that maximizes the correlation of
%            movie and activity at source peak pixels.
%     blur   the footprints are blurred by the Gaussian (sigma from
%            blurSigmas) for which the movie-on-activity slope of the flux
%            in each source's core (<= 1.5 px from its peak pixel) equals
%            that of the flux 1.5-3.5 px out (median over sources of the
%            ratio, blurShapeRatio): a match of shape that does not depend
%            on the activity's scale. The 80 Hz frames are slightly blurrier
%            than the 200 Hz data the footprints were fitted to.
%     scale  none: the activity is GIAnT's, unscaled (activityGain 1).
%            Older exports (without activitySlope) multiplied it by the disk-flux
%            slope below, clamped to 0.5-1.5, and chose the blur that made
%            the peak-pixel slope match that gain. A least-squares slope
%            tracks the movie-activity correlation, not the scale, so that
%            gain removed about half of GIAnT's activity.
%   QC per trial, all for the activity as written (after activityGain):
%   activitySlope, the movie-on-activity slope of the flux in a 4 px disk
%   around each source at blur 0 (median over sources; the old exports'
%   gain before its clamp); the slope and correlation at peak pixels
%   (peakSlope, peakR) and for the disk flux (diskSlope, diskR), below 1
%   when the activity is noisy, since a least-squares slope is pulled toward
%   0 by any error in the activity; the event-triggered residual at source
%   peaks as a fraction of the event-triggered movie, with events picked on
%   the activity (eventResidualFraction, near 0 when the activity is
%   removed); and the selection-safe event ratios of ringEventRatios
%   (events picked on the movie 3-5 px from each source; sources touching
%   GIAnT fit artifacts left out): activity / movie event amplitude at
%   <= 1.5 px (eventRatioCore) and 1.5-2.5 px (eventRatio2px), with their
%   source and event counts (eventRatioSources, eventRatioEvents,
%   eventRatioArtifactSources). Both ratios are near 1 when the activity has
%   the movie's size; the selection favours spatially broad events.
%
%   Name-Value options:
%     saveDir     folder holding motion_correction/ and source_extraction/
%                 (default: the trial table's savedr if it exists, else the
%                 trial table's folder)
%     rawDir      folder of the raw SLAP2 files, to read linesPerCycle
%                 (default: the trial table's datadr; without it the kernel
%                 is truncated at 3*dt)
%     paths       imaging paths to export (default all)
%     trials      trials to export (default all that SiLo analysed)
%     channels    channels whose movie and activity are written (default 1,
%                 the glutamate channel when activityChannel is 1)
%     blurSigmas  candidate blur sigmas in px (default 0:0.25:1.5; a trial
%                 whose best candidate is an end of the range shows it as a
%                 blurShapeRatio away from 1)
%     shiftRange, shiftStep  sub-pixel offset search around the integer
%                 offset of the mean images (default 1 and 0.25 px)
%     varFac      'mean' (per-pixel mean, default) or 'full' (every frame)
%     deflate     HDF5 deflate level (default 1: 40% faster to write than 4,
%                 and 5% larger files)
%     label       session label stored with the data (default: the name of
%                 the trial table's datadr folder)
%     giantRef    GIAnT commit that made the SiLo outputs, stored as metadata
%     ignoreSources  export every path as if SiLo had found no sources (zero
%                 activity; for checks of the raw residual statistics)
%     nWorkers    trials exported at once on a parallel pool (default 1:
%                 one after the other); each needs several GB of memory
%                 (about 15 GB for a 2-channel 845 x 320 px trial of 28 s)

p = inputParser;
addParameter(p, 'saveDir', '');
addParameter(p, 'rawDir', '');
addParameter(p, 'paths', []);
addParameter(p, 'trials', []);
addParameter(p, 'channels', 1);
addParameter(p, 'blurSigmas', 0:0.25:1.5);
addParameter(p, 'shiftRange', 1);
addParameter(p, 'shiftStep', 0.25);
addParameter(p, 'varFac', 'mean');
addParameter(p, 'deflate', 1);
addParameter(p, 'label', '');
addParameter(p, 'giantRef', '');
addParameter(p, 'ignoreSources', false);
addParameter(p, 'nWorkers', 1);
parse(p, varargin{:});
o = p.Results;

tt = loadStructFromH5(trialTableFn);
if isempty(o.saveDir)
    if isfield(tt, 'savedr') && exist(char(tt.savedr), 'dir')
        o.saveDir = char(tt.savedr);
    else
        o.saveDir = fileparts(trialTableFn);
    end
end
if isempty(o.rawDir) && isfield(tt, 'datadr')
    o.rawDir = char(tt.datadr);
end
if isempty(o.label)
    [~, o.label] = fileparts(strip(o.rawDir, 'right', filesep));
end
mocoDr = fullfile(o.saveDir, 'motion_correction');
es = fullfile(o.saveDir, 'source_extraction', 'experiment_summary.h5');
pts = fullfile(o.saveDir, 'source_extraction', 'per_trial_summary.h5');
prm = readParams(es);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

nPaths = size(tt.filename, 1);
paths = o.paths;
if isempty(paths)
    paths = 1:nPaths;
end
summary = struct('label', o.label, 'giantRef', o.giantRef, 'trials', {{}});
for DMDix = paths(:)'
    grp = sprintf('/Path%d', DMDix);
    try
        nf = double(h5read(es, [grp '/frame_info/trial_num_frames']));
    catch
        warning('exportF0TrainingData:NoPath', 'No SiLo output for path %d; skipped', DMDix);
        continue
    end
    try
        assert(~o.ignoreSources);
        profInfo = h5info(pts, [grp '/sources/spatial/per_trial_profiles']);
        profSz = profInfo.Dataspace.Size; % [nTrials nSources 1 rows cols]
        fl = double(h5read(es, [grp '/frame_info/frame_line_idxs']));
    catch
        warning('exportF0TrainingData:NoSources', 'No SiLo sources for path %d; exporting with zero activity', DMDix);
        % without sources SiLo writes zero frame counts; the trials are those with a registered movie
        reg = tt.motion_correction.fn_reg_ds;
        nf = double(arrayfun(@(k) strlength(string(reg{DMDix, k})) > 0, 1:size(reg, 2)));
        profSz = [numel(nf) 0 1 0 0];
        fl = [];
    end
    pathDir = fullfile(outDir, sprintf('Path%d', DMDix));
    if ~exist(pathDir, 'dir')
        mkdir(pathDir);
    end
    trials = o.trials;
    if isempty(trials)
        trials = find(nf(:)' > 0);
    end
    % every trial's footprints, read once for the path (see readTrialProfiles)
    if profSz(2) > 0
        tRead = tic;
        Fall = readTrialProfiles(pts, [grp '/sources/spatial/per_trial_profiles'], profSz);
        fprintf('[exportF0] %s path %d: footprints of %d trials x %d sources read in %.0f s\n', ...
            o.label, DMDix, profSz(1), profSz(2), toc(tRead));
    else
        Fall = cell(numel(nf), 1);
    end
    trials = trials(:)';
    trials = trials(trials <= numel(nf));
    trials = trials(nf(trials) > 0);
    nW = min(o.nWorkers, numel(trials));
    if nW > 1 % trials at once; the summary is written once the path is done
        pool = gcp('nocreate');
        if isempty(pool)
            pool = parpool('Processes', nW);
        end
        nW = min(nW, pool.NumWorkers);
        Ft = Fall(trials);
        qs = cell(1, numel(trials));
        parfor (i = 1:numel(trials), nW)
            qs{i} = exportOne(tt, trials(i), DMDix, nf, fl, profSz, Ft{i}, es, grp, mocoDr, prm, o, pathDir);
        end
        summary.trials = [summary.trials, qs];
        writeSummary(outDir, summary);
    else
        for t = trials
            summary.trials{end+1} = exportOne(tt, t, DMDix, nf, fl, profSz, Fall{t}, es, grp, mocoDr, prm, o, pathDir);
            writeSummary(outDir, summary);
        end
    end
end
end

function q = exportOne(tt, t, DMDix, nf, fl, profSz, H, es, grp, mocoDr, prm, o, pathDir)
%one trial, with its log lines; a failure is returned as {path, trial, error}
fprintf('[exportF0] %s path %d trial %d\n', o.label, DMDix, t);
try
    q = exportTrial(tt, t, DMDix, nf, fl, profSz, H, es, grp, mocoDr, prm, o, ...
        fullfile(pathDir, sprintf('trial_%03d.h5', t)));
    fprintf(['[exportF0] path %d trial %d: shift [%.2f %.2f], blur %.2f px (slope ratio %.2f); activity unscaled: ' ...
        'disk-flux slope at blur 0 %.2f, peak slope %.2f (r %.2f), disk-flux slope %.2f (r %.2f), event residual %.2f; ' ...
        'ring events: activity/movie %.2f at the core, %.2f at 1.5-2.5 px (%d sources, %d events)\n'], DMDix, t, ...
        q.shiftRC, q.blurSigma, q.blurShapeRatio, q.activitySlope, q.peakSlope, q.peakR, q.diskSlope, q.diskR, ...
        q.eventResidualFraction, q.eventRatioCore, q.eventRatio2px, q.eventRatioSources, q.eventRatioEvents);
catch ME
    fprintf(2, '[exportF0] path %d trial %d failed: %s\n', DMDix, t, ME.message);
    q = struct('path', DMDix, 'trial', t, 'error', ME.message);
end
end

function writeSummary(outDir, summary)
fid = fopen(fullfile(outDir, 'export_summary.json'), 'w');
fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);
end

function q = exportTrial(tt, t, DMDix, nf, fl, profSz, H, es, grp, mocoDr, prm, o, outFn)
% H: this trial's footprints on the consensus canvas, a sparse (rows*cols) x nS
% matrix (readTrialProfiles)
nS = profSz(2); % 0: no sources, zero activity
%--- GIAnT outputs for this trial (200 Hz, consensus canvas)
if nS > 0
    f200 = sum(nf(1:t-1)) + (1:nf(t));
    nCh200 = h5info(es, [grp '/sources/temporal/dF_denoised']).Dataspace.Size(2);
    X200 = double(h5read(es, [grp '/sources/temporal/dF_denoised'], [1 1 f200(1)], [nS nCh200 nf(t)]));
    lines200 = fl(f200);
    disc200 = squeeze(all(isnan(X200(:,1,:)), 1));
    X200(isnan(X200)) = 0;
    if nnz(H) == 0
        error('no footprints for this trial');
    end
    H = struct('H', H, 'sz', [profSz(4) profSz(5)]);
end

%--- the 80 Hz registered movie and its alignment data
fn = tt.motion_correction.fn_reg_ds{DMDix, t};
if endsWith(fn, '.h5')
    desc = h5info(fullfile(mocoDr, fn));
    IM = h5read(fullfile(mocoDr, fn), ['/' desc.Datasets(1).Name]);
else
    IM = ScanImageTiffWrapper(fullfile(mocoDr, fn));
end
IM = single(reshape(IM, size(IM,1), size(IM,2), prm.numChannels, []));
[R, C, ~, T] = size(IM);
meanAll = reshape(mean(IM, 4, 'omitnan'), R, C, []); % R x C x channels
aData = loadStructFromH5(fullfile(mocoDr, tt.motion_correction.fn_adata{DMDix, t}));
dsLines = double(aData.DSframes(:))';

%discarded frames, exactly as loadAndProcessTrialAsync
tmp = aData.recNegErr(:) - medfilt1(aData.recNegErr(:), round(4*prm.alignHz));
tmp = -tmp ./ (min(-0.005, prctile(tmp, 5)));
win = 2*ceil(0.025*prm.alignHz) + 1;
discard = imclose(imdilate(tmp > prm.motionThresh, ones(win,1)) & (tmp > prm.motionThresh/2), ones(win,1));
discard(1:ceil(prm.discardInitial_s*prm.alignHz)) = true;
act = squeeze(IM(:,:,prm.activityChannel,:));
act(:,:,discard) = nan;
valid = mean(isnan(act), 3) < prm.nanThresh;
clear act

Y1 = reshape(IM(:,:,prm.activityChannel,:), R*C, T);
meanT = mean(reshape(IM(:,:,prm.activityChannel,:), R, C, T), 3, 'omitnan');
gain = 1; % the activity is written unscaled (see the help)
if nS == 0 % no activity to map, align or fit
    Y1(:, discard) = nan;
    Hs = sparse(R*C, 0);
    dt = NaN; half = NaN; lpc = NaN; best.shift = [0 0];
    bestSig = struct('sigma', NaN, 'ratio', NaN, 'slope', NaN, 'r', NaN);
    actSlope = NaN; cfD = NaN; rD = NaN; etaFrac = NaN; nEv = 0; pr = []; pc = [];
else
%--- the getImages kernel, from 200 Hz frames to 80 Hz frames
if numel(dsLines) > 1
    dt = ceil(mean(diff(dsLines)));
else
    dt = ceil(median(diff(lines200))*prm.analyzeHz/prm.alignHz);
end
lpc = linesPerCycle(tt, t, DMDix, o.rawDir);
half = max(3*dt, lpc);
ii = []; jj = []; ww = []; discW = zeros(1, T);
for j = 1:T
    d = lines200(:)' - dsLines(j);
    sel = find(abs(d) <= half);
    w = exp(-abs(d(sel))/dt);
    discW(j) = sum(w(disc200(sel))) / max(sum(w), eps);
    keep = ~disc200(sel);
    if any(keep)
        ii = [ii sel(keep)]; jj = [jj repmat(j, 1, nnz(keep))]; ww = [ww w(keep)/sum(w(keep))]; %#ok<AGROW>
    end
end
Wk = sparse(ii, jj, ww, numel(lines200), T);
discard = discard(:)' | discW > 0.5;
X80 = cell(1, nCh200);
for ch = 1:nCh200
    X80{ch} = reshape(X200(:, ch, :), nS, []) * Wk; % sources x 80 Hz frames (squeeze would turn one source into a column)
end

%--- consensus -> trial offset and blur
Y1(:, discard) = nan;
% mean_im is [1 channels rows cols]: squeeze would also drop the channel
% dimension of a one-channel recording, so reshape to channels x pixels
meanC = h5read(es, [grp '/visualizations/mean_im']);
meanC = reshape(meanC, [], profSz(4)*profSz(5));
meanC = reshape(meanC(prm.activityChannel, :), profSz(4), profSz(5));
sh0 = integerOffset(meanT, meanC, 5);
hpf = @(x) x - movmedian(x, 81, 2, 'omitnan');
best = struct('r', -inf);
for dr = sh0(1) + (-o.shiftRange:o.shiftStep:o.shiftRange)
    for dc = sh0(2) + (-o.shiftRange:o.shiftStep:o.shiftRange)
        Hs = placeFootprints(H, [dr dc], 0, R, C);
        [sl, r] = peakFit(Hs, X80{prm.activityChannel}, Y1, hpf);
        if median(r, 'omitnan') > best.r
            best = struct('r', median(r, 'omitnan'), 'shift', [dr dc]);
        end
    end
end
%the disk-flux slope at blur 0 (QC only: the old exports' gain); then the blur
%whose core and 1.5-3.5 px slopes agree, with the cores at the unblurred peaks
H0 = placeFootprints(H, best.shift, 0, R, C);
cfD = diskFit(H0, X80{prm.activityChannel}, Y1, hpf, R, C);
actSlope = median(cfD, 'omitnan');
[~, pk0] = max(H0, [], 1);
[pr0, pc0] = ind2sub([R C], full(pk0(:)));
clear H0
bestSig = struct('d', inf, 'sigma', o.blurSigmas(1), 'ratio', NaN);
ratios = nan(size(o.blurSigmas));
for k = 1:numel(o.blurSigmas)
    Hs = placeFootprints(H, best.shift, o.blurSigmas(k), R, C);
    ratios(k) = median(shapeFit(Hs, X80{prm.activityChannel}, Y1, hpf, valid, pr0, pc0), 'omitnan');
    if abs(log(ratios(k))) < bestSig.d
        bestSig = struct('d', abs(log(ratios(k))), 'sigma', o.blurSigmas(k), 'ratio', ratios(k));
    end
end
fprintf('[exportF0] path %d trial %d: core / 1.5-3.5 px slope ratio by blur %s: %s\n', DMDix, t, ...
    mat2str(o.blurSigmas), mat2str(round(ratios, 3)));
Hs = placeFootprints(H, best.shift, bestSig.sigma, R, C); % (R*C) x nS sparse, unscaled
[sl, r] = peakFit(Hs, X80{prm.activityChannel}, Y1, hpf);
bestSig.slope = median(sl, 'omitnan'); bestSig.r = median(r, 'omitnan');
[~, pk] = max(Hs, [], 1); pk = full(pk(:));
[pr, pc] = ind2sub([R C], pk);

%--- QC of the activity as written: disk flux and event-triggered residual
A1 = Hs(pk, :) * X80{prm.activityChannel}; % activity at the peak pixels (nS x T)
[cfD, rD] = diskFit(Hs, X80{prm.activityChannel}, Y1, hpf, R, C);
etaY = zeros(1, 25); etaR = etaY; nEv = 0;
for s = 1:nS
    x = X80{prm.activityChannel}(s, :);
    thr = prctile(x, 99);
    ev = find(x(2:end-1) > thr & x(2:end-1) >= x(1:end-2) & x(2:end-1) > x(3:end)) + 1;
    ev = ev(ev > 8 & ev <= T - 16);
    y = double(Y1(pk(s), :)); a = full(A1(s, :));
    for e = ev
        wy = y(e-8:e+16); wr = wy - a(e-8:e+16);
        if any(isnan(wy)), continue, end
        etaY = etaY + wy - mean(wy(1:6)); etaR = etaR + wr - mean(wr(1:6)); nEv = nEv + 1;
    end
end
etaFrac = max(etaR(9:13)) / max(etaY(9:13));
end

%--- crop and write
rows = find(any(valid, 2)); cols = find(any(valid, 1));
r0 = rows(1); r1 = rows(end); c0 = cols(1); c1 = cols(end);
Rc = r1 - r0 + 1; Cc = c1 - c0 + 1;
%selection-safe event ratios (sources whose 17 px box fits in the crop)
if nS > 0
    Xa = X80{prm.activityChannel};
    rq = ringEventRatios(Y1, @(idx) full(Hs(idx, :) * Xa), full(Hs * mean(Xa, 2)), valid, pr, pc, discard, [r0 r1 c0 c1]);
    clear Xa
else
    rq = struct('core', NaN, 'near', NaN, 'nSources', 0, 'nEvents', 0, 'nArtifactSources', 0);
end
crop = @(M) M(r0:r1, c0:c1, :, :);
if exist(outFn, 'file'), delete(outFn); end
chans = o.channels(:)';
mov = zeros(Rc, Cc, T, numel(chans), 'single');
actv = zeros(Rc, Cc, T, numel(chans), 'single');
if nS > 0 % the activity is computed only at footprint pixels inside the crop (zero elsewhere)
    [cr, cc] = ndgrid(r0:r1, c0:c1);
    inCrop = sub2ind([R C], cr(:), cc(:));
    sup = find(any(Hs(inCrop, :), 2));
end
for k = 1:numel(chans)
    m = crop(squeeze(IM(:,:,chans(k),:)));
    m(:,:,discard) = nan;
    mov(:,:,:,k) = m;
    if nS > 0
        a = zeros(Rc*Cc, T, 'single');
        a(sup, :) = single(full(Hs(inCrop(sup), :) * X80{min(chans(k), nCh200)}));
        actv(:,:,:,k) = reshape(a, Rc, Cc, T);
    end
end
clear IM
movSz = [Cc Rc T numel(chans)]; % explicit, so a single channel keeps its axis
writeArr(outFn, '/movie', permute(mov, [2 1 3 4]), [min(64,Cc) min(64,Rc) min(256,T) 1], o.deflate, movSz);
writeArr(outFn, '/activity', permute(actv, [2 1 3 4]), [min(64,Cc) min(64,Rc) min(256,T) 1], o.deflate, movSz);
clear mov actv
mf = medfilt2(meanT, [3 3]);
labeled = ~isnan(meanT) & mf > 3*prctile(mf(~isnan(mf)), 25);
if nnz(valid & ~labeled) < 0.05*nnz(valid)
    % a dim trial: the canvas's 25th percentile is so low that (nearly)
    % every valid pixel counts as labeled, which leaves the noise statistics
    % no background pixels; take the percentile over the valid pixels
    labeled = ~isnan(meanT) & mf > 3*prctile(mf(valid & ~isnan(mf)), 25);
end
support = reshape(full(any(Hs, 2)), R, C);
writeArr(outFn, '/mask/valid', uint8(crop(valid)'), [], o.deflate);
writeArr(outFn, '/mask/labeled', uint8(crop(labeled)'), [], o.deflate);
writeArr(outFn, '/mask/footprint_support', uint8(crop(support)'), [], o.deflate);
mi = permute(single(crop(meanAll)), [2 1 3]); % size keeps the channel axis of one-channel recordings
writeArr(outFn, '/mean_image', mi, [], o.deflate, [size(mi, 1) size(mi, 2) size(meanAll, 3)]);
writeArr(outFn, '/frame/discard', uint8(discard(:)), [], o.deflate);
writeArr(outFn, '/frame/motion_row', single(aData.motionDSr(:)), [], o.deflate);
writeArr(outFn, '/frame/motion_col', single(aData.motionDSc(:)), [], o.deflate);
writeArr(outFn, '/frame/rec_neg_err', single(aData.recNegErr(:)), [], o.deflate);
for f = {'onlineMotionXshift', 'onlineMotionYshift', 'onlineMotionZshift'}
    if isfield(aData, 'slap2') && isfield(aData.slap2, f{1})
        writeArr(outFn, ['/frame/' f{1}], single(aData.slap2.(f{1})(:)), [], o.deflate);
    end
end
gm = mean(Y1(reshape(valid, [], 1), :), 1, 'omitnan');
writeArr(outFn, '/frame/valid_mean', single(gm(:)), [], o.deflate);
vf = crop(aData.slap2.varFacDS);
if strcmpi(o.varFac, 'full')
    writeArr(outFn, '/varfac', permute(single(vf), [2 1 3]), [min(64,Cc) min(64,Rc) min(256,T)], o.deflate);
else
    writeArr(outFn, '/varfac', single(mean(vf, 3, 'omitnan'))', [], o.deflate);
end
if nS > 0
    writeArr(outFn, '/sources/peak_row', int32(pr(:) - r0), [], o.deflate);
    writeArr(outFn, '/sources/peak_col', int32(pc(:) - c0), [], o.deflate);
end

q = struct('path', DMDix, 'trial', t, 'file', outFn, 'frames', T, 'rows', Rc, 'cols', Cc, ...
    'cropOriginRC', [r0 c0] - 1, 'nSources', nS, 'validPixels', nnz(valid), ...
    'kernelDtLines', dt, 'kernelHalfWidthLines', half, 'linesPerCycle', lpc, ...
    'shiftRC', best.shift, 'blurSigma', bestSig.sigma, 'blurShapeRatio', bestSig.ratio, ...
    'activityGain', gain, 'activitySlope', actSlope, ...
    'peakSlope', bestSig.slope, 'peakR', bestSig.r, ...
    'diskSlope', median(cfD, 'omitnan'), 'diskR', median(rD, 'omitnan'), ...
    'eventResidualFraction', etaFrac, 'nEvents', nEv, ...
    'eventRatioCore', rq.core, 'eventRatio2px', rq.near, 'eventRatioSources', rq.nSources, ...
    'eventRatioEvents', rq.nEvents, 'eventRatioArtifactSources', rq.nArtifactSources, ...
    'discardedFrames', nnz(discard), ...
    'channels', chans, 'activityChannel', prm.activityChannel, 'alignHz', prm.alignHz, ...
    'photonScale', prm.photonScale, 'label', o.label, 'giantRef', o.giantRef);
for fn2 = fieldnames(q)'
    v = q.(fn2{1});
    if ~strcmp(fn2{1}, 'file')
        h5writeatt(outFn, '/', fn2{1}, v);
    end
end
end

function Fall = readTrialProfiles(pts, ds, profSz)
%every trial's footprints in one pass. SiLo's per_trial_profiles is
%[trials sources 1 rows cols], uncompressed and contiguous, with trials
%varying fastest on disk: reading one trial walks the whole dataset (tens of
%GB for a 15 min session), once per trial. Blocks of columns are contiguous,
%so they are read in turn (about 2 GB each). Returns, per trial, a sparse
%(rows*cols) x sources matrix (NaN as 0).
nT = profSz(1); nS = profSz(2); Rk = profSz(4); Ck = profSz(5);
cb = max(1, floor(2^31 / (4 * nT * nS * Rk))); % columns per block
P = cell(0, 1); S = P; T = P; V = P;
for c0 = 1:cb:Ck
    n = min(cb, Ck - c0 + 1);
    B = h5read(pts, ds, [1 1 1 1 c0], [nT nS 1 Rk n]);
    B = reshape(B, nT, nS, Rk * n);
    B(isnan(B)) = 0;
    k = find(B);
    [tk, sk, pk] = ind2sub([nT nS Rk*n], k);
    T{end+1} = tk; S{end+1} = sk; P{end+1} = pk + (c0 - 1) * Rk; V{end+1} = double(B(k)); %#ok<AGROW>
end
clear B
T = vertcat(T{:}); S = vertcat(S{:}); P = vertcat(P{:}); V = vertcat(V{:});
Fall = repmat({sparse(Rk*Ck, nS)}, nT, 1);
if isempty(T), return, end
[T, ord] = sort(T); S = S(ord); P = P(ord); V = V(ord);
edges = [0; find(diff(T)); numel(T)];
for i = 1:numel(edges) - 1
    j = edges(i)+1:edges(i+1);
    Fall{T(j(1))} = sparse(P(j), S(j), V(j), Rk*Ck, nS);
end
end

function Hs = placeFootprints(F, shift, sigma, R, C)
%footprints from the consensus canvas into trial coordinates,
%trial(r + shift(1), c + shift(2)) = consensus(r, c), optionally blurred;
%returns a sparse (R*C) x nSources matrix. F.H is the sparse (rows*cols) x
%nSources footprint matrix on the canvas (size F.sz). Each source is
%interpolated on a crop of the canvas that holds every grid neighbour of its
%sample points, which gives the same values as interpolating the whole canvas.
nS = size(F.H, 2);
Rk = F.sz(1); Ck = F.sz(2);
ii = cell(nS, 1); jj = ii; vv = ii;
pad = 3 + ceil(3*sigma) + ceil(max(abs(shift)));
m = pad + ceil(max(abs(shift))) + 2; % crop margin around the footprint
for s = 1:nS
    [k, ~, w] = find(F.H(:, s));
    if isempty(k), continue, end
    [rr, cc] = ind2sub([Rk Ck], k);
    ra = max(1, min(rr) - pad):min(Rk, max(rr) + pad);
    ca = max(1, min(cc) - pad):min(Ck, max(cc) + pad);
    gr = max(1, min(rr) - m):min(Rk, max(rr) + m);
    gc = max(1, min(cc) - m):min(Ck, max(cc) + m);
    h = zeros(numel(gr), numel(gc));
    h(sub2ind(size(h), rr - gr(1) + 1, cc - gc(1) + 1)) = w;
    [CC, RR] = meshgrid(ca, ra);
    patch = interp2(gc, gr, h, CC - shift(2), RR - shift(1), 'linear', 0);
    if sigma > 0
        patch = imgaussfilt(patch, sigma);
    end
    ok = patch > 1e-6 * max(patch(:)) & RR <= R & CC <= C;
    ii{s} = sub2ind([R C], RR(ok), CC(ok)); jj{s} = repmat(s, nnz(ok), 1); vv{s} = patch(ok);
end
Hs = sparse(vertcat(ii{:}), vertcat(jj{:}), vertcat(vv{:}), R*C, nS);
end

function [cf, r] = diskFit(Hs, X80, Y1, hpf, R, C)
%slope and correlation of the movie on the activity for the flux in a disk of
%radius 4 px around each source's peak pixel (high-passed)
nS = size(Hs, 2);
[~, pk] = max(Hs, [], 1); pk = full(pk(:));
[pr, pc] = ind2sub([R C], pk);
[sr, sc] = find(reshape(any(Hs, 2), R, C));
cf = nan(nS, 1); r = cf;
for s = 1:nS
    near = hypot(sr - pr(s), sc - pc(s)) <= 4;
    in = sub2ind([R C], sr(near), sc(near));
    fy = sum(double(Y1(in, :)), 1);
    fa = full(sum(Hs(in, :), 1)) * X80;
    yh = hpf(fy); ah = hpf(fa); ok = isfinite(yh) & isfinite(ah);
    if nnz(ok) > 100 && any(ah(ok))
        cf(s) = sum(yh(ok).*ah(ok)) / sum(ah(ok).^2); r(s) = corr(yh(ok)', ah(ok)');
    end
end
end

function rho = shapeFit(Hs, X80, Y1, hpf, valid, pr, pc)
%per source, the movie-on-activity slope of the flux in the core (valid pixels
%<= 1.5 px from (pr, pc)) over that of the flux 1.5-3.5 px out (high-passed;
%NaN with fewer than 5 core or 12 outer pixels, or a slope <= 0). Scaling the
%activity scales both slopes, so the ratio measures only how the activity's
%spatial profile compares with the movie's: above 1 when the activity is too
%broad, below 1 when it is too concentrated.
[R, C] = size(valid);
[DC, DR] = meshgrid(-3:3, -3:3);
d = hypot(DR(:), DC(:));
nS = size(Hs, 2);
rho = nan(nS, 1);
for s = 1:nS
    rr = pr(s) + DR(:); cc = pc(s) + DC(:);
    in = d <= 3.5 & rr >= 1 & rr <= R & cc >= 1 & cc <= C;
    idx = sub2ind([R C], rr(in), cc(in)); ds = d(in);
    ok = valid(idx); idx = idx(ok); ds = ds(ok);
    parts = {idx(ds <= 1.5), idx(ds > 1.5)};
    if numel(parts{1}) < 5 || numel(parts{2}) < 12, continue, end
    sl = nan(1, 2);
    for k = 1:2
        fy = sum(double(Y1(parts{k}, :)), 1);
        fa = full(sum(Hs(parts{k}, :), 1)) * X80;
        yh = hpf(fy); ah = hpf(fa); okt = isfinite(yh) & isfinite(ah);
        if nnz(okt) > 100 && any(ah(okt))
            sl(k) = sum(yh(okt).*ah(okt)) / sum(ah(okt).^2);
        end
    end
    if all(sl > 0)
        rho(s) = sl(1) / sl(2);
    end
end
end

function [sl, r] = peakFit(Hs, X80, Y1, hpf)
%slope and correlation of the movie on the activity at each source's peak
%pixel (high-passed), with every source's contribution at that pixel
[~, pk] = max(Hs, [], 1); pk = full(pk(:));
A = full(Hs(pk, :)) * X80;
y = double(Y1(pk, :));
yh = hpf(y); ah = hpf(A);
ok = isfinite(yh) & isfinite(ah);
yz = yh; yz(~ok) = 0; az = ah; az(~ok) = 0;
sl = sum(yz.*az, 2) ./ sum(az.^2, 2);
r = nan(numel(pk), 1);
for i = 1:numel(pk)
    if nnz(ok(i,:)) > 100 && std(ah(i, ok(i,:))) > 0
        r(i) = corr(yh(i, ok(i,:))', ah(i, ok(i,:))');
    end
end
end

function sh = integerOffset(A, B, maxShift)
%integer (row, col) offset with A(r + sh(1), c + sh(2)) best matching B(r, c)
best = -inf; sh = [0 0];
[R, C] = size(A);
Bp = B(1:min(R, size(B,1)), 1:min(C, size(B,2)));
for dr = -maxShift:maxShift
    for dc = -maxShift:maxShift
        a = A(max(1,1+dr):min(R,R+dr), max(1,1+dc):min(C,C+dc));
        b = Bp(max(1,1-dr):min(R,R-dr), max(1,1-dc):min(C,C-dc));
        n = min(size(a), size(b)); a = a(1:n(1), 1:n(2)); b = b(1:n(1), 1:n(2));
        ok = isfinite(a) & isfinite(b);
        if nnz(ok) < 100, continue, end
        rr = corr(double(a(ok)), double(b(ok)));
        if rr > best, best = rr; sh = [dr dc]; end
    end
end
end

function n = linesPerCycle(tt, t, DMDix, rawDir)
%lines per SLAP2 acquisition cycle from the raw file header (0 if unreadable)
n = 0;
try
    fn = tt.source_extraction.fn_raw{DMDix, t};
    if ~isfile(fn)
        [~, nm, ext] = fileparts(fn);
        fn = fullfile(rawDir, [nm ext]);
    end
    S2 = slap2.Slap2DataFile(fn);
    n = double(S2.hMultiDataFiles.header.linesPerCycle);
catch ME
    warning('exportF0TrainingData:linesPerCycle', 'linesPerCycle unavailable (%s); kernel truncated at 3*dt', ME.message);
end
end

function prm = readParams(es)
names = {'alignHz', 'analyzeHz', 'numChannels', 'activityChannel', 'motionThresh', ...
    'discardInitial_s', 'nanThresh', 'photonScale'};
for k = 1:numel(names)
    prm.(names{k}) = double(h5read(es, ['/params/' names{k}]));
end
end

function writeArr(fn, name, data, chunk, deflate, sz)
%write an array (vectors as 1-D datasets); empty chunk = one chunk; sz
%(optional) gives the dataset size when trailing singleton axes must stay
if nargin < 6
    sz = size(data);
    if isvector(data)
        sz = numel(data);
    end
end
if isempty(chunk)
    chunk = sz;
end
chunk = max(1, min(chunk(1:numel(sz)), sz));
h5create(fn, name, sz, 'Datatype', class(data), 'ChunkSize', chunk, ...
    'Deflate', deflate, 'Shuffle', true);
h5write(fn, name, data);
end
