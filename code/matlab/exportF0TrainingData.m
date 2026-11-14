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
%   plus <outDir>/export_summary.json with per-trial QC. Arrays are written
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
%            movie and activity at source peak pixels. The footprints are
%            then blurred by the Gaussian (sigma from blurSigmas) that brings
%            the movie-on-activity slope at peak pixels closest to 1: the
%            80 Hz frames are slightly blurrier than the 200 Hz data the
%            footprints were fitted to.
%   QC per trial: that slope and correlation; the same for the flux in a
%   4 px disk around each source (insensitive to blur and offset); and the
%   event-triggered residual at source peaks as a fraction of the
%   event-triggered movie (near 0 when the activity is removed).
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
%     blurSigmas  candidate blur sigmas in px (default 0:0.25:1)
%     shiftRange, shiftStep  sub-pixel offset search around the integer
%                 offset of the mean images (default 1 and 0.25 px)
%     varFac      'mean' (per-pixel mean, default) or 'full' (every frame)
%     deflate     HDF5 deflate level (default 4)
%     label       session label stored with the data (default: the name of
%                 the trial table's datadr folder)
%     giantRef    GIAnT commit that made the SiLo outputs, stored as metadata

p = inputParser;
addParameter(p, 'saveDir', '');
addParameter(p, 'rawDir', '');
addParameter(p, 'paths', []);
addParameter(p, 'trials', []);
addParameter(p, 'channels', 1);
addParameter(p, 'blurSigmas', 0:0.25:1);
addParameter(p, 'shiftRange', 1);
addParameter(p, 'shiftStep', 0.25);
addParameter(p, 'varFac', 'mean');
addParameter(p, 'deflate', 4);
addParameter(p, 'label', '');
addParameter(p, 'giantRef', '');
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
        profInfo = h5info(pts, [grp '/sources/spatial/per_trial_profiles']);
    catch
        warning('exportF0TrainingData:NoPath', 'No SiLo sources for path %d; skipped', DMDix);
        continue
    end
    fl = double(h5read(es, [grp '/frame_info/frame_line_idxs']));
    profSz = profInfo.Dataspace.Size; % [nTrials nSources 1 rows cols]
    pathDir = fullfile(outDir, sprintf('Path%d', DMDix));
    if ~exist(pathDir, 'dir')
        mkdir(pathDir);
    end
    trials = o.trials;
    if isempty(trials)
        trials = find(nf(:)' > 0);
    end
    for t = trials(:)'
        if t > numel(nf) || nf(t) == 0
            continue
        end
        fprintf('[exportF0] %s path %d trial %d\n', o.label, DMDix, t);
        try
            q = exportTrial(tt, t, DMDix, nf, fl, profSz, es, pts, grp, mocoDr, prm, o, ...
                fullfile(pathDir, sprintf('trial_%03d.h5', t)));
            summary.trials{end+1} = q;
            fprintf(['[exportF0]   peak slope %.2f (r %.2f), disk-flux slope %.2f (r %.2f), shift [%.2f %.2f], ' ...
                'blur %.2f px, event residual %.2f\n'], q.peakSlope, q.peakR, q.diskSlope, q.diskR, ...
                q.shiftRC, q.blurSigma, q.eventResidualFraction);
        catch ME
            fprintf(2, '[exportF0]   failed: %s\n', ME.message);
            summary.trials{end+1} = struct('path', DMDix, 'trial', t, 'error', ME.message);
        end
        fid = fopen(fullfile(outDir, 'export_summary.json'), 'w');
        fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
        fclose(fid);
    end
end
end

function q = exportTrial(tt, t, DMDix, nf, fl, profSz, es, pts, grp, mocoDr, prm, o, outFn)
nS = profSz(2);
%--- GIAnT outputs for this trial (200 Hz, consensus canvas)
f200 = sum(nf(1:t-1)) + (1:nf(t));
nCh200 = h5info(es, [grp '/sources/temporal/dF_denoised']).Dataspace.Size(2);
X200 = double(h5read(es, [grp '/sources/temporal/dF_denoised'], [1 1 f200(1)], [nS nCh200 nf(t)]));
lines200 = fl(f200);
disc200 = squeeze(all(isnan(X200(:,1,:)), 1));
X200(isnan(X200)) = 0;
H = h5read(pts, [grp '/sources/spatial/per_trial_profiles'], [t 1 1 1 1], [1 nS 1 profSz(4) profSz(5)]);
H = reshape(double(H), nS, profSz(4), profSz(5));
H(isnan(H)) = 0;
if ~any(H(:))
    error('no footprints for this trial');
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
    X80{ch} = squeeze(X200(:, ch, :)) * Wk; % sources x 80 Hz frames
    if nS == 1, X80{ch} = X80{ch}(:)'; end
end

%--- consensus -> trial offset and blur
Y1 = reshape(IM(:,:,prm.activityChannel,:), R*C, T);
Y1(:, discard) = nan;
meanT = mean(reshape(IM(:,:,prm.activityChannel,:), R, C, T), 3, 'omitnan');
meanC = squeeze(h5read(es, [grp '/visualizations/mean_im']));
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
bestSig = struct('d', inf);
for sg = o.blurSigmas
    Hs = placeFootprints(H, best.shift, sg, R, C);
    [sl, r] = peakFit(Hs, X80{prm.activityChannel}, Y1, hpf);
    if abs(median(sl, 'omitnan') - 1) < bestSig.d
        bestSig = struct('d', abs(median(sl, 'omitnan') - 1), 'sigma', sg, 'slope', median(sl, 'omitnan'), 'r', median(r, 'omitnan'));
    end
end
Hs = placeFootprints(H, best.shift, bestSig.sigma, R, C); % (R*C) x nS sparse
[~, pk] = max(Hs, [], 1); pk = full(pk(:));

%--- QC: disk flux and event-triggered residual
A1 = Hs * X80{prm.activityChannel}; % (R*C) x T, sparse rows outside support
[pr, pc] = ind2sub([R C], pk);
[sr, sc] = find(reshape(any(Hs, 2), R, C));
cfD = nan(nS, 1); rD = cfD;
for s = 1:nS
    in = sub2ind([R C], sr(hypot(sr - pr(s), sc - pc(s)) <= 4), sc(hypot(sr - pr(s), sc - pc(s)) <= 4));
    fy = sum(double(Y1(in, :)), 1); fa = full(sum(A1(in, :), 1));
    yh = hpf(fy); ah = hpf(fa); ok = isfinite(yh) & isfinite(ah);
    if nnz(ok) > 100 && any(ah(ok))
        cfD(s) = sum(yh(ok).*ah(ok)) / sum(ah(ok).^2); rD(s) = corr(yh(ok)', ah(ok)');
    end
end
etaY = zeros(1, 25); etaR = etaY; nEv = 0;
for s = 1:nS
    x = X80{prm.activityChannel}(s, :);
    thr = prctile(x, 99);
    ev = find(x(2:end-1) > thr & x(2:end-1) >= x(1:end-2) & x(2:end-1) > x(3:end)) + 1;
    ev = ev(ev > 8 & ev <= T - 16);
    y = double(Y1(pk(s), :)); a = full(A1(pk(s), :));
    for e = ev
        wy = y(e-8:e+16); wr = wy - a(e-8:e+16);
        if any(isnan(wy)), continue, end
        etaY = etaY + wy - mean(wy(1:6)); etaR = etaR + wr - mean(wr(1:6)); nEv = nEv + 1;
    end
end
etaFrac = max(etaR(9:13)) / max(etaY(9:13));

%--- crop and write
rows = find(any(valid, 2)); cols = find(any(valid, 1));
r0 = rows(1); r1 = rows(end); c0 = cols(1); c1 = cols(end);
Rc = r1 - r0 + 1; Cc = c1 - c0 + 1;
crop = @(M) M(r0:r1, c0:c1, :, :);
if exist(outFn, 'file'), delete(outFn); end
chans = o.channels(:)';
mov = zeros(Rc, Cc, T, numel(chans), 'single');
actv = zeros(Rc, Cc, T, numel(chans), 'single');
for k = 1:numel(chans)
    m = crop(squeeze(IM(:,:,chans(k),:)));
    m(:,:,discard) = nan;
    mov(:,:,:,k) = m;
    a = full(Hs * X80{min(chans(k), nCh200)});
    actv(:,:,:,k) = crop(reshape(single(a), R, C, T));
end
clear IM
movSz = [Cc Rc T numel(chans)]; % explicit, so a single channel keeps its axis
writeArr(outFn, '/movie', permute(mov, [2 1 3 4]), [min(64,Cc) min(64,Rc) min(256,T) 1], o.deflate, movSz);
writeArr(outFn, '/activity', permute(actv, [2 1 3 4]), [min(64,Cc) min(64,Rc) min(256,T) 1], o.deflate, movSz);
clear mov actv
labeled = medfilt2(meanT, [3 3]);
labeled = ~isnan(meanT) & labeled > 3*prctile(labeled(~isnan(labeled)), 25);
support = reshape(full(any(Hs, 2)), R, C);
writeArr(outFn, '/mask/valid', uint8(crop(valid)'), [], o.deflate);
writeArr(outFn, '/mask/labeled', uint8(crop(labeled)'), [], o.deflate);
writeArr(outFn, '/mask/footprint_support', uint8(crop(support)'), [], o.deflate);
writeArr(outFn, '/mean_image', permute(single(crop(meanAll)), [2 1 3]), [], o.deflate);
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
writeArr(outFn, '/sources/peak_row', int32(pr(:) - r0), [], o.deflate);
writeArr(outFn, '/sources/peak_col', int32(pc(:) - c0), [], o.deflate);

q = struct('path', DMDix, 'trial', t, 'file', outFn, 'frames', T, 'rows', Rc, 'cols', Cc, ...
    'cropOriginRC', [r0 c0] - 1, 'nSources', nS, 'validPixels', nnz(valid), ...
    'kernelDtLines', dt, 'kernelHalfWidthLines', half, 'linesPerCycle', lpc, ...
    'shiftRC', best.shift, 'blurSigma', bestSig.sigma, ...
    'peakSlope', bestSig.slope, 'peakR', bestSig.r, ...
    'diskSlope', median(cfD, 'omitnan'), 'diskR', median(rD, 'omitnan'), ...
    'eventResidualFraction', etaFrac, 'nEvents', nEv, 'discardedFrames', nnz(discard), ...
    'channels', chans, 'activityChannel', prm.activityChannel, 'alignHz', prm.alignHz, ...
    'photonScale', prm.photonScale, 'label', o.label, 'giantRef', o.giantRef);
for fn2 = fieldnames(q)'
    v = q.(fn2{1});
    if ~strcmp(fn2{1}, 'file')
        h5writeatt(outFn, '/', fn2{1}, v);
    end
end
end

function Hs = placeFootprints(H, shift, sigma, R, C)
%footprints from the consensus canvas into trial coordinates,
%trial(r + shift(1), c + shift(2)) = consensus(r, c), optionally blurred;
%returns a sparse (R*C) x nSources matrix
nS = size(H, 1);
ii = cell(nS, 1); jj = ii; vv = ii;
pad = 3 + ceil(3*sigma) + ceil(max(abs(shift)));
for s = 1:nS
    h = squeeze(H(s, :, :));
    [rr, cc] = find(h);
    if isempty(rr), continue, end
    ra = max(1, min(rr) - pad):min(size(h,1), max(rr) + pad);
    ca = max(1, min(cc) - pad):min(size(h,2), max(cc) + pad);
    [CC, RR] = meshgrid(ca, ra);
    patch = interp2(1:size(h,2), 1:size(h,1), h, CC - shift(2), RR - shift(1), 'linear', 0);
    if sigma > 0
        patch = imgaussfilt(patch, sigma);
    end
    ok = patch > 1e-6 * max(patch(:)) & RR <= R & CC <= C;
    ii{s} = sub2ind([R C], RR(ok), CC(ok)); jj{s} = repmat(s, nnz(ok), 1); vv{s} = patch(ok);
end
Hs = sparse(vertcat(ii{:}), vertcat(jj{:}), vertcat(vv{:}), R*C, nS);
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
