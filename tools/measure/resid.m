function resid(trialIx)
%RESID Prototype of the training target: the 80 Hz aligned movie of one
%780125 trial minus GIAnT's fitted activity (per-trial footprints x
%dF_denoised, 200 Hz -> 80 Hz), then measurements on dendrite pixels:
%  - units/timing check (regression of the movie on the activity)
%  - how much events bias the current localization baseline
%  - residual structure, and how well simple blind-spot predictors built from
%    temporal and spatial context explain the smoothed residual
if nargin < 1, trialIx = 10; end
rng(1);
D = 'C:/GIAnT_testdata/780125_Neuron1_2025-03-14/processed';
addpath(genpath('C:/GitHub/GIAnT-MATLAB-pspline'));
es = fullfile(D, 'source_extraction', 'experiment_summary.h5');
pt = fullfile(D, 'source_extraction', 'per_trial_summary.h5');
nm = sprintf('E1T%dDMD1', trialIx);

%--- movie (channel 1) and alignment data
IM = ScanImageTiffWrapper(fullfile(D, 'motion_correction', [nm '_REGISTERED_DOWNSAMPLED-80Hz.tif']));
IM = reshape(IM, size(IM,1), size(IM,2), 2, []);
Y = single(squeeze(IM(:,:,1,:))); clear IM
[R, C, T] = size(Y);
a = loadStructFromH5(fullfile(D, 'motion_correction', [nm '_ALIGNMENTDATA.h5']));
dsLines = double(a.DSframes(:));
valid = mean(isnan(Y),3) < 0.33;
mT = mean(Y, 3, 'omitnan');

%--- trial -> consensus shift, from the mean images
M = h5read(es, '/Path1/visualizations/mean_im'); mC = squeeze(M(1,1,:,:));
best = -inf;
for dr = -5:5
    for dc = -5:5
        A1 = mT(max(1,1+dr):min(R,R+dr), max(1,1+dc):min(C,C+dc));
        B1 = mC(max(1,1-dr):min(R,R-dr), max(1,1-dc):min(C,C-dc));
        ok = isfinite(A1) & isfinite(B1);
        r = corr(double(A1(ok)), double(B1(ok)));
        if r > best, best = r; sh = [dr dc]; end
    end
end
fprintf('SHIFT trial->consensus [%d %d], r = %.3f\n', sh, best);

%--- per-trial footprints into trial coordinates
P = h5read(pt, '/Path1/sources/spatial/per_trial_profiles', [trialIx 1 1 1 1], [1 Inf 1 Inf Inf]);
P = reshape(P, size(P,2), R, C); P(isnan(P)) = 0;
nS = size(P,1);
Ht = zeros(R*C, nS, 'single');
for s = 1:nS
    h = circshift(squeeze(P(s,:,:)), sh); %trial(r+dr) = consensus(r)
    Ht(:,s) = h(:);
end
sup = any(Ht > 0, 2);
Hs = sparse(double(Ht(sup,:)));

%--- activity at 200 Hz, then 80 Hz
nf = double(h5read(es, '/Path1/frame_info/trial_num_frames'));
fl = double(h5read(es, '/Path1/frame_info/frame_line_idxs'));
ix = sum(nf(1:trialIx-1)) + (1:nf(trialIx));
X = h5read(es, '/Path1/sources/temporal/dF_denoised', [1 1 ix(1)], [Inf 1 numel(ix)]);
X = double(reshape(X, nS, [])); disc200 = any(isnan(X), 1); X(isnan(X)) = 0;
lines200 = fl(ix);
A200 = Hs * X; % support pixels x 200 Hz frames
Yv = reshape(Y, R*C, T);
cands = {'1 frame, starting at DSframe', 0, 134; '1 frame, centred', -67, 134; '2 frames, centred', -134, 268; '2 frames, starting at DSframe', 0, 268};
src = zeros(nS,1); %brightest footprint pixel of each source, within support rows
supIx = find(sup);
for s = 1:nS, [~, k] = max(Ht(supIx, s)); src(s) = k; end
yS = double(Yv(supIx(src), :));
hpf = @(x) x - movmedian(x, 81, 2, 'omitnan');
bestR = -inf;
for k = 1:size(cands,1)
    Wm = sparse(numel(lines200), T);
    for j = 1:T
        sel = lines200 >= dsLines(j) + cands{k,2} & lines200 < dsLines(j) + cands{k,2} + cands{k,3};
        if any(sel), Wm(sel, j) = 1/nnz(sel); end
    end
    A80 = full(A200(src,:) * Wm);
    yh = hpf(yS); ah = hpf(A80); ok = isfinite(yh) & isfinite(ah);
    cf = sum(yh.*ah.*ok, 2)./sum(ah.^2.*ok, 2);
    rr = arrayfun(@(i) corr(yh(i,ok(i,:))', ah(i,ok(i,:))'), 1:nS);
    fprintf('ACTIVITY->80 Hz window "%s": movie-on-activity slope median %.2f [%.2f-%.2f], r median %.2f\n', cands{k,1}, median(cf,'omitnan'), prctile(cf,10), prctile(cf,90), median(rr,'omitnan'));
    if median(rr,'omitnan') > bestR, bestR = median(rr,'omitnan'); Wbest = Wm; kb = k; gainA = median(cf,'omitnan'); end
end
for L = -2:2
    A80 = circshift(full(A200(src,:) * Wbest), L, 2);
    yh = hpf(yS); ah = hpf(A80); ok = isfinite(yh) & isfinite(ah);
    rr = arrayfun(@(i) corr(yh(i,ok(i,:))', ah(i,ok(i,:))'), 1:nS);
    fprintf('  lag %+d frames: r median %.3f\n', L, median(rr,'omitnan'));
end
A80 = zeros(R*C, T, 'single'); A80(sup,:) = single(gainA * full(A200 * Wbest));
fprintf('GAIN: activity scaled by %.2f (median movie-on-activity slope) before subtracting\n', gainA);
disc80 = full(double(~disc200) * Wbest) < 0.5;
Res = Yv - A80; Res(:, disc80) = nan;
Res = reshape(Res, R, C, T); A80 = reshape(A80, R, C, T);
clear Yv A200

%--- dendrite pixels: valid and labeled (as processAllTrials_Async does)
lab = medfilt2(mT, [3 3]); lab = valid & lab > 3*prctile(lab(valid), 25);
supM = reshape(sup, R, C);
fprintf('PIXELS valid %d, labeled %d, footprint support %d (labeled & support %d)\n', nnz(valid), nnz(lab), nnz(supM), nnz(lab & supM));
activeFrac = mean(reshape(A80, R*C, T) > 0, 2);

%--- event bias of the current localization baseline (localizeSources_vIM:
%movmean over ceil(denoiseWindow/2) frames, then movmedian over the baseline
%window), from the movie vs from the residual, at each source's peak pixel
w1 = ceil(ceil(0.2*80)/2); w2 = ceil(4*80);
pk = supIx(src);
yP = double(reshape(Y, R*C, T)); yP = yP(pk, :); rP = double(reshape(Res, R*C, T)); rP = rP(pk,:);
yP(:, disc80) = nan;
bY = movmedian(movmean(yP, w1, 2, 'omitnan'), w2, 2, 'omitnan');
bR = movmedian(movmean(rP, w1, 2, 'omitnan'), w2, 2, 'omitnan');
bias = mean(bY - bR, 2, 'omitnan') ./ mean(bR, 2, 'omitnan');
aP = double(reshape(A80, R*C, T)); aP = aP(pk,:);
fprintf('EVENT BIAS of the current baseline at source peak pixels: (movie - residual)/residual median %+.1f%% [10th-90th %+.1f%% to %+.1f%%]; corr with activity level %.2f\n', ...
    100*median(bias,'omitnan'), 100*prctile(bias,10), 100*prctile(bias,90), corr(bias, mean(aP,2)./mean(bR,2,'omitnan'), 'rows','complete'));
yF = fillmissing(yP, 'linear', 2, 'EndValues', 'nearest');
bSF = splitFreqLocal(yF, 16, ceil(320/16));
biasSF = mean(bSF - bR, 2, 'omitnan') ./ mean(bR, 2, 'omitnan');
fprintf('EVENT BIAS of splitFreq (lower-envelope) baseline on the movie, vs the residual reference: median %+.1f%% [10th-90th %+.1f%% to %+.1f%%]\n', ...
    100*median(biasSF,'omitnan'), 100*prctile(biasSF,10), 100*prctile(biasSF,90));
%how often each source pixel's activity is above 10% of its F0 at 80 Hz
fprintf('ACTIVITY at source peak pixels: fraction of 80 Hz frames with activity > 10%% of baseline median %.2f; mean activity / baseline median %.3f\n', ...
    median(mean(aP > 0.1*bR, 2)), median(mean(aP,2)./mean(bR,2,'omitnan')));

%--- residual structure on labeled pixels
labIx = find(lab);
smp = labIx(randperm(numel(labIx), min(3000, numel(labIx))));
[sr, sc] = ind2sub([R C], smp);
Rv = reshape(Res, R*C, T);
rs = double(Rv(smp, :));
mu = mean(rs, 2, 'omitnan');
e = rs - movmean(rs, 17, 2, 'omitnan');
vhf = var(e, 0, 2, 'omitnan');
s = ''; for L = 1:4, s = [s sprintf(' lag%d %.2f', L, rowcorr(e(:,1:end-L), e(:,1+L:end)))]; end %#ok<AGROW>
fprintf('RESIDUAL (labeled) mean median %.1f; fast-residual temporal autocorr:%s; var/mean median %.2f\n', median(mu), s, median(vhf./mu,'omitnan'));
%smoothed residual (0.2 s), detrended by its 4 s median (what a slow baseline catches)
Rs = movmean(rs, 17, 2, 'omitnan');
slow = movmedian(Rs, 321, 2, 'omitnan');
band = Rs - slow;
nvBand = vhf * (17/16) / 17 * 2.6; %noise variance of a 17-frame mean, with the measured noise autocorrelation
fprintf('RESIDUAL band (0.2-4 s): std/mean median %.3f; noise share of band variance median %.2f\n', median(std(band,0,2,'omitnan')./mu,'omitnan'), median(nvBand./var(band,0,2,'omitnan'),'omitnan'));
fprintf('RESIDUAL slow (>4 s, movmedian of 0.2 s mean): std/mean median %.3f\n', median(std(slow,0,2,'omitnan')./mu,'omitnan'));
z = (band - mean(band,2,'omitnan'))./std(band,0,2,'omitnan'); z(isnan(z)) = 0;
Cm = (z*z')/T; Dm = hypot(sr-sr', sc-sc');
edges = [1 2 4 8 16 32 64 128 1024]; s = '';
for k = 1:numel(edges)-1
    sel = triu(Dm>=edges(k) & Dm<edges(k+1), 1);
    if nnz(sel) > 20, s = [s sprintf(' %d-%d px %.3f', edges(k), edges(k+1), mean(Cm(sel)))]; end %#ok<AGROW>
end
fprintf('RESIDUAL band spatial corr vs distance (labeled pixels):%s\n', s);
zs = (slow - mean(slow,2,'omitnan'))./std(slow,0,2,'omitnan'); zs(isnan(zs)) = 0;
Cs = (zs*zs')/T; s = '';
for k = 1:numel(edges)-1
    sel = triu(Dm>=edges(k) & Dm<edges(k+1), 1);
    if nnz(sel) > 20, s = [s sprintf(' %d-%d px %.3f', edges(k), edges(k+1), mean(Cs(sel)))]; end %#ok<AGROW>
end
fprintf('RESIDUAL slow spatial corr vs distance (labeled pixels):%s\n', s);

%--- blind-spot predictors of the smoothed residual, in z-scores
%target: each labeled pixel's 0.2 s-smoothed residual minus its 4 s moving
%median, divided by its SD. Temporal context: the same pixel at
%16 < |dt| <= 16+W frames. Spatial context: mean over labeled pixels at
%4 < d <= rOut px, same time.
[lr, lc] = ind2sub([R C], labIx);
RL = double(Rv(labIx, :));
RLs = movmean(RL, 17, 2, 'omitnan');
bL = RLs - movmedian(RLs, 321, 2, 'omitnan');
zL = (bL - mean(bL,2,'omitnan')) ./ std(bL,0,2,'omitnan');
[~, si] = ismember(smp, labIx);
zt = zL(si, :);
tgt = zt(:, 200:end-200);
cT = {}; nmT = {};
for W = [16 48 144 304]
    k = [ones(1,W) zeros(1,33) ones(1,W)]/(2*W); %blind spot |dt| <= 16
    p = conv2(1, k, fillmissing(zt, 'constant', 0), 'same') ./ conv2(1, k, double(isfinite(zt)), 'same');
    cT{end+1} = p(:, 200:end-200); nmT{end+1} = sprintf('T%d', W); %#ok<AGROW>
end
zL0 = zL; okL = isfinite(zL0); zL0(~okL) = 0;
cS = {}; nmS = {};
for rOut = [8 16 32 64 128]
    [idx, dst] = rangesearch([lr lc], [sr sc], rOut);
    ii = []; jj = [];
    for i = 1:numel(idx)
        keep = dst{i} > 4; ii = [ii; repmat(i, nnz(keep), 1)]; jj = [jj; idx{i}(keep)']; %#ok<AGROW>
    end
    Wn = sparse(ii, jj, 1, numel(smp), numel(labIx));
    p = (Wn * zL0) ./ (Wn * double(okL)); p((Wn * double(okL)) < 5) = nan;
    cS{end+1} = p(:, 200:end-200); nmS{end+1} = sprintf('S%d', rOut); %#ok<AGROW>
end
vt = tgt(:); okt = isfinite(vt);
r2 = @(Xc) fitR2(Xc, vt, okt);
noiseShare = median(nvBand ./ var(band, 0, 2, 'omitnan'), 'omitnan');
fprintf('BLIND-SPOT PREDICTION of the 0.2 s-smoothed residual (relative), pooled R^2; noise share of target variance median %.2f\n', noiseShare);
s = ''; for k = 1:numel(cT), s = [s sprintf(' %s %.3f', nmT{k}, r2({cT{k}}))]; end, fprintf('  temporal only:%s\n', s); %#ok<AGROW>
s = ''; for k = 1:numel(cS), s = [s sprintf(' %s %.3f', nmS{k}, r2({cS{k}}))]; end, fprintf('  spatial only:%s\n', s); %#ok<AGROW>
s = ''; for k = 1:numel(cS), s = [s sprintf(' T304+%s %.3f', nmS{k}, r2({cT{end}, cS{k}}))]; end, fprintf('  both:%s\n', s); %#ok<AGROW>
fprintf('  all temporal + all spatial: %.3f\n', r2([cT cS]));
end

function v = fitR2(Xc, y, ok)
Xm = cell2mat(cellfun(@(c) c(:), Xc, 'UniformOutput', false));
ok = ok & all(isfinite(Xm), 2);
b = Xm(ok,:) \ y(ok);
v = 1 - var(y(ok) - Xm(ok,:)*b) / var(y(ok));
end

function r = rowcorr(A, B)
ok = isfinite(A) & isfinite(B); A(~ok) = 0; B(~ok) = 0; n = sum(ok,2);
ma = sum(A,2)./n; mb = sum(B,2)./n;
A = (A - ma).*ok; B = (B - mb).*ok;
r = mean(sum(A.*B,2) ./ sqrt(sum(A.^2,2).*sum(B.^2,2)), 'omitnan');
end
