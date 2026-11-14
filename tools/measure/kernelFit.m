function kernelFit(trialIx)
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
addpath(genpath('C:/GitHub/GIAnT-MATLAB-pspline')); rmpath(genpath('C:/GitHub/GIAnT-MATLAB-pspline/tests')); addpath('C:/git/slap2','-begin');
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
tt = loadStructFromH5(fullfile(D, 'trial_table.h5'));
S2 = slap2.Slap2DataFile(fullfile('C:/GIAnT_testdata/780125_Neuron1_2025-03-14/raw', tt.filename{1, trialIx}));
linesPerCycle = double(S2.hMultiDataFiles.header.linesPerCycle);
fprintf('linesPerCycle %d, line rate %.0f Hz\n', linesPerCycle, 134*80);
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
cands = {'boxcar 1 frame, centred', -67, 134, 0; 'laplace dt=100', 0, 0, 100; 'laplace dt=134 (getImages)', 0, 0, 134; 'laplace dt=180', 0, 0, 180};
src = zeros(nS,1); %brightest footprint pixel of each source, within support rows
supIx = find(sup);
for s = 1:nS, [~, k] = max(Ht(supIx, s)); src(s) = k; end
yS = double(Yv(supIx(src), :));
hpf = @(x) x - movmedian(x, 81, 2, 'omitnan');
bestR = -inf;
for k = 1:size(cands,1)
    Wm = sparse(numel(lines200), T);
    for j = 1:T
        if cands{k,4} == 0
            sel = lines200 >= dsLines(j) + cands{k,2} & lines200 < dsLines(j) + cands{k,2} + cands{k,3};
            if any(sel), Wm(sel, j) = 1/nnz(sel); end
        else
            b = cands{k,4}; d = lines200 - dsLines(j); sel = abs(d) <= max(3*b, linesPerCycle);
            w = exp(-abs(d(sel))/b); Wm(sel, j) = w/sum(w);
        end
    end
    A80 = full(A200(src,:) * Wm);
    yh = hpf(yS); ah = hpf(A80); ok = isfinite(yh) & isfinite(ah);
    cf = sum(yh.*ah.*ok, 2)./sum(ah.^2.*ok, 2);
    rr = arrayfun(@(i) corr(yh(i,ok(i,:))', ah(i,ok(i,:))'), 1:nS);
    fprintf('ACTIVITY->80 Hz window "%s": movie-on-activity slope median %.2f [%.2f-%.2f], r median %.2f\n', cands{k,1}, median(cf,'omitnan'), prctile(cf,10), prctile(cf,90), median(rr,'omitnan'));
    if median(rr,'omitnan') > bestR, bestR = median(rr,'omitnan'); Wbest = Wm; kb = k; gainA = median(cf,'omitnan'); end
end
