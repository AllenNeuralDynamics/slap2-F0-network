function spatialFit(trialIx)
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
%getImages kernel (two-sided exponential, dt = 134 lines) from 200 Hz to 80 Hz
b = 134; Wm = sparse(numel(lines200), T);
for j = 1:T
    d = lines200 - dsLines(j); sel = abs(d) <= max(3*b, linesPerCycle);
    w = exp(-abs(d(sel))/b); Wm(sel, j) = w/sum(w);
end
X80 = X * Wm; % sources x 80 Hz frames
hpf = @(x) x - movmedian(x, 81, 2, 'omitnan');
Yv = reshape(Y, R*C, T);
P2 = permute(P, [2 3 1]); % R x C x nS (consensus)
[cc0, rr0] = meshgrid(1:C, 1:R);
res = [];
for dr = 0.5:0.25:1.5
    for dc = -0.5:0.25:0.5
        Hs2 = zeros(R, C, nS, 'single');
        for s = 1:nS, Hs2(:,:,s) = interp2(P2(:,:,s), cc0 - dc, rr0 - dr, 'linear', 0); end
        Hm = reshape(Hs2, R*C, nS);
        [~, pk] = max(Hm, [], 1);
        A = double(Hm(pk, :)) * X80; y = double(Yv(pk, :));
        yh = hpf(y); ah = hpf(A); ok = isfinite(yh) & isfinite(ah);
        cf = sum(yh.*ah.*ok, 2)./sum(ah.^2.*ok, 2);
        rr = arrayfun(@(i) corr(yh(i,ok(i,:))', ah(i,ok(i,:))'), 1:nS);
        res(end+1,:) = [dr dc median(cf,'omitnan') median(rr,'omitnan')]; %#ok<AGROW>
    end
end
[~, ib] = max(res(:,4));
fprintf('SUBPIXEL shift search (peak pixel, getImages kernel): best [%.2f %.2f] slope %.2f r %.3f; integer [1 0]: slope %.2f r %.3f\n', res(ib,:), res(res(:,1)==1 & res(:,2)==0, 3:4));
%flux in a disk of radius 4 around each peak, at the best shift
dr = res(ib,1); dc = res(ib,2);
Hs2 = zeros(R, C, nS, 'single');
for s = 1:nS, Hs2(:,:,s) = interp2(P2(:,:,s), cc0 - dc, rr0 - dr, 'linear', 0); end
Hm = reshape(Hs2, R*C, nS); [~, pk] = max(Hm, [], 1);
supIx = find(any(Hm > 0, 2)); Hsup = sparse(double(Hm(supIx,:)));
Asup = Hsup * X80; Ysup = double(Yv(supIx, :));
[sr, sc] = ind2sub([R C], supIx); [pr, pc] = ind2sub([R C], pk(:));
cfD = nan(nS,1); rD = cfD; cfB = cfD; rB = cfD;
for s = 1:nS
    in = hypot(sr - pr(s), sc - pc(s)) <= 4;
    fy = sum(Ysup(in,:), 1, 'omitnan'); fa = sum(Asup(in,:), 1);
    yh = hpf(fy); ah = hpf(fa); ok = isfinite(yh) & isfinite(ah) & all(isfinite(Ysup(in,:)),1);
    cfD(s) = sum(yh(ok).*ah(ok))/sum(ah(ok).^2); rD(s) = corr(yh(ok)', ah(ok)');
end
fprintf('DISK FLUX (radius 4 px) slope median %.2f [%.2f-%.2f], r median %.3f\n', median(cfD,'omitnan'), prctile(cfD,10), prctile(cfD,90), median(rD,'omitnan'));
%peak pixel with the activity blurred spatially by a Gaussian
for sg = [0.5 0.75 1.0]
    A = zeros(nS, T);
    for s = 1:nS
        h = imgaussfilt(Hs2(:,:,s), sg); A(s,:) = double(h(pk(s))) * X80(s,:);
        others = setdiff(1:nS, s);
    end
    %include all sources' blurred contributions at each peak
    Hb = zeros(R*C, nS, 'single'); for s = 1:nS, h = imgaussfilt(Hs2(:,:,s), sg); Hb(:,s) = h(:); end
    A = double(Hb(pk, :)) * X80; y = double(Yv(pk, :));
    yh = hpf(y); ah = hpf(A); ok = isfinite(yh) & isfinite(ah);
    cf = sum(yh.*ah.*ok, 2)./sum(ah.^2.*ok, 2); rr = arrayfun(@(i) corr(yh(i,ok(i,:))', ah(i,ok(i,:))'), 1:nS);
    fprintf('BLUR sigma %.2f px: peak-pixel slope median %.2f, r median %.3f\n', sg, median(cf,'omitnan'), median(rr,'omitnan'));
end
end
