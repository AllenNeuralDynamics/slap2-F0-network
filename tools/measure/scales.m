function scales(trialName)
%SCALES Noise and background scales in one 780125 trial's 80 Hz aligned movie,
%and footprint / event scales from the GIAnT outputs, to size a blind-spot
%baseline network.
if nargin < 1, trialName = 'E1T10DMD1'; end
rng(0);
D = 'C:/GIAnT_testdata/780125_Neuron1_2025-03-14/processed';
addpath(genpath('C:/GitHub/GIAnT-MATLAB-pspline'));
es = fullfile(D, 'source_extraction', 'experiment_summary.h5');
hz = 80;

%--- footprints and sources (consensus frame)
P = h5read(es, '/Path1/sources/spatial/profiles'); P = squeeze(P); % src x rows x cols
coords = h5read(es, '/Path1/sources/spatial/coords');
nS = size(P,1); r50 = nan(nS,1); r90 = r50; r95 = r50;
[cc, rr] = meshgrid(1:size(P,3), 1:size(P,2));
for s = 1:nS
    h = squeeze(P(s,:,:)); h(isnan(h)) = 0; h = max(h,0);
    if sum(h(:)) <= 0, continue, end
    m = h(:)/sum(h(:)); r0 = sum(m.*rr(:)); c0 = sum(m.*cc(:));
    d = hypot(rr(:)-r0, cc(:)-c0); [ds, ix] = sort(d); cm = cumsum(m(ix));
    r50(s) = ds(find(cm>=0.5,1)); r90(s) = ds(find(cm>=0.9,1)); r95(s) = ds(find(cm>=0.95,1));
end
dd = squareform(pdist(coords(:,1:2))); dd(1:nS+1:end) = inf; nn = min(dd,[],2);
fprintf('FOOTPRINT radius containing 50/90/95%% of mass: median %.1f / %.1f / %.1f px (90%%: 10th-90th pct %.1f-%.1f)\n', ...
    median(r50,'omitnan'), median(r90,'omitnan'), median(r95,'omitnan'), prctile(r90,10), prctile(r90,90));
fprintf('SOURCES %d; nearest-neighbour distance median %.1f px (10th %.1f)\n', nS, median(nn), prctile(nn,10));

%--- events: rate per source and channel-2 response at channel-1 events
ev = h5read(es, '/Path1/sources/temporal/events'); % src x ch x frames (200 Hz)
den = h5read(es, '/Path1/sources/temporal/dF_denoised');
F0 = h5read(es, '/Path1/sources/temporal/F0');
e1 = squeeze(ev(:,1,:)); ok = ~isnan(e1);
thr = 3*mad(e1(ok & e1>0),1); % events well above the small-amplitude floor
rate = sum(e1>thr & ok, 2) ./ (sum(ok,2)/200);
fprintf('EVENTS (ch1, >3 MAD of positive events): rate per source median %.1f /s (10th-90th %.1f-%.1f); fraction of 200 Hz frames with an event %.3f\n', ...
    median(rate), prctile(rate,10), prctile(rate,90), median(sum(e1>thr & ok,2)./sum(ok,2)));
dff1 = squeeze(den(:,1,:)) ./ squeeze(F0(:,1,:)); dff2 = squeeze(den(:,2,:)) ./ squeeze(F0(:,2,:));
c12 = nan(nS,1);
for s = 1:nS
    o = ok(s,:) & isfinite(dff1(s,:)) & isfinite(dff2(s,:));
    if nnz(o) > 100, c12(s) = corr(dff1(s,o)', dff2(s,o)'); end
end
fprintf('dF/F (denoised): ch1 std median %.3f, ch2 std median %.3f; corr(ch1, ch2) per source median %.2f\n', ...
    median(std(dff1,0,2,'omitnan')), median(std(dff2,0,2,'omitnan')), median(c12, 'omitnan'));
%event-triggered average of ch1 and ch2 dF/F around big ch1 events
win = -20:60; eta1 = zeros(1,numel(win)); eta2 = eta1; n = 0;
for s = 1:nS
    t = find(e1(s,:) > 5*thr); t = t(t>-win(1) & t<size(e1,2)-win(end));
    for k = t
        a1 = dff1(s,k+win); a2 = dff2(s,k+win);
        if any(isnan([a1 a2])), continue, end
        eta1 = eta1 + a1 - mean(a1(1:15)); eta2 = eta2 + a2 - mean(a2(1:15)); n = n+1;
    end
end
eta1 = eta1/n; eta2 = eta2/n;
[~, p1] = max(eta1); [~, p2] = max(eta2);
k10 = find(eta1(p1:end) < 0.1*eta1(p1), 1) + p1 - 1;
fprintf('EVENT-TRIGGERED (n=%d, 200 Hz): ch1 peak dF/F %.3f at %+d ms, below 10%% of peak by %+d ms; ch2 peak %.3f at %+d ms, ch2 at +200 ms %.3f\n', ...
    n, eta1(p1), win(p1)*5, win(k10)*5, eta2(p2), win(p2)*5, eta2(win==40));

%--- the 80 Hz aligned movie
fn = [trialName '_REGISTERED_DOWNSAMPLED-80Hz.tif'];
IM = ScanImageTiffWrapper(fullfile(D, 'motion_correction', fn));
IM = reshape(IM, size(IM,1), size(IM,2), 2, []);
a = loadStructFromH5(fullfile(D, 'motion_correction', [trialName '_ALIGNMENTDATA.h5']));
vF = a.slap2.varFacDS;
Y1 = squeeze(IM(:,:,1,:)); Y2 = squeeze(IM(:,:,2,:)); clear IM
[R, C, T] = size(Y1);
valid = mean(isnan(Y1),3) < 0.33;
fprintf('MOVIE %s: %d x %d px, %d frames (%.1f s), valid pixels %d (%.0f%% of frame)\n', trialName, R, C, T, T/hz, nnz(valid), 100*mean(valid(:)));
%background pixels: valid, >10 px from every source centre, with valid pixels 3 px right/down
farMask = true(R, C);
for s = 1:nS
    farMask(max(1,round(coords(s,1))-10):min(R,round(coords(s,1))+10), max(1,round(coords(s,2))-10):min(C,round(coords(s,2))+10)) = false;
end
bg = valid & farMask; bg([1:3 end-2:end],:) = false; bg(:,[1:3 end-2:end]) = false;
bg = bg & circshift(valid,[0 -3]) & circshift(valid,[-3 0]) & circshift(valid,[-3 -3]);
bgIx = find(bg); bgIx = bgIx(randperm(numel(bgIx), min(4000, numel(bgIx))));
[br, bc] = ind2sub([R C], bgIx);
pix = @(Y, r, c) reshape(Y(sub2ind([R C], r, c) + (0:T-1)*R*C), numel(r), T); % pixels x frames
y = double(pix(Y1, br, bc));
mY = mean(y, 2, 'omitnan');
fprintf('BACKGROUND sample %d px: mean F median %.2f (10th-90th %.2f-%.2f)\n', numel(bgIx), median(mY), prctile(mY,10), prctile(mY,90));

%noise: fast residual e = Y - 0.2 s moving mean
hp = @(x) x - movmean(x, 17, 2, 'omitnan');
e = hp(y);
offs = [0 1; 1 0; 1 1; 0 2; 2 0; 0 3; 3 0; 3 3];
s = '';
for k = 1:size(offs,1)
    e2 = hp(double(pix(Y1, br+offs(k,1), bc+offs(k,2))));
    s = [s sprintf(' (%d,%d) %.3f', offs(k,1), offs(k,2), rowcorr(e, e2))]; %#ok<AGROW>
end
fprintf('NOISE spatial corr of fast residual (0.2 s high-pass) at offsets (row,col):%s\n', s);
s = '';
for L = 1:5
    s = [s sprintf(' lag %d: %.3f', L, rowcorr(e(:,1:end-L), e(:,1+L:end)))]; %#ok<AGROW>
end
fprintf('NOISE temporal autocorr of fast residual (80 Hz frames; the high-pass alone gives about -0.06):%s\n', s);
v = var(e, 0, 2, 'omitnan'); okv = isfinite(v) & mY > 0;
b = polyfit(mY(okv), v(okv), 1);
fprintf('VARIANCE vs mean (fast residual): slope %.3f, intercept %.3f; var/mean median %.2f\n', b(1), b(2), median(v(okv)./mY(okv)));
vf = double(pix(vF, br, bc));
cvv = std(vf,0,2,'omitnan')./mean(vf,2,'omitnan');
fprintf('varFacDS over time per pixel: CV median %.2f (10th-90th %.2f-%.2f); frames with varFac > 2x the pixel median: %.1f%%\n', ...
    median(cvv,'omitnan'), prctile(cvv,10), prctile(cvv,90), 100*mean(vf > 2*median(vf,2,'omitnan'), 'all', 'omitnan'));
ev2 = movmean(e.^2, 80, 2, 'omitnan'); pv = movmean(mY.*vf, 80, 2, 'omitnan');
rr2 = nan(size(e,1),1);
for i = 1:size(e,1), rr2(i) = corr(ev2(i,:)', pv(i,:)', 'rows', 'complete'); end
cvE = std(ev2,0,2,'omitnan')./mean(ev2,2,'omitnan');
fprintf('TIME-VARYING noise: CV over time of 1 s noise variance median %.2f; corr with mean*varFac median %.2f\n', median(cvE,'omitnan'), median(rr2,'omitnan'));

%background fluctuation: band component (0.2-4 s)
lp = movmean(y, 17, 2, 'omitnan');
bnd = lp - movmedian(lp, 321, 2, 'omitnan');
nv = v/17; % noise variance left after a 17-frame mean
fprintf('BAND 0.2-4 s: std/mean median %.3f; band variance / (noise variance after 0.2 s mean) median %.2f\n', median(std(bnd,0,2,'omitnan')./mY,'omitnan'), median(var(bnd,0,2,'omitnan')./nv,'omitnan'));
lags = [1 2 4 8 16 32 64 160];
s = '';
for L = lags
    s = [s sprintf(' %dfr %.2f', L, rowcorr(bnd(:,1:end-L), bnd(:,1+L:end)))]; %#ok<AGROW>
end
fprintf('BAND temporal autocorr:%s\n', s);
z = (bnd - mean(bnd,2,'omitnan'))./std(bnd,0,2,'omitnan'); z(isnan(z)) = 0;
Cm = (z*z')/T; Dm = hypot(br-br', bc-bc');
edges = [1 2 4 8 16 32 64 128 256 1024]; s = '';
for k = 1:numel(edges)-1
    sel = triu(Dm>=edges(k) & Dm<edges(k+1), 1);
    if nnz(sel) > 20, s = [s sprintf(' %d-%d px %.2f (n=%d)', edges(k), edges(k+1), mean(Cm(sel)), nnz(sel))]; end %#ok<AGROW>
end
fprintf('BAND spatial corr vs distance:%s\n', s);
ze = (e - mean(e,2,'omitnan'))./std(e,0,2,'omitnan'); ze(isnan(ze)) = 0;
Ce = (ze*ze')/T; s = '';
for k = 1:numel(edges)-1
    sel = triu(Dm>=edges(k) & Dm<edges(k+1), 1);
    if nnz(sel) > 20, s = [s sprintf(' %d-%d px %.3f', edges(k), edges(k+1), mean(Ce(sel)))]; end %#ok<AGROW>
end
fprintf('FAST (<0.2 s) spatial corr vs distance:%s\n', s);
g = mean(bnd, 1, 'omitnan');
r2g = nan(size(bnd,1),1);
for i = 1:size(bnd,1), r2g(i) = corr(bnd(i,:)', g', 'rows', 'complete')^2; end
fprintf('GLOBAL: variance of band component explained by the background-sample mean: median %.2f\n', median(r2g, 'omitnan'));
y2 = double(pix(Y2, br, bc)); lp2 = movmean(y2, 17, 2, 'omitnan'); bnd2 = lp2 - movmedian(lp2, 321, 2, 'omitnan');
e2c = hp(y2); r12 = nan(size(bnd,1),1); r12e = r12;
for i = 1:size(bnd,1)
    r12(i) = corr(bnd(i,:)', bnd2(i,:)', 'rows', 'complete');
    r12e(i) = corr(e(i,:)', e2c(i,:)', 'rows', 'complete');
end
fprintf('CHANNEL 2 same pixel: band corr median %.2f, fast-residual corr median %.3f; ch2 mean F median %.2f\n', median(r12,'omitnan'), median(r12e,'omitnan'), median(mean(y2,2,'omitnan')));
%how much band variance a linear fit explains from (a) the pixel's own band
%8 frames away on both sides, (b) the mean band of background pixels 8-32 px
%away at the same time, (c) both
r2 = nan(size(bnd,1), 3);
for i = 1:size(bnd,1)
    nb = Dm(i,:) >= 8 & Dm(i,:) < 32; if nnz(nb) < 3, continue, end
    tgt = bnd(i, 9:end-8)'; own = [bnd(i, 1:end-16)' bnd(i, 17:end)']; nbr = mean(bnd(nb, 9:end-8), 1, 'omitnan')';
    X = {[own ones(size(tgt))], [nbr ones(size(tgt))], [own nbr ones(size(tgt))]};
    for j = 1:3
        okr = all(isfinite([X{j} tgt]), 2); bb = X{j}(okr,:) \ tgt(okr);
        r2(i,j) = 1 - var(tgt(okr) - X{j}(okr,:)*bb) / var(tgt(okr));
    end
end
fprintf('PREDICT band at a background pixel (R^2, median): own trace 8 frames away (both sides) %.2f | mean of bg pixels 8-32 px away, same time %.2f | both %.2f\n', median(r2,'omitnan'));
end

function r = rowcorr(A, B)
%mean over rows of the correlation of A(i,:) with B(i,:), NaN-aware
ok = isfinite(A) & isfinite(B); A(~ok) = 0; B(~ok) = 0; n = sum(ok,2);
ma = sum(A,2)./n; mb = sum(B,2)./n;
A = (A - ma).*ok; B = (B - mb).*ok;
r = mean(sum(A.*B,2) ./ sqrt(sum(A.^2,2).*sum(B.^2,2)), 'omitnan');
end
