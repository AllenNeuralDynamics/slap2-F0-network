function locbase_bdiag(trialTablePath, giantDir, DMDix, trialIx, paramsIn, resultsFn, photonScale)
%LOCBASE_BDIAG splitFreq vs movmedian IMb by pixel brightness, and at the selected sources.
%   Per pixel: time-median of IMb (each setting) and time-mean of IMs.
%   Reports median IMb_splitFreq / IMb_movmedian by brightness quintile of
%   the valid pixels, and at the harness's movmedian sources (resultsFn,
%   consensus frame -> this trial's frame with its motOutput).

pathDirs = strsplit(path, pathsep);
otherGiant = pathDirs(contains(pathDirs, 'GIAnT-MATLAB'));
if ~isempty(otherGiant), rmpath(otherGiant{:}); end
addpath(genpath(giantDir));
params = setParams('SILo', paramsIn);
dr = fileparts(trialTablePath);
tt = loadStructFromH5(trialTablePath);
mocodr = fullfile(dr, 'motion_correction');
[~, stem, ext] = fileparts(tt.motion_correction.fn_reg_ds{DMDix, trialIx});
fn = [stem ext];
if ~exist(fullfile(mocodr, fn), 'file')
    if exist(fullfile(mocodr, [stem '.tif']), 'file'), fn = [stem '.tif']; else, fn = [stem '.h5']; end
end
[~, adStem] = fileparts(tt.motion_correction.fn_adata{DMDix, trialIx});
a0 = loadStructFromH5(fullfile(mocodr, [adStem '.h5']), '/slap2/varFacDS');
numChannels = a0.numChannels;
if endsWith(fn, '.h5')
    desc = h5info([mocodr filesep fn]);
    IM = h5read([mocodr filesep fn], ['/', desc.Datasets.Name]);
else
    IM = ScanImageTiffWrapper([mocodr filesep fn]);
end
IM = reshape(IM, size(IM,1), size(IM,2), numChannels, []);
IM = squeeze(IM(:,:,params.activityChannel,:));
fnStemEnd = strfind(fn, '_REGISTERED') -1;
aData = loadStructFromH5([mocodr filesep fn(1:fnStemEnd) '_ALIGNMENTDATA.h5']);
params.alignHz = aData.alignHz;
tmp = aData.recNegErr(:)- medfilt1(aData.recNegErr(:), round(4*params.alignHz));
tmp = -tmp./(min(-0.005, prctile(tmp,5)));
window = 2*ceil(0.025*params.alignHz)+1;
discardFrames = imclose(imdilate(tmp>params.motionThresh, ones(window,1)) & (tmp>(params.motionThresh/2)), ones(window,1));
IM(:,:,discardFrames) = nan;
if isfield(aData, 'slap2') && isfield(aData.slap2, 'varFacDS'), vIM = aData.slap2.varFacDS; else, vIM = ones(size(IM), 'like', IM); end
clear aData
baselineWindow = ceil(params.baselineWindow_Glu_s.*params.alignHz);
denoiseWindow = ceil(params.denoiseWindow_s.*params.alignHz);
valid = mean(isnan(IM),3)<params.nanThresh;
IM(repmat(~valid, 1, 1, size(IM,3))) = nan;
nans = isnan(IM);
vIM(nans) = nan;
IMs = smoothdata(IM./vIM, 3, 'movmean', ceil(denoiseWindow/2), 'omitnan');
vIM = 1./smoothdata(1./vIM, 3, 'movmean', ceil(denoiseWindow/2), 'omitnan');
IMs = IMs.*vIM;
clear IM vIM
Bm = smoothdata(IMs, 3, 'movmedian', baselineWindow, 'omitnan');
sz = size(IMs); Y = reshape(IMs, [], sz(3));
Bs = nan(size(Y), 'like', Y);
px = find(sum(~isnan(Y), 2) >= 2);
Bs(px,:) = splitFreq(fillmissing(Y(px,:), 'linear', 2, 'EndValues', 'nearest'), denoiseWindow, ceil(baselineWindow/denoiseWindow));
Bs = reshape(Bs, sz); clear Y
Bm(nans) = nan; Bs(nans) = nan;
mBm = median(Bm, 3, 'omitnan'); mBs = median(Bs, 3, 'omitnan');
bright = mean(IMs, 3, 'omitnan');
ratio = mBs./mBm;
v = valid & isfinite(ratio) & bright > 0;
q = prctile(bright(v), [0 20 40 60 80 100]);
fprintf('denoiseWindow %d frames, LPfactor %d, photonScale %g\n', denoiseWindow, ceil(baselineWindow/denoiseWindow), photonScale);
for k = 1:5
    sel = v & bright >= q(k) & bright <= q(k+1);
    fprintf('brightness quintile %d (%.3g-%.3g, ~%.2f photons/page): median IMb_splitFreq/IMb_movmedian %.3f\n', ...
        k, q(k), q(k+1), median(bright(sel))*denoiseWindow/photonScale, median(ratio(sel)));
end
R = load(resultsFn);
P = R.(sprintf('Path%d', DMDix));
mot = P.motOutput(:, trialIx);
src = P.movmedian.sources;
r = round(src.R + mot(1)); c = round(src.C + mot(2));
ok = r >= 1 & r <= sz(1) & c >= 1 & c <= sz(2);
ix = sub2ind(sz(1:2), r(ok), c(ok));
rs = ratio(ix);
fprintf('at %d movmedian sources: median IMb_splitFreq/IMb_movmedian %.3f (p25 %.3f, p75 %.3f), brightness pct among valid px %.0f\n', ...
    sum(isfinite(rs)), median(rs, 'omitnan'), prctile(rs, 25), prctile(rs, 75), 100*mean(bright(v) < median(bright(ix), 'omitnan')));
end
