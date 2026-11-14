function locbase_zdiag(trialTablePath, giantDir, DMDix, trialIx, paramsIn, outFn)
%LOCBASE_ZDIAG Residual (Z-score) offset of the two localization baselines on one trial.
%   Repeats localizeSources_vIM up to the Z-score (IMf = (IM - IMb)./stdIM)
%   for movmedian and splitFreq IMb, and reports the distribution of the
%   Z-scores over observed pixel-frames, and after the exponential matched
%   filter and DoG (the image that the activity image's maxima come from).

pathDirs = strsplit(path, pathsep);
otherGiant = pathDirs(contains(pathDirs, 'GIAnT-MATLAB'));
if ~isempty(otherGiant), rmpath(otherGiant{:}); end
addpath(genpath(giantDir));
rmpath(genpath(fullfile(giantDir, '.git')));
rmpath(genpath(fullfile(giantDir, 'tests')));
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
%load as loadAndProcessTrialAsync
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

%localizeSources_vIM up to IMb
tau = params.tau_s.*params.alignHz;
sigma = params.sigma_px;
baselineWindow = ceil(params.baselineWindow_Glu_s.*params.alignHz);
denoiseWindow = ceil(params.denoiseWindow_s.*params.alignHz);
nT = size(IM,3);
valid = mean(isnan(IM),3)<params.nanThresh;
IMf = IM; clear IM
IMf(repmat(~valid, 1, 1, nT)) = nan;
nans = isnan(IMf);
nanFrac = mean(nans,3);
vIM(nans) = nan;
IMs = smoothdata(IMf./vIM, 3, 'movmean', ceil(denoiseWindow/2), 'omitnan');
vIM = 1./smoothdata(1./vIM, 3, 'movmean', ceil(denoiseWindow/2), 'omitnan');
IMs = IMs.*vIM;

names = {'movmedian', 'splitFreq'};
res = struct();
for k = 1:2
    if k == 1
        IMb = smoothdata(IMs, 3, 'movmedian', baselineWindow, 'omitnan');
    else
        sz = size(IMs); Y = reshape(IMs, [], sz(3));
        B = nan(size(Y), 'like', Y);
        px = find(sum(~isnan(Y), 2) >= 2);
        B(px,:) = splitFreq(fillmissing(Y(px,:), 'linear', 2, 'EndValues', 'nearest'), denoiseWindow, ceil(baselineWindow/denoiseWindow));
        IMb = reshape(B, sz); clear B Y
    end
    firstValidFrames = find(any(~nans, [1 2]),500, 'first');
    varIM = var(IMs(:,:,firstValidFrames),0,3,"omitmissing");
    varIM(nanFrac>0.4) = nan;
    Vb = params.VIF*prctile(varIM, 10, 'all');
    varPred = mean(IMb(:,:,firstValidFrames),3,'omitmissing').* mean(vIM(:,:,firstValidFrames),3,'omitmissing');
    selBright = varPred>prctile(varPred(:), 90);
    Vk = prctile((varIM(selBright)-(Vb/params.VIF))./varPred(selBright), 10);
    Z = (IMf - IMb)./sqrt(max(0,Vk.*IMb.*vIM)+Vb);
    zv = Z(~nans);
    r.Vk = Vk; r.Vb = Vb;
    r.z_median = median(zv, 'omitnan');
    r.z_mean = mean(zv, 'omitnan');
    r.z_fracpos = mean(zv > 0, 'omitnan');
    r.IMb_over_IMs_median = median(IMb(~nans)./IMs(~nans), 'omitnan');
    %matched filter and DoG as localizeSources_vIM
    gamma = exp(-1/tau);
    mem = max(0,gamma*Z(:,:,end));
    for t = size(Z,3):-1:1
        Zt = Z(:,:,t); nt = isnan(Zt); Zt(nt) = mem(nt);
        Z(:,:,t) = gamma*mem + (1-gamma)*Zt; mem = Z(:,:,t);
    end
    Z(nans) = 0;
    Z = imgaussfilt(Z, [sigma sigma]);
    Z = Z - imgaussfilt(Z, 5*[sigma sigma]);
    dv = Z(~nans);
    r.dog_median = median(dv, 'omitnan');
    r.dog_mean = mean(dv, 'omitnan');
    r.dog_std = std(dv, 'omitnan');
    %per-pixel time-mean of the DoG image: spatial structure the offset creates
    pm = sum(Z.*~nans, 3)./max(1, sum(~nans, 3)); pm(~valid) = nan;
    r.dog_pixelmean_std = std(pm(valid), 'omitnan');
    r.valid_frac = mean(valid(:));
    res.(names{k}) = r;
    fprintf('%s: Z median %.3f mean %.3f frac>0 %.3f | IMb/IMs %.3f | DoG mean %.4f std %.4f, per-pixel-mean std %.4f (valid px %.2f)\n', ...
        names{k}, r.z_median, r.z_mean, r.z_fracpos, r.IMb_over_IMs_median, r.dog_mean, r.dog_std, r.dog_pixelmean_std, r.valid_frac);
    clear Z IMb
end
save(outFn, '-struct', 'res');
end
