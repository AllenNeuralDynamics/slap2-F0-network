function locbase_trials(trialTablePath, outDir, giantDir, jobs, settings, paramsIn)
%LOCBASE_TRIALS Per-trial stage of SILo's localization, once per baseline setting.
%   locbase_trials(trialTablePath, outDir, giantDir, jobs, settings, paramsIn)
%   jobs     - n x 2 [DMDix trialIx]
%   settings - cellstr of localizationBaseline values, e.g. {'movmedian','splitFreq'}
%   paramsIn - struct of SILo params (as passed to SILo)
% For each job the registered movie is loaded once (as in
% loadAndProcessTrialAsync) and localizeSources_vIM is run for each setting.
% Writes outDir/trial_D<d>_T<t>.mat. Nothing is written next to the data.

setupPath(giantDir);
if nargin < 6 || isempty(paramsIn)
    paramsIn = struct();
end
if ischar(jobs)
    jobs = str2num(jobs); %#ok<ST2NM>
end
params = setParams('SILo', paramsIn);
params.minBaseline = 0.01;

dr = fileparts(trialTablePath);
trialTable = loadStructFromH5(trialTablePath);
mocodr = fullfile(dr, 'motion_correction');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

for jx = 1:size(jobs, 1)
    DMDix = jobs(jx, 1); trialIx = jobs(jx, 2);
    outFn = fullfile(outDir, sprintf('trial_D%d_T%d.mat', DMDix, trialIx));
    if exist(outFn, 'file')
        fprintf('exists, skipping: %s\n', outFn);
        continue
    end
    fnAdata = trialTable.motion_correction.fn_adata{DMDix, trialIx};
    [~, stem] = fileparts(fnAdata);
    aData0 = loadStructFromH5(fullfile(mocodr, [stem '.h5']), '/slap2/varFacDS'); %metadata only
    numChannels = aData0.numChannels;
    clear aData0
    params.numChannels = numChannels;
    [~, regStem, regExt] = fileparts(trialTable.motion_correction.fn_reg_ds{DMDix, trialIx});
    fn = [regStem regExt];
    if ~exist(fullfile(mocodr, fn), 'file')
        if exist(fullfile(mocodr, [regStem '.tif']), 'file'), fn = [regStem '.tif']; else, fn = [regStem '.h5']; end
    end
    tTrial = tic;
    R = struct();
    [R.meanIM, R.discardFrames, R.alignHz, perSetting, R.timing] = loadAndLocalize(mocodr, fn, numChannels, params, settings);
    R.settings = settings;
    R.perSetting = perSetting;
    R.fn = fn; R.DMDix = DMDix; R.trialIx = trialIx;
    R.totalTime = toc(tTrial);
    save(outFn, '-struct', 'R', '-v7.3');
    fprintf('[locbase] D%d T%d done in %.1f s (%s)\n', DMDix, trialIx, R.totalTime, ...
        strjoin(cellfun(@(s, t) sprintf('%s %.1f s', s, t), settings, num2cell(R.timing.localize), 'UniformOutput', false), ', '));
end
end

function [meanIM, discardFrames, alignHz, perSetting, timing] = loadAndLocalize(mocoDr, fn, numChannels, params, settings)
% Copy of loadAndProcessTrialAsync (GIAnT main d2fb85c), with the movie
% loaded once and localizeSources_vIM run once per baseline setting.
t0 = tic;
if endsWith(fn, '.h5')
    desc = h5info([mocoDr filesep fn]);
    IM = h5read([mocoDr filesep fn], ['/', desc.Datasets.Name]);
else
    IM = ScanImageTiffWrapper([mocoDr filesep fn]);
end
IM = reshape(IM, size(IM,1), size(IM,2), numChannels, []); %deinterleave;

meanIM= mean(IM,4, 'omitnan');
nanPx = repmat(mean(isnan(IM), [3 4])>params.nanThresh, 1,1,size(IM,3));
meanIM(nanPx) = nan;

IM = squeeze(IM(:,:,params.activityChannel,:));

%load alignment data
fnStemEnd = strfind(fn, '_REGISTERED') -1;
aData = loadStructFromH5([mocoDr filesep fn(1:fnStemEnd) '_ALIGNMENTDATA.h5']);
aData.dsFac = 1; %SLAP2 data does not have downsampling per se
params.dsFac = aData.dsFac;
params.alignHz = aData.alignHz;
alignHz = aData.alignHz;
nInitFrames = ceil(params.discardInitial_s * params.alignHz); %initial frames to discard

%discard motion frames
tmp = aData.recNegErr(:)- medfilt1(aData.recNegErr(:), round(4*params.alignHz));
tmp = -tmp./(min(-0.005, prctile(tmp,5)));
thresh = params.motionThresh;
window = 2*ceil(0.025*params.alignHz)+1;
discardFrames = imclose(imdilate(tmp>thresh, ones(window,1)) & (tmp>(thresh/2)), ones(window,1));
discardFrames(1:nInitFrames) = true;
IM(:,:,discardFrames) = nan;
if isfield(aData, 'slap2') && isfield(aData.slap2, 'varFacDS')
    vIM = aData.slap2.varFacDS;
else
    vIM = [];
end
clear aData
timing.load = toc(t0);

perSetting = struct('IMc', {}, 'peaks', {});
timing.localize = nan(1, numel(settings));
for sx = 1:numel(settings)
    p = params;
    p.localizationBaseline = settings{sx};
    t1 = tic;
    try
        [IMc, peaks] = localizeSources_vIM(IM, vIM, p);
    catch ME
        fprintf(2, 'localizeSources_vIM failed (%s); retrying with NaN borders as loadAndProcessTrialAsync does\n', ME.message);
        IM2 = IM;
        IM2([1 end],:,:) = nan;
        IM2(:,[1 end],:) = nan;
        [IMc, peaks] = localizeSources_vIM(IM2, vIM, p);
        clear IM2
    end
    timing.localize(sx) = toc(t1);
    perSetting(sx).IMc = IMc;
    perSetting(sx).peaks = peaks;
end
end

function setupPath(giantDir)
pathDirs = strsplit(path, pathsep);
otherGiant = pathDirs(contains(pathDirs, 'GIAnT-MATLAB'));
if ~isempty(otherGiant)
    rmpath(otherGiant{:});
end
addpath(genpath(giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(giantDir, '.git')));
if exist(fullfile(giantDir, 'tests'), 'dir')
    rmpath(genpath(fullfile(giantDir, 'tests')));
end
end
