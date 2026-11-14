function locbase_select(trialTablePath, trialsDir, outFn, giantDir, paramsIn, annotationsFn)
%LOCBASE_SELECT Cross-trial stage of SILo's localization and source selection.
%   Reads the per-trial results of locbase_trials (trialsDir/trial_D<d>_T<t>.mat)
%   and, for every imaging path and every baseline setting in them, repeats
%   SILo (GIAnT main d2fb85c, SILo.m lines ~172-290): consensus template,
%   cross-trial alignment of the mean images, averaged activity image, 5x5
%   median subtraction, getActImPeaks (soma ROIs excluded), unique seeds,
%   selPix and keepSources. The alignment uses only the mean images, so it
%   is shared by all settings. Per-trial peaks are moved into the consensus
%   frame. Saves everything to outFn.
%   annotationsFn - annotations.h5 to read user ROIs from (read only), or ''.

pathDirs = strsplit(path, pathsep);
otherGiant = pathDirs(contains(pathDirs, 'GIAnT-MATLAB'));
if ~isempty(otherGiant), rmpath(otherGiant{:}); end
addpath(genpath(giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(giantDir, '.git')));
rmpath(genpath(fullfile(giantDir, 'tests')));

if nargin < 5 || isempty(paramsIn), paramsIn = struct(); end
if nargin < 6, annotationsFn = ''; end
params = setParams('SILo', paramsIn);
params.minBaseline = 0.01;

trialTable = loadStructFromH5(trialTablePath);
nDMDs = size(trialTable.filename, 1);
nTrials = size(trialTable.filename, 2);

ROIs = [];
if ~isempty(annotationsFn) && exist(annotationsFn, 'file')
    ROIs = loadAnnotationsH5(annotationsFn);
end

out = struct();
for DMDix = 1:nDMDs
    %per-trial results; keepTrials = trials whose results exist
    keepTrials = false(1, nTrials);
    T = cell(1, nTrials);
    for trialIx = 1:nTrials
        fn = fullfile(trialsDir, sprintf('trial_D%d_T%d.mat', DMDix, trialIx));
        if exist(fn, 'file')
            T{trialIx} = load(fn);
            keepTrials(trialIx) = true;
        end
    end
    if ~any(keepTrials)
        continue
    end
    settings = T{find(keepTrials, 1)}.settings;
    numChannels = size(T{find(keepTrials, 1)}.meanIM, 3);
    mIM = cell(1, nTrials);
    for trialIx = find(keepTrials)
        mIM{trialIx} = T{trialIx}.meanIM;
    end

    %Assemble same-sized mean images (SILo uses the activity images' width;
    %they have the mean images' size)
    aIM1 = cell(1, nTrials);
    for trialIx = find(keepTrials)
        aIM1{trialIx} = T{trialIx}.perSetting(1).IMc;
    end
    szm1 = max(cellfun(@(x)size(x,1),mIM)); szm2 = max(cellfun(@(x)size(x,2), aIM1));
    meanIM = nan(szm1,szm2,numChannels, nTrials);
    for trialIx = 1:nTrials
        tmp =  mIM{trialIx};
        meanIM(1:size(tmp,1),1:size(tmp,2),:,trialIx) = tmp;
    end

    %Make template
    maxshift = 5;
    M = squeeze(sum(meanIM, 3));
    samples = find(keepTrials); samples = samples(unique(round(linspace(1,length(samples),20))));
    template = makeConsensusTemplate(M(:,:,samples), maxshift);

    %align all mean images to template (offsets do not depend on the setting)
    meanAligned = [];
    corrCoeff = nan(1,nTrials);
    motOutput = nan(2,nTrials);
    Mpad = nan([size(template) size(M,3)]);
    Mpad(maxshift+(1:size(M,1)), maxshift+(1:size(M,2)),:) = M;
    fillval = min(template(:),[], 'omitnan')-1;
    tFFT = fft2(max(template, fillval));
    [rr,cc] = ndgrid(1:size(meanIM,1), 1:size(meanIM,2));
    aligned = false(1, nTrials);
    for trialIx = nTrials:-1:1
        if ~keepTrials(trialIx) || all(isnan(aIM1{trialIx}), 'all')
            continue
        end
        output1 = dftregistration_clipped(tFFT, fft2(max(Mpad(:,:,trialIx), fillval)),1,80);
        mot1 = [-output1(3) -output1(4)];
        [motOutput(:,trialIx), corrCoeff(trialIx)] = xcorr2_nans(Mpad(:,:,trialIx), template, round(mot1'), maxshift);
        for chIx = 1:size(meanIM,3)
            meanAligned(:,:,chIx,trialIx) = interp2(meanIM(:,:,chIx,trialIx), cc+motOutput(2,trialIx), rr+motOutput(1,trialIx)); %#ok<AGROW>
        end
        aligned(trialIx) = true;
    end
    if size(meanAligned, 4) < nTrials
        meanAligned(:,:,:,end+1:nTrials) = 0;
    end

    pathOut = struct();
    pathOut.motOutput = motOutput;
    pathOut.corrCoeff = corrCoeff;
    pathOut.keepTrials = keepTrials;
    pathOut.meanIM = [];
    pathOut.settings = settings;

    for sx = 1:numel(settings)
        activIM = nan(szm1,szm2,1, nTrials);
        for trialIx = find(keepTrials)
            tmp = T{trialIx}.perSetting(sx).IMc;
            activIM(1:size(tmp,1),1:size(tmp,2),:,trialIx) = tmp;
        end
        actAligned = nan(size(meanIM,1), size(meanIM,2),1,nTrials);
        for trialIx = find(aligned)
            actAligned(:,:,1,trialIx) = interp2(activIM(:,:,1,trialIx), cc+motOutput(2,trialIx), rr+motOutput(1,trialIx));
        end

        %identify outliers in alignment quality to determine valid trials
        corrThresh = min(0.90, median(corrCoeff, 'omitnan')-2*std(corrCoeff, 'omitmissing'));
        actValidPix = squeeze(mean(~isnan(actAligned(:,:,1,:)), [1 2]));
        validTrials= find(corrCoeff(:)>corrThresh & actValidPix(:)>mean(actValidPix)/2);
        if isempty(pathOut.meanIM)
            pathOut.meanIM = mean(meanAligned(:,:,:,validTrials),4, 'omitnan');
        end

        %select sources on aligned activity image
        actIMraw = mean(actAligned(:,:,:,validTrials), 4, 'includenan');
        normFac = 10^(floor(log10(max(actIMraw(:))))-1);
        actIM = actIMraw ./ normFac;
        nanFrac = mean(isnan(actAligned(:,:,:,validTrials)), 4);
        actIM(nanFrac>params.nanThresh) = nan;
        medIM = nanmedfilt2(actIM, 5.*[1 1]);
        actIM = actIM-medIM; %subtract a local baseline
        sz = size(actIM);

        %user ROIs into the consensus frame
        somaMask = false(size(actIM));
        if ~isempty(ROIs) && DMDix <= numel(ROIs)
            R1 = roisToConsensusFrame(ROIs(DMDix), trialTable.motion_correction.fn_reg_ds(DMDix,:), motOutput, sz(1:2));
            for rix = 1:numel(R1.roiData)
                if contains(upper(R1.roiData{rix}.Label), 'SOMA')
                    tmp = R1.roiData{rix}.mask;
                    somaMask(1:size(tmp,1), 1:size(tmp,2)) = somaMask(1:size(tmp,1), 1:size(tmp,2)) | tmp;
                end
            end
        end

        thetaf = getActImPeaks(actIM,params.peakth,somaMask,params.minPeakDistance);

        sources = struct('R', [], 'C', [], 'V', []);
        totalPix = sum(~isnan(actIM(:)) & ~somaMask(:));
        if totalPix == 0 || isempty(thetaf)
            k = 0;
        else
            sources.R = round(thetaf(:,2));
            sources.C = round(thetaf(:,3));
            sources.V = thetaf(:,1);
            [~, uniqueIx] = unique([sources.R(:) sources.C(:)], 'rows', 'stable');
            sources.R = sources.R(uniqueIx);
            sources.C = sources.C(uniqueIx);
            sources.V = sources.V(uniqueIx);
            k = length(sources.R);
        end
        nPeaksBeforePrune = k;

        selPix = false([sz(1:2) k]);
        selRadius = ceil(2*params.dXY);
        for sourceIx = k:-1:1
            r0 = round(sources.R(sourceIx));
            c0 = round(sources.C(sourceIx));
            selPix(r0,c0,sourceIx) = true;
            selPix(:,:,sourceIx) = imdilate(selPix(:,:,sourceIx), strel('disk',selRadius));
        end
        pxAlwaysValid = mean(isnan(meanAligned(:,:,1,validTrials)),4)<params.nanThresh;
        selPix = selPix & repmat(pxAlwaysValid, 1, 1, k);
        centerValid = pxAlwaysValid(sub2ind(size(pxAlwaysValid), sources.R, sources.C));
        keepSources = squeeze(sum(selPix, [1 2]))>5 & centerValid(:);
        if k > 0
            sources.R = sources.R(keepSources);
            sources.C = sources.C(keepSources);
            sources.V = sources.V(keepSources);
        end
        fprintf('Path %d, %s: %d sources (%d peaks before pruning), normFac %g\n', DMDix, settings{sx}, numel(sources.R), nPeaksBeforePrune, normFac);

        %per-trial peaks in the consensus frame: aligned(r,c) = trial(r+mot(1), c+mot(2))
        perTrialPeaks = cell(1, nTrials);
        for trialIx = find(aligned)
            P = T{trialIx}.perSetting(sx).peaks;
            if isempty(P) || ~isfield(P, 'row')
                perTrialPeaks{trialIx} = zeros(0, 3);
                continue
            end
            perTrialPeaks{trialIx} = [P.row(:) - motOutput(1,trialIx), P.col(:) - motOutput(2,trialIx), P.val(:)];
        end

        S = struct();
        S.sources = sources;
        S.nPeaksBeforePrune = nPeaksBeforePrune;
        S.actIM = actIM;
        S.actIMraw = actIMraw;
        S.normFac = normFac;
        S.validTrials = validTrials;
        S.somaMask = somaMask;
        S.pxAlwaysValid = pxAlwaysValid;
        S.perTrialPeaks = perTrialPeaks;
        S.localizeTime = cellfun(@(t) t.timing.localize(sx), T(keepTrials));
        pathOut.(matlab.lang.makeValidName(settings{sx})) = S;
    end
    out.(sprintf('Path%d', DMDix)) = pathOut;
end
save(outFn, '-struct', 'out', '-v7.3');
fprintf('saved %s\n', outFn);

%flat copy for python (scipy.io.loadmat): 1-indexed MATLAB coordinates
flat = struct();
for pf = fieldnames(out)'
    P = out.(pf{1});
    flat.([pf{1} '_meanIM']) = P.meanIM;
    flat.([pf{1} '_motOutput']) = P.motOutput;
    flat.([pf{1} '_corrCoeff']) = P.corrCoeff;
    for sx = 1:numel(P.settings)
        sn = matlab.lang.makeValidName(P.settings{sx});
        S = P.(sn);
        pre = [pf{1} '_' sn];
        flat.([pre '_RCV']) = [S.sources.R(:) S.sources.C(:) S.sources.V(:)];
        flat.([pre '_actIM']) = S.actIM;
        flat.([pre '_actIMraw']) = S.actIMraw;
        flat.([pre '_normFac']) = S.normFac;
        flat.([pre '_validTrials']) = S.validTrials(:)';
        flat.([pre '_nPeaksBeforePrune']) = S.nPeaksBeforePrune;
        pk = zeros(0, 4);
        for trialIx = 1:numel(S.perTrialPeaks)
            if ~isempty(S.perTrialPeaks{trialIx})
                pk = [pk; repmat(trialIx, size(S.perTrialPeaks{trialIx}, 1), 1) S.perTrialPeaks{trialIx}]; %#ok<AGROW>
            end
        end
        flat.([pre '_peaks']) = pk;
        flat.([pre '_localizeTime']) = S.localizeTime;
    end
end
[d, n] = fileparts(outFn);
save(fullfile(d, [n '_flat.mat']), '-struct', 'flat', '-v7');
end
