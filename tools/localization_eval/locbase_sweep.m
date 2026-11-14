function locbase_sweep(resultsFn, giantDir, ths, paramsIn)
%LOCBASE_SWEEP Re-pick sources from locbase_select's activity images at other peakth values.
%   Same steps as SILo after the activity image: getActImPeaks (soma ROIs
%   excluded), rounding, unique seeds, selPix/keepSources. Writes
%   <resultsFn stem>_sweep_flat.mat with <Path>_<setting>_th<k>_RCV (k = 10*peakth).

pathDirs = strsplit(path, pathsep);
otherGiant = pathDirs(contains(pathDirs, 'GIAnT-MATLAB'));
if ~isempty(otherGiant), rmpath(otherGiant{:}); end
addpath(genpath(giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(giantDir, '.git')));
rmpath(genpath(fullfile(giantDir, 'tests')));
if nargin < 4 || isempty(paramsIn), paramsIn = struct(); end
params = setParams('SILo', paramsIn);

R = load(resultsFn);
flat = struct();
for pf = fieldnames(R)'
    P = R.(pf{1});
    for sx = 1:numel(P.settings)
        sn = matlab.lang.makeValidName(P.settings{sx});
        S = P.(sn);
        actIM = S.actIM;
        mu = median(actIM, 'all', 'omitmissing');
        sig = mad(actIM(~isnan(actIM)), 1, 'all') ./ 0.6741891400433162;
        flat.([pf{1} '_' sn '_bg']) = [mu sig];
        for th = ths(:)'
            thetaf = getActImPeaks(actIM, th, S.somaMask, params.minPeakDistance);
            RCV = zeros(0, 3);
            if ~isempty(thetaf)
                RCV = [round(thetaf(:,2)) round(thetaf(:,3)) thetaf(:,1)];
                [~, ix] = unique(RCV(:,1:2), 'rows', 'stable');
                RCV = RCV(ix, :);
                sz = size(actIM);
                k = size(RCV, 1);
                selRadius = ceil(2*params.dXY);
                keep = false(k, 1);
                for i = 1:k
                    m = false(sz);
                    m(RCV(i,1), RCV(i,2)) = true;
                    m = imdilate(m, strel('disk', selRadius)) & S.pxAlwaysValid;
                    keep(i) = sum(m(:)) > 5 && S.pxAlwaysValid(RCV(i,1), RCV(i,2));
                end
                RCV = RCV(keep, :);
            end
            flat.(sprintf('%s_%s_th%d_RCV', pf{1}, sn, round(10*th))) = RCV;
            fprintf('%s %s peakth %g: %d sources\n', pf{1}, sn, th, size(RCV, 1));
        end
    end
end
flat.ths = ths(:)';
[d, n] = fileparts(resultsFn);
save(fullfile(d, [n '_sweep_flat.mat']), '-struct', 'flat', '-v7');
end
