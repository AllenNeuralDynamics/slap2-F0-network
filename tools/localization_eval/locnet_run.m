function R = locnet_run(trialFn, predFn, names, outFn, giantDir, peakths)
%LOCNET_RUN Localize one exported (or simulated) trial with several baselines.
%   R = locnet_run(trialFn, predFn, names, outFn, giantDir, peakths)
%   trialFn   trial file in the F0-network export format (movie, masks, varfac;
%             simulated trials also have truth/baseline)
%   predFn    output of `slap2-f0 predict` for that trial (baseline/<name> at
%             the valid pixels), a cell array of them (the first file holding
%             a baseline is used), or '' when only movmedian/truth are wanted
%   names     cell array: 'movmedian' (GIAnT's 4 s moving median), 'truth'
%             (simulations), or names of baselines in predFn
%   outFn     .mat file for the results
%   giantDir  GIAnT-MATLAB checkout whose SiLo parameters and getActImPeaks
%             are used (localization itself is localizeSources_ext, a copy of
%             main's localizeSources_vIM with an optional external baseline)
%   peakths   peak thresholds to sweep (default 4:1:20); the default
%             threshold (params.peakth) is always included
%   R.(name): P (sources at params.peakth), sweep (one P per peakth),
%   summary (the image getActImPeaks thresholds), seconds.

if nargin < 6 || isempty(peakths), peakths = 4:20; end
pathDirs = strsplit(path, pathsep);
other = pathDirs(contains(pathDirs, 'GIAnT-MATLAB') | contains(pathDirs, 'bench-code'));
if ~isempty(other), rmpath(other{:}); end
addpath(genpath(giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(giantDir, '.git')));
rmpath(genpath(fullfile(giantDir, 'tests')));
addpath(fileparts(mfilename('fullpath')));

params = setParams('SILo', struct('isSLAP2', true));
params.alignHz = 80;
peakths = unique([peakths(:)' params.peakth]);

% movie (1, T, R, C) in HDF5 order -> R x C x T
t0 = tic;
IM = permute(squeeze(h5read(trialFn, '/movie')), [2 1 3]);
[nr, nc, T] = size(IM);
valid = logical(h5read(trialFn, '/mask/valid')');
discard = logical(h5read(trialFn, '/frame/discard'));
IM(repmat(~valid, 1, 1, T)) = nan;
IM(:, :, discard) = nan;
vf = double(h5read(trialFn, '/varfac'));
if ndims(vf) == 3, vf = mean(permute(vf, [2 1 3]), 3, 'omitnan'); else, vf = vf'; end
vf(~valid | ~(vf > 0)) = nan;
vIM = repmat(single(vf), 1, 1, T);
% exported trials are cropped to the box around the valid pixels; SiLo sees the
% whole frame, with NaN margins, so pad (a peak's subpixel position can
% otherwise round to outside the image, which getActImPeaks' caller indexes)
pad = 16;
padIm = @(A) padarray(A, [pad pad 0], nan);
IM = padIm(IM);
vIM = padIm(vIM);
R.load_seconds = toc(t0);
R.pad = pad;
R.trial = trialFn;
R.pred = predFn;
R.peakths = peakths;
R.params = params;
R.giantDir = giantDir;
src = h5info(trialFn);
if any(strcmp({src.Groups.Name}, '/sources'))
    R.gt_row = double(h5read(trialFn, '/sources/peak_row')) + 1;  % 0-based -> MATLAB
    R.gt_col = double(h5read(trialFn, '/sources/peak_col')) + 1;
end

for k = 1:numel(names)
    name = names{k};
    IMb = [];
    switch name
        case 'movmedian'
        case 'truth'
            IMb = padIm(permute(h5read(trialFn, '/truth/baseline'), [2 1 3]));
        otherwise
            fns = cellstr(predFn);
            fn = '';
            for j = 1:numel(fns)
                try
                    h5info(fns{j}, ['/baseline/' name]);
                    fn = fns{j};
                    break
                catch
                end
            end
            if isempty(fn), error('locnet_run:noBaseline', 'no baseline/%s in the prediction files', name); end
            rows = double(h5read(fn, '/rows')) + 1;
            cols = double(h5read(fn, '/cols')) + 1;
            b = h5read(fn, ['/baseline/' name]);  % (T, N) in HDF5 -> N x T
            if isa(b, 'uint16') || ~isfloat(b), b = single(b); end
            IMb = nan(nr * nc, T, 'single');
            IMb(sub2ind([nr nc], rows, cols), :) = single(b);
            IMb = padIm(reshape(IMb, nr, nc, T));
    end
    t1 = tic;
    [~, P, summary] = localizeSources_ext(IM, vIM, params, IMb);
    P.row = P.row - pad; P.col = P.col - pad;  % back to the trial's coordinates
    summary = summary(pad + 1:end - pad, pad + 1:end - pad);
    P = rmfield(P, 'peakIM');
    sweep = cell(1, numel(peakths));
    for j = 1:numel(peakths)
        th = getActImPeaks(summary, peakths(j), [], params.minPeakDistance);
        sweep{j} = struct('row', th(:, 2), 'col', th(:, 3), 'val', th(:, 1));
    end
    R.(matlab.lang.makeValidName(name)) = struct('P', P, 'sweep', {sweep}, 'summary', summary, ...
        'seconds', toc(t1), 'name', name);
    fprintf('%s: %s: %d sources at peakth %g (%.0f s)\n', trialFn, name, numel(P.row), params.peakth, toc(t1));
    clear IMb
end
save(outFn, '-struct', 'R', '-v7.3');
% the same without images, for scoring in Python
J = rmfield(R, intersect(fieldnames(R), {'params'}));
for k = 1:numel(names)
    f = matlab.lang.makeValidName(names{k});
    J.(f) = rmfield(J.(f), 'summary');
end
J.names = names;
J.peakth_default = params.peakth;
fid = fopen(regexprep(outFn, '\.mat$', '.json'), 'w');
fwrite(fid, jsonencode(J));
fclose(fid);
end
