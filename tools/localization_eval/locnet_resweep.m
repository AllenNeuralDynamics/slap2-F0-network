function locnet_resweep(matFns, peakths, giantDir)
%LOCNET_RESWEEP Redo locnet_run's threshold sweep from its saved summary images.
%   locnet_resweep(matFns, peakths, giantDir): for each locnet_run .mat file
%   (cell array or wildcard), getActImPeaks at the given thresholds (e.g.
%   1.5:0.5:20) on each baseline's summary image; rewrites the JSON next to it.
if ischar(matFns)
    d = dir(matFns);
    matFns = fullfile({d.folder}, {d.name});
end
pathDirs = strsplit(path, pathsep);
other = pathDirs(contains(pathDirs, 'GIAnT-MATLAB') | contains(pathDirs, 'bench-code'));
if ~isempty(other), rmpath(other{:}); end
addpath(genpath(giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(giantDir, 'tests')));
for k = 1:numel(matFns)
    R = load(matFns{k});
    peakths = unique([peakths(:)' R.params.peakth]);
    names = setdiff(fieldnames(R), {'load_seconds', 'trial', 'pred', 'peakths', 'params', 'giantDir', 'gt_row', 'gt_col', 'pad'});
    J = rmfield(R, intersect(fieldnames(R), {'params'}));
    J.peakths = peakths;
    J.names = {};
    for j = 1:numel(names)
        e = R.(names{j});
        if ~isstruct(e) || ~isfield(e, 'summary'), continue, end
        sweep = cell(1, numel(peakths));
        for t = 1:numel(peakths)
            th = getActImPeaks(e.summary, peakths(t), [], R.params.minPeakDistance);
            sweep{t} = struct('row', th(:, 2), 'col', th(:, 3), 'val', th(:, 1));
        end
        e.sweep = sweep;
        J.(names{j}) = rmfield(e, 'summary');
        J.names{end + 1} = e.name;
    end
    J.peakth_default = R.params.peakth;
    fid = fopen(regexprep(matFns{k}, '\.mat$', '.json'), 'w');
    fwrite(fid, jsonencode(J));
    fclose(fid);
    fprintf('%s: %d thresholds\n', matFns{k}, numel(peakths));
end
end
