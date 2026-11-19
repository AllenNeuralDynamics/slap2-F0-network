function [rawDir, sel] = selectAcquisitionsSLAP2(dyn, linkDir)
%SELECTACQUISITIONSSLAP2  The acquisitions of a SLAP2 session usable as training data.
%   [rawDir, sel] = SELECTACQUISITIONSSLAP2(dyn, linkDir) checks every
%   acquisition in dyn (a session's slap2/dynamic_data folder) and fills
%   linkDir with links to dyn's contents, leaving out the files of excluded
%   acquisitions. buildTrialTableSLAP2 takes every .dat file directly in its
%   folder as a trial; subfolders such as reference_stack only provide the
%   reference images GIAnT needs, so they are linked as they are. rawDir is
%   linkDir, or '' when the whole session is excluded (linkDir is then not
%   made).
%
%   Excluded:
%   - the whole session, if any .meta file under dyn has an integration ROI
%     (imagingMode 'Integrate'), even alongside raster ROIs;
%   - reference stacks, whatever else the metadata says: an acquisition whose
%     name says so (refstack, ref_stack, structure, reference), whose
%     annotation says type ref_stack or a full-field raster, or whose .meta
%     is a simple raster or a stack (enableStack);
%   - acquisitions whose .meta file is missing or unreadable, since they
%     can't be checked.
%   An acquisition is all files named <acquisition>_DMD<k>...; it is excluded
%   if any of its paths is. Its -REFERENCE images stay linked.
%
%   sel: session_excluded, reason, integration_meta (the .meta files with
%   integration ROIs), and acquisitions (name, used, reasons, n_dat).
%
%   Remove linkDir with removeLinkDir, never with rmdir(..., 's'): on
%   Windows the links are junctions and hard links into the raw data.

sel = struct('session_excluded', false, 'reason', '', 'integration_meta', {{}}, 'acquisitions', {{}});
rawDir = '';

% the whole session: any integration ROI in any .meta file
metas = dir(fullfile(dyn, '**', '*.meta'));
info = containers.Map();
for k = 1:numel(metas)
    fn = fullfile(metas(k).folder, metas(k).name);
    mi = slap2MetaInfo(fn);
    info(fn) = mi;
    if mi.nIntegrate > 0
        sel.integration_meta{end+1} = sprintf('%s (%d integration ROIs)', relPath(fn, dyn), mi.nIntegrate);
    end
end
if ~isempty(sel.integration_meta)
    sel.session_excluded = true;
    sel.reason = ['integration scans in ' strjoin(sel.integration_meta, ', ')];
    return
end

% each acquisition with .dat files directly in dyn
d = dir(dyn);
d = d(~startsWith({d.name}, '.'));
files = d(~[d.isdir]);
isDat = endsWith({files.name}, '.dat', 'IgnoreCase', true);
datNames = {files(isDat).name};
datAcq = cellfun(@acqName, datNames, 'UniformOutput', false);
datStem = cellfun(@pathStem, datNames, 'UniformOutput', false);
acqs = unique(datAcq, 'stable');
excluded = {};
for a = 1:numel(acqs)
    reasons = {};
    if ~isempty(regexpi(acqs{a}, 'ref_?stack|structur|reference', 'once'))
        reasons{end+1} = 'reference stack (name)'; %#ok<AGROW>
    end
    stems = unique(datStem(strcmp(datAcq, acqs{a})), 'stable');
    for st = stems
        ann = fullfile(dyn, [st{1} '.annotation.json']);
        if exist(ann, 'file')
            try
                j = jsondecode(fileread(ann));
                if isfield(j, 'type') && strcmpi(char(j.type), 'ref_stack')
                    reasons{end+1} = sprintf('reference stack (%s: type ref_stack)', [st{1} '.annotation.json']); %#ok<AGROW>
                end
                if isfield(j, 'slap2_mode') && contains(lower(char(j.slap2_mode)), 'full-field')
                    reasons{end+1} = sprintf('reference stack (%s: %s)', [st{1} '.annotation.json'], char(j.slap2_mode)); %#ok<AGROW>
                end
            catch ME
                reasons{end+1} = sprintf('unreadable %s: %s', [st{1} '.annotation.json'], ME.message); %#ok<AGROW>
            end
        end
        mfn = fullfile(dyn, [st{1} '.meta']);
        if ~isKey(info, mfn)
            reasons{end+1} = sprintf('no %s.meta', st{1}); %#ok<AGROW>
            continue
        end
        mi = info(mfn);
        if ~isempty(mi.error)
            reasons{end+1} = sprintf('unreadable %s.meta: %s', st{1}, mi.error); %#ok<AGROW>
        elseif mi.isSimpleRaster
            reasons{end+1} = sprintf('reference stack (%s.meta: simple raster)', st{1}); %#ok<AGROW>
        elseif mi.enableStack
            reasons{end+1} = sprintf('reference stack (%s.meta: stack)', st{1}); %#ok<AGROW>
        end
    end
    used = isempty(reasons);
    if ~used, excluded{end+1} = acqs{a}; end %#ok<AGROW>
    sel.acquisitions{end+1} = struct('name', acqs{a}, 'used', used, 'reasons', {reasons}, ...
        'n_dat', nnz(strcmp(datAcq, acqs{a})));
end
if numel(excluded) == numel(acqs)
    sel.session_excluded = true;
    if isempty(acqs)
        sel.reason = 'no .dat files';
    else
        sel.reason = 'no usable acquisition';
    end
    return
end

% links to everything but the excluded acquisitions' files (their
% -REFERENCE images stay, for GIAnT's reference search)
if exist(linkDir, 'dir')
    removeLinkDir(linkDir);
end
mkdir(linkDir);
for k = 1:numel(d)
    if ~d(k).isdir && ismember(acqName(d(k).name), excluded) && ~contains(d(k).name, 'REFERENCE')
        continue
    end
    makeLink(fullfile(dyn, d(k).name), fullfile(linkDir, d(k).name), d(k).isdir);
end
rawDir = linkDir;
end

function a = acqName(name)
%the acquisition a file belongs to: its name up to _DMD<k>, else without extension
tok = regexp(name, '^(.*)_DMD\d', 'tokens', 'once');
if isempty(tok)
    [~, a] = fileparts(name);
else
    a = tok{1};
end
end

function s = pathStem(name)
%the acquisition and path: name up to and including _DMD<k>
tok = regexp(name, '^(.*_DMD\d)', 'tokens', 'once');
if isempty(tok)
    [~, s] = fileparts(name);
else
    s = tok{1};
end
end

function r = relPath(fn, root)
r = fn;
if startsWith(fn, root)
    r = strip(fn(numel(root)+1:end), 'left', filesep);
end
end

function makeLink(src, dst, isDir)
if ispc % junctions and hard links need no special rights
    if isDir
        [st, out] = system(sprintf('cmd /c mklink /J "%s" "%s"', dst, src));
    else
        [st, out] = system(sprintf('cmd /c mklink /H "%s" "%s"', dst, src));
    end
else
    [st, out] = system(sprintf('ln -s "%s" "%s"', src, dst));
end
if st ~= 0
    error('selectAcquisitionsSLAP2:link', 'Could not link %s to %s: %s', dst, src, strtrim(out));
end
end
