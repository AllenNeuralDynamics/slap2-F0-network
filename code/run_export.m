function run_export(varargin)
% RUN_EXPORT  Make F0-network training data from every SLAP2 session in dataDir.
%
%   Called by the `run` shell script on Code Ocean, e.g.
%     run_export('giantDir','/scratch/giant','maxDuration_s','180', ...)
%
%   For each attached SLAP2 session (<asset>/slap2/dynamic_data/*.dat):
%     selectAcquisitionsSLAP2 -> buildTrialTableSLAP2 ->
%     trimTrialTableSLAP2 (first maxDuration_s; 0 = all) ->
%     MultiRoiRegistration -> SILo -> exportF0TrainingData
%   with the same GIAnT settings as the GIAnT integration-test capsule,
%   except registration with the online motion correction undone
%   (compensateOnline) and sources seeded by the whitened matched filter
%   (peakMethod 'wmf', wmfPeakth). Registered movies are written as
%   deflated HDF5 (saveTif 0): they are mostly NaN outside the ROIs and
%   shrink about 9x, which matters for long sessions (GIAnT's working files
%   are several hundred GB per 30 min). selectAcquisitionsSLAP2 leaves out
%   reference stacks, and the whole session if it has any integration scan
%   (run_info.json says what was left out and why; status 'excluded').
%   Outputs go to resultsDir/<session>/: the exported trials
%   (Path<k>/trial_<NNN>.h5), export_summary.json (per-trial QC),
%   run_info.json (status, stage timings) and SiLo's trial_table.h5 and
%   experiment_summary.h5 (per_trial_summary.h5 too if keepGiantOutputs is
%   'all'). provenance.json records the GIAnT commit and environment.
%   A failure in one session is recorded and the next session still runs.

str = @(x) ischar(x) || isstring(x);
p = inputParser;
addParameter(p, 'giantDir', '/opt/GIAnT-MATLAB', str);
addParameter(p, 'giantRef', '', str);
addParameter(p, 'maxDuration_s', 1800);
addParameter(p, 'compensateOnline', 1);
addParameter(p, 'saveTif', 0);
addParameter(p, 'peakMethod', 'wmf', str);
addParameter(p, 'wmfPeakth', 3);
addParameter(p, 'datasets', 'all', str);
addParameter(p, 'nWorkers', 0);
addParameter(p, 'channels', '1', str);
addParameter(p, 'keepGiantOutputs', 'summary', str);
addParameter(p, 'dataDir', '/root/capsule/data', str);
addParameter(p, 'resultsDir', '/root/capsule/results', str);
addParameter(p, 'scratchDir', '/root/capsule/scratch', str);
parse(p, varargin{:});
opts = p.Results;
opts.maxDuration_s = num(opts.maxDuration_s);
opts.compensateOnline = logical(num(opts.compensateOnline));
opts.saveTif = logical(num(opts.saveTif));
opts.peakMethod = char(opts.peakMethod);
opts.wmfPeakth = num(opts.wmfPeakth);
opts.nWorkers = num(opts.nWorkers);
if opts.nWorkers <= 0
    opts.nWorkers = min(32, feature('numcores'));
end
opts.channels = str2num(char(opts.channels)); %#ok<ST2NM> e.g. '1' or '[1 2]'

% exactly one GIAnT checkout on the path, without its unit-test fakes
pathDirs = strsplit(path, pathsep);
other = pathDirs(contains(pathDirs, 'GIAnT-MATLAB') | contains(pathDirs, [filesep 'giant' filesep]));
if ~isempty(other), rmpath(other{:}); end
addpath(genpath(opts.giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(opts.giantDir, '.git')));
if exist(fullfile(opts.giantDir, 'tests'), 'dir')
    rmpath(genpath(fullfile(opts.giantDir, 'tests')));
end
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab'));

if ~exist(opts.resultsDir, 'dir'), mkdir(opts.resultsDir); end
prov = provenance(opts);
writeJson(fullfile(opts.resultsDir, 'provenance.json'), prov);

sessions = findSessions(opts.dataDir, opts.datasets);
fprintf('Found %d SLAP2 sessions\n', numel(sessions));
for k = 1:numel(sessions)
    processSession(sessions(k), opts, prov.giant_commit);
end
end

function sessions = findSessions(dataDir, selection)
sessions = struct('name', {}, 'path', {});
d = dir(dataDir);
d = d([d.isdir] & ~startsWith({d.name}, '.'));
selected = strtrim(strsplit(char(selection), ','));
for k = 1:numel(d)
    if ~strcmpi(selection, 'all') && ~any(strcmp(selected, d(k).name))
        continue
    end
    dyn = fullfile(dataDir, d(k).name, 'slap2', 'dynamic_data');
    if exist(dyn, 'dir') && ~isempty(dir(fullfile(dyn, '*.dat')))
        sessions(end+1) = struct('name', d(k).name, 'path', dyn); %#ok<AGROW>
    end
end
end

function processSession(s, opts, giantCommit)
outDir = fullfile(opts.resultsDir, s.name);
% GIAnT's working files (registered movies, alignment data: several GB per
% trial and path) go under /results, which has room for them, as in the
% integration-test capsule; /scratch filled up after 3 trials. They are
% deleted once the export has read them.
work = fullfile(opts.resultsDir, '_work', s.name);
linkDir = fullfile(work, 'raw'); % links to the session's usable acquisitions
removeLinkDir(linkDir); % before any recursive delete of work
for dd = {outDir, work}
    if exist(dd{1}, 'dir'), rmdir(dd{1}, 's'); end
    mkdir(dd{1});
end
info = struct('session', s.name, 'source_path', s.path, 'status', 'running', 'failed_stage', '', ...
    'error', '', 'stages', struct(), 'acquisitions', struct(), 'trim', struct(), ...
    'start_time', char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss')));
fprintf('\n===== %s =====\n', s.name);
stage = '';
try
    stage = 'select_acquisitions'; t = tic;
    [rawDir, info.acquisitions] = selectAcquisitionsSLAP2(s.path, linkDir);
    info.stages.select_acquisitions = toc(t);
    for a = 1:numel(info.acquisitions.acquisitions)
        q = info.acquisitions.acquisitions{a};
        if q.used, how = 'used'; else, how = ['left out: ' strjoin(q.reasons, '; ')]; end
        fprintf('  acquisition %s (%d .dat files): %s\n', q.name, q.n_dat, how);
    end
    if isempty(rawDir)
        info.status = 'excluded';
        fprintf('Session left out: %s\n', info.acquisitions.reason);
    else
        tt = fullfile(work, 'trial_table.h5');
        stage = 'trial_table'; t = tic;
        buildTrialTableSLAP2(rawDir, work, 1);
        info.stages.trial_table = toc(t);

        stage = 'trim'; t = tic;
        if opts.maxDuration_s > 0 && isfinite(opts.maxDuration_s)
            info.trim = trimTrialTableSLAP2(tt, opts.maxDuration_s);
        else
            info.trim = struct('maxDuration_s', 0, 'note', 'not trimmed');
        end
        info.stages.trim = toc(t);

        stage = 'motion_correction'; t = tic;
        MultiRoiRegistration(tt, struct('nWorkers', opts.nWorkers, 'overwriteExisting', true, ...
            'compensateOnline', opts.compensateOnline, 'saveTif', opts.saveTif));
        info.stages.motion_correction = toc(t);

        stage = 'source_extraction'; t = tic;
        SILo(tt, struct('isSLAP2', true, 'drawUserRois', true, 'allowAnnotationGUI', false, ...
            'nWorkers', opts.nWorkers, 'peakMethod', opts.peakMethod, 'wmfPeakth', opts.wmfPeakth));
        info.stages.source_extraction = toc(t);

        stage = 'export'; t = tic;
        info.export = exportF0TrainingData(tt, outDir, 'channels', opts.channels, ...
            'label', s.name, 'giantRef', giantCommit);
        info.stages.export = toc(t);
        info.status = 'ok';
    end
catch err
    info.status = 'failed';
    info.failed_stage = stage;
    info.error = getReport(err, 'extended', 'hyperlinks', 'off');
    fprintf(2, 'FAILED at %s: %s\n', stage, err.message);
end
info.end_time = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));

% keep SiLo's small outputs with the training data; the registered movies
% (GBs per trial) stay in scratch
keep = {'trial_table.h5', 'trial_table_untrimmed.h5', fullfile('source_extraction', 'experiment_summary.h5')};
if strcmpi(opts.keepGiantOutputs, 'all')
    keep{end+1} = fullfile('source_extraction', 'per_trial_summary.h5');
end
if ~strcmpi(opts.keepGiantOutputs, 'none')
    for k = 1:numel(keep)
        src = fullfile(work, keep{k});
        if exist(src, 'file')
            dst = fullfile(outDir, 'giant', keep{k});
            if ~exist(fileparts(dst), 'dir'), mkdir(fileparts(dst)); end
            copyfile(src, dst);
        end
    end
end
writeJson(fullfile(outDir, 'run_info.json'), info);
try
    removeLinkDir(linkDir); % the links first: rmdir(..., 's') could follow them
    rmdir(work, 's');
catch err
    fprintf(2, 'Could not clean up %s: %s\n', work, err.message);
end
if exist(fileparts(work), 'dir') && numel(dir(fileparts(work))) <= 2
    rmdir(fileparts(work));
end
end

function prov = provenance(opts)
prov = struct('giant_ref', opts.giantRef);
[~, sha] = system(sprintf('git -C "%s" rev-parse HEAD', opts.giantDir));
prov.giant_commit = strtrim(sha);
% on Code Ocean /code is not a git checkout; record the capsule and run
prov.codeocean_capsule_id = getenv('CO_CAPSULE_ID');
prov.codeocean_computation_id = getenv('CO_COMPUTATION_ID');
[st, sha] = system(sprintf('git -C "%s" rev-parse HEAD', fileparts(fileparts(mfilename('fullpath')))));
if st == 0, prov.slap2_f0_commit = strtrim(sha); else, prov.slap2_f0_commit = ''; end
prov.matlab_version = version;
prov.num_cores = feature('numcores');
prov.n_workers = opts.nWorkers;
prov.max_duration_s = opts.maxDuration_s;
prov.compensate_online = opts.compensateOnline;
prov.save_tif = opts.saveTif;
prov.peak_method = opts.peakMethod;
prov.wmf_peakth = opts.wmfPeakth;
prov.channels = opts.channels;
prov.date = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
end

function writeJson(fn, s)
try
    txt = jsonencode(s, 'PrettyPrint', true);
catch
    if isfield(s, 'export'), s.export = 'not serializable'; end
    txt = jsonencode(s, 'PrettyPrint', true);
end
fid = fopen(fn, 'w');
fprintf(fid, '%s', txt);
fclose(fid);
end

function y = num(x)
if isnumeric(x), y = x; else, y = str2double(x); end
end
