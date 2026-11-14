function locbase_prepsim(simDir, outDir, giantDir, capsuleCodeDir, maxDuration_s, startFrame)
%LOCBASE_PREPSIM Trim and register one iGluSnFR simulation the way the integration-test capsule does.
%   trimSimulation -> buildTrialTable -> StripRegistration (capsule settings),
%   keeping the registered downsampled movie. startFrame (default 1) > 1
%   takes the window starting there instead of the first maxDuration_s
%   seconds (an extra, non-capsule window of the same simulation).
%   The trimmed crop and the registered raw movie are deleted afterwards
%   (localization reads only the downsampled movie).

if nargin < 6 || isempty(startFrame), startFrame = 1; end
if ischar(maxDuration_s), maxDuration_s = str2double(maxDuration_s); end
if ischar(startFrame), startFrame = str2double(startFrame); end

pathDirs = strsplit(path, pathsep);
otherGiant = pathDirs(contains(pathDirs, 'GIAnT-MATLAB'));
if ~isempty(otherGiant), rmpath(otherGiant{:}); end
addpath(genpath(giantDir));
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(genpath(fullfile(giantDir, '.git')));
rmpath(genpath(fullfile(giantDir, 'tests')));
addpath(capsuleCodeDir);

simFile = dir(fullfile(simDir, 'SIMULATION_*.h5'));
simFile = simFile(~contains({simFile.name}, 'groundtruth'));
simParams = jsondecode(fileread(fullfile(simDir, 'simulation_parameters.json')));
cropDir = fullfile(outDir, 'crop');
regDir = fullfile(outDir, 'out');
if ~exist(cropDir, 'dir'), mkdir(cropDir); end
if ~exist(regDir, 'dir'), mkdir(regDir); end

t = tic;
src = fullfile(simDir, simFile(1).name);
dst = fullfile(cropDir, simFile(1).name);
if startFrame == 1
    info.trim = trimSimulation(src, dst, maxDuration_s, simParams.frametime);
else
    info.trim = trimWindow(src, dst, startFrame, round(maxDuration_s / simParams.frametime));
end
info.trim.startFrame = startFrame;
info.stages.trim = toc(t);

t = tic; buildTrialTable(cropDir, regDir, false, simFile(1).name);
info.stages.trial_table = toc(t);

mcParams = struct('maxshift', 50, 'clipShift', 10, 'removeLines', 4, 'ds_time', 1, ...
    'frameRate', 1/simParams.frametime, 'overwriteExisting', true, 'saveTif', false, ...
    'nWorkers', 1);
t = tic; info.params.motion_correction = StripRegistration(fullfile(regDir, 'trial_table.h5'), mcParams);
info.stages.motion_correction = toc(t);
info.ground_truth = fullfile(simDir, strrep(simFile(1).name, '.h5', '_groundtruth.h5'));
info.frametime_s = simParams.frametime;

fid = fopen(fullfile(outDir, 'prep_info.json'), 'w');
fprintf(fid, '%s', jsonencode(info, 'PrettyPrint', true));
fclose(fid);

delete(dst);
raw = dir(fullfile(regDir, 'motion_correction', '*REGISTERED_RAW*'));
for k = 1:numel(raw)
    delete(fullfile(raw(k).folder, raw(k).name));
end
fprintf('[prepsim] %s start %d done: trim %.0f s, registration %.0f s\n', simDir, startFrame, info.stages.trim, info.stages.motion_correction);
end

function info = trimWindow(srcFile, dstFile, startFrame, nKeep)
% trimSimulation, for frames startFrame:startFrame+nKeep-1
desc = h5info(srcFile);
dsName = ['/' desc.Datasets(1).Name];
sz = desc.Datasets(1).Dataspace.Size;
nKeep = min(nKeep, sz(end) - startFrame + 1);
start = ones(1, numel(sz)); start(end) = startFrame;
count = sz; count(end) = nKeep;
mov = h5read(srcFile, dsName, start, count);
if exist(dstFile, 'file'), delete(dstFile); end
chunk = sz; chunk(end) = min(nKeep, 1000);
h5create(dstFile, dsName, count, 'Datatype', class(mov), 'ChunkSize', chunk);
h5write(dstFile, dsName, mov);
info = struct('nFramesUntrimmed', sz(end), 'nFramesKept', nKeep, 'datasetName', dsName);
fprintf('Trimmed simulation window: frames %d-%d\n', startFrame, startFrame + nKeep - 1);
end
