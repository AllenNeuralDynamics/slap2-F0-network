function info = trimTrialTableSLAP2(pathToTrialTable, maxDuration_s)
%TRIMTRIALTABLESLAP2 Keep only the first maxDuration_s seconds of a SLAP2 session.
%   info = TRIMTRIALTABLESLAP2(pathToTrialTable, maxDuration_s) rewrites the
%   trial_table.h5 written by buildTrialTableSLAP2 so that downstream stages
%   only read the first maxDuration_s seconds of imaging. Raw data are not
%   copied: GIAnT reads each analysis trial from slap2_info.first_line to
%   slap2_info.last_line, so trimming the table is enough.
%
%   Whole analysis trials are kept in order until the duration is reached;
%   the last kept trial is shortened so the total matches maxDuration_s.
%   This works the same way for triggered (one file per trial) and
%   continuous (~20 s analysis trials) acquisitions.
%
%   The untrimmed table is kept next to it as trial_table_untrimmed.h5.

trialTable = loadStructFromH5(pathToTrialTable);
[dr, stem, ext] = fileparts(pathToTrialTable);
copyfile(pathToTrialTable, fullfile(dr, [stem '_untrimmed' ext]));

linePeriod_s = readLinePeriod(fullfile(trialTable.datadr, trialTable.filename{1,1}));

firstLine = double(trialTable.slap2_info.first_line);
lastLine = double(trialTable.slap2_info.last_line);
nLines = lastLine - firstLine + 1;
trialDur_s = max(nLines, [], 1) .* linePeriod_s; % trials are shared across paths
nTrialsIn = numel(trialDur_s);

info = struct();
info.linePeriod_s = linePeriod_s;
info.maxDuration_s = maxDuration_s;
info.nTrialsUntrimmed = nTrialsIn;
info.durationUntrimmed_s = sum(trialDur_s);

startTime_s = [0 cumsum(trialDur_s(1:end-1))];
keep = find(startTime_s < maxDuration_s);
if isempty(keep)
    error('trimTrialTableSLAP2:Empty', 'No trials start before %g s.', maxDuration_s);
end

% shorten the last kept trial so the total is maxDuration_s
k = keep(end);
remaining_s = maxDuration_s - startTime_s(k);
if remaining_s < trialDur_s(k)
    keepLines = max(1, floor(remaining_s / linePeriod_s));
    lastLine(:, k) = min(lastLine(:, k), firstLine(:, k) + keepLines - 1);
end

trialTable.filename = trialTable.filename(:, keep);
trialTable.true_trial_ix = trialTable.true_trial_ix(:, keep);
trialTable.epoch = trialTable.epoch(:, keep);
s2 = trialTable.slap2_info;
s2.first_line = firstLine(:, keep);
s2.last_line = lastLine(:, keep);
for fn = {'trial_end_time_from_pc', 'trial_start_time_inferred'}
    if isfield(s2, fn{1}) && numel(s2.(fn{1})) == nTrialsIn
        s2.(fn{1}) = s2.(fn{1})(keep);
    end
end
trialTable.slap2_info = s2;

saveStructToH5(trialTable, pathToTrialTable);

nLinesKept = s2.last_line - s2.first_line + 1;
info.nTrialsKept = numel(keep);
info.durationKept_s = sum(max(nLinesKept, [], 1)) * linePeriod_s;
info.lastTrialShortened = remaining_s < trialDur_s(k);
fprintf('Trimmed trial table: kept %d of %d trials, %.1f of %.1f s\n', ...
    info.nTrialsKept, nTrialsIn, info.durationKept_s, info.durationUntrimmed_s);
end

function linePeriod_s = readLinePeriod(datFile)
hDat = slap2.Slap2DataFile(datFile);
if isprop(hDat, 'hDataFile')
    linePeriod_s = hDat.hDataFile.metaData.linePeriod_s;
elseif isprop(hDat, 'hMultiDataFiles')
    linePeriod_s = hDat.hMultiDataFiles.metaData.linePeriod_s;
else
    error('trimTrialTableSLAP2:Metadata', 'Could not read linePeriod_s from %s', datFile);
end
end
