function summarize_meta(dr, outfn)
% One JSON line per SLAP2 .meta file: ROI imaging modes, shapes and raster flags.
fid = fopen(outfn, 'w');
d = dir(fullfile(dr, '*.meta'));
for i = 1:numel(d)
    fn = fullfile(dr, d(i).name);
    o = struct('file', d(i).name);
    try
        s = load(fn, '-mat');
        ac = s.AcquisitionContainer;
        rois = {};
        if isfield(ac, 'ROIs') && isfield(ac.ROIs, 'rois')
            rois = ac.ROIs.rois;
            if ~iscell(rois), rois = num2cell(rois); end
        end
        nR = 0; nI = 0; nO = 0; areas = []; names = {}; rates = []; zs = []; shp = {};
        for r = 1:numel(rois)
            roi = rois{r};
            m = '';
            if isfield(roi, 'imagingMode'), m = roi.imagingMode; end
            if ~ischar(m), try m = char(string(m)); catch, m = ''; end, end
            if strcmpi(m, 'Raster'), nR = nR + 1;
            elseif strncmpi(m, 'Integr', 6), nI = nI + 1;
            else, nO = nO + 1;
            end
            if isfield(roi, 'name') && ischar(roi.name), names{end+1} = roi.name; end %#ok<AGROW>
            if isfield(roi, 'targetRate'), rates(end+1) = double(roi.targetRate(1)); end %#ok<AGROW>
            if isfield(roi, 'z'), zs(end+1) = double(roi.z(1)); end %#ok<AGROW>
            if isfield(roi, 'shapeData')
                sd = roi.shapeData;
                shp{end+1} = sprintf('%s%s', class(sd), mat2str(size(sd))); %#ok<AGROW>
                a = size(sd, 1);
                areas(end+1) = a; %#ok<AGROW>
            end
        end
        o.nROIs = numel(rois); o.nRaster = nR; o.nIntegrate = nI; o.nOther = nO;
        planName = 'none'; pp = [];
        if isfield(ac, 'ParsePlan') && isstruct(ac.ParsePlan), pp = ac.ParsePlan; planName = 'ParsePlan';
        elseif isfield(ac, 'AcquisitionPlan') && isstruct(ac.AcquisitionPlan), pp = ac.AcquisitionPlan; planName = 'AcquisitionPlan';
        end
        o.plan = planName;
        o.isSimpleRaster = [];
        o.rasterSizeXY = [];
        if ~isempty(pp)
            if isfield(pp, 'isSimpleRaster'), o.isSimpleRaster = logical(pp.isSimpleRaster); end
            if isfield(pp, 'rasterSizeXY'), o.rasterSizeXY = double(pp.rasterSizeXY(:)'); end
        end
        o.fovSize = [];
        if isfield(ac, 'ScannerParameters') && isfield(ac.ScannerParameters, 'fovSize')
            o.fovSize = double(ac.ScannerParameters.fovSize(:)');
        end
        o.roiPixTotal = sum(areas);
        o.roiPixMax = max([areas 0]); o.roiPixMedian = median(areas); o.shapeTypes = {};
        un = unique(names);
        o.roiNames = un(1:min(5, numel(un)));
        o.targetRates = unique(rates);
        o.nZ = numel(unique(zs));
        if isfield(s, 'acqDuration_s'), o.acqDuration_s = double(s.acqDuration_s); end
        if isfield(s, 'acqDurationCycles'), o.acqDurationCycles = double(s.acqDurationCycles); end
        if isfield(s, 'enableStack'), o.enableStack = logical(s.enableStack); end
        if isfield(s, 'channelsSave'), o.channelsSave = double(s.channelsSave(:)'); end
    catch ME
        o.error = ME.message;
    end
    fprintf(fid, '%s\n', jsonencode(o));
end
fclose(fid);
end
