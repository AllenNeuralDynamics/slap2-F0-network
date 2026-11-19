function m = slap2MetaInfo(metaFn)
%SLAP2METAINFO  What a SLAP2 .meta file says about its acquisition.
%   m = SLAP2METAINFO(metaFn) loads the .meta file (a MAT v7.3 file) and
%   returns a struct with fields
%     nROIs, nRaster, nIntegrate, nOther  ROI counts by imagingMode
%                                         ('Raster', 'Integrate', other)
%     isSimpleRaster  the acquisition plan is a simple (full-field) raster
%     enableStack     the acquisition is a z-stack
%     error           '' or why the file could not be read
%   Reference stacks are simple rasters or stacks; multiROI recordings are
%   neither. A recording with integration ROIs has nIntegrate > 0.

m = struct('nROIs', 0, 'nRaster', 0, 'nIntegrate', 0, 'nOther', 0, ...
    'isSimpleRaster', false, 'enableStack', false, 'error', '');
try
    s = load(metaFn, '-mat');
catch ME
    m.error = ME.message;
    return
end
if ~isfield(s, 'AcquisitionContainer')
    m.error = 'no AcquisitionContainer';
    return
end
ac = s.AcquisitionContainer;
rois = {};
if isfield(ac, 'ROIs') && isstruct(ac.ROIs) && isfield(ac.ROIs, 'rois')
    rois = ac.ROIs.rois;
    if ~iscell(rois), rois = num2cell(rois); end
end
m.nROIs = numel(rois);
for r = 1:numel(rois)
    mode = '';
    if isstruct(rois{r}) && isfield(rois{r}, 'imagingMode'), mode = rois{r}.imagingMode; end
    if ~ischar(mode)
        try, mode = char(string(mode)); catch, mode = ''; end
    end
    if strcmpi(mode, 'Raster')
        m.nRaster = m.nRaster + 1;
    elseif strncmpi(mode, 'Integr', 6)
        m.nIntegrate = m.nIntegrate + 1;
    else
        m.nOther = m.nOther + 1;
    end
end
for planName = {'ParsePlan', 'AcquisitionPlan'}
    if isfield(ac, planName{1}) && isstruct(ac.(planName{1}))
        pp = ac.(planName{1});
        if isfield(pp, 'isSimpleRaster') && ~isempty(pp.isSimpleRaster)
            m.isSimpleRaster = logical(pp.isSimpleRaster(1));
        end
        break
    end
end
if isfield(s, 'enableStack') && ~isempty(s.enableStack)
    m.enableStack = logical(s.enableStack(1));
end
end
