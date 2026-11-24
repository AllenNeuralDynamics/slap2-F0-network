function q = ringEventRatios(Y, actAt, meanA, valid, pr, pc, discard, bounds)
%RINGEVENTRATIOS Activity / movie event amplitude near sources, with events picked on the movie 3-5 px away.
%
%   q = ringEventRatios(Y, actAt, meanA, valid, pr, pc, discard, bounds)
%
%   A selection-safe comparison of GIAnT's activity with the movie. Events
%   are picked on the movie in a ring 3-5 px from each source's peak pixel:
%   pixels that carry almost none of the source's footprint, so neither the
%   core's photon noise nor the activity estimate's noise selects them.
%   (This selection favours spatially broad events.) At each event the
%   amplitude is the mean of frames -1..+1 minus the mean of frames -8..-3,
%   for the movie and for the activity, at every valid pixel of the box.
%
%   Inputs:
%     Y        (R*C) x T movie, NaN on discarded frames
%     actAt    function handle: actAt(idx) is the activity at the linear
%              pixel indices idx, numel(idx) x T
%     meanA    (R*C) x 1 time-mean of the activity (all frames)
%     valid    R x C logical
%     pr, pc   the sources' peak pixels (1-based rows and columns)
%     discard  1 x T logical, discarded frames
%     bounds   [r0 r1 c0 c1]: a source counts only if its 17 x 17 px box
%              lies inside these rows and columns (the export's crop);
%              default the whole grid
%
%   Returns a struct:
%     core, near   summed activity amplitude over summed movie amplitude, at
%                  <= 1.5 px and 1.5-2.5 px from the peak (NaN without events)
%     nSources     sources with at least 3 ring events
%     nEvents      ring events over those sources
%     nArtifactSources  sources left out because a pixel within 4 px is a
%                  GIAnT fit artifact (mean activity > 3 x max(mean movie, 1))
%
%   Sources are left out when their peak pixel is not valid, the box leaves
%   bounds, or fewer than 12 ring / 5 core pixels are valid. A ring event is
%   a local maximum, at frames 161..T-160, of the ring's mean (frames with a
%   NaN ring pixel left out) minus its 81-frame running median, above that
%   signal's 99th percentile, with no discarded frame within -8..+16.
%   This is the reference analysis ring_eta.py (activity-fit plan, 2026-11-24)
%   ported line for line.

[R, C] = size(valid);
T = size(Y, 2);
if nargin < 8 || isempty(bounds)
    bounds = [1 R 1 C];
end
RAD = 8; EDGE = 160;
[DC, DR] = meshgrid(-RAD:RAD, -RAD:RAD);
dist = hypot(DR(:), DC(:));
discard = logical(discard(:)');
mm = mean(Y, 2, 'omitnan');
bad = ~isnan(mm) & meanA(:) > 3 * max(mm, 1);
numM = zeros(1, 2); numA = zeros(1, 2);
bins = [0 1.5; 1.5 2.5];
q = struct('core', NaN, 'near', NaN, 'nSources', 0, 'nEvents', 0, 'nArtifactSources', 0);
for s = 1:numel(pr)
    r0 = pr(s); c0 = pc(s);
    if r0 - RAD < bounds(1) || r0 + RAD > bounds(2) || c0 - RAD < bounds(3) || c0 + RAD > bounds(4) || ~valid(r0, c0)
        continue
    end
    idx = sub2ind([R C], r0 + DR(:), c0 + DC(:));
    v = valid(idx);
    if any(bad(idx(dist <= 4)))
        q.nArtifactSources = q.nArtifactSources + 1;
        continue
    end
    ring = v & dist >= 3 & dist <= 5;
    core = v & dist <= 1.5;
    if nnz(ring) < 12 || nnz(core) < 5
        continue
    end
    mr = double(Y(idx(ring), :));
    sel = mean(mr, 1);
    sel(any(isnan(mr), 1)) = NaN;
    filled = sel;
    filled(isnan(filled)) = median(sel, 'omitnan');
    sel = sel - medianNearest(filled, 81);
    thr = percentileLinear(sel(~isnan(sel)), 99);
    ev = [];
    for e = EDGE + 1:T - EDGE
        if sel(e) > thr && sel(e) >= sel(e-1) && sel(e) > sel(e+1) && ~any(discard(e-8:e+16))
            ev(end+1) = e; %#ok<AGROW>
        end
    end
    if numel(ev) < 3
        continue
    end
    Mb = double(Y(idx, :));
    Ab = double(actAt(idx));
    for e = ev
        am = mean(Mb(:, e-1:e+1), 2) - mean(Mb(:, e-8:e-3), 2);
        aa = mean(Ab(:, e-1:e+1), 2) - mean(Ab(:, e-8:e-3), 2);
        okp = v & isfinite(am);
        for i = 1:2
            k = okp & dist >= bins(i, 1) & dist < bins(i, 2);
            numM(i) = numM(i) + sum(am(k)); numA(i) = numA(i) + sum(aa(k));
        end
    end
    q.nSources = q.nSources + 1;
    q.nEvents = q.nEvents + numel(ev);
end
if q.nEvents > 0
    q.core = numA(1) / numM(1);
    q.near = numA(2) / numM(2);
end
end

function m = medianNearest(x, w)
%running median of width w (odd), the ends padded with the end values
%(scipy.ndimage.median_filter, mode 'nearest')
h = (w - 1) / 2;
x = x(:)';
m = movmedian([repmat(x(1), 1, h) x repmat(x(end), 1, h)], w, 'Endpoints', 'discard');
end

function v = percentileLinear(x, p)
%percentile with linear interpolation between order statistics (numpy's default)
x = sort(x(:));
n = numel(x);
pos = (n - 1) * p / 100 + 1;
lo = floor(pos); hi = min(lo + 1, n);
v = x(lo) + (pos - lo) * (x(hi) - x(lo));
end
