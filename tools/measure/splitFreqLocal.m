function [LP,HP] = splitFreqLocal(A, denoiseWindow, LPfactor)
nPages= floor(size(A,2)./denoiseWindow);
totSamps = nPages*denoiseWindow;
t = 1:size(A,2);

if nPages < 2 %trial too short to interpolate between pages; use a constant low-pass component
    LP = repmat(mean(A,2,'omitmissing'), 1, size(A,2));
    HP = A-LP;
    return
end

a = reshape(A(:,1:totSamps), size(A,1), denoiseWindow, nPages); % #pixels x #samps/page x #pages
Ma = squeeze(mean(a,2, 'omitmissing'));
SMa = smoothdata(Ma,2, 'lowess',LPfactor, 'omitmissing');
resid = Ma-SMa;
lowVals = resid<=ordfilt2(resid, max(2,ceil(0.15*LPfactor)), ones([1 LPfactor]));
Ma(~lowVals) = nan;
SMa = smoothdata(Ma,2, 'movmean',LPfactor, 'omitmissing');
for iter = 1:3
    selNans = isnan(SMa);
    if any(selNans(:))
        tmp = smoothdata(SMa,2, 'movmean',LPfactor, 'omitmissing');
        SMa(selNans) = tmp(selNans); 
    else
        break
    end
end
tDS = (denoiseWindow+1)/2 + (denoiseWindow).*(0:size(SMa,2)-1);
LP = nan(size(A));
for pxIx = 1:size(LP,1)
    LP(pxIx,:) = interp1(tDS,SMa(pxIx,:)',t,'linear','extrap');
end
HP = A-LP;
end

