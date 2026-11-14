% Dependencies for the GIAnT checkout. GIAnT itself is added by
% run_export, so that the requested commit is the only copy on the path.
addpath(genpath('/opt/Slap2DataReader'));
addpath(genpath('/opt/NoRMCorre'));
addpath(genpath('/opt/Fast_Tiff_Write'));

c = parcluster("Processes");
c.NumWorkers = 512;
saveProfile(c);
