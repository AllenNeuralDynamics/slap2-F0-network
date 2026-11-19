function removeLinkDir(linkDir)
%REMOVELINKDIR  Remove a folder of links made by selectAcquisitionsSLAP2.
%   Each entry is removed without following it: symbolic links and Windows
%   junctions to folders are unlinked (never emptied), and links to files
%   (symbolic or hard) are deleted, which leaves the raw data alone. Any
%   regular file GIAnT may have written there is deleted too. Fails rather
%   than recursing into a real folder.

if ~exist(linkDir, 'dir')
    return
end
d = dir(linkDir);
d = d(~ismember({d.name}, {'.', '..'}));
for k = 1:numel(d)
    p = fullfile(linkDir, d(k).name);
    if ispc
        if d(k).isdir
            [st, out] = system(sprintf('cmd /c rmdir "%s"', p)); % a junction: removes the link only
        else
            [st, out] = system(sprintf('cmd /c del /f /q "%s"', p));
        end
    else
        [st, out] = system(sprintf('rm -f "%s"', p)); % no -r: refuses a real folder
    end
    if st ~= 0
        error('removeLinkDir:failed', 'Could not remove %s: %s', p, strtrim(out));
    end
end
rmdir(linkDir); % now empty
end
