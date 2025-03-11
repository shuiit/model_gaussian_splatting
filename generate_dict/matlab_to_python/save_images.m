function save_images(sp,save_path)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
frameSize = [800,1280]

for cam = 1:1:4
for frame = 1:1:length(sp{cam}.frames)
     indim = sp{cam}.frames(frame).indIm;

    if cam == 1
        indim(:,1) = 801 - sp{cam}.frames(frame).indIm(:,1);

    end

    im_name = sprintf('P%dCAM%d.mat',frame,cam);
    [Im] = ImfromSp(frameSize,indim);
    im = im2gray(Im/255/255);
    bg = Im == 0;
    im = im.*(1-bg*1);
    % imwrite(uint8(im * 255),[save_path,im_name],'Quality', 100);
    save([save_path,im_name],'im')

end
end
end