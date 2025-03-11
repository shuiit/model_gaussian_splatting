function [Im] = ImfromSp(frameSize,indIm)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
Im=zeros(frameSize);
IndIm=sub2ind(frameSize,indIm(:,1),indIm(:,2));
if size(indIm,2) == 2
  Im(IndIm) = 1;  
else
Im(IndIm)=indIm(:,3);
end
end

