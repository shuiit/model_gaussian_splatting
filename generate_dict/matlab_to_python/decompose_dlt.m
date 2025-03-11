function [R,K,X0,Ht] = decompose_dlt(coefs,ew_rotation)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


H=[coefs(1),coefs(2),coefs(3);coefs(5),coefs(6),coefs(7);coefs(9),coefs(10),coefs(11)];
h=[coefs(4);coefs(8);1];

X0 = -inv(H)*h;

[R,K] = QR_Decomposition(inv(H));
% change_ax_dir = [sum(sign(ew_rotation) + sign(R))] + 1;
% change_ax_dir(change_ax_dir==1) = -1;

change_ax_dir = sign(diag(ew_rotation' * R)); % We multiply the matrices to get the dot product of each axis (by using only the diagonal), we could have just calculated the dot product
change_ax_dir(change_ax_dir == 0) = 1; % Avoid zero values
Rot_to_ew = [change_ax_dir(1)*1,0,0;0,change_ax_dir(2)*1,0;0,0,change_ax_dir(3)*1];
Rot_to_stan = [1,0,0;0,-1,0;0,0,-1];
Ht = inv(H)';

K=inv(K);
K = K*Rot_to_stan*Rot_to_ew/K(3,3);

K(2,2) = -K(2,2);
K(2,3) = 801 -K(2,3);

K = K/K(3,3);
R = Rot_to_stan*Rot_to_ew*R';

end