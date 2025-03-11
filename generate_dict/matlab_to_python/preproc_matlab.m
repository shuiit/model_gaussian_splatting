%% load hull
clear
close all
clc

exp = '2024_11_12_darkan'
path = 'H:\My Drive\dark 2022\2024_11_12_darkan\hull\hull_Reorder\'
easyWand_name = 'coefs_12_11_24_easyWandData.mat'


movie = 30
mov_name = sprintf('mov%d',movie)
struct_file_name = sprintf('\\Shull_mov%d',movie)
load([path,mov_name,'\hull_op\',struct_file_name])

hull3d_file_name = sprintf('\\hull3d_mov%d',movie)
load([path,mov_name,'\hull_op\',hull3d_file_name])

load([path,easyWand_name])

save_path = [path,mov_name,'_',exp,'\','images','\']

save_path_parent =  'G:\My Drive\Research\gaussian_splatting\gaussian_splatting_input\'

save_images_dir = [save_path_parent,mov_name,'_',exp,'\','images','\'];
mkdir(save_images_dir)
% load sparse
for cam = 1:1:4
    

    sparse_file = sprintf('\\mov%d_cam%d_sparse.mat',movie,cam)
    sp{cam} = load([path,mov_name,sparse_file])
    im_name = [mov_name,'_bg','.mat']
    bg = sp{cam}.metaData.bg;
    save([save_images_dir,im_name],'bg')

end

%%
path = 'G:\My Drive\Research\gaussian_splatting\gaussian_splatting_input\'
save_path = [path,mov_name,'_',exp,'\','3d_pts','\']
mkdir(save_path)

hull_mat_file(hull3d.body.body4plot,[save_path,'body.mat'],hull3d.frames);

hull_mat_file(hull3d.rightwing.hull.hull3d,[save_path,'rwing.mat'],hull3d.frames);
hull_mat_file(hull3d.leftwing.hull.hull3d,[save_path,'lwing.mat'],hull3d.frames);

real_coord(Shull,[save_path,'real_coord.mat'])
%%
path = 'G:\My Drive\Research\gaussian_splatting\gaussian_splatting_input\'

save_path = [path,mov_name,'_',exp,'\','images','\']
mkdir(save_path)
save_images(sp,save_path)


%% world axes - from coefs
path = 'G:\My Drive\Research\gaussian_splatting\gaussian_splatting_input\'
frame_sparse = 1000

frame = find(Shull.frames == frame_sparse);
body = hull3d.body.body4plot{frame};
wing_left = hull3d.leftwing.hull.hull3d{frame};
wing_right = hull3d.rightwing.hull.hull3d{frame};


real_coords = Shull.real_coord{frame}
body_3d = [real_coords{1}(body(:,1))',real_coords{2}(body(:,2))',real_coords{3}(body(:,3))']
wing_left_3d = [real_coords{1}(wing_left(:,1))',real_coords{2}(wing_left(:,2))',real_coords{3}(wing_left(:,3))']
wing_right_3d = [real_coords{1}(wing_right(:,1))',real_coords{2}(wing_right(:,2))',real_coords{3}(wing_right(:,3))']
ew2lab = Shull.rotmat_EWtoL;
fly = [body_3d;wing_left_3d;wing_right_3d];
fly_h = [fly,ones(size(fly,1),1)];

for j= 1:1:4
save_path = [path,mov_name,'_',exp,'\','camera_KRX0']
[R,K,X0,H] = decompose_dlt(easyWandData.coefs(:,j),easyWandData.rotationMatrices(:,:,j)');
camera(:,:,j) = [K,R,X0];
rotation(:,:,j) = R; 
translation(:,:,j) = X0; 
if j == 1
    K(2,3) = 800-K(2,3)
        K(2,2) = -K(2,2)

end
pmdlt{j} = [K*R,-K*R*X0];

end
plot_camera(rotation,translation,[1,0,0;0,1,0;0,0,1],'standard wand')
save(save_path,'camera');
cam = 1
im = ImfromSp([800,1280],sp{cam}.frames(frame_sparse).indIm);

pt2d = pmdlt{cam}*fly_h';
pt2d =( pt2d./pt2d(3,:))';
figure;
imshow(im2gray(im/255/255));hold on
scatter(pt2d(:,1),pt2d(:,2),'r.')

