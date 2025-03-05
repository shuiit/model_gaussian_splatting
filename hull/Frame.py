import cv2
import numpy as np
from PIL import Image
from Camera import Camera
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
class Frame(Camera):
    def __init__(self,path,frames_dict,frame,cam_num):
        """
        Initialize a Frame object by loading the corresponding image and voxels and processing pixel data.

        Args:
            path (str): The directory path to the images.
            im_name (str): The name of the image file.
            idx (int): Unique identifier for the image.
        """
      

        self.image_name = frames_dict[frame][0][cam_num]['name']
        self.image_id = cam_num
        self.path = path
        self.frame = frame
        self.im = Image.open(f'{self.path}/{self.image_name}').convert('L')

        super().__init__()
        self.load_from_dict(frames_dict,frame,cam_num)
        self.camera_number = cam_num
        y,x = np.where((np.array(self.im) < 250) )
        self.pixels = np.vstack([y,x]).T
        self.cm = np.mean(self.pixels,0)

    
    
    def calculate_bounding_box(self,cm,delta_xy):
        top_left = np.array([max(0,cm[0] - delta_xy), max(0,cm[1]-delta_xy)])
        bottom_right =  np.array([min(self.image_size[0],cm[0] + delta_xy), min(self.image_size[1],cm[1] + delta_xy)])

        if (bottom_right != 0).any():
            top_left = np.minimum(bottom_right - delta_xy*2,top_left) 
        return np.hstack((top_left,bottom_right))
        
        # bottom_right = np.array([min(self.image_size[0],bottom_right[0]),min(self.image_size[1],bottom_right[1])])


        # self.bounding_box = [max(0,cm[1] - delta_xy), max(0,cm[0]-delta_xy), max(0,cm[1] - delta_xy) + delta_xy*2 , max(0,cm[0]-delta_xy) + delta_xy*2] # [top left, bottom right]



    def map_3d_2d(self,points_3d, use_zbuff = True):
        """
        Map 3D voxel positions to 2D pixel coordinates and store relevant data.

        Args:
            croped_image (bool, optional): Whether to use cropped image pixels. Default is False.
        """
        # voxels,pixels_of_voxels = self.z_buffer(croped_camera_matrix = croped_image) if use_zbuff == True else self.map_no_zbuff(croped_camera_matrix = croped_image)
        pixels_of_voxels = self.project_on_image(points_3d)

        original_projected_pixels = np.vstack((pixels_of_voxels,np.fliplr(self.pixels ))) # project pixels
        [non_intersect_pixels,cnt] = np.unique(original_projected_pixels,axis = 0,return_counts=True) # identify non intersecting pixels
        non_intersect_pixels = non_intersect_pixels[cnt == 1,:] 

        all_pixels = np.vstack((pixels_of_voxels, non_intersect_pixels)) if use_zbuff == True else pixels_of_voxels
        all_3d_idx = np.full(all_pixels.shape[0], -1)
        all_3d_idx[0:points_3d.shape[0]] = points_3d[:,3]

        self.pixel_with_idx = np.column_stack((all_pixels, all_3d_idx))
        self.voxels_with_idx = np.column_stack((points_3d,np.full(pixels_of_voxels.shape[0],self.image_id),np.arange(pixels_of_voxels.shape[0])))

        # determine the color of every pixel that has a mapping, 
        idx = self.pixel_with_idx[:,2] != -1
        pixels = self.pixel_with_idx[idx,0:3].astype(int)
        self.color_of_pixel =  np.array(np.array(self.image))[pixels[:,1],pixels[:,0]]

    
    def map_no_zbuff(self,croped_camera_matrix = False):
        
        voxels_cam = np.matmul(self.world_to_cam, self.points_in_ew_frame_homo.T).T
        projected = self.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
        pxls = np.round(projected) 
        
        idx_sorted_by_z = voxels_cam[:,2].argsort()
        voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
        return voxels_sorted_by_z,pxls[idx_sorted_by_z,:]

    def z_buffer(self,croped_camera_matrix = False):
        """
        Compute the z-buffer for 3D points, projecting them onto the 2D image plane.

        Args:
            croped_camera_matrix (bool, optional): Whether to use a cropped camera matrix. Default is False.

        Returns:
            tuple: A tuple containing sorted voxel positions and pixel coordinates.
        """
        voxels_cam = np.matmul(self.world_to_cam, self.points_in_ew_frame_homo.T).T
        projected = self.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
        pxls = np.round(projected) 
        idx_sorted_by_z = voxels_cam[:,2].argsort()
        voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
        [pixels,idx] = np.unique(pxls[idx_sorted_by_z,:], axis=0,return_index=True)
        return voxels_sorted_by_z[idx,:],pixels
    

    

    def save_croped_images(self):
        image_rgb = self.image.convert('RGB')
        im_np = np.array(image_rgb)
        idx = np.where(im_np[:,:,0] == 0)
        im_np[idx[0],idx[1],0] = 200
        image = Image.fromarray(im_np)
        image.save(f'{self.path}/input_data_for_gs/images/{self.image_name}', format='JPEG', subsampling=0, quality=100)


    
    def generate_base_image(self):

        return {'id' :self.image_id, 'qvec' : self.qvec.copy(), 'tvec' : self.t.T[0].copy(),
                            'camera_id': self.camera_number, 'name' : self.image_name
                            }


    


    


    





