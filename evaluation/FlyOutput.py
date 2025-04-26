
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from Frame import Frame
from plyfile import PlyData
import numpy as np
import Utils
from scipy.linalg import svd
from skimage.measure import LineModelND, ransac
import render.sh_utils as sh_utils


class FlyOutput:
    def __init__(self,image_path,frame,input_dir,output_angles_weights,frame0,iteration,file_name,deg = 0,**kwargs):

        # super().__init__(image_path,frame,cam,**kwargs)
        self.frames = [Frame(image_path,frame,cam,**kwargs) for cam in range(4)] 


        self.ew_to_lab = self.frames[0].ew_to_lab
        ply_path = f'{input_dir}/{frame}/{file_name}/point_cloud/iteration_{iteration}/point_cloud.ply'
        vertices = PlyData.read(ply_path)["vertex"]
        property_names = vertices.data.dtype.names
        self.sh = np.column_stack([vertices[key] for key in property_names if 'rest' in key or 'dc' in key]) 
        self.color = sh_utils.rgb_from_sh(deg,self.sh)
        self.xyz = np.column_stack((vertices['x'],vertices['y'],vertices['z']))
        self.xyz_rotated = (self.ew_to_lab @ self.xyz.T).T
        self.idx_parts = [np.sum(output_angles_weights['weights'][frame - frame0][iteration][:,idx:idx + 3],axis = 1) == 1 for idx in range(0,9,3)]
        self.opacity = 1 / (1 + np.exp(-vertices["opacity"]))

        self.body = self.xyz_rotated[self.idx_parts[0],:]
        self.right_wing = self.xyz_rotated[self.idx_parts[1],:]
        self.left_wing = self.xyz_rotated[self.idx_parts[2],:]

        self.body_ew = self.xyz[self.idx_parts[0],:]
        self.right_wing_ew = self.xyz[self.idx_parts[1],:]
        self.left_wing_ew = self.xyz[self.idx_parts[2],:]
        self.frame0 = frame0


    def intersect_all_cams(frames,intersected,tol = 1):
        for cam in range(4):
            intersected = Utils.intersection_per_cam(frames, cam, intersected, tol=tol) 
        return intersected

 
    def add_interest_point(self,interest_point_h5_path):
        return [frame.interest_point_crop(interest_point_h5_path,frame0 = self.frame0) for frame in self.frames]
        

    def intersect_interest_pixels(self):
        camera_center_to_pixel = np.stack([frame.camera_center_to_pixel_ray(frame.interest_points) for frame in self.frames])
        cam_center = np.hstack([frame.X0 for frame in self.frames]).T 
        projected_wing = np.argsort([np.unique(frame.project_with_proj_mat(self.right_wing_ew)[:,0:2].astype(int),axis = 0).shape[0] for frame in self.frames])
        self.interest_points_3d = np.vstack([Utils.triangulate_least_square(cam_center[projected_wing[0:2],:],camera_center_to_pixel[projected_wing[0:2],idx,:]) for idx in range(camera_center_to_pixel.shape[1])])
        self.rotated_points_3d = (self.ew_to_lab @ np.vstack(self.interest_points_3d).T).T

    def intersect_projections(self):
        """intersect all points to get only points that are projected on the image (of the camera), do it for the easywand FoR and then rotate to lab
        """
        for attr in ['body', 'right_wing', 'left_wing']:
            ew = getattr(self, f"{attr}_ew")
            ew = self.intersect_all_cams(self.frames, ew, tol=1)
            lab = (self.ew_to_lab @ ew.T).T
            setattr(self, f"{attr}_ew", ew)
            setattr(self, attr, lab)



    # def find_closest_gaussian_to_point(self):
    #     sorted_dist = [np.argsort(np.sqrt(np.sum((self.xyz_rotated - point)**2, axis = 1)))[0:1] for point in self.rotated_points_3d]
    #     self.gaussian_closest_to_interest = np.vstack([np.mean(self.xyz_rotated[sorted_idx,:], axis = 0) for sorted_idx in sorted_dist])
    #     self.gaussian_closest_to_interest_ew = (self.ew_to_lab.T @ np.vstack(self.gaussian_closest_to_interest).T).T
    #     self.dist_3d = np.hstack([np.sqrt(np.sum((self.xyz_rotated[min_idx,:] - point)**2, axis = 1))[0] for point,min_idx in zip(self.rotated_points_3d,sorted_dist)])


    def get_all_interest_2d_projection(self, interest_points =  [0,1,2,3,4,5,7,8,9,10,11,12,13,15,16,17]):
        self.projected_interest = np.stack([frame.project_with_proj_mat(self.interest_points_3d)[:,0:2] for frame in self.frames])
        self.interest_points = np.stack([ frame.interest_points for frame in self.frames])
        self.projected_interest_gaussians = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])
        self.dist_from_interest_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points[:,interest_points,:])**2,axis = 2))
        self.dist_from_interest_point = np.sqrt(np.sum((self.gaussian_closest_to_interest_ew - self.interest_points_3d[interest_points,:])**2,axis = 1))


    def interest_load_and_intersect(self,interest_point_h5_path, num_of_bins= 20):
        self.add_interest_point(interest_point_h5_path)
        self.intersect_interest_pixels()
      

    def get_principle_axes(self,frame_xyz):
        body_cm = np.mean(frame_xyz,axis = 0)
        body_centered = frame_xyz - body_cm
        U, S, Vt = svd(body_centered, full_matrices=False)
        return Vt

    def get_axis_orientation(self,axis,points_from,points_to):
        direction = (np.mean(points_to,axis = 0) - points_from)/np.linalg.norm(np.mean(points_to,axis = 0) - points_from)
        return -axis if np.dot(direction,axis) < 0 else axis
    
    
    def reorient_axis(self,points,direction,percent_bot = 0.05,percent_top = 0.1):
        projected_on_body = np.dot(points,direction)
        min_points = min(projected_on_body)
        max_points = max(projected_on_body)
        perc_of_body_length_bot = (max_points - min_points)*percent_bot
        perc_of_body_length_top = (max_points - min_points)*percent_top

        bottom = points[(projected_on_body  < (min_points + perc_of_body_length_bot)),:]
        top = points[(projected_on_body  > (max_points - perc_of_body_length_top)),:]
        x_ax_points = np.vstack([np.mean(bottom,axis = 0),np.mean(top,axis = 0)])
        x_ax = np.mean(top,axis = 0) - np.mean(bottom,axis = 0)
        return x_ax/np.linalg.norm(x_ax),bottom,top,x_ax_points
    
    def wing_span_chord(self,wing_xyz):
        
        wing_axes = self.get_principle_axes(wing_xyz)
        wing_span = self.get_axis_orientation(wing_axes[0],self.body_cm,wing_xyz)
        wing_chord = self.get_axis_orientation(wing_axes[1],[[0,0,0]],[[0,0,1]])
        return wing_span,wing_chord
    
    def get_indices_le_te(self,projected_on_chord,real_indices,bin_indices,idx):
        max_of_bin = np.argmax(projected_on_chord[bin_indices == idx])
        min_of_bin = np.argmin(projected_on_chord[bin_indices == idx])
        return  real_indices[bin_indices == idx][max_of_bin], real_indices[bin_indices == idx][min_of_bin]
        
    
    def get_wing_le_te(self,xyz,span,chord, perc_wing = 0.7,num_of_bins = 100):

        projected_on_span = np.dot(xyz,span)

        half_wing = perc_wing*(max(projected_on_span) - min(projected_on_span))
        xyz_for_le = xyz[projected_on_span < (min(projected_on_span) + half_wing),:]
        projected_on_span = np.dot(xyz_for_le,span)
        projected_on_chord = np.dot(xyz_for_le,chord)


        diff = (max(projected_on_span) - min(projected_on_span))/num_of_bins
        bin_edges = np.arange(np.min(projected_on_span), np.max(projected_on_span) + diff, diff)
        bin_indices = np.digitize(projected_on_span, bins=bin_edges)
        real_indices = np.array(range(len(projected_on_chord)))
        le_coord,te_coord = [],[]
        visited = set()
        for idx in bin_indices:
            if idx not in visited:
                visited.add(idx)
                real_idx_le,real_idx_te = self.get_indices_le_te(projected_on_chord,real_indices,bin_indices,idx)
                le_coord.append(xyz_for_le[real_idx_le,:])
                te_coord.append(xyz_for_le[real_idx_te,:])

        return np.vstack(le_coord),np.vstack(te_coord)



    def ransac_for_le(self,wing_le):
        
        model_robust, inliers = ransac(wing_le, LineModelND, min_samples=2, residual_threshold=5/100000, max_trials=1000)
        origin, direction = model_robust.params
        return origin, direction
    

    def wing_le(self,wing_xyz,span,chord,**kwargs):
        
        wing_le,wing_te = self.get_wing_le_te(wing_xyz,span,chord,**kwargs)
        wing_le2,wing_te2 = self.get_wing_le_te(wing_xyz,chord,span,**kwargs)
        wing_le = np.vstack((wing_le,wing_le2))
        wing_te = np.vstack((wing_te,wing_te2))

        wing_origin, r_wing_direction = self.ransac_for_le(wing_le)
        return wing_origin, r_wing_direction,wing_le,wing_te


    def calc_wing_le_te(self,num_of_bins = 20,**kwargs):
        self.body_cm = np.mean(self.body,axis = 0)
        xbody = self.get_principle_axes(self.body)[0]
        self.xbody = self.get_axis_orientation(xbody,[[0,0,0]],[[0,0,1]])
        self.xbody,self.bottom,self.top,self.xbody_points = self.reorient_axis(self.body,self.xbody)


        self.interest_on_xbody = np.dot(self.rotated_points_3d[16:,:] - self.body_cm,self.xbody[:,np.newaxis])*self.xbody + self.body_cm

        
        self.body_interest_gaussian = np.vstack((self.xbody_points[0], self.xbody_points[1]))
        self.body_interest_gaussian_ew = (self.ew_to_lab.T @ self.body_interest_gaussian.T).T

        right_wing_span,right_wing_chord = self.wing_span_chord(self.right_wing)
        left_wing_span,left_wing_chord = self.wing_span_chord(self.left_wing)

        self.right_wing_origin, self.right_wing_direction,self.right_wing_le,self.right_wing_te = self.wing_le(self.right_wing,right_wing_span,right_wing_chord,perc_wing = 1, num_of_bins = num_of_bins)
        self.left_wing_origin, self.left_wing_direction,self.left_wing_le,self.left_wing_te = self.wing_le(self.left_wing,left_wing_span,left_wing_chord,perc_wing = 1, num_of_bins = num_of_bins)
        

    def dist_points(self,x1,x2):
        return np.sqrt(np.sum((x1 - x2)**2, axis = 1))


    def closest_point_to_interest_boundary(self,wing_boundary,points):   

        gaussian_closest_to_interest = np.vstack((wing_boundary[np.argmin(self.dist_points(wing_boundary,point)),:] for point in points))
        gaussian_closest_to_interest_ew = (self.ew_to_lab.T @ np.vstack(gaussian_closest_to_interest).T).T
        dist_gaus_interest = self.dist_points(gaussian_closest_to_interest[1:,:],gaussian_closest_to_interest[0:-1,:])
        dist_interest = self.dist_points(points[1:,:],points[0:-1,:])   
        return  gaussian_closest_to_interest,gaussian_closest_to_interest_ew,dist_gaus_interest,dist_interest


    def wings_interest_point(self, left_wing = [0,1,2,3,4,5,7],right_wing = [8,9,10,11,12,13,15] ):

        left_wing = self.closest_point_to_interest_boundary(np.vstack((self.left_wing_le,self.left_wing_te)),self.rotated_points_3d[left_wing,:])
        right_wing = self.closest_point_to_interest_boundary(np.vstack((self.right_wing_le,self.right_wing_te)),self.rotated_points_3d[right_wing,:])

        self.gaussian_closest_to_interest = np.vstack((left_wing[0],right_wing[0],self.body_interest_gaussian))
        self.gaussian_closest_to_interest_ew = np.vstack((left_wing[1],right_wing[1],self.body_interest_gaussian_ew))
        self.dist_gaus_interest = np.hstack((left_wing[2],right_wing[2]))
        self.dist_interest = np.vstack((left_wing[3],right_wing[3]))


    

        
