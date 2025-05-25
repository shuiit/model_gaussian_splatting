
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
import pickle
from scipy.signal import savgol_filter

class FlyOutput:
    def __init__(self,image_path,frame,input_dir,output_angles_weights,frame0,iteration,file_name,letedict = None,deg = 0,skip_frames = 1,**kwargs, ):

        # super().__init__(image_path,frame,cam,**kwargs)
        self.frames = [Frame(image_path,frame,cam,**kwargs) for cam in range(4)] 

        self.frame_num = frame
        self.ew_to_lab = self.frames[0].ew_to_lab
        ply_path = f'{input_dir}/{frame}/{file_name}/point_cloud/iteration_{iteration}/point_cloud.ply'
        vertices = PlyData.read(ply_path)["vertex"]
        property_names = vertices.data.dtype.names
        self.sh = np.column_stack([vertices[key] for key in property_names if 'rest' in key or 'dc' in key]) 
        self.color = sh_utils.rgb_from_sh(deg,self.sh)
        grayscale = (self.color[:,0] - self.color[:,0].min()) / (self.color[:,0].max() - self.color[:,0].min())
        self.grayscale = grayscale[grayscale <1]
        self.xyz = np.column_stack((vertices['x'],vertices['y'],vertices['z']))
        self.xyz_rotated = (self.ew_to_lab @ self.xyz.T).T
        if output_angles_weights is not None:
            self.idx_parts = [np.sum(output_angles_weights['weights'][(frame - frame0)//skip_frames][iteration][:,idx:idx + 3],axis = 1) == 1 for idx in range(0,9,3)]
            self.body = self.xyz_rotated[self.idx_parts[0],:]
            self.right_wing = self.xyz_rotated[self.idx_parts[1],:]
            self.left_wing = self.xyz_rotated[self.idx_parts[2],:]

            self.body_ew = self.xyz[self.idx_parts[0],:]
            self.right_wing_ew = self.xyz[self.idx_parts[1],:]
            self.left_wing_ew = self.xyz[self.idx_parts[2],:]
            self.calc_wing_le_te(letedict['num_of_bins'] ,letedict['perc_wing_for_le'],letedict['wing_length_snip'])
            xbody = self.get_principle_axes(self.body)[0]
            self.xbody = self.get_axis_orientation(xbody,[[0,0,0]],[[0,0,1]])
            self.xbody,bottom,top,x_ax_points= self.reorient_axis(self.body,xbody)

        
        self.opacity = 1 / (1 + np.exp(-vertices["opacity"]))

        self.frame0 = frame0


    def intersect_all_cams(frames,intersected,tol = 1):
        for cam in range(4):
            intersected = Utils.intersection_per_cam(frames, cam, intersected, tol=tol) 
        return intersected

 
    def add_interest_point(self,interest_point_h5_path):
        return [frame.interest_point_crop(interest_point_h5_path,frame0 = self.frame0) for frame in self.frames]
        

    # def triangulate_interest_pixels(self):
    #     camera_center_to_pixel = np.stack([frame.camera_center_to_pixel_ray(frame.interest_points) for frame in self.frames])
    #     cam_center = np.hstack([frame.X0 for frame in self.frames]).T 
    #     projected_wing = np.argsort([np.unique(frame.project_with_proj_mat(self.right_wing_ew)[:,0:2].astype(int),axis = 0).shape[0] for frame in self.frames])
    #     self.interest_points_3d = np.vstack([Utils.triangulate_least_square(cam_center[projected_wing[0:2],:],camera_center_to_pixel[projected_wing[0:2],idx,:]) for idx in range(camera_center_to_pixel.shape[1])])
    #     self.rotated_points_3d = (self.ew_to_lab @ np.vstack(self.interest_points_3d).T).T

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


    def get_all_interest_2d_projection(self):
        self.projected_interest = np.stack([np.fliplr(frame.project_with_proj_mat(self.interest_points_3d)[:,0:2]) for frame in self.frames])
        
        self.projected_gaussians_closest_to_interest = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])
        self.projected_interest_closest_to_gaussians = np.stack([frame.project_with_proj_mat(self.interest_points_closest_to_gaussian_ew)[:,0:2] for frame in self.frames])

        
        
        self.dist_from_interest_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points)**2,axis = 2))



        self.dist_from_interest_point = np.sqrt(np.sum((self.gaussian_closest_to_interest_ew - self.interest_points_3d)**2,axis = 1))
        self.dist_interest_from_projected = np.sqrt(np.sum((self.projected_interest - self.interest_points)**2,axis = 2))

        self.projected_interest_gaussians = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])
        self.dist_from_gaussians_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points)**2,axis = 2))
        self.dist_from_gaussians_point = np.sqrt(np.sum((self.interest_points_closest_to_gaussian_ew - self.interest_points_3d)**2,axis = 1))

    def interest_load_and_intersect_amitai(self,interest_point_h5_path):
        self.add_interest_point(interest_point_h5_path)
        self.triangulate_interest_pixels()


    # def load_parimiter(self,dict_path):
    #     dict_path = dict_path if os.path.exists(f'{dict_path}/wing1_gt_points_frame{self.frame_num}.pkl') else None
    #     with open(f'{dict_path}/wing1_gt_points_frame{self.frame_num}.pkl','rb') as f1 \
    #     ,open(f'{dict_path}/wing2_gt_points_frame{self.frame_num}.pkl','rb') as f2:
    #         interest_points_wing1,interest_points_wing2 = pickle.load(f1),pickle.load(f2)
    #     self.interest_point_shape = [len(interest_points_wing1) for interest_points_wing1 in interest_points_wing1.values()]
    #     interest_points = {key: interest_points_wing1[key] + interest_points_wing2[key] for key in interest_points_wing1}
    #     return interest_points

    # def add_interest_points_and_triangulate(self,interest_points):
    #     [frame.add_interest_point(np.fliplr(np.vstack(interest_point))) for interest_point, frame in zip(interest_points.values(),self.frames)]
    #     self.triangulate_interest_pixels()

    # def define_wings_interest_points(self):
    #     mean_wing = np.mean(self.right_wing,axis = 0)
    #     interest_points = self.rotated_points_3d
    #     wings = [interest_points[0:self.interest_point_shape[0],:],interest_points[self.interest_point_shape[0]:,:]]

    #     wings_idx = [list(range(self.interest_point_shape[0])),list(range(self.interest_point_shape[0],interest_points.shape[0]))]

    #     mean_interest = [np.atleast_2d(np.mean(wing,axis = 0)) for wing in wings]
    #     dist_interest_wing = [Utils.dist_points(mean_wing,mean_interest) for mean_interest in mean_interest]
    #     idx_right_wing = np.argsort(np.hstack(dist_interest_wing))
    #     self.interest_left_wing_boundry = wings_idx[idx_right_wing[1]]
    #     self.interest_right_wing_boundry = wings_idx[idx_right_wing[0] ]
  

    # def cyclic_sort(self,points,span,chord):
    #     points_2d = Utils.project_to_plane(points, np.mean(points,axis = 0), span, chord)
    #     cyclic_points = Utils.rotational_sort(points_2d, np.mean(points_2d,axis = 0), clockwise=True)
    #     first_index = np.argmin(np.dot(span,points[cyclic_points].T))
    #     return  np.roll(cyclic_points,first_index)
         

    def zscore(self,points):
        nrml = np.cross(self.right_wing_span,self.right_wing_chord)
        pts_on_nrml = np.dot(points,nrml)
        std = np.std(pts_on_nrml)
        mean = np.mean(pts_on_nrml)
        return points[((pts_on_nrml - mean)/std) < 1.5]

    # def zsocre_ol_calc_indices(self):
    #     rwing_poins = self.zscore(np.vstack((self.right_wing_le,self.right_wing_te)))
    #     lwing_points = self.zscore(np.vstack((self.left_wing_le,self.left_wing_te)))

    #     interest_right_wing = self.rotated_points_3d[self.interest_right_wing_boundry,:]
    #     interest_left_wing = self.rotated_points_3d[self.interest_left_wing_boundry,:]

    #     indices_right_wing_bound = self.cyclic_sort(rwing_poins,self.right_wing_span,self.right_wing_chord)
    #     indices_left_wing_bound = self.cyclic_sort(lwing_points,self.left_wing_span,self.left_wing_chord)

    #     indices_interest_right_wing = self.cyclic_sort(interest_right_wing,self.right_wing_span,self.right_wing_chord)
    #     indices_interest_left_wing = self.cyclic_sort(interest_left_wing,self.left_wing_span,self.left_wing_chord)

    #     left_bound = lwing_points[indices_left_wing_bound]
    #     right_bound = rwing_poins[indices_right_wing_bound]

    #     dists = np.linalg.norm(np.diff(left_bound, axis=0), axis=1)
    #     max_dist_l = np.argmax(dists)
    #     dists = np.linalg.norm(np.diff(right_bound, axis=0), axis=1)
    #     max_dist_r = np.argmax(dists)
    #     indices_interest_right_wing =np.roll(indices_interest_right_wing,max_dist_r)
    #     indices_interest_left_wing =np.roll(indices_interest_left_wing,max_dist_l)

        # left_bound = np.vstack([savgol_filter(pts, 19, 2) for pts in left_bound .T]).T
        # right_bound = np.vstack([savgol_filter(pts, 19, 2) for pts in right_bound .T]).T

        # return left_bound,right_bound,interest_left_wing[indices_interest_left_wing],interest_right_wing[indices_interest_right_wing]


    # def fit_interest_and_gs(self):
    #     wing_gs_left,wing_gs_right,interest_lw,interest_rw = self.zsocre_ol_calc_indices()
    #     fitted_gs = np.vstack([Utils.fit_all_points(wing,skip_points = 10,num_of_fit_point = 100, degree = 2) for wing in [wing_gs_left,wing_gs_right]])
    #     fitted_interest = np.vstack([Utils.fit_all_points(interest,skip_points = 3,num_of_fit_point = 100, degree = 2) for interest in [interest_lw,interest_rw]])
    #     return fitted_gs,fitted_interest
    

    # def closest_dist_to_line(self,interest_left_wing,wing_bound):
    #     line = interest_left_wing[0] - interest_left_wing[1]
    #     line = line / np.linalg.norm(line)
    #     point_to_origin = wing_bound - interest_left_wing[0]
    #     projected = np.dot(point_to_origin,line)
    #     dist = point_to_origin - (line*np.atleast_2d(projected).T)
    #     min_idx = np.argmin(np.linalg.norm(dist,axis = 1))
    #     return wing_bound[min_idx],interest_left_wing[0] + line*np.atleast_2d(projected[min_idx]).T


    # def calculate_dist_interest_gs(self):

    #     fitted_gs,fitted_interest = self.fit_interest_and_gs()

    #     interest_points_closest_to_gaussian = np.vstack((fitted_interest[np.argmin(Utils.dist_points(fitted_interest,point)),:] for point in fitted_gs))
    #     gaussians_points_closest_to_gaussian = np.vstack((fitted_gs[np.argmin(Utils.dist_points(fitted_gs,point)),:] for point in fitted_interest))


    #     self.dist_3d_interest_to_gs,self.dist_2d_interest_to_gs = self.find_3d_2d_dist( fitted_gs, interest_points_closest_to_gaussian)
    #     self.dist_3d_gs_to_interest,self.dist_2d_gs_to_interest = self.find_3d_2d_dist( fitted_interest, gaussians_points_closest_to_gaussian)

    def rotate_to_ew_and_project(self,points):
        points_closest_to_fitted_ew =  (self.ew_to_lab.T @ np.vstack(points).T).T
        return np.stack([frame.project_with_proj_mat(points_closest_to_fitted_ew)[:,0:2] for frame in self.frames])

        
    def find_3d_2d_dist(self, fited_points, closest_to_fitted):
        points_closest_to_fitted = np.vstack((closest_to_fitted[np.argmin(Utils.dist_points(closest_to_fitted,point)),:] for point in fited_points))
        dist_3d_from_fited = Utils.dist_points(points_closest_to_fitted,fited_points)
        projected_fitted = self.rotate_to_ew_and_project(fited_points)
        projected_closest_to_fitted = self.rotate_to_ew_and_project(points_closest_to_fitted)
        dist_2d = np.hstack([Utils.dist_points(projected_fitted[cam],(projected_closest_to_fitted[cam])) for cam in range(len(projected_closest_to_fitted))])
        return dist_3d_from_fited,dist_2d




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
        wing_le_full = np.vstack((wing_le,wing_le2))
        wing_te_full = np.vstack((wing_te,wing_te2))

        wing_origin, r_wing_direction = self.ransac_for_le(wing_le)
        return wing_origin, r_wing_direction,wing_le_full,wing_te_full

    def get_perc_of_bound(self,wing_bound,wing_all,wing_span, wing_length = 0.1):

        dot_on_le = np.dot(wing_span,wing_all.T)
        length = (np.max(dot_on_le) - np.min(dot_on_le))
        dot_bound = np.dot(wing_span,wing_bound.T)
        return wing_bound[(dot_bound  - np.min(dot_on_le))> length*wing_length,:]

    def run_ransac_and_snip_wing(self, wing_pts,num_of_bins = 20,perc_wing_for_le = 1,wing_length_snip = 0.1):
        wing_span,wing_chord = self.wing_span_chord(wing_pts)
        wing_origin, wing_direction,wing_le,wing_te = self.wing_le(wing_pts,wing_span,wing_chord,perc_wing = perc_wing_for_le, num_of_bins = num_of_bins)
        
        wing_direction = np.sign(np.dot(wing_span,wing_direction))*wing_direction
        
        wing_le = self.get_perc_of_bound(wing_le,wing_pts,wing_direction, wing_length = wing_length_snip)
        wing_te = self.get_perc_of_bound(wing_te,wing_pts,wing_direction, wing_length = wing_length_snip)
        return wing_origin,wing_direction,wing_chord,wing_le,wing_te




    def calc_wing_le_te(self,num_of_bins = 20,perc_wing_for_le = 1,wing_length_snip = 0.1):
        self.body_cm = np.mean(self.body,axis = 0)
        xbody = self.get_principle_axes(self.body)[0]
        self.xbody = self.get_axis_orientation(xbody,[[0,0,0]],[[0,0,1]])
        self.xbody,self.bottom,self.top,self.xbody_points = self.reorient_axis(self.body,self.xbody)
        # self.interest_on_xbody = np.dot(self.rotated_points_3d[16:,:] - self.body_cm,self.xbody[:,np.newaxis])*self.xbody + self.body_cm

        # self.body_interest_gaussian = np.vstack((self.xbody_points[0], self.xbody_points[1]))
        # self.body_interest_gaussian_ew = (self.ew_to_lab.T @ self.body_interest_gaussian.T).T

        self.right_wing_origin,self.right_wing_span,self.right_wing_chord,self.right_wing_le,self.right_wing_te = self.run_ransac_and_snip_wing( self.right_wing,num_of_bins = num_of_bins,perc_wing_for_le = perc_wing_for_le,wing_length_snip = wing_length_snip)
        self.left_wing_origin,self.left_wing_span,self.left_wing_chord,self.left_wing_le,self.left_wing_te = self.run_ransac_and_snip_wing( self.left_wing,num_of_bins = num_of_bins,perc_wing_for_le = perc_wing_for_le,wing_length_snip = wing_length_snip)
        
        wing_bound_rw = np.unique(np.vstack((self.right_wing_le,self.right_wing_te)),axis = 0)
        wing_bound_lw = np.unique(np.vstack((self.left_wing_le,self.left_wing_te)), axis = 0)

        wing_bound_rw = self.zscore(wing_bound_rw)
        wing_bound_lw = self.zscore(wing_bound_lw)

        self.right_wing_boundary = Utils.cyclic_sort(wing_bound_rw,self.right_wing_span,self.right_wing_chord)
        self.left_wing_boundary = Utils.cyclic_sort(wing_bound_lw,self.left_wing_span,self.left_wing_chord)

    # def get_wing_origin(self):
    #      np.dot(self.body,)


    def closest_point_to_interest_boundary(self,wing_boundary,points):   

        gaussian_closest_to_interest = np.vstack((wing_boundary[np.argmin(Utils.dist_points(wing_boundary,point)),:] for point in points))
        gaussian_closest_to_interest_ew = (self.ew_to_lab.T @ np.vstack(gaussian_closest_to_interest).T).T
        dist_gaus_interest = Utils.dist_points(gaussian_closest_to_interest[1:,:],gaussian_closest_to_interest[0:-1,:])
        dist_interest = Utils.dist_points(points[1:,:],points[0:-1,:])   
        return  gaussian_closest_to_interest,gaussian_closest_to_interest_ew,dist_gaus_interest,dist_interest


    def wings_interest_point(self, left_wing = [0,1,2,3,4,5,7],right_wing = [8,9,10,11,12,13,15] ):

        left_wing = self.closest_point_to_interest_boundary(np.vstack((self.left_wing_le,self.left_wing_te)),self.rotated_points_3d[left_wing,:])
        right_wing = self.closest_point_to_interest_boundary(np.vstack((self.right_wing_le,self.right_wing_te)),self.rotated_points_3d[right_wing,:])

        self.gaussian_closest_to_interest = np.vstack((left_wing[0],right_wing[0],self.body_interest_gaussian))
        self.gaussian_closest_to_interest_ew = np.vstack((left_wing[1],right_wing[1],self.body_interest_gaussian_ew))
        self.dist_gaus_interest = np.hstack((left_wing[2],right_wing[2]))
        self.dist_interest = np.vstack((left_wing[3],right_wing[3]))


    

        
