
import pickle
import os
from FlyOutput import FlyOutput
import Utils
import numpy as np

class Evaluation(FlyOutput):
    def __init__(self,interest_points_path,image_path,frame,input_dir,output_angles_weights,frame0,iteration,file_name,**kwargs):
        super().__init__(image_path,frame,input_dir,output_angles_weights,frame0,iteration,file_name,deg = 0,skip_frames = 1,**kwargs)
        interest_points = self.load_parimiter(interest_points_path)
        [frame.add_interest_point(np.fliplr(np.vstack(interest_point))) for interest_point, frame in zip(interest_points.values(),self.frames)]
        self.triangulate_interest_pixels()
        self.define_wings_interest_points()
        self.att_to_calc_interest = [
            "interest_on_bound_rw", "interest_on_bound_lw", "interest_right_wing_boundry","interest_left_wing_boundry",    
        ]

        self.att_to_calc_bound = [
            "bound_on_interest_rw", "bound_on_interest_lw", "right_wing_boundary","left_wing_boundary",    
        ]
        self.projection_tasks = [
            ("interest_right_wing_boundry", "right_wing_boundary", "interest_on_bound_rw"),
            ("interest_left_wing_boundry", "left_wing_boundary", "interest_on_bound_lw"),
            ("right_wing_boundary", "interest_right_wing_boundry", "bound_on_interest_rw"),
            ("left_wing_boundary", "interest_left_wing_boundry", "bound_on_interest_lw"),]



    def load_parimiter(self,dict_path):
        dict_path = dict_path if os.path.exists(f'{dict_path}/wing1_gt_points_frame{self.frame_num}.pkl') else None
        with open(f'{dict_path}/wing1_gt_points_frame{self.frame_num}.pkl','rb') as f1 \
        ,open(f'{dict_path}/wing2_gt_points_frame{self.frame_num}.pkl','rb') as f2:
            interest_points_wing1,interest_points_wing2 = pickle.load(f1),pickle.load(f2)
        self.interest_point_shape = [len(interest_points_wing1) for interest_points_wing1 in interest_points_wing1.values()]
        interest_points = {key: interest_points_wing1[key] + interest_points_wing2[key] for key in interest_points_wing1}
        return interest_points
    

    
    def triangulate_interest_pixels(self):
        camera_center_to_pixel = np.stack([frame.camera_center_to_pixel_ray(frame.interest_points) for frame in self.frames])
        cam_center = np.hstack([frame.X0 for frame in self.frames]).T 
        projected_wing = np.argsort([np.unique(frame.project_with_proj_mat(self.right_wing_ew)[:,0:2].astype(int),axis = 0).shape[0] for frame in self.frames])
        self.interest_points_3d = np.vstack([Utils.triangulate_least_square(cam_center[projected_wing[0:2],:],camera_center_to_pixel[projected_wing[0:2],idx,:]) for idx in range(camera_center_to_pixel.shape[1])])
        self.rotated_points_3d = (self.ew_to_lab @ np.vstack(self.interest_points_3d).T).T


    def define_wings_interest_points(self):
        mean_wing = np.mean(self.right_wing,axis = 0)
        interest_points = self.rotated_points_3d
        wings = [interest_points[0:self.interest_point_shape[0],:],interest_points[self.interest_point_shape[0]:,:]]

        wings_idx = [list(range(self.interest_point_shape[0])),list(range(self.interest_point_shape[0],interest_points.shape[0]))]

        mean_interest = [np.atleast_2d(np.mean(wing,axis = 0)) for wing in wings]
        dist_interest_wing = [Utils.dist_points(mean_wing,mean_interest) for mean_interest in mean_interest]
        idx_right_wing = np.argsort(np.hstack(dist_interest_wing))
        # self.interest_left_wing_boundry = interest_points[wings_idx[idx_right_wing[1]]]
        # self.interest_right_wing_boundry = interest_points[wings_idx[idx_right_wing[0] ]]
        interest_right = interest_points[wings_idx[idx_right_wing[0]]]
        point_on_vec_right_bound = np.dot(np.vstack([self.right_wing_le,self.right_wing_le]),self.right_wing_span)
        point_on_vec_right_interest = np.dot(interest_right,self.right_wing_span)
        idx_to_keep = np.where((point_on_vec_right_interest > min(point_on_vec_right_bound)) & (point_on_vec_right_interest < max(point_on_vec_right_bound)))
        point_on_vec_right_interest = interest_right[idx_to_keep[0],:]


        interest_left = interest_points[wings_idx[idx_right_wing[1]]]
        point_on_vec_left_bound = np.dot(np.vstack([self.left_wing_le,self.left_wing_le]),self.left_wing_span)
        point_on_vec_left_interest = np.dot(interest_left,self.left_wing_span)
        idx_to_keep = np.where((point_on_vec_left_interest > min(point_on_vec_left_bound)) & (point_on_vec_left_interest < max(point_on_vec_left_bound)))
        point_on_vec_left_interest = interest_left[idx_to_keep[0],:]



        min_max_bound_on_span_left = np.dot(np.vstack([self.left_wing_le,self.left_wing_le]),self.left_wing_span)

        self.interest_right_wing_boundry = Utils.cyclic_sort(point_on_vec_right_interest,self.right_wing_span,self.right_wing_chord)
        self.interest_left_wing_boundry = Utils.cyclic_sort(point_on_vec_left_interest,self.left_wing_span,self.left_wing_chord)

        self.interest_right_wing_boundry = np.vstack((self.interest_right_wing_boundry,self.interest_right_wing_boundry[0]))
        self.interest_left_wing_boundry = np.vstack((self.interest_left_wing_boundry,self.interest_left_wing_boundry[0]))

        # self.interest_right_wing_boundry = self.zscore(interest_right_wing_boundry)
        # self.interest_left_wing_boundry = self.zscore(interest_left_wing_boundry)



    def point_to_segment_projection(self,point, origin, point_line):
        line = point_line - origin # the line - a vector
        point_to_origin = point - origin # a vector from the point to the lines origin
        line_sq_length = np.dot(line, line) # project the vector from the origin to the point on the line
        t = np.dot(point_to_origin, line) / line_sq_length
        if 0 <= t <= 1:
            projection = origin + t * line
            dist = np.linalg.norm(point - projection)
            return dist
        else:
            return float('inf')
        


# dist_closest_interest_to_gs = []

    def get_indices_closest_points_to_line(self, points,points_of_line):
        return np.argmin([self.point_to_segment_projection(points, points_of_line[k], points_of_line[k+1]) for k in range(points_of_line.shape[0] - 1)])


    def run_all_points_get_closest(self, points, points_of_line):
        return [self.get_indices_closest_points_to_line(points[idx], points_of_line) for idx in range(points.shape[0])]



    def get_projected_points_on_line(self,points_to_project_on_line,line_points,indices):
        return np.vstack([Utils.project_point_on_line(points_to_project_on_line[idx],line_points,indices[idx]) for idx in range(len(indices))])

    def get_projected_and_store(self,frame, source_attr, target_attr, output_attr):
        source = getattr(frame, source_attr)
        target = getattr(frame, target_attr)
        closest_indices = frame.run_all_points_get_closest(source, target)
        projected = frame.get_projected_points_on_line(source, target, closest_indices)
        setattr(frame, output_attr, projected)
        return closest_indices,projected

# now we need to find the 3d point on the closest line - to calculate the 2d distance


        
    def zscore(self,points):
        nrml = np.cross(self.right_wing_span,self.right_wing_chord)
        pts_on_nrml = np.dot(points,nrml)
        std = np.std(pts_on_nrml)
        mean = np.mean(pts_on_nrml)
        return points[((pts_on_nrml - mean)/std) < 1.5]
    
    
    def calculate_repreojection_error(self,att_to_calc):
        fitted = np.vstack((getattr(self,att_to_calc[0]),getattr(self, att_to_calc[1])))
        original = np.vstack((getattr(self, att_to_calc[2]),getattr(self, att_to_calc[3])))
        bound_on_ew = (self.ew_to_lab.T @ fitted.T ).T
        interest_to_ew = (self.ew_to_lab.T @ original.T).T
        projected_interest = [np.fliplr(frame2d.project_with_proj_mat(interest_to_ew)[:,0:2]) for frame2d in self.frames]
        projected_gs = [np.fliplr(frame2d.project_with_proj_mat(bound_on_ew)[:,0:2]) for frame2d in self.frames]
        return np.sqrt(np.sum((np.vstack(projected_interest) - np.vstack(projected_gs))**2, axis = 1))
    
    def calculate_3d_dist(self,att_to_calc):
        
        fitted = np.vstack((getattr(self,att_to_calc[0]),getattr(self, att_to_calc[1])))
        original = np.vstack((getattr(self, att_to_calc[2]),getattr(self, att_to_calc[3])))
        return np.sqrt(np.sum((fitted - original)**2, axis = 1))*1000

    
    def get_all_interest_2d_projection(self):
        self.projected_interest = np.stack([np.fliplr(frame.project_with_proj_mat(self.interest_points_3d)[:,0:2]) for frame in self.frames])
        self.projected_gaussians_closest_to_interest = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])

        
    
        self.dist_from_interest_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points)**2,axis = 2))



        self.dist_from_interest_point = np.sqrt(np.sum((self.gaussian_closest_to_interest_ew - self.interest_points_3d)**2,axis = 1))
        self.dist_interest_from_projected = np.sqrt(np.sum((self.projected_interest - self.interest_points)**2,axis = 2))

        self.projected_interest_gaussians = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])
        self.dist_from_gaussians_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points)**2,axis = 2))
        self.dist_from_gaussians_point = np.sqrt(np.sum((self.interest_points_closest_to_gaussian_ew - self.interest_points_3d)**2,axis = 1))

    def calculate_error(self):
        self.error2d_gt_on_boundary_to_gt = self.calculate_repreojection_error(self.att_to_calc_interest) # distance between gt on banudary to baundary
        self.error3d_gt_on_boundary_to_gt = self.calculate_3d_dist(self.att_to_calc_interest)
        self.error2d_boundary_on_gt_to_boundary= self.calculate_repreojection_error(self.att_to_calc_bound) # distance between boundary on gt to gt
        self.error3d_boundary_on_gt_to_boundary = self.calculate_3d_dist(self.att_to_calc_bound)
        