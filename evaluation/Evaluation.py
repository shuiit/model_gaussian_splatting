
import pickle
import os
from FlyOutput import FlyOutput
import Utils
import numpy as np
import math

class Evaluation(FlyOutput):
    def __init__(self,interest_points_path,image_path,frame,input_dir,output_angles_weights,frame0,iteration,file_name,letedict = None,**kwargs):
        super().__init__(image_path,frame,input_dir,output_angles_weights,frame0,iteration,file_name,deg = 0,skip_frames = 1,letedict = letedict,**kwargs)
        interest_points = self.load_parimiter(interest_points_path)
        [frame.add_interest_point(np.fliplr(np.vstack(interest_point))) for interest_point, frame in zip(interest_points.values(),self.frames)]
        self.triangulate_interest_pixels()
        self.define_wings_interest_points()
        self.letedict = letedict

        self.projection_tasks = [
            ("right_wing_tagged_le", "right_wing_boundary_le", "interest_on_bound_rw_le"),
            ("left_wing_tagged_le", "left_wing_boundary_le", "interest_on_bound_lw_le"),
            ("right_wing_boundary_le", "right_wing_tagged_le", "bound_on_interest_rw_le"),
            ("left_wing_boundary_le", "left_wing_tagged_le", "bound_on_interest_lw_le"),
            ("right_wing_tagged_te", "right_wing_boundary_te", "interest_on_bound_rw_te"),
            ("left_wing_tagged_te", "left_wing_boundary_te", "interest_on_bound_lw_te"),
            ("right_wing_boundary_te", "right_wing_tagged_te", "bound_on_interest_rw_te"),
            ("left_wing_boundary_te", "left_wing_tagged_te", "bound_on_interest_lw_te"),]



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


    def trim_tagged_to_wing(self, tagged, boundary, span):
        point_on_vec_bound = np.dot(boundary,span)
        point_on_vec_tagged = np.dot(tagged,span)
        idx_to_keep = np.where((point_on_vec_tagged > min(point_on_vec_tagged)) & (point_on_vec_tagged < max(point_on_vec_bound)))
        return tagged[idx_to_keep[0],:]



    def reorder_boundary(self,wing_name):


        wing_side = wing_name.split('_wing')[0]
        span = getattr(self,f'{wing_side}_gt_wing_span')
        chord = getattr(self,f'{wing_side}_gt_wing_chord')

        points = getattr(self,wing_name)



        le = getattr(self,f'{wing_side}_wing_le') 
        
        

        centered_points =  points - np.mean(points,axis = 0)

        
        projected_on_chord = np.dot(centered_points,chord)
        projected_on_span = np.dot(centered_points,span)
        points_projected = np.column_stack((projected_on_span,projected_on_chord))

        le = points[points_projected[:,1] > 0,:]
        centered_le = le- np.mean(points,axis = 0)
        
        le_projected_on_span = np.dot(centered_le,span)
        
        rotationl_idx = Utils.rotational_sort(points_projected,np.mean(points_projected,axis = 0))

        idx_to_roll = points.shape[0]- np.argmin(np.abs(projected_on_span[rotationl_idx] - np.min(le_projected_on_span)))

        points_projected = np.roll(points_projected[rotationl_idx,:],idx_to_roll,axis = 0)
        return np.roll(points[rotationl_idx,:],idx_to_roll,axis = 0)
        

        # setattr(self,wing_name,points)
        # le = points[points_projected[:,1] > 0,:]
        # setattr(self,f'{wing_name}_le',le)
        # setattr(self,f'{wing_name}_te',np.vstack((le[-1,:],points[points_projected[:,1] < 0,:])))



    def define_wings_interest_points(self):
        mean_wing = np.mean(self.right_wing,axis = 0)
        interest_points = self.rotated_points_3d
        wings = [interest_points[0:self.interest_point_shape[0],:],interest_points[self.interest_point_shape[0]:,:]]
        wings_idx = [list(range(self.interest_point_shape[0])),list(range(self.interest_point_shape[0],interest_points.shape[0]))]

        mean_interest = [np.atleast_2d(np.mean(wing,axis = 0)) for wing in wings]
        dist_interest_wing = [Utils.dist_points(mean_wing,mean_interest) for mean_interest in mean_interest]
        idx_right_wing = np.argsort(np.hstack(dist_interest_wing))

    
        self.right_wing_tagged = interest_points[wings_idx[idx_right_wing[0]]]
        self.left_wing_tagged = interest_points[wings_idx[idx_right_wing[1]]]

        right_gt_wing_span_chord = self.wing_span_chord(self.right_wing_tagged)
        left_gt_wing_span_chord = self.wing_span_chord(self.left_wing_tagged)




        self.right_gt_wing_span = right_gt_wing_span_chord[0]
        self.right_gt_wing_chord = right_gt_wing_span_chord[1]

        self.left_gt_wing_span = left_gt_wing_span_chord[0]
        self.left_gt_wing_chord = left_gt_wing_span_chord[1]

        # self.right_wing_tagged = self.trim_tagged_to_wing(interest_points[wings_idx[idx_right_wing[0]]], np.vstack([self.right_wing_le,self.right_wing_le]), gt_span_right[0])
        # self.left_wing_tagged = self.trim_tagged_to_wing(interest_points[wings_idx[idx_right_wing[1]]], np.vstack([self.left_wing_le,self.left_wing_le]), gt_span_left[1])
        
        
        # self.right_wing_boundary = self.trim_tagged_to_wing(self.right_wing_boundary,interest_points[wings_idx[idx_right_wing[0]]], gt_span_right[0])
        # self.left_wing_boundary = self.trim_tagged_to_wing(self.left_wing_boundary,interest_points[wings_idx[idx_right_wing[1]]], gt_span_left[1])
        

        for wing_name in ['right_wing_tagged','left_wing_tagged','right_wing_boundary','left_wing_boundary']:
            reordered = self.reorder_boundary(wing_name)
            span = getattr(self,f'{wing_name.split("_wing")[0]}_gt_wing_span')
            chord = getattr(self,f'{wing_name.split("_wing")[0]}_gt_wing_chord')

            le = reordered[0:reordered.shape[0]//2 + 1,:]
            te = reordered[reordered.shape[0]//2:,:]
            setattr(self,f'{wing_name}',reordered)

            setattr(self,f'{wing_name}_le',le)
            setattr(self,f'{wing_name}_te',te)



        
        # self.interest_left_wing_boundry = interest_points[wings_idx[idx_right_wing[1]]]
        # self.interest_right_wing_boundry = interest_points[wings_idx[idx_right_wing[0] ]]


        # min_max_bound_on_span_left = np.dot(np.vstack([self.left_wing_le,self.left_wing_le]),self.left_wing_span)

        # self.interest_right_wing_boundry = Utils.cyclic_sort(point_on_vec_right_interest,self.right_wing_span,self.right_wing_chord)
        # self.interest_left_wing_boundry = Utils.cyclic_sort(point_on_vec_left_interest,self.left_wing_span,self.left_wing_chord)

        # self.interest_right_wing_boundry = np.vstack((self.interest_right_wing_boundry,self.interest_right_wing_boundry[0]))
        # self.interest_left_wing_boundry = np.vstack((self.interest_left_wing_boundry,self.interest_left_wing_boundry[0]))

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
        # add if all inf return None/inf...
        projection = [self.point_to_segment_projection(points, points_of_line[k], points_of_line[k+1]) for k in range(points_of_line.shape[0] - 1)]
        all_inf = all(math.isinf(p) for p in projection)
        if all_inf == True: 
            return float('inf')
        else:
            return np.argmin(projection)


    def run_all_points_get_closest(self, points, points_of_line):
        return [self.get_indices_closest_points_to_line(points[idx], points_of_line) for idx in range(points.shape[0])]


    def get_projected_points_on_line(self,points_to_project_on_line,line_points,indices):
        
        dist = []
        for idx,val in enumerate(indices):
            if val == float('inf'):
                closest_point = np.argmin(np.linalg.norm(points_to_project_on_line[idx] - line_points,axis = 1))
                dist.append(line_points[closest_point])
            else:
                dist.append(Utils.project_point_on_line(points_to_project_on_line[idx],line_points,indices[idx]))
        #ADD - if inf dist from vlosest point
        return np.vstack(dist)

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
        fitted = getattr(self,att_to_calc[0])
        original = getattr(self, att_to_calc[1])
        bound_on_ew = (self.ew_to_lab.T @ fitted.T ).T
        interest_to_ew = (self.ew_to_lab.T @ original.T).T
        projected_interest = [np.fliplr(frame2d.project_with_proj_mat(interest_to_ew)[:,0:2]) for frame2d in self.frames]
        projected_gs = [np.fliplr(frame2d.project_with_proj_mat(bound_on_ew)[:,0:2]) for frame2d in self.frames]
        return np.sqrt(np.sum((np.vstack(projected_interest) - np.vstack(projected_gs))**2, axis = 1))
    
    def calculate_3d_dist(self,att_to_calc):
        return np.sqrt(np.sum((getattr(self,att_to_calc[0]) - getattr(self,att_to_calc[1]))**2, axis = 1))*1000

    
    def get_all_interest_2d_projection(self):
        self.projected_interest = np.stack([np.fliplr(frame.project_with_proj_mat(self.interest_points_3d)[:,0:2]) for frame in self.frames])
        self.projected_gaussians_closest_to_interest = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])

        
    
        self.dist_from_interest_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points)**2,axis = 2))



        self.dist_from_interest_point = np.sqrt(np.sum((self.gaussian_closest_to_interest_ew - self.interest_points_3d)**2,axis = 1))
        self.dist_interest_from_projected = np.sqrt(np.sum((self.projected_interest - self.interest_points)**2,axis = 2))

        self.projected_interest_gaussians = np.stack([frame.project_with_proj_mat(self.gaussian_closest_to_interest_ew)[:,0:2] for frame in self.frames])
        self.dist_from_gaussians_point_2d = np.sqrt(np.sum((self.projected_interest_gaussians[...,::-1] - self.interest_points)**2,axis = 2))
        self.dist_from_gaussians_point = np.sqrt(np.sum((self.interest_points_closest_to_gaussian_ew - self.interest_points_3d)**2,axis = 1))

    def calculate_error(self, suffix, model_name):
        

        att_to_calc_tagged_rw = [f"interest_on_bound_rw_{suffix}",f"right_wing_tagged_{suffix}"] 
        att_to_calc_tagged_lw = [f"interest_on_bound_lw_{suffix}",f"left_wing_tagged_{suffix}"]

        att_to_calc_gs_rw = [f'bound_on_interest_rw_{suffix}',f'right_wing_boundary_{suffix}' ]
        att_to_calc_gs_lw = [f'bound_on_interest_lw_{suffix}',f'left_wing_boundary_{suffix}']

        if 'right' in model_name: 
            calc_error_part = [att_to_calc_tagged_rw,att_to_calc_gs_rw]
        elif 'left' in model_name:
            calc_error_part = [att_to_calc_tagged_lw,att_to_calc_gs_lw]
        else:
            calc_error_part = [att_to_calc_tagged_rw,att_to_calc_tagged_lw,att_to_calc_gs_rw,att_to_calc_gs_lw]


 
        error_3d = [self.calculate_3d_dist(tagged_gs) for tagged_gs in calc_error_part]
        error_2d = [self.calculate_repreojection_error(tagged_gs) for tagged_gs in calc_error_part]
        return error_3d,error_2d
    
    def calculate_chamfler(self, model_name):
        num_wings = 1 if ('right' in model_name )| ('left' in model_name) else 2
        self.error_3d_le,self.error_2d_le = self.calculate_error('le', model_name)
        self.error_3d_te,self.error_2d_te = self.calculate_error('te', model_name)

        self.error_3d_chamfer_wing = sum([np.mean(np.hstack((self.error_3d_te[idx],self.error_3d_le[idx])))/num_wings for idx in range(len(self.error_3d_le))])*1000 # le_te right- gs to bound + le_te_left_gs_to_bound + le_te right- bound to gs + le_te_left bound to gs
        self.error_2d_chamfer_wing = sum([np.mean(np.hstack((self.error_2d_te[idx],self.error_2d_le[idx])))/num_wings for idx in range(len(self.error_2d_le))]) # le_te right- gs to bound + le_te_left_gs_to_bound + le_te right- bound to gs + le_te_left bound to gs
        

    # def calculate_chamfer_dist(self):
    #    ( np.mean(self.error2d_gt_on_boundary_to_gt) + np.mean(self.error3d_gt_on_boundary_to_gt))/2

    def homog_and_zbuff(self,points):
        homo_points = np.column_stack((points,np.ones((points.shape[0],1))))
        zbuff_ew=[self.frames[cam].z_buffer(homo_points)[0] for cam in range(4)]
        return np.vstack([np.dot(self.ew_to_lab,zbuff[:,0:3].T).T for zbuff in zbuff_ew])

    def load_hull_calc_xbody_dot_per_idx(self,zbuff_hull):
        """load body hull for each index, calculate the dot product of the X axis and the chamfler distance

        Args:
            zbuff_hull (dict): a dictionary with the body hull in index 0 and the x axis in index 1
        """
        self.load_hull_gs_body( zbuff_hull[0])
        self.hull_xbody_gs = np.abs(np.dot(self.xbody,zbuff_hull[1]))



    def load_hull_gs_body(self, hull):
        self.hull_ew = hull
        # self.zbuff_hull_lab = self.homog_and_zbuff(hull)
        self.zbuff_body = self.homog_and_zbuff(self.body_ew)
        self.cm_zbuff_body = np.mean(self.zbuff_body,axis = 0)
        self.cm_hull_ew = np.mean(self.hull_ew,axis = 0)

    def calculate_chamfler_body(self):
        self.closest_points_hull_to_body = Utils.find_closest_points_inptclouds(self.hull_ew,self.zbuff_body)
        self.closest_points_body_to_hull = Utils.find_closest_points_inptclouds(self.zbuff_body,self.hull_ew)
        errors_3d_1 = np.sqrt(np.sum((self.closest_points_hull_to_body - self.hull_ew)**2,axis = 1))  
        errors_3d_2 = np.sqrt(np.sum((self.closest_points_body_to_hull - self.zbuff_body)**2,axis = 1))  
        self.error_3d_chamfer_body = (np.mean(errors_3d_1*1000*1000) + np.mean(errors_3d_2*1000*1000))

         
    

    def calculate_chamfler_body_2d(self):
        
        errors_2d_1 = self.calc2d_error_body( self.closest_points_hull_to_body,self.hull_ew)
        errors_2d_2 = self.calc2d_error_body( self.closest_points_body_to_hull,self.zbuff_body)
        self.error_2d_chamfer_body = (np.mean(errors_2d_1) + np.mean(errors_2d_2))

        
    

    
    def calc2d_error_body(self,fitted,original):
        bound_on_ew = (self.ew_to_lab.T @ fitted.T ).T
        interest_to_ew = (self.ew_to_lab.T @ original.T).T
        projected_interest = [np.fliplr(frame2d.project_with_proj_mat(interest_to_ew)[:,0:2]) for frame2d in self.frames]
        projected_gs = [np.fliplr(frame2d.project_with_proj_mat(bound_on_ew)[:,0:2]) for frame2d in self.frames]
        diff = np.vstack(projected_interest) - np.vstack(projected_gs)
        return np.linalg.norm(diff, axis=1)  # Euclidean reprojection error per point
        # errors_2d_l1 = np.sum(np.abs(diff),axis = 1)  # Euclidean reprojection error per point
    


    def add_z_buff_to_class(self,hull_to_compare):
        self.zbuff_hull = self.homog_and_zbuff( hull_to_compare) 
        self.zbuff_model = self.homog_and_zbuff( self.xyz) 
        
