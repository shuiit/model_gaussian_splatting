
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
import time


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
            self.body_cm = np.mean(self.body,axis = 0)

            self.right_wing, self.left_wing= {},{}

            self.right_wing['xyz_lab'] = self.xyz_rotated[self.idx_parts[1],:]
            self.left_wing['xyz_lab'] = self.xyz_rotated[self.idx_parts[2],:]
            self.right_wing['mean'] = np.mean(self.right_wing['xyz_lab'] ,axis = 0)
            self.left_wing['mean'] = np.mean(self.left_wing['xyz_lab'] ,axis = 0)


            self.body_ew = self.xyz[self.idx_parts[0],:]
            self.right_wing['xyz_ew'] = self.xyz[self.idx_parts[1],:]
            self.left_wing['xyz_ew'] = self.xyz[self.idx_parts[2],:]
            # self.calc_wing_le_te(letedict['num_of_bins'] ,letedict['perc_wing_for_le'],letedict['wing_length_snip'])
            xbody = self.get_principle_axes(self.body)[0]
            self.xbody = self.get_axis_orientation(xbody,[[0,0,0]],[[0,0,1]])
            self.xbody,bottom,top,x_ax_points= self.reorient_axis(self.body,xbody)

                
        self.opacity = 1 / (1 + np.exp(-vertices["opacity"]))

        self.frame0 = frame0



    # def zscore(self,points):
    #     nrml = np.cross(self.right_wing['span'],self.right_wing['chord'])
    #     pts_on_nrml = np.dot(points,nrml)
    #     std = np.std(pts_on_nrml)
    #     mean = np.mean(pts_on_nrml)
    #     return points[((pts_on_nrml - mean)/std) < 1.5]

    def rotate_to_ew_and_project(self,points):
        points_closest_to_fitted_ew =  (self.ew_to_lab.T @ np.vstack(points).T).T
        return np.stack([frame.project_with_proj_mat(points_closest_to_fitted_ew)[:,0:2] for frame in self.frames])


    def get_principle_axes(self,frame_xyz):
        body_cm = np.mean(frame_xyz,axis = 0)
        body_centered = frame_xyz - body_cm
        U, S, Vt = svd(body_centered, full_matrices=False)
        return Vt

    def get_axis_orientation(self,axis,points_from,points_to):
        direction = (np.mean(points_to,axis = 0) - points_from)/np.linalg.norm(np.mean(points_to,axis = 0) - points_from)
        return -axis if np.dot(direction,axis) < 0 else axis
    
    
    def reorient_axis(self,points,direction,percent_bot = 0.2,percent_top = 0.2):
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
    
    def wing_span_chord(self,wing):
        
        wing_axes = self.get_principle_axes(wing)
        wing_span = self.get_axis_orientation(wing_axes[0],self.body_cm,wing)
        wing_chord = self.get_axis_orientation(wing_axes[1],[[0,0,0]],[[0,0,1]])
        return wing_span/np.linalg.norm(wing_span),wing_chord/np.linalg.norm(wing_chord)
    

    def get_span(self,wing):

        approx_span, wing['chord'] = self.wing_span_chord(wing['xyz_lab'])
        projected_span = np.dot(wing['xyz_lab'] - wing['mean'],approx_span)
        tip_side = np.max(projected_span)
        root_side = np.min(projected_span)
        wing_size = np.abs(tip_side - root_side)
        tip_chunck = wing['xyz_lab'][projected_span > (tip_side - wing_size*0.1),:]
        wing['tip_mean'] = np.mean(tip_chunck,axis = 0)
        wing['span'] = (wing['tip_mean'] - wing['mean'] ) / np.linalg.norm(wing['tip_mean'] - wing['mean'])
         


    def get_le_te(self,wing):
        projected_chord = np.dot(wing['xyz_lab'] - wing['mean'],wing['chord'])
        wing['le'] = wing['xyz_lab'][projected_chord > 0]
        wing['te'] = wing['xyz_lab'][projected_chord <= 0]



    def bin_le_te(self,le_te,wing, num_of_bins = 50):
        
        projected_le_te = np.dot(le_te - wing['mean'],wing['span'])
        diff = (max(projected_le_te) - min(projected_le_te))/num_of_bins
        bin_edges = np.arange(np.min(projected_le_te), np.max(projected_le_te) + diff, diff)
        return np.digitize(projected_le_te, bins=bin_edges)

    def get_max_le_te(self,bin_indices_le,idx,projected_chord,le_te):
        le_bin = le_te[bin_indices_le == idx]
        if len(le_bin) > 0:
            return le_bin[np.argmax(projected_chord[bin_indices_le == idx])]
        else:
            return [None]*3


    def get_le_te_bins(self,le_te,wing,num_of_bins):
        le_te_sign = 1 if le_te == 'le' else -1
        bin_indices_le = self.bin_le_te(wing[le_te], wing,num_of_bins)
        le_te_projected_chord = le_te_sign*np.dot(wing[le_te] -  wing['mean'],wing['chord'])
        le_te_bins = [self.get_max_le_te(bin_indices_le,idx,le_te_projected_chord,wing[le_te]) for idx in bin_indices_le] 
        wing[f'{le_te}_bins'] = np.vstack(le_te_bins)

    def approx_le(self,wing, perc_wing_for_le = 0.7, key = 'le_ransac'):
        le_projected_span = np.dot(wing[f'le_bins'] -  wing['mean'], wing['span'])
        tip_side = np.max(le_projected_span)
        root_side = np.min(le_projected_span)

        root_le = wing['le_bins'][le_projected_span < (root_side + (tip_side - root_side)*perc_wing_for_le)]


        model_robust, inliers = ransac(root_le, LineModelND, min_samples=2, residual_threshold=10/100000, max_trials=50)
        origin, direction = model_robust.params
        wing[key] = [origin,direction]


    def check_direction_span_ransac(self, wing):
        if np.dot(wing['le_ransac'][1],wing['span']) < 0:
            wing['le_ransac'][1] = -wing['le_ransac'][1]
        

    def get_wing_root(self,wing,perc_wing_for_root = 0.1):
        le_projected_span = np.dot(wing[f'le_bins'] -  wing['mean'], wing['span'])
        tip_side = np.max(le_projected_span)
        root_side = np.min(le_projected_span)
        root_le = wing['le_bins'][le_projected_span < (root_side + (tip_side - root_side)*perc_wing_for_root)]
        wing['root'] = np.mean(root_le,axis=0)
        wing['root_le'] = wing['le_ransac'][0] + wing['le_ransac'][1]*np.dot(wing['le_ransac'][1],(wing['root'] - wing['le_ransac'][0]))


    def calculate_chord(self,wing):
        wing['tip_le'] =  wing['le_ransac'][0] + wing['le_ransac'][1]*np.dot(wing['le_ransac'][1],(wing['tip_mean'] - wing['le_ransac'][0]))

        v1 = wing['tip_le'] - wing['tip_mean']
        v2 = wing['root_le'] - wing['tip_mean']
        norm_le = np.cross(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2))
        chord = np.cross(wing['le_ransac'][1],norm_le)
        wing['chord'] = chord /  np.linalg.norm(chord)


    def calculate_ybody(self,perc_wing_for_root = 0.1):
        self.get_wing_root(self.right_wing,perc_wing_for_root = perc_wing_for_root)
        self.get_wing_root(self.left_wing,perc_wing_for_root = perc_wing_for_root)

        ybody = self.right_wing['root_le'] - self.left_wing['root_le']
        ybody = ybody/np.linalg.norm(ybody)

        self.ybody = ybody 



    def calculate_zbody(self):
        zbody = np.cross(self.ybody,self.xbody)
        zbody = zbody/np.linalg.norm(zbody)
        self.zbody = zbody
        self.xbody = np.cross(self.zbody,self.ybody)
        self.xbody = self.xbody/np.linalg.norm(self.xbody)
        self.sp_normal = self.rodrigues_rotate_vector(self.zbody, self.ybody, -45*np.pi/180)
        

    def calculate_phi(self,wing, left):
        sign_left = -1 if left== 1 else 1
        le_on_sp = self.project_on_plane( self.sp_normal, wing['le_ransac'][1])
        xbody_on_sp = self.project_on_plane( self.sp_normal, self.xbody)
        ybody_on_sp = np.cross(xbody_on_sp,self.sp_normal)
        ybody_on_sp = ybody_on_sp / np.linalg.norm(ybody_on_sp)
        xle = np.dot(le_on_sp,xbody_on_sp)
        yle = np.dot(le_on_sp,ybody_on_sp)
        phi = np.arctan2(sign_left*yle,xle) % (2*np.pi)*180/np.pi
        wing['phi'] = phi if phi < 250 else phi - 360


    def calculate_theta(self,wing):
        wing['theta'] = 90 - np.arccos(np.dot( self.sp_normal,wing['le_ransac'][1]))*180/np.pi

    def project_on_plane(self, normal, vector):
        projected = vector - np.dot(normal,vector)*normal
        return projected/np.linalg.norm(projected)




    def calcultae_psi(self,wing,left ):
        if left == 1:
            le_sp_normal = np.cross(self.sp_normal, wing['le_ransac'][1])
            signy = -1
        else:
            le_sp_normal = np.cross( wing['le_ransac'][1],self.sp_normal)
            signy = 1

        le_sp_normal = le_sp_normal / np.linalg.norm(le_sp_normal)
        sp_chord = np.cross(wing['le_ransac'][1],le_sp_normal)
        sp_chord = sp_chord / np.linalg.norm(sp_chord)

        ypsi = signy*np.dot(wing['chord'],sp_chord)
        xpsi = np.dot(wing['chord'],le_sp_normal)
        psi = np.arctan2(ypsi,xpsi) 
        psi = np.unwrap(psi)  
        wing['psi'] = np.mod(psi, 2*np.pi) * 180/np.pi 

    def rodrigues_rotate_vector(self,v, k, theta):
        """
        Rotates a vector v by an angle theta around a unit vector axis k.

        Args:
            v (np.array): The 3D vector to rotate.
            k (np.array): The 3D unit vector representing the rotation axis.
            theta (float): The rotation angle in radians.

        Returns:
            np.array: The rotated 3D vector.
        """
        k = k / np.linalg.norm(k)  # Ensure k is a unit vector
        v_rotated = v * np.cos(theta) + \
                    np.cross(k, v) * np.sin(theta) + \
                    k * np.dot(k, v) * (1 - np.cos(theta))
        return v_rotated


    def wings_parameters(self, wing):
        t0 = time.time()
        self.get_span(wing)
        self.get_le_te(wing)
        self.get_le_te_bins('le',wing,num_of_bins = 50)
        self.get_le_te_bins('te',wing,num_of_bins = 50)
        self.approx_le(wing)
        self.check_direction_span_ransac( wing)

        
    # def wings_parameters(self, wing):
    #     timings = {}

    #     t0 = time.perf_counter()
    #     self.get_span(wing)
    #     timings['get_span'] = time.perf_counter() - t0

    #     t0 = time.perf_counter()
    #     self.get_le_te(wing)
    #     timings['get_le_te'] = time.perf_counter() - t0

    #     t0 = time.perf_counter()
    #     self.get_le_te_bins('le', wing, num_of_bins=50)
    #     timings['get_le_te_bins_le'] = time.perf_counter() - t0

    #     t0 = time.perf_counter()
    #     self.get_le_te_bins('te', wing, num_of_bins=50)
    #     timings['get_le_te_bins_te'] = time.perf_counter() - t0

    #     t0 = time.perf_counter()
    #     self.approx_le(wing)
    #     timings['approx_le'] = time.perf_counter() - t0

    #     t0 = time.perf_counter()
    #     self.check_direction_span_ransac(wing)
    #     timings['check_direction_span_ransac'] = time.perf_counter() - t0

    #     # attach timings to the wing for later inspection
        # wing['timings'] = timings
        

    def calculate_wing_angles(self,wing, left):
        self.calculate_phi(wing, left)
        self.calculate_theta(wing)
        self.calcultae_psi(wing,left )
        

    

