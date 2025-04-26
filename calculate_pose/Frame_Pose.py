
import numpy as np
from scipy.linalg import svd
from skimage.measure import LineModelND, ransac


class Frame_Pose():

    def __init__(self,xyz_rotated,frame,idx_parts,frame0):
        """Initialize the camera object with parameters, intrinsic and extrinsic matrices."""
        frame_idx = idx_parts[frame - frame0]
        self.frame = frame
        self.body_xyz = xyz_rotated[frame - frame0][frame_idx[0],:]
        self.right_wing_xyz = xyz_rotated[frame - frame0][frame_idx[1],:]
        self.left_wing_xyz = xyz_rotated[frame - frame0][frame_idx[2],:]
        
        self.body_cm = np.mean(self.body_xyz,axis = 0)
        xbody = self.get_principle_axes(self.body_xyz)[0]
        self.xbody = self.get_axis_orientation(xbody,[[0,0,0]],[[0,0,1]])
        self.xbody,bottom,top = self.reorient_axis(self.body_xyz,xbody,percent = 0.2)

        self.right_wing_span,self.right_wing_chord = self.wing_span_chord(self.right_wing_xyz)
        self.left_wing_span,self.left_wing_chord = self.wing_span_chord(self.left_wing_xyz)
        
        self.right_wing_origin, self.right_wing_direction,self.right_wing_le = self.wing_le(self.right_wing_xyz,self.right_wing_span,self.right_wing_chord)
        self.left_wing_origin, self.left_wing_direction,self.left_wing_le = self.wing_le(self.left_wing_xyz,self.left_wing_span,self.left_wing_chord)



    def get_principle_axes(self,frame_xyz):
        body_cm = np.mean(frame_xyz,axis = 0)
        body_centered = frame_xyz - body_cm
        U, S, Vt = svd(body_centered, full_matrices=False)
        return Vt

    def get_axis_orientation(self,axis,points_from,points_to):
        direction = (np.mean(points_to,axis = 0) - points_from)/np.linalg.norm(np.mean(points_to,axis = 0) - points_from)
        return -axis if np.dot(direction,axis) < 0 else axis
    

    
    def reorient_axis(self,points,direction,percent = 0.2):
        projected_on_body = np.dot(points,direction)
        min_points = min(projected_on_body)
        max_points = max(projected_on_body)
        perc_of_body_length = (max_points - min_points)*percent
        bottom = points[(projected_on_body  < (min_points + perc_of_body_length)),:]
        top = points[(projected_on_body  > (max_points - perc_of_body_length)),:]
        x_ax = np.mean(top,axis = 0) - np.mean(bottom,axis = 0)
        return x_ax/np.linalg.norm(x_ax),bottom,top
    
    def wing_span_chord(self,wing_xyz):
        
        wing_axes = self.get_principle_axes(wing_xyz)
        wing_span = self.get_axis_orientation(wing_axes[0],self.body_cm,wing_xyz)
        wing_chord = self.get_axis_orientation(wing_axes[1],[[0,0,0]],[[0,0,1]])
        return wing_span,wing_chord
    

    
    def get_wing_le(self,xyz,span,chord, perc_wing = 0.7):

        projected_on_span = np.dot(xyz,span)

        half_wing = perc_wing*(max(projected_on_span) - min(projected_on_span))
        xyz_for_le = xyz[projected_on_span < (min(projected_on_span) + half_wing),:]
        projected_on_span = np.dot(xyz_for_le,span)
        projected_on_chord = np.dot(xyz_for_le,chord)


        diff = (max(projected_on_span) - min(projected_on_span))/100
        bin_edges = np.arange(np.min(projected_on_span), np.max(projected_on_span) + diff, diff)
        bin_indices = np.digitize(projected_on_span, bins=bin_edges)
        real_indices = np.array(range(len(projected_on_chord)))
        coord = []
        for idx in bin_indices:
            max_of_bin = np.argmax(projected_on_chord[bin_indices == idx])
            real_idx = real_indices[bin_indices == idx][max_of_bin]
            coord.append(xyz_for_le[real_idx,:])

        return np.vstack(coord)


    def ransac_for_le(self,wing_le):
        
        model_robust, inliers = ransac(wing_le, LineModelND, min_samples=2, residual_threshold=5/100000, max_trials=1000)
        origin, direction = model_robust.params
        return origin, direction
    

    def wing_le(self,wing_xyz,span,chord):
        wing_le = self.get_wing_le(wing_xyz,span,chord)
        wing_origin, r_wing_direction = self.ransac_for_le(wing_le)
        return wing_origin, r_wing_direction,wing_le


        
