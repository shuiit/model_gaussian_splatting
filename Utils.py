# 
import numpy as np
import scipy
import pickle
from scipy.spatial import cKDTree
from math import atan2
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
import scipy.io as sio
import pandas as pd


def find_closest_points_inptclouds(points1,points2):
    # Two 3D point clouds: points1 (Nx3), points2 (Mx3)
    tree = cKDTree(points2)

    # For each point in A, find closest in B
    distances, indices = tree.query(points1)  # distances: (N,), indices: (N,)

    # Closest points from B
    closest_points = points2[indices]
    return closest_points

def rotate_vector_direction_and_clip(rotation_matrix, vector_points, scale_vector):
    
    rotated_vector = np.dot(rotation_matrix,vector_points.T).T

    vector_dir = np.array(rotated_vector[0] - rotated_vector[1] )
    vector_dir_norm= (vector_dir/np.linalg.norm(vector_dir))

    return rotated_vector + vector_dir_norm*scale_vector



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])



def triangulate_least_square(origins,end_of_vectors):
    # triangulate all lines to find the closest 3d point with least square
    # we define a 3d vector ab and a point p
    # the distance between the point and the vector: d^2 = |p-a|^2 - |(p-a).T *(b-a)/|(b-a)|^2 where (p-a).T *(b-a)/|(b-a) is the projection of ap on ab
    # d^2 = |p-a|^2 - |(p-a).T *(b-a)/|(b-a)|| = |(p-a)*(p-a).T| - |(p-a).T *(b-a)/|(b-a)||^2 = |(p-a)*(p-a).T| - |(p-a).T *n||^2 where n = (b-a)/|(b-a)|
    # we sum the squared distances and get 
    # sum(di^2) = sum(|(p-a)*(p-a).T| - |(p-a).T *n||^2)
    # we want to find the minimum of the sums of distences - the point that is closest to all lines so we differentiate with respect to p and get: 


    # sum([2*(p-a) - 2*[(p-a)^T*n]]*n) = 0
    # sum(p-ai) = sum(n*n^T)*(p-a) --> sum(n*n.T - I)*p = sum((n*n.T - I)*a) --> S*p = C (n*n.T is the outer product, not dot) for every vector we multiply it with itself to get vx^2,vxy,vxz,vy^2...


    #** we can also calculate the distance d using cross product: we define a vector ab and a point p, we know that |ab X ap| will result the area of a parallalegram. 
    # we also know that d*|ab| is the area of a parallalegram --> d*|ab| = |ab X ap| --> d = |ab X ap|/|ab| which is the distace between the point p and the vector ab
    # (we can differenciate the same way using an identity for the cross - https://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines )


    n = (end_of_vectors - origins)/np.linalg.norm(end_of_vectors - origins, axis = 1)[:,np.newaxis]
    inner = [np.outer(n_row,n_row.T) - np.eye(3) for n_row in n]
    s = np.sum(inner,axis = 0)
    c = np.sum([np.dot(mat,vec) for mat,vec in zip(inner,origins)],axis = 0)
    return  np.linalg.solve(s,c)

def dist_points(x1,x2):
    return np.sqrt(np.sum((x1 - x2)**2, axis = 1))

def project_point_on_line(points_to_project_on_line,line_points,indices):

    points_of_line = line_points[indices:indices + 2]
    line = (points_of_line[1] - points_of_line[0])/np.linalg.norm((points_of_line[1] - points_of_line[0]))
    return np.dot(points_to_project_on_line - points_of_line[0],line)*line + points_of_line[0]


def cyclic_sort(points,span,chord):
    points_2d = project_to_plane(points, np.mean(points,axis = 0), span, chord)
    cyclic_points = rotational_sort(points_2d, np.mean(points_2d,axis = 0), clockwise=True)
    first_index = np.argmin(np.dot(span,points[cyclic_points].T))
    return  points[np.roll(cyclic_points,first_index)]
         

def point_to_segment_projection(point, origin, point_line):
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


def project_to_plane(points, origin, x_axis, y_axis):
    centered = points - origin
    x_coords = np.dot(centered, x_axis)
    y_coords = np.dot(centered, y_axis)
    return np.stack((x_coords, y_coords), axis=1)

def fit_poly(pts, degree = 2, num_of_fit_point = 1000):

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.insert(np.cumsum(dists), 0, 0)  # insert 0 at the beginning
    p = [np.polyfit(t, pts, degree) for pts in pts.T]
    t_fit = np.linspace(t[0], t[-1], num_of_fit_point)
    return np.vstack([np.polyval(p, t_fit) for p in p]).T


def fit_all_points(points,skip_points = 3, **kwargs):
    
    pts_to_fit = [points[k:k+skip_points] for k in range(0,points.shape[0],skip_points) ]
    return  np.vstack([fit_poly(pts, **kwargs) for pts in pts_to_fit[:-1] ])


def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    #https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python 
    # from Boris Gorelik
    return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(list_of_xy_coords, centre_of_rotation_xy_coord, clockwise=True):
    cx,cy=centre_of_rotation_xy_coord
    angles = [atan2(x-cx, y-cy) for x,y in list_of_xy_coords]
    indices = argsort(angles)
    # if clockwise:
    #     return [list_of_xy_coords[i] for i in indices]
    # else:
    #     return [list_of_xy_coords[i] for i in indices[::-1]]
    return indices

def intersection_per_cam(frames_per_cam, cam_num, ptcloud_volume, tol=1.0):
    """Efficiently finds intersecting 3D points projected onto a camera image plane."""
    
    # ptsv = frames_per_cam[cam_num].homogenize_coordinate(ptcloud_volume)
    pt2dv = frames_per_cam[cam_num].project_with_proj_mat(ptcloud_volume)[:,0:2]
    pt2dv = np.fliplr(pt2dv)  # Flip x-y coordinates if needed

    # Build KDTree for fast pixel search
    pixel_tree = cKDTree(frames_per_cam[cam_num].pixels)

    # Find pixels that are close to projected 2D points
    indices = pixel_tree.query_ball_point(pt2dv, r=tol)
    
    # Convert list of indices to a mask for filtering
    valid_mask = np.array([len(n) > 0 for n in indices])

    return ptcloud_volume[valid_mask]


def delete_after_projection(frames_per_cam,pt_cloud):
    for idx in range(4):
        pt_cloud = frames_per_cam[idx].intersection_per_cam(pt_cloud)
    return pt_cloud


def pickle_file(dict, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict, f)

def stack_filter_hist_points_2d(frames_list, top_perc_ol,points_to_plot_rwing,points_to_plot_lwing):
    hist_points = np.stack([frame.dist_from_interest_point_2d for frame in frames_list ])
    hist_points = np.stack([[np.sort(hist_points[:,cam,idx])[0:int(len(hist_points[:,cam,idx]) - top_perc_ol*len(hist_points[:,cam,idx]))] for cam in  range(4)] for idx in range(hist_points.shape[2])])
    hist_points = np.swapaxes(hist_points,0,2)
    return  [np.vstack((hist_points[:,:,points_to_plot_rwing],hist_points[:,:,points_to_plot_lwing]))[:,:,idx].flatten() for idx,points in enumerate(points_to_plot_lwing)]



def stack_filter_hist_points_3d(frames_list, top_perc_ol,points_to_plot_rwing,points_to_plot_lwing):
    """Sort and cut off top percentage for outlier removal."""
    hist_points = np.stack([frame.dist_from_interest_point for frame in frames_list ])*1000
    hist_points = np.stack([np.sort(hist_points[:,idx])[0:int(len(hist_points[:,idx]) - top_perc_ol*len(hist_points[:,idx]))] for idx in range(hist_points.shape[1])])
    return [np.vstack((hist_points[points_to_plot_rwing,:],hist_points[points_to_plot_lwing,:]))[idx,:].flatten() for idx,points in enumerate(points_to_plot_lwing)]


def stack_filter_hist_all_2d(frames_list,points_to_plot,top_perc_ol):
    hist_points = np.stack([frame.dist_from_interest_point_2d[:,points_to_plot] for frame in frames_list ])
    hist_points = np.stack([[np.sort(hist_points[:,cam,idx])[0:int(len(hist_points[:,cam,idx]) - top_perc_ol*len(hist_points[:,cam,idx]))] for cam in  range(4)] for idx in range(hist_points.shape[2])])
    return np.swapaxes(hist_points,0,2)


def stack_filter_hist_all_3d(frames_list,top_perc_ol):
    hist_points_3d = np.stack([frame.dist_from_interest_point for frame in frames_list ])
    return np.stack([np.sort(hist_points_3d[:,idx])[0:int(len(hist_points_3d[:,idx]) - top_perc_ol*len(hist_points_3d[:,idx]))] for idx in range(hist_points_3d.shape[1])])


def make_body_hull_file(nominal_initial_angles,file_path_save):


    hull_body = {}

    for mov_frame in list(nominal_initial_angles.keys())[1:]:
        mov = int(mov_frame.split('_')[1]) 
        frame_num = int(mov_frame.split('_')[3]) 
        hull_path = f'H:/My Drive/dark 2022/2023_08_09_60ms/hull/hull_Reorder/mov{mov}/hull_op/'
        hull = sio.loadmat(f'{hull_path}/hull3d_mov{mov}')['hull3d']
        shull = sio.loadmat(f'{hull_path}/Shull_mov{mov}')['Shull']
        frame_in_hull = np.where(shull['frames'][0][0][0] == frame_num - 1)[0][0]
        hull_idx = hull['body'][0][0]['body4plot'][0][0][frame_in_hull][0]
        hull_3d = np.vstack([np.array(shull['real_coord'][0][0][0][frame_in_hull][idx][0])[0][np.array(hull_idx[:,idx])] for idx in range(3)]).T
        hull_body[mov_frame] = hull_3d

    pickle_file(hull_body,file_path_save)


def load_body_hull_calc_xbody(frame,hull_mov):
    # load body ground truth, do z buffer to get only the outside
    hull = frame.homog_and_zbuff( hull_mov)
    xbody = frame.get_principle_axes(hull)[0]
    xbody = frame.get_axis_orientation(xbody,[[0,0,0]],[[0,0,1]])
    xbody,bottom,top,x_ax_points= frame.reorient_axis(hull,xbody)
    return [hull, xbody]

def get_camfer_stats(angle_name,delta_angles,chamfer):
    return pd.DataFrame(data = {f'delta {angle_name}': delta_angles,'mean': np.mean(chamfer,axis = 0) ,'std': np.std(chamfer,axis = 0),
                   'median':np.median(chamfer,axis = 0),'max':np.max(chamfer,axis = 0),'min':np.min(chamfer,axis = 0)})


# def intersection_per_cam(frames_per_cam,cam_num,ptcloud_volume):    
#     ptsv = frames_per_cam[cam_num].homogenize_coordinate(ptcloud_volume)
#     pt2dv = frames_per_cam[cam_num].project_on_image(ptsv)
#     pt2dv = np.fliplr(pt2dv)
#     pts_for_unique = np.vstack((frames_per_cam[cam_num].pixels,np.unique(pt2dv.astype(int),axis = 0)))
#     v,cnt = np.unique(pts_for_unique,return_counts = True,axis = 0)
#     projected_on_image = v[cnt > 1]
#     all_indices = np.vstack(np.argwhere(np.all(pt2dv.astype(int) == repeated_group, axis=1)) for repeated_group in projected_on_image)
#     return ptcloud_volume[all_indices[:,0]]