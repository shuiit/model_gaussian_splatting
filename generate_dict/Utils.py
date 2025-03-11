
import numpy as np
import scipy
import pickle



def rotate_vector_direction_and_clip(rotation_matrix, vector_points, scale_vector):
    
    rotated_vector = np.dot(rotation_matrix,vector_points.T).T

    vector_dir = np.array(rotated_vector[0] - rotated_vector[1] )
    vector_dir_norm= (vector_dir/np.linalg.norm(vector_dir))

    return rotated_vector + vector_dir_norm*scale_vector



def get_dict_for_points3d(frames):
    """
    Create dictionaries mapping 3D voxel positions and their mean colors.

    This function processes voxel data from multiple frames to generate a dictionary
    that associates unique voxel identifiers with their 3D positions and another dictionary 
    that maps voxel identifiers to their corresponding mean color values.

    Args:
        frames (dict): A dictionary containing frame data, where each key is an image identifier 
                       and each value has attributes for voxel indices and pixel colors.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary mapping voxel identifiers to their 3D positions (list of floats).
            - dict: A dictionary mapping voxel identifiers to their mean color values (list of floats).
    """

    # generates the dictionary of all 3d-2d mappings from every frame. 
    colors = np.hstack([frames[im_name].color_of_pixel for im_name in frames.keys()])
    all_voxels = np.column_stack((np.vstack(([(frames[im_name].voxels_with_idx) for im_name in frames.keys()])),colors))
    unique_voxels  = np.unique(all_voxels[:,0:4],axis = 0)

    voxel_dict = {vxl[3]:list(vxl[0:3]) for vxl in unique_voxels}
    [voxel_dict[vxl[3]].extend(vxl[4:-1]) for vxl in all_voxels]

    # calculate mean color
    colors_dict = {vxl[3]:[] for vxl in unique_voxels}
    for vxl in all_voxels:
        colors_dict[vxl[3]].extend(vxl[6:])
    colors_dict = {key: [np.mean(np.array(values)).astype(int)]*3 for key, values in colors_dict.items()}
    return voxel_dict,colors_dict



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

def load_hull(body_wing,path):
    """
    Load the 3D hull points for a specified body part from a .mat file.

    Args:
        body_wing (str): The name of the body part ('body', 'rwing', or 'lwing').
        path (str): The directory path where the .mat file is located.

    Returns:
        numpy.ndarray: An array containing the 3D hull points for the specified body part.
    """
    return scipy.io.loadmat(f'{path}/3d_pts/{body_wing}.mat')['hull']




def pickle_file(dict, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict, f)

