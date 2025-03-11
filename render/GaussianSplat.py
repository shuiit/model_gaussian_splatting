
import numpy as np
import scipy.io
from plyfile import PlyData, PlyElement
import Plotters
import sh_utils

class GaussianSplat():
    """
        Initializes the GaussianSplat class with vertices, camera parameters, and Gaussian parameters.
        
        Args:
            path (str): Path to the .ply file containing vertex data.
            vertices (np.array): Array of vertices. Loaded from file if not provided.
            block_xy (list): Block size for 2D grid representation.
            image_size (list): Size of the image in pixels.
            sh (np.array): Spherical harmonics coefficients. Loaded from file if not provided.
        """
    def __init__(self,path = None,vertices = None,block_xy = [16,16], image_size = [160,160],sh = None):
        self.path = path
        self.vertices =  {prop: np.array(PlyData.read(path)["vertex"].data[prop].tolist()) for prop in PlyData.read(path)["vertex"].data.dtype.names} if vertices is None else vertices
        self.xyz = np.column_stack((self.vertices["x"], self.vertices["y"], self.vertices["z"]))
        if 'scale_2' not in self.vertices.keys():
            self.vertices['scale_2'] = self.vertices["scale_1"] * 0 + np.log(1.6e-10) 
        self.scale = np.exp(np.column_stack(([self.vertices["scale_0"], self.vertices["scale_1"], self.vertices["scale_2"]])))  
        self.opacity = 1 / (1 + np.exp(-self.vertices["opacity"]))#self.vertices["opacity"]         
        self.rot = np.column_stack([self.vertices["rot_0"], self.vertices["rot_1"], self.vertices["rot_2"], self.vertices["rot_3"]])
        self.rot = self.rot/np.linalg.norm(self.rot,axis = 1)[:,np.newaxis]
        self.sh = np.column_stack([self.vertices[key] for key in self.vertices.keys() if 'rest' in key or 'dc' in key]) if sh is None else sh
        self.image_size = image_size
        self.block_xy = block_xy
        self.grid = [int((self.image_size[0] + self.block_xy[0] - 1)/self.block_xy[0]),int((self.image_size[1] + self.block_xy[1] - 1)/self.block_xy[1])]
        self.get_color(0)
        

    def rearange_gs(self,idx_to_rearange):
        """
        Rearranges the Gaussian splats based on the given indices.
        
        Args:
            idx_to_rearange (np.array): Array of indices to rearrange vertices and associated properties.
        """
        # self.vertices = self.vertices[idx_to_rearange]
        self.xyz = self.xyz[idx_to_rearange]
        self.scale = self.scale[idx_to_rearange]
        self.opacity = self.opacity[idx_to_rearange]
        self.rot = self.rot[idx_to_rearange]
        self.color = self.color[idx_to_rearange]
        self.sh = self.sh[idx_to_rearange]


    def projection_filter(self,frames,point3d,**kwargs):
        """
        Filters projected 3D points to exclude background pixels across multiple frames.
        
        Args:
            frames (dict): Dictionary of frame objects for projection filtering.
            point3d (np.array): 3D points to project.
        
        Returns:
            np.array: Boolean array indicating which points are not in the background for any frame.
        """
        return np.column_stack([image.filter_projections_from_bg(point3d,**kwargs) for image in frames.values()]).any(axis = 1) == False


    def filter(self,filter_by,**kwargs):
        """
        Creates a new GaussianSplat object filtered by a boolean array.
        
        Args:
            filter_by (np.array): Boolean array to filter vertices.
        
        Returns:
            GaussianSplat: New GaussianSplat instance with filtered vertices.
        """
        filtered_vertices = {key: self.vertices[key][filter_by] for key in  self.vertices.keys() }
        return GaussianSplat(vertices = filtered_vertices, sh = self.sh[filter_by,:],**kwargs)

    def save_gs(self,name = '_filtered'):
        """
        Saves the filtered vertices to a new .ply file.
        
        Args:
            name (str): Suffix for the output filename.
        """
        dtype = [(key, "f4") for key in self.vertices.keys()]
        # Create a structured array
        structured_array = np.zeros(len(next(iter(self.vertices.values()))), dtype=dtype)
        for key in self.vertices:
            structured_array[key] = self.vertices[key]
        filtered_element = PlyElement.describe(structured_array, "vertex")
        PlyData([filtered_element]).write(f'{self.path.split(".ply")[0]}{name}.ply')

    def q_array_to_rotmat(self,q):
        """
        Converts quaternion array to rotation matrix.
        
        Args:
            q (np.array): Array of quaternions.
        
        Returns:
            np.array: Corresponding rotation matrix.
        """
        return np.column_stack(
            [1.0 - 2.0 * (q[:,2] * q[:,2] + q[:,3] * q[:,3]), 2.0 * (q[:,1]  * q[:,2] - q[:,0] * q[:,3] ), 2.0 * (q[:,1] * q[:,3]  + q[:,0] * q[:,2]),
            2.0 * (q[:,1] * q[:,2] + q[:,0] * q[:,3] ), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,3]  * q[:,3] ), 2.0 * (q[:,2] * q[:,3]  - q[:,0] * q[:,1]),
            2.0 * (q[:,1] * q[:,3]  - q[:,0] * q[:,2]), 2.0 * (q[:,2] * q[:,3]  + q[:,0] * q[:,1]), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,2] * q[:,2])])


    def calc_cov3d(self):
        """
        Calculates 3D covariance matrices based on scale and rotation (quaternion) for each Gaussian.
        """
        scale = np.eye(3) * self.scale[:,np.newaxis,:] # size of the gaussians along X,Y,Z
        q = (self.rot.T/ np.linalg.norm(self.rot,axis =1)).T # orientation of the gaussian (quaternion)
        rot_mat = self.q_array_to_rotmat(q) # converts the array of quaternions into rotation matrices
        rot_mat = rot_mat.reshape(rot_mat.shape[0],3,3) 
        scale_rot = rot_mat @ scale   # scales the rotation matrix -> generates a gaussian in the orientation of q. 
        self.cov3d = scale_rot @ scale_rot.transpose(0,2,1) # the covariance matrix 

    def calc_cov2d(self,camera, image_size = [160,160]):
        """
        Calculates 2D covariance matrices for each Gaussian projected onto an image plane.
        
        Args:
            camera: Camera object with intrinsic and extrinsic parameters.
            image_size (list): Size of the output image.
        """
        # camera parameters:
        fxfy = [camera.K[0,0],camera.K[1,1]]
        tan_fov = np.array([camera.focal2fov(focal, size) for focal,size in zip(fxfy,image_size)])
        viewmat = camera.world_to_cam

        # project 3d coordinates and clip to keep only pixels in screen space
        projected = np.matmul(viewmat , np.column_stack((self.xyz,np.ones(self.xyz.shape[0]))).T).T
        limxy = np.tile(tan_fov*1.3,(projected.shape[0],1))
        projected[:, :2] = np.minimum(limxy, np.maximum(-limxy, projected[:,0:2]/projected[:,2:])) *projected[:,2:]

        # calculate the Jacobian - the change in reprojection. they use the Jacobian to define the projection. its a Taylor expansion.
        jacobian = self.calc_jacobian(fxfy[0],fxfy[1],projected)
        jacobian = jacobian.reshape((jacobian.shape[0],3,3))

        # The original covariance matrix : V
        # we multipy by M, the projection matrix. we get: MVM^T 
        # this transdorms from object to camera, then apply the V (covariance) transformation, scaling and shearing the data
        # then multiply by M again, going back to camera FoR
        # Jacobian: used as a taylor expansion, to generate alocal linear transformation around a pixel. the Jacobian is trated as anew transformation 
        # and it is multiplied by the covariance matrix. 
        tile_viwe_mat = np.tile(viewmat[0:3,0:3].T,(jacobian.shape[0],1,1))
        T = tile_viwe_mat @ jacobian
        cov = T.transpose(0,2,1) @ self.cov3d.transpose(0,2,1) @ T
        # cov = jacobian @ tile_viwe_mat @ self.cov3d.transpose(0,2,1) @ tile_viwe_mat.transpose(0,2,1) @  jacobian.transpose(0,2,1)
        self.cov2d_matrix = cov
        self.cov2d = np.squeeze(np.dstack((cov[:,0,0],cov[:,0,1],cov[:,1,1])))

        # Adjust covariance and compute conic parameters
        # We use conic parameters to plot the gaussian on 2d. 
        # conic: Ax^2 + Bxy + Cy^2 = sigma_y^2*x^2 + sigma_xy*xy + sigma_x^2*y^2
        # we devide by the area to scale the reprojection. (the covariance will be the same if we have mm or m, we want to be in the right scaling)
        self.cov2d[:,0] = self.cov2d[:,0]+ 0.3
        self.cov2d[:,2] = self.cov2d[:,2]+ 0.3

        self.det = self.cov2d[:,0]*self.cov2d[:,2] - self.cov2d[:,1]*self.cov2d[:,1]
        self.inv_det = 1/self.det
        self.conic = np.column_stack((self.cov2d[:,2]*self.inv_det,-self.cov2d[:,1]*self.inv_det,self.cov2d[:,0]*self.inv_det,self.opacity))

        
        self.radius = self.compute_radius()
        self.projected = projected

    def compute_radius(self):
        """
        Computes the radius for each Gaussian splat based on covariance values.
        
        Returns:
            np.array: Radius values for each Gaussian splat.
        """
        mid = 0.5*(self.cov2d[:,0] + self.cov2d[:,2])
        lambda1 = mid + np.sqrt(np.maximum(0.1,mid * mid - self.det))
        lambda2 = mid - np.sqrt(np.maximum(0.1,mid * mid - self.det))
        return np.ceil(3.0 * np.sqrt(np.maximum(lambda1, lambda2)))


    def calc_jacobian(self,fx,fy,projected):
        """
        Calculates the Jacobian for each Gaussian splat based on projected points and camera parameters.
        
        Args:
            fx (float): Focal length along x-axis.
            fy (float): Focal length along y-axis.
            projected (np.array): Projected 3D points.
        
        Returns:
            np.array: Jacobian matrices for each projected point.
        """
        zero_np = np.zeros((projected.shape[0],1))
        return np.column_stack((
            fx / projected[:,2:], zero_np, - (fx * projected[:,0:1]) / (projected[:,2:] ** 2),
            zero_np, fy / projected[:,2:], - (fy * projected[:,1:2]) / (projected[:,2:] ** 2),
            zero_np, zero_np, zero_np
        ))


    def get_rect(self,cam):   
        """
        Calculates the upper-left and bottom-right corners of the bounding boxes for each projected point.

        Args:
            cam: Camera object used for projecting 3D points onto the image plane.
            
        Returns:
            Tuple of np.ndarray: 
                - Upper-left corner coordinates (xy_up_left_corner) of bounding boxes.
                - Bottom-right corner coordinates (xy_bot_right_corner) of bounding boxes.
        """
        pixel = cam.project_with_proj_mat(self.xyz)[:,0:2]
        xy_up_left_corner = np.minimum(self.grid,(np.maximum(0,pixel - self.radius[:,np.newaxis]) / self.block_xy).astype(int))
        xy_bot_right_corner = np.minimum(self.grid,((np.maximum(0,(pixel + self.radius[:,np.newaxis] + self.block_xy - 1) / self.block_xy)))).astype(int)
        return xy_up_left_corner,xy_bot_right_corner


    def get_color(self,deg, **kwargs):
        """
        Computes RGB color values from spherical harmonics coefficients for each vertex.

        Args:
            deg (int): Degree of spherical harmonics to consider for color computation.
            **kwargs: Additional keyword arguments for `sh_utils.rgb_from_sh`.
            
        Sets:
            self.color (np.array): Array of RGB color values computed for each vertex.
        """
        self.color = sh_utils.rgb_from_sh(deg,self.sh, **kwargs)


    def calculate_T_2d(self,camera):
        # calculate T: 
        
        # T stands for (WH)^T in Eq.9 - projmat transforms from camera to NDC (screen coordinates)
        # T is the transformation of every gaussian from tangent plane to NDC (here its pixel), its homogebnus coordinates. with the rotation matrix 
        # representing the axes and the translation vector representing the location of the center of each gaussian (in camera coordinates). 
        # 1. Scale (defines the splat)
        # 2. [Scale * axes | mean_splat xyz] transformation to world
        # 3. multiply 2 by camera.full_proj_transform.T (transformation to camera (perspective))
        # 4. multiply by ndc2pix to get the projection in pixels

        # get the direction of the axes and the scale of each axis of the gaussian (world)
        rotations = self.build_scaling_rotation(self.scale, self.rot) 
        self.rotation = rotations
        self.normal =  rotations[:,:,2]
        # use the Z direction of the 2d splat as a normal to the splat. rotate it to camera axes
        self.normal_to_splat_camera = np.dot(camera.world_to_cam[:,:3],rotations[:,:,2].T ).T

        # rotate the xyz points to camera 
        self.p_orig = np.dot(camera.world_to_cam[:,:3],self.xyz.T ).T

        normal_surface_direction = -np.sum(self.p_orig*self.normal_to_splat_camera,axis = 1)
        self.normal_to_splat_camera[normal_surface_direction < 0] = -self.normal_to_splat_camera[normal_surface_direction < 0]
        self.normal_to_splat_camera = self.normal_to_splat_camera/np.linalg.norm(self.normal_to_splat_camera,axis = 1)[:,np.newaxis]

        # define the transformation matrix of the gaussian from object to world (in homogenues coordinates)
        splat2world = np.hstack((np.transpose(rotations[:,:,0:2],(0,2,1)),self.xyz[:,np.newaxis,:])) 
        splat2world = np.concatenate((splat2world,np.tile(np.array([[0,0,1]]).T,(splat2world.shape[0],1,1))),2)
        # get the transformation from world to ndc (using the projection matrix)
        world2ndc = camera.full_proj_transform.T
        # get the transformation from ndc to pixel
        ndc2pix = np.vstack([[ self.image_size[0]/2,0,0,( self.image_size[0]-1)/2],[0, self.image_size[1]/2,0,( self.image_size[1]-1)/2],[0,0,0,1]])
        
        # Multiply: splat2world * world2ndc - transformation from splat to ndc
        temp = splat2world @ np.tile(world2ndc,(splat2world.shape[0],1,1))  # (4, 3) @ (4, 4) = (4, 4)
        # Multiply: splat2ndc * ndc2pix - transformation from splat to pixel
        self.T = temp @ np.tile(ndc2pix.T,(splat2world.shape[0],1,1))   # (4, 4) @ (4, 3) = (4, 3)


        # Next, We calculate the radius of the gaussian. We normalize by w to get homogeneus coordinates. In addition we flip Z axis (not sure why) 
        # we calculate the distance from the camera to the gaussian mean (this is w, the last row of a homogenues coordinate, deviding by it will give perspective view)
        # Notice that the rotation is scaled (in build_scaling_rotation) and is not normalized.

        # self.center - the projectes mean of the gaussian (with flipped z)
        # half_extend - used to calculate the radius of the gaussian, we take 3 sigma. because the ratation is scaled 
        # we calculate the distance for each axis and can get the 3 sigma by multiplying each distance. (we also devide by w to get the prespective view)

        temp_point = np.tile([9,9,-1],(self.T.shape[0],1))
        distance  = np.sum(temp_point*self.T[..., 2] * self.T[..., 2],-1)
        f = (1 / distance[:,np.newaxis]) * temp_point
        self.center = np.column_stack((np.sum(f * self.T[..., 0] * self.T[...,2],1),np.sum(f * self.T[..., 1] * self.T[...,2],1),np.sum(f * self.T[..., 1] * self.T[...,2],1)))
        axes_dist = np.column_stack((np.sum(f * self.T[..., 0] * self.T[...,0],1),np.sum(f * self.T[..., 1] * self.T[...,1],1),np.sum(f * self.T[..., 2] * self.T[...,2],1)))

        half_extend = np.sqrt(np.maximum(self.center * self.center - axes_dist, 1e-4)) * 3
        self.radius_2d = np.ceil(np.maximum(np.maximum(half_extend[:,0], half_extend[:,1]), 3 * 2))
        self.axes = half_extend


    def build_scaling_rotation(self,s, r):
        L = np.zeros((s.shape[0], 3, 3))
        R = self.build_rotation(r)

        L[:,0,0] = s[:,0]
        L[:,1,1] = s[:,1]
        L[:,2,2] = s[:,2]

        L = R @ L
        return L

    def build_rotation(self,r):
        norm = np.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = np.zeros((q.shape[0], 3, 3))

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

