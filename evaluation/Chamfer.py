

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from shapely.geometry import Point, Polygon
import Utils
class Chamfer():
    def __init__(self, boundary_gt, surface_model, span, chord,remove_outliers = True,num_of_parts = 7):
        self.boundary = boundary_gt
        self.surface = surface_model
        self.span = span
        self.chord = chord



        try:
            devided_wing = self.devide_wings_to_parts(self.surface,span,num_of_parts = num_of_parts)

            self.surface_parts = self.get_parts_wing(self.surface,devided_wing,span)
            # self.boundary_parts = self.get_parts_wing(self.boundary,devided_wing,span)
            if remove_outliers == True:
                self.surface_parts = [self.ransac_for_ol(data) for data in self.surface_parts[2:]]
            else:
                self.surface_parts = np.vstack(self.surface_parts[2:])
        except:
            self.surface_parts = self.surface
        try:
            self.inside_boundary()
            self.center,self.normal = self.fit_plane_and_axes(self.boundary)
        except:
            return None

    def interpulate_wing(self):
        interp_vec = (self.boundary[0:-1,:] + self.boundary[1:,:])/2
        interleaved = np.empty((self.boundary.shape[0] + interp_vec.shape[0], self.boundary.shape[1]))
        interleaved[0::2] = self.boundary
        interleaved[1::2] = interp_vec
        return interleaved



    def devide_wings_to_parts(self,wing,span,num_of_parts = 8):
        projected_on_span = np.dot(wing,span)
        len_span = max(projected_on_span) - min(projected_on_span)
        projected_parts = [projected_on_span[((projected_on_span - min(projected_on_span)) <= (len_span*idx/num_of_parts)) & ((projected_on_span - min(projected_on_span)) >= (len_span*(idx-1)/num_of_parts))]  for idx in range(1,num_of_parts + 1)]
        parts_bound = []

        projected_parts = [projected_part for projected_part in projected_parts if len(projected_part) > 10 ]
        for idx in range(len(projected_parts) - 1): 
            parts_bound.append([min(projected_parts[idx]),min(projected_parts[idx + 1])])
        parts_bound.append([min(projected_parts[-1]),max(projected_parts[-1])])
        return parts_bound
    

    
    
    
    def get_parts_wing(self,wing,devided_wing,span):
        projected_on_span = np.dot(wing,span)
        devided_bool = [(projected_on_span >= devided_wing[0]) & (projected_on_span <= devided_wing[1]) for devided_wing in devided_wing]
        return  [wing[devided_bool] for devided_bool in devided_bool]
    





    def zscore(self,data, axis=0):
        """
        Compute the z-score of the input data along the specified axis.

        Parameters:
            data (array-like): Input data.
            axis (int): Axis along which to compute the z-score.

        Returns:
            ndarray: Z-scored data.
        """
        data = np.asarray(data)
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, ddof=0, keepdims=True)
        return (data - mean) / std





    def ransac_for_ol(self,data):
        np.random.seed(0)
        n_samples = len(self.zscore(np.dot(self.chord,data.T)))
        x = np.linspace(0, 10, n_samples)
        y = self.zscore(np.dot(self.chord,data.T))


        X = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Fit line using RANSAC
        model = RANSACRegressor(LinearRegression(), residual_threshold=1.0, random_state=42)
        model.fit(X, y)

        # Predict line
        line_x = np.linspace(0, 12, 100).reshape(-1, 1)
        line_y = model.predict(line_x)

        # Plot
        inlier_mask = model.inlier_mask_
        outlier_mask = ~inlier_mask

        # plt.scatter(X[inlier_mask], y[inlier_mask], color='blue', label='Inliers')
        # plt.scatter(X[outlier_mask], y[outlier_mask], color='red', label='Outliers')
        # plt.plot(line_x, line_y, color='green', linewidth=2, label='RANSAC line')
        # plt.legend()
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.title("RANSAC Line Fitting")
        # plt.show()



        return data[inlier_mask]
    


    def inside_boundary(self):


        
        gt_points_for_polygon = np.vstack(self.boundary)
        x_gt = np.dot(gt_points_for_polygon,self.span)
        y_gt = np.dot(gt_points_for_polygon,self.chord)


        model_points_for_polygon = np.vstack(self.surface_parts)
        x_model = np.dot(model_points_for_polygon,self.span)
        y_model = np.dot(model_points_for_polygon,self.chord)

        model_points = np.vstack((x_model,y_model)).T

        closed_points = np.vstack((x_gt,y_gt)).T
        closed_points = np.vstack((closed_points,closed_points[0]))

        # Shapely wants the polygon without the repeat
        spline_polygon = Polygon(closed_points[:-1])

        # Test points
        test_points = model_points[:,0:2]

        # Check and store results
        inside_mask = []
        for point_coords in test_points:
            point = Point(point_coords)
            is_inside = spline_polygon.contains(point)
            inside_mask.append(is_inside)

            
        self.inside_bound = np.vstack(self.surface_parts)[np.array(inside_mask)]
        self.outside_bound = np.vstack(self.surface_parts)[~np.array(inside_mask)]


    
    def fit_plane_and_axes(self,points):
        # points: Nx3 numpy array
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        U, S, Vt = np.linalg.svd(centered)
        
        # Right singular vectors = principal directions
        x_axis = Vt[0]  # First principal direction
        y_axis = Vt[1]  # Second principal direction
        normal = Vt[2]  # Normal to the plane (third component)

        return centroid,normal
    

    def dist_from_plane(self):
        np.dot(self.inside_bound,self.normal)


    def fit_plane_and_axes(self,points):
        # points: Nx3 numpy array
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        U, S, Vt = np.linalg.svd(centered)
        
        # Right singular vectors = principal directions
        x_axis = Vt[0]  # First principal direction
        y_axis = Vt[1]  # Second principal direction
        normal = Vt[2]  # Normal to the plane (third component)

        return centroid,normal


    def calculate_chamfer(self):
        centroid,normal = self.fit_plane_and_axes(self.boundary)

        
        add_cham =0    
        if  (hasattr(self,'inside_bound')) | (hasattr(self,'outside_bound')):
            
            if hasattr(self,'inside_bound'):
                
                centroid, _ = self.fit_plane_and_normal(self.boundary)
                tris = self.triangle_fan_fast(self.boundary, centroid)
                centers, normals = self.triangle_centers_and_normals(tris)
                add_cham = self.mean_abs_plane_distance(self.inside_bound, centers, normals) * 1e6  # to microns


                # inside = np.abs(np.dot(self.inside_bound - centroid,normal))
                # self.chamfer_inside = inside*1000*1000
                # add_cham = (np.sum(inside)/inside.shape[0])*1000*1000

            if len(self.outside_bound) > 0 :
                if hasattr(self,'inside_bound'):
                    closest_to_outside = Utils.find_closest_points_inptclouds(self.outside_bound,self.boundary)
                    dist = np.sqrt(np.sum((self.outside_bound - closest_to_outside)**2,axis = 1))
                    outside = (dist)
                    self.chamfer_outside = outside*1000*1000

                    add_cham += (np.sum(outside)/outside.shape[0])*1000*1000

        if add_cham == 0:
            self.sum_chamfer = 'Fail'
            return 'Fail'
        else:
            self.sum_chamfer = add_cham
            return self.sum_chamfer
        

    def calculate_chamfer_gt_to_model(self):
        closest_to_outside = Utils.find_closest_points_inptclouds(self.boundary,self.surface)
        dist = np.sqrt(np.sum((self.boundary - closest_to_outside)**2,axis = 1))
        self.chamfer_gt_to_model = dist*1000*1000
        self.sum_chamfer_nn = np.sum(dist)/dist.shape[0]
        return self.sum_chamfer_nn



    def fit_plane_and_normal(self,points: np.ndarray):
        """Return centroid and plane normal via PCA."""
        centroid = points.mean(axis=0)
        U, S, Vt = np.linalg.svd(points - centroid, full_matrices=False)
        normal = Vt[-1]  # smallest singular vector
        # ensure unit normal
        normal /= np.linalg.norm(normal)
        return centroid, normal

    def triangle_fan_fast(self,boundary: np.ndarray, center: np.ndarray):
        """
        Build a triangle fan as an array of shape (T, 3, 3),
        where T == len(boundary).
        """
        p = boundary
        q = np.roll(boundary, -1, axis=0)
        T = len(boundary)
        centers = np.broadcast_to(center, (T, 3))
        tris = np.stack([centers, p, q], axis=1)  # (T, 3, 3)
        return tris

    def triangle_centers_and_normals(self,tris: np.ndarray):
        """
        Given triangles (T,3,3), return per-triangle centers (T,3)
        and unit normals (T,3).
        """
        e1 = tris[:, 1] - tris[:, 0]         # (T,3)
        e2 = tris[:, 2] - tris[:, 0]         # (T,3)
        n = np.cross(e1, e2)                 # (T,3)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n_unit = n / n_norm
        centers = tris.mean(axis=1)          # (T,3)
        return centers, n_unit

    def mean_abs_plane_distance(self,points: np.ndarray, plane_points: np.ndarray, plane_normals: np.ndarray):
        """
        For each point, compute |(p - plane_point_j) · n_j| to every plane j,
        take the min over j, then average over points.
        points:        (N,3)
        plane_points:  (T,3) one point per plane (triangle center here)
        plane_normals: (T,3) unit normals per plane
        Returns scalar mean distance.
        """
        # diff: (N,T,3)
        diff = points[:, None, :] - plane_points[None, :, :]
        # signed distances: (N,T)
        d = np.abs(np.sum(diff * plane_normals[None, :, :], axis=2))
        d_min = d.min(axis=1)  # (N,)
        return d_min.mean()


# # Convert mask to color (green if inside, red if outside)
# colors = ['green' if inside else 'red' for inside in inside_mask]

# # Plot
# plt.figure(figsize=(6, 6))
# plt.plot(closed_points[:, 0], closed_points[:, 1], 'k-', label="Polygon")
# plt.scatter(test_points[:, 0], test_points[:, 1], c=colors, s=100, edgecolors='k', label="Test Points")
# for i, pt in enumerate(test_points):
#     plt.text(pt[0] + 0.02, pt[1] + 0.02, f"{i}", fontsize=9)

# plt.axis('equal')
# plt.title("Polygon and Test Points\n(Green = Inside, Red = Outside)")
# plt.legend()
# plt.grid(True)
# plt.show()


