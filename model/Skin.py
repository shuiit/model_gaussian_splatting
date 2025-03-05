import os.path
import open3d as o3d
import torch
import numpy as np

class Skin():
    def __init__(self, path_to_mesh, scale=1, constant_weight=False, color='green'):
        self.scale = scale
        self.load_skin(path_to_mesh)
        self.constant_weight = constant_weight
        self.color = color

    def add_bones(self, joints_of_bone):
        self.bones = joints_of_bone

    def load_skin(self, path_to_mesh):
        if os.path.isfile(f'{path_to_mesh}'):
            skin = self.load_mesh(f'{path_to_mesh}')  
        self.ptcloud_skin = torch.tensor(skin[:, 0:3] * self.scale, dtype=torch.float32,device = 'cuda') 
        self.skin_normals = torch.tensor((skin[:, 3:].T / torch.norm(skin[:, 3:-1], dim=1)).T, dtype=torch.float32,device = 'cuda') 

    def load_mesh(self, path_to_mesh):
        mesh = o3d.io.read_triangle_mesh(path_to_mesh)
        pt_cloud = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        normals = torch.tensor(np.asarray(mesh.vertex_normals), dtype=torch.float32)
        pt_cloud = torch.unique(pt_cloud, dim=0)
        

        pt_cloud, inverse_indices = torch.unique(pt_cloud, dim=0, return_inverse=True)

        idx = torch.nonzero(inverse_indices == torch.arange(inverse_indices.size(0), device=inverse_indices.device)).squeeze()

        normals = normals[idx, :]
        
        return torch.cat((pt_cloud, normals), dim=1)

    def translate_ptcloud_skin(self, translation):
        self.ptcloud_skin = self.ptcloud_skin - translation

    def calculate_weights_dist(self, bones=None):
        bones = bones if bones else self.bones
        weights = torch.stack([joint.bone.calculate_dist_from_bone(self.ptcloud_skin) for joint in bones]).T
        idx = torch.argmin(weights, dim=1)
        weights = torch.zeros((weights.shape[0], len(self.bones)))
        weights[torch.arange(weights.shape[0]), idx] = 1
        self.weights = weights.cuda()

    def calculate_weights_constant(self):
        self.weights = torch.zeros((self.ptcloud_skin.shape[0], len(self.bones))).cuda()
        self.weights[:, self.bones.index(self.constant_weight)] = 1

    def rotate_skin_points(self):
        points_homo = torch.cat([self.ptcloud_skin, torch.ones(self.ptcloud_skin.shape[0], 1).cuda()], dim=1)
        rotated_points = [joint.rotate_to_new_position(weight[:, None], points_homo) for weight, joint in zip(self.weights.T, self.bones)]
        return torch.sum(torch.stack(rotated_points), dim=0)[:, 0:3]

    def rotate_skin_normals(self):
        normals_homo = torch.cat([self.skin_normals, torch.ones(self.skin_normals.shape[0], 1).cuda()], dim=1)
        normals_rotated = [joint.rotate_normal_to_new_position(weight[:, None], normals_homo) for weight, joint in zip(self.weights.T, self.bones)]
        normals_rotated = torch.sum(torch.stack(normals_rotated), dim=0)[:, 0:3]
        return (normals_rotated.T / torch.norm(normals_rotated, dim=1)).T

    def get_part(self, part, points):
        return points[self.ptcloud_part_idx == self.parts[part], :]

