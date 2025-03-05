import torch
import numpy as np
from model.Bone import Bone


class Joint:
    def __init__(self, translation, rotation, parent=None, end_joint_of_bone=True, rotation_order='zyx', scale=1, color='green'):
        self.child = []
        self.parent = parent
        self.rotation_order = list(rotation_order)

        # Convert rotation and translation to tensors
        self.local_angles = torch.tensor(rotation, dtype=torch.float32, device='cuda',requires_grad= True)  # rotation angles as tensor
        self.local_rotation = torch.tensor(self.rotation_matrix(rotation[0], rotation[1], rotation[2]), dtype=torch.float32, device='cuda',requires_grad= True)  # rotation matrix as tensor
        self.translation_from_parent = torch.tensor(translation, dtype=torch.float32, device='cuda') * scale  # translation as tensor
        self.local_translation = translation * scale  # translation as tensor

        # Store other properties
        self.scale = scale
        self.color = color
        self.end_joint_of_bone = end_joint_of_bone
        self.local_transformation = self.transformation_matrix()
        self.global_transformation = self.get_global_transformation(rest_bind=True)
        self.get_global_point()
        self.bone = None
        
        # Update children
        self.update_child()
    
    def get_and_assign_bones(self, visited = None):
        visited = visited or set()
            
        if self in visited:
            return []
        visited.add(self)
        if self.end_joint_of_bone:
            self.parent.bone = Bone(self.parent,self )
        bones = [self.parent] if self.end_joint_of_bone else []
        for child in self.child: 
            bones += child.get_and_assign_bones(visited)
        return bones
    

    def get_list_of_joints(self, visited = None,joints = None):
        visited = visited or set()
        joints = joints or []
        joints.append(self)
        visited.add(self)

        for child in self.child: 
            if child not in visited:
                child.get_list_of_joints(visited,joints)
        return joints


    def update_child(self):
        if self.parent is None:
            return
        self.parent.update_child()
        if self not in self.parent.child:
            self.parent.child.append(self)

    # Other methods remain similar, but replace numpy operations with PyTorch tensor operations
    def set_local_rotation(self, angles):
        self.local_rotation = self.rotation_matrix(angles[0], angles[1], angles[2])
        self.local_transformation = self.transformation_matrix()

    def set_local_translation(self, translation):
        self.translation_from_parent = torch.tensor(translation, dtype=torch.float32, device='cuda')
        self.local_transformation = self.transformation_matrix()

    def get_global_transformation(self, rest_bind=False):
        if self.parent is None:
            return self.local_transformation
        self.global_transformation = self.parent.get_global_transformation(rest_bind=rest_bind)
        self.global_transformation = torch.matmul(self.global_transformation, self.local_transformation)
        if rest_bind:
            self.bind_transformation = self.global_transformation
        return self.global_transformation

    def get_global_point(self, point=[0, 0, 0, 1]):
        if point == [0, 0, 0, 1]:
            self.global_origin = torch.matmul(self.global_transformation, torch.tensor(point, dtype=torch.float32, device='cuda'))[:3]
        return torch.matmul(self.global_transformation, torch.tensor(point, dtype=torch.float32, device='cuda'))[:3]

    def rotate_to_new_position(self, weight, points_homo):
        transformation_rest = torch.inverse(self.bind_transformation)
        rotated_points = torch.matmul(transformation_rest, points_homo.T)
        return weight * torch.matmul(self.global_transformation, rotated_points).T

    def update_rotation(self):
        self.get_global_transformation()
        self.get_global_point()

    def rotate_normal_to_new_position(self, weight, normal):
        transformation_rest = torch.inverse(self.bind_transformation)
        transformation_rest_to_global = torch.matmul(self.global_transformation, transformation_rest).T
        rotated_points_inv = torch.inverse(transformation_rest_to_global)
        return weight * torch.matmul(rotated_points_inv, normal.T).T

    # Rotation matrix using PyTorch
    def rotation_matrix(self, yaw, pitch, roll):
        roll = roll * np.pi / 180
        pitch = pitch * np.pi / 180
        yaw = yaw * np.pi / 180
        mat = {}
        mat['x'] = torch.tensor([[1, 0, 0], [0, torch.cos(roll), -torch.sin(roll)], [0, torch.sin(roll), torch.cos(roll)]], dtype=torch.float32, device='cuda')
        mat['y'] = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)], [0, 1, 0], [-torch.sin(pitch), 0, torch.cos(pitch)]], dtype=torch.float32, device='cuda')
        mat['z'] = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0], [torch.sin(yaw), torch.cos(yaw), 0], [0, 0, 1]], dtype=torch.float32, device='cuda')

        rotation_matrix = torch.matmul(mat[self.rotation_order[0]], torch.matmul(mat[self.rotation_order[1]], mat[self.rotation_order[2]]))
        return rotation_matrix

    def transformation_matrix(self):
        return torch.cat((torch.column_stack((self.local_rotation, self.translation_from_parent)), torch.tensor([0, 0, 0, 1], dtype=torch.float32, device='cuda').view(1, 4)), dim=0)
