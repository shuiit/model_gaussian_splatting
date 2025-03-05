import torch

class Bone():
    def __init__(self, parent_joint, child_joint):
        self.parent = parent_joint
        self.child = child_joint

    @property
    def bone_points(self):
        # Convert global origins of parent and child joints to tensors
        return torch.vstack([self.parent.global_origin, self.child.global_origin])

    @property
    def length(self):
        # Calculate the length using torch operations
        return torch.norm(self.parent.global_origin - self.child.global_origin)

    @property
    def direction(self):
        # Calculate the direction using torch operations
        displacement = self.parent.global_origin - self.child.global_origin
        return displacement / torch.norm(displacement) if torch.norm(displacement) != 0 else torch.zeros_like(displacement)

    def update_bone(self):
        # Update the bone's direction and bone points
        self.direction = (self.parent.global_origin - self.child.global_origin) / self.length
        self.bone_points = torch.vstack([self.parent.global_origin, self.child.global_origin])

    def calculate_dist_from_bone(self, points):
        # Convert points to tensor if not already a tensor
        points = torch.tensor(points).cuda() if not isinstance(points, torch.Tensor) else torch.tensor(points).cuda()

        points_to_bone_origin = points - self.bone_points[0]  # vector between the 3d points and the bone origin (not normalized)
        bone_vector = (self.bone_points[1] - self.bone_points[0])  # The vector representing the bone (from origin to end, not normalized)

        t = torch.matmul(points_to_bone_origin, bone_vector.T) / self.length**2  # Project 'points_to_bone_origin' onto 'bone_vector'
        t = torch.clamp(t, 0, 1)[:, None]  # clip to get the closest point, if its on the bone, its between [0,1] else its outside the bone
        closest_point = self.bone_points[0] + t * bone_vector  # Determine the closest point, if its outside the bone take the endpoint as the closest point

        # Calculate the distance using torch operations
        return torch.norm(points - closest_point, dim=1)
