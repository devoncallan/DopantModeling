import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize


class Fibril:
    
    def __init__(self, center: np.ndarray, direction: np.ndarray, radius: float, length: float):
        """Initialize fibril object

        Args:
            center (np.ndarray): Center of fibril 
            direction (np.ndarray): 3D angle vector 
            radius (float): radius of the fibril
            length (float): length of the fibril
        """
        self.center = center 
        self.direction = direction + 1e-40 # Add 1e-40 to avoid numerical errors with divison by zero?
        self.radius = radius
        self.length = length
        self.volume = 0
        self.voxel_volume = 0
        # Rotation matrix describes how to rotate primitive cylinder to self.direction
        self.rotation_matrix = None

        # will be 3D vector of orientation of each voxel
        self.orientation_theta = 0
        self.orientation_psi = 0
        self.set_random_orientation()
        # self.orientation = self.get_random_orientation()
        # self.color = self.get_color_from_orientation()

        self.intersection_mesh = None
        self.fibril_mesh       = None
        self.voxel_grid        = None
        self.voxel_mesh        = None

        self.make_intersection_mesh()

    def set_random_orientation(self):
        orientation = np.random.random(3)
        orientation /= np.linalg.norm(orientation)
        self.orientation_theta = np.arccos(orientation[2])
        self.orientation_psi = np.arctan2(orientation[1], orientation[0])

    # def get_random_orientation(self):
    #     orientation = np.random.random(3)
    #     orientation /= np.linalg.norm(orientation)
    #     return orientation
    
    # def get_color_from_orientation(self):
    #     theta = np.arccos(self.orientation[2])
    #     psi = np.arctan2(self.orientation[1], self.orientation[0])
    #     r = np.sin(theta)*np.cos(psi)
    #     g = np.sin(theta)*np.sin(psi)
    #     b = np.cos(theta)
    #     return np.array([r, g, b])
    
    def transform_mesh(self, mesh, current_direction: np.ndarray(shape=3), override=False):
        """ Rotate given mesh from current_direction to self.direction and translate mesh to center.
        Assumes given mesh is centered at the origin

        Args:
            mesh (_type_): _description_
            current_direction (_type_): Current direction of fibril
            override (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # If rotation matrix has not been defined, calculate the rotation matrix required to rotate fibril to self.direction
        if self.rotation_matrix is None or override:
            rotation_angles = trimesh.transformations.angle_between_vectors(current_direction, self.direction)
            rotation_axis   = trimesh.transformations.vector_product(current_direction, self.direction)
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angles, rotation_axis)
            self.rotation_matrix = rotation_matrix
        # Translation matrix to translate mesh from origin to self.center
        translation_matrix = trimesh.transformations.translation_matrix(self.center)
        
        mesh.apply_transform(self.rotation_matrix)
        mesh.apply_transform(translation_matrix)
        
        return mesh
    
    def make_intersection_mesh(self):
        # Make a cylinder mesh for intersection calculations with specified radius, length, and direction 
        mesh = trimesh.primitives.Cylinder(radius=self.radius, height=self.length, use_embree=True)
        mesh = self.transform_mesh(mesh, mesh.direction)
        self.intersection_mesh = mesh
        return mesh

    def make_tapered_cylinder_mesh(self):
        N = 64  # Number of vertices along midline
        N_taper = int(N / 4)  # Number of points to taper at the ends
        shape_k1 = 1.  # Changes shape at midpoint
        shape_k2 = 1.  # Changes shape at endpoint

        u = np.linspace(0, self.length, N) - self.length / 2
        theta = np.linspace(0, 2 * np.pi, N)

        # Midline as straight line
        X = np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

        # Define the tapered radius as a function of midline
        t = np.linspace(0, np.pi / 2, N_taper)
        x_t = N_taper * np.cos(t)**shape_k1
        r_t = self.radius * np.sin(t)**shape_k2

        # Resample to get equally spaced points
        f = interp1d(x_t, r_t, kind='cubic', fill_value='extrapolate')
        x_taper = np.linspace(0, N_taper, N_taper)
        r_taper = f(x_taper) # interpolate y to fit new x (t_idxs)

        # Concatenate sections to get radius as function of midline
        r = np.concatenate([
            r_taper[::-1], # tapered section
            np.ones(N - 2 * N_taper) * self.radius, # midsection (constant radius)
            r_taper # tapered section
        ])

        # Calculate a TNB frame from the midline
        T = normalize(np.gradient(X, axis=0))
        N = normalize(np.cross(T, np.ones_like(T) * [0, 0, 1]))
        B = np.cross(T, N)

        # Generate surface
        x = X[:, 0] + r * (np.outer(np.cos(theta), N[:, 0]) + np.outer(np.sin(theta), B[:, 0]))
        y = X[:, 1] + r * (np.outer(np.cos(theta), N[:, 1]) + np.outer(np.sin(theta), B[:, 1]))
        z = X[:, 2] + r * (np.outer(np.cos(theta), N[:, 2]) + np.outer(np.sin(theta), B[:, 2]))

        x_p = np.ravel(x)
        y_p = np.ravel(y)
        z_p = np.ravel(z)
        pts = np.stack([x_p, y_p, z_p], axis=1)

        mesh = trimesh.PointCloud(pts).convex_hull
        mesh = self.transform_mesh(mesh, [1, 0, 0], override=True)
        # self.fibril_mesh = mesh
        return mesh

    def make_fibril_mesh(self, voxelize=True, isCylinder=False):
        if isCylinder:
            self.fibril_mesh = self.make_intersection_mesh()
        else:
            self.fibril_mesh = self.make_tapered_cylinder_mesh()

        self.volume = self.fibril_mesh.volume

<<<<<<< HEAD
    def make_voxelized_fibril_mesh(self):
=======
    def make_voxelized_fibril_mesh(self, pitch_nm: float):
>>>>>>> 15085b1047ea41e264d0356ec880ee7719a6e354
        self.voxel_grid = self.fibril_mesh.voxelized(pitch=1).fill()
        self.voxel_mesh = self.voxel_grid.as_boxes()
        self.voxel_volume = self.voxel_mesh.volume
        # print(f'Voxel vvol: {self.voxel_volume}')

# class NormalFibril(Fibril):

#     def __init__(self):
#         # rotation angle in radians
#         self.theta = np.random.uniform(-np.pi, np.pi) # rotation about y
#         self.psi   = np.random.uniform(-np.pi, np.pi) # rotation about z

#         self.gr

        
class TypeOneFibril(Fibril):
    # Core-shell angle directed certain way

    def __init__():
        return None

class TypeTwoFibril(Fibril):

    def __init__():
        return None