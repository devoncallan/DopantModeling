import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trimesh
from Fibril import Fibril
import copy

class Morphology:

    # Type I

    # Type II
    
    # Initialize morphology
    """
    Dimensions:
    x_dim_nm - 
    y_dim_nm - 
    z_dim_nm - z dimension of morphology box in nm
    pitch_nm - 
    """


    def __init__(self, x_dim_nm: float, y_dim_nm: float, z_dim_nm: float, pitch_nm: float):
        """Initialize morphology object

        Args:
            x_dim_nm (float): x dimension of morphology box in nm
            y_dim_nm (float): y dimension of morphology box in nm
            z_dim_nm (float): z dimension of morphology box in nm
            pitch_nm (float): size of voxels in morphology in nm
        """
        # Dimensions of morphology in nm
        self.x_dim_nm = x_dim_nm
        self.y_dim_nm = y_dim_nm
        self.z_dim_nm = z_dim_nm
        
        # Dimensions of morphology in voxels
        self.pitch_nm = pitch_nm # Dimension of voxel in nm
        self.x_dim = int(round(x_dim_nm / pitch_nm))
        self.y_dim = int(round(y_dim_nm / pitch_nm))
        self.z_dim = int(round(z_dim_nm / pitch_nm))
        self.box_volume = self.x_dim * self.y_dim * self.z_dim
        self.dims  = np.array([self.x_dim, self.y_dim, self.z_dim])
        
        self.initialize_box()
        
        self.fibrils = []        # # Material 1 - P3HT
        # self.mat1_Vfrac = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # self.mat1_S     = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # self.mat1_theta = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # self.mat1_psi   = np.zeros((self.z_dim, self.y_dim, self.x_dim))

        # # Stores RGB values at each voxel (for morphology movie)
        # self.mat1_orientation = np.zeros((self.z_dim, self.y_dim, self.x_dim, 3))

        # # Material 2 - Vacuum 
        # self.mat2_Vfrac = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # self.mat2_S     = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # self.mat2_theta = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # self.mat2_psi   = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        

    def initialize_box(self):
        """Initialize voxelized box and bounding box
        """
        
        # Initialize morphology box as numpy array
        # Each xyz coordinate contains (material, x_orientation, y_orientation, z_orientation)
        self.box = np.zeros((self.x_dim, self.y_dim, self.z_dim, 4))

        # Set up bounding box mesh to detect out of bounds collisions
        self.bounding_box = trimesh.primitives.Box(extents=[self.x_dim, self.y_dim, self.z_dim])
        
        # Translate bounding box to center
        center = [self.x_dim/2, self.y_dim/2, self.z_dim/2]
        translation_matrix = trimesh.transformations.translation_matrix(center)
        self.bounding_box.apply_transform(translation_matrix)
        
        # Create bounding box path for visualization
        self.bounding_path = self.bounding_box.as_outline()
        self.bounding_path.colors = [[100,100,100]]

    # Check if any points are within the mesh  
    def check_mesh_contains_any(self, mesh, points):
        return np.any(mesh.contains(points))

    # Checks if all points are within the mesh
    def check_mesh_contains_all(self, mesh, points):
        return np.all(mesh.contains(points))
    
    # Check that the vertices of a given mesh are within the bounding box
    def check_mesh_within_bounding_box(self, mesh):
        return self.check_mesh_contains_all(self.bounding_box, mesh.vertices)

    def is_initial_mesh_valid(self, initial_mesh):
        
        def check_initial_mesh_intersection_with_fibrils(initial_mesh):
            points = initial_mesh.vertices

            # Sample surface points of fibril
            surface, face_index = trimesh.sample.sample_surface_even(initial_mesh, 150)
            points = np.concatenate((points, np.array(surface)))
            for fibril in self.fibrils:
                if self.check_mesh_contains_any(fibril.intersection_mesh, points):
                    return True
            return False
        
        return self.check_mesh_within_bounding_box(initial_mesh) and not check_initial_mesh_intersection_with_fibrils(initial_mesh)

    def is_mesh_valid(self, mesh):

        def check_mesh_intersection_with_fibrils(mesh):
            points = mesh.vertices
            for fibril in self.fibrils:
                if self.check_mesh_contains_any(fibril.intersection_mesh, points):
                    return True
            return False
        
        return self.check_mesh_within_bounding_box(mesh) and not check_mesh_intersection_with_fibrils(mesh)
    
    def set_model_parameters(self, radius_nm_avg: float, radius_nm_std: float, max_num_fibrils: int, 
                             fibril_length_range_nm: list):
        self.radius_nm_avg = radius_nm_avg
        self.radius_nm_std = radius_nm_std
        self.radius_avg = radius_nm_avg / self.pitch_nm
        self.radius_std = radius_nm_std / self.pitch_nm
        self.max_num_fibrils = max_num_fibrils
        self.min_fibril_length_nm = min(fibril_length_range_nm)
        self.max_fibril_length_nm = max(fibril_length_range_nm)
        self.min_fibril_length = self.min_fibril_length_nm / self.pitch_nm
        self.max_fibril_length = self.max_fibril_length_nm / self.pitch_nm
        
    def get_random_point(self):
        return self.dims * np.random.rand(3)
    
    def get_random_direction(self):
        theta = (90 + np.random.normal(0, 1.0))/180 * np.pi
        phi   = np.random.uniform(0, np.pi)
        direction = np.asarray([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        # direction = np.random.rand(3)
        return direction / np.linalg.norm(direction)
    
    def new_fibril(self):
        """ Initializes a new fibril object.

        Returns:
            Fibril: new fibril object 
        """

        # Set starting center of new fibril
        center = self.get_random_point()

        # Set long axis fibril direction
        direction = self.get_random_direction()

        # Set radius from specified distribution
        radius = np.random.normal(self.radius_avg, self.radius_std)

        # Initialize length to be minimum fibril length
        length = self.min_fibril_length

        return Fibril(center, direction, radius, length)

    def grow_fibril(self, fibril: Fibril, grow_direction: int):
        """ Grow the fibril in the grow_direction until intersection (with another fibril or the bounding box)
        or the maximum fibril length is reached.

        Args:
            fibril (Fibril): Fibril object that undergoes growth
            grow_direction (int): +1, grow left. -1, grow right.

        Returns:
            Fibril: Fibril object after growth
        """
        
        # Check that fibril is not intersecting and is the length is less than max length
        while (self.is_mesh_valid(fibril.intersection_mesh) and fibril.length <= self.max_fibril_length):
            fibril.length += self.pitch_nm
            fibril.center += self.pitch_nm * fibril.direction * grow_direction / 2
            fibril.make_intersection_mesh()
        
        # Returns false if fibril could not grow beyond initial (minimum) fibril length
        if (fibril.length != self.min_fibril_length):
            # Fibril growth terminated due to exceeding maximum fibril length
            if (fibril.length > self.max_fibril_length):
                diff = fibril.length - self.max_fibril_length
                fibril.center -= self.pitch_nm * diff * fibril.direction * grow_direction / 2
                fibril.length = self.max_fibril_length
                fibril.make_intersection_mesh()
            # Fibril growth terminated due to intersection with another fibril or bounding box
            else:
                fibril.length -= self.pitch_nm
                fibril.center -= self.pitch_nm * fibril.direction * grow_direction / 2
                fibril.make_intersection_mesh()

        return fibril
    
    def fill_model(self):
        """ Fill morphology with fibrils
        """

        for i in range(self.max_num_fibrils):
            # Initialize a new fibril 
            fibril = self.new_fibril()
        
            # Check if fibril is valid. If not, keep generating new fibril until it is valid.
            # Fibril is invalid if it intersects with the bounding box or any existing fibrils
            while not self.is_initial_mesh_valid(fibril.intersection_mesh):
                fibril = self.new_fibril()
            
            # Save a copy of the fibril in its initial state
            init_fibril = copy.deepcopy(fibril)
        
            # Grow fibrils in each direction until the fibril intersects with any other fibrils or
            # the bounding box or until the fibril reaches its maximum length
            fibril = self.grow_fibril(fibril, +1) # Grow fibril to the left
            fibril = self.grow_fibril(fibril, -1) # Grow fibril to the right

            fibril.make_fibril_mesh()

            print(f'-- Fibril {i} --')
            # print(f'Mesh  volume: {fibril.volume}')
            # print(f'Voxel volume: {fibril.voxel_volume}')
            # print(f'Difference (%): {np.abs(fibril.voxel_volume - fibril.volume)/fibril.volume*100}')

            self.fibrils.append(fibril)

    def voxelize_model(self):
        for fibril in self.fibrils:
            fibril.make_voxelized_fibril_mesh() 

    def set_fibril_orientations(self):
        for i in range(len(self.fibrils)):
            self.fibrils[i].set_random_orientation()

    # def create_material_matrices (self):
    #     self.mat1_orientation = np.zeros((self.z_dim, self.y_dim, self.x_dim, 4))
    #     for fibril in self.fibrils:
    #         indices = np.array(fibril.voxel_mesh.vertices, dtype=int)
    #         for index in indices:
    #             index = np.flip(index)
    #             if index[0] < self.z_dim and index[1] < self.y_dim and index[2] < self.x_dim:
    #                 self.mat1_Vfrac[tuple(index)] = 1
    #                 self.mat1_S[tuple(index)] = 1
    #                 self.mat1_theta[tuple(index)] = fibril.orientation_theta
    #                 self.mat1_psi[tuple(index)] = fibril.orientation_psi
    #                 self.mat1_orientation[tuple(index)] = [fibril.color[0], fibril.color[1], fibril.color[2], 1]
    #     self.mat2_Vfrac = 1 - self.mat1_Vfrac  

    def get_scene(self, show_voxelized=False, show_bounding_box=False):

        scene_mesh_list = []
        for fibril in self.fibrils:
            if show_voxelized:
                scene_mesh_list.append(fibril.voxel_mesh)
            else:
                scene_mesh_list.append(fibril.fibril_mesh)

        if show_bounding_box:
            scene_mesh_list.append(self.bounding_path)
        return trimesh.Scene(scene_mesh_list)
    
    # def get_morphology_movie(self):
    #     fig = plt.figure(dpi=300)
    #     def update_fig(i):
    #         fig.clear()
    #         plt.imshow(self.mat1_orientation[i,:,:,:])
    #     anim = animation.FuncAnimation(fig, update_fig, self.z_dim)
    #     anim.save("Morphology.mp4", fps=16)
    #     plt.close()

    # def analyze_crystallinity(self):
    #     return None

