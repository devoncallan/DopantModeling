import numpy as np
import pickle
import matplotlib.pyplot as plt
from Fibril import Fibril
from Morphology import Morphology

class ReducedFibril:

    def __init__(self, f: Fibril):
        
        self.fibril_indices = np.array(f.voxel_mesh.vertices, dtype=int)
        self.direction = f.direction
        self.length = f.length
        self.radius = f.radius

        self.orientation_theta = 0
        self.orientation_psi   = 0
        self.color = None
        self.set_fibril_orientation()
        
    def set_fibril_orientation(self):
        orientation = self.direction
        self.orientation_theta = np.arccos(orientation[2])
        self.orientation_psi   = np.arctan2(orientation[1], orientation[0])

        # Map orientation angles to rgb colorspace
        r = np.sin(self.orientation_theta)*np.cos(self.orientation_psi)
        g = np.sin(self.orientation_theta)*np.sin(self.orientation_psi)
        b = np.cos(self.orientation_theta)
        self.color = np.array([r, g, b])

class ReducedMorphology:

    def __init__(self, m: Morphology):

        # Dimensions of morphology in nm
        self.x_dim_nm = m.x_dim_nm
        self.y_dim_nm = m.y_dim_nm
        self.z_dim_nm = m.z_dim_nm
        self.pitch_nm = m.pitch_nm

        # Dimensions of morphology in voxels
        self.x_dim = int(round(self.x_dim_nm / self.pitch_nm))
        self.y_dim = int(round(self.y_dim_nm / self.pitch_nm))
        self.z_dim = int(round(self.z_dim_nm / self.pitch_nm))
        self.dims  = np.array([self.x_dim, self.y_dim, self.z_dim])

        # Morphology parameters
        self.radius_nm_avg = m.radius_nm_avg
        self.radius_nm_std = m.radius_nm_std
        self.min_fibril_length_nm = m.min_fibril_length_nm
        self.max_fibril_length_nm = m.max_fibril_length_nm

        self.fibrils = []

        for fibril in m.fibrils:
            self.fibrils.append(ReducedFibril(fibril))
        self.num_fibrils = len(self.fibrils)

    def pickle(self):

        dimension_str = str(int(self.x_dim_nm)) + "x" + str(int(self.y_dim_nm)) + "x" + str(int(self.z_dim_nm)) + "nm"
        pitch_str = "pitch" + str(self.pitch_nm) + "nm"
        radius_str = "rad" + str(self.radius_nm_avg) + "nm"
        radius_std_str = "std" + str(self.radius_nm_std) + "nm"
        num_fibrils_str = str(int(self.num_fibrils)) + "fib"
        length_str = str(int(self.min_fibril_length_nm)) + "-" + str(int(self.max_fibril_length_nm)) + "nm"

        filename = dimension_str + "_" + pitch_str + "_" + radius_str + "_" + radius_std_str + "_" + num_fibrils_str + "_" + length_str + ".pickle"

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def plot_fibril_histogram(self):
        orientation_thetas = [fibril.orientation_theta for fibril in self.fibrils]
        orientation_psis = [fibril.orientation_psi for fibril in self.fibrils]
        lengths = [fibril.length for fibril in self.fibrils]
        radii = [fibril.radius for fibril in self.fibrils]
    
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
        axs[0, 0].hist(orientation_thetas, bins=20, edgecolor='black')
        axs[0, 0].set_title('Histogram of Orientation Theta')
        axs[0, 0].set_xlabel('Theta (rad)')
        axs[0, 0].set_ylabel('Frequency')
    
        axs[0, 1].hist(orientation_psis, bins=20, edgecolor='black')
        axs[0, 1].set_title('Histogram of Orientation Psi')
        axs[0, 1].set_xlabel('Psi (rad)')
        axs[0, 1].set_ylabel('Frequency')
    
        axs[1, 0].hist(lengths, bins=20, edgecolor='black')
        axs[1, 0].set_title('Histogram of Fibril Lengths')
        axs[1, 0].set_xlabel('Length')
        axs[1, 0].set_ylabel('Frequency')
    
        axs[1, 1].hist(radii, bins=20, edgecolor='black')
        axs[1, 1].set_title('Histogram of Fibril Radii')
        axs[1, 1].set_xlabel('Radius')
        axs[1, 1].set_ylabel('Frequency')
    
        plt.tight_layout()
        plt.show()
    
        return fig, axs