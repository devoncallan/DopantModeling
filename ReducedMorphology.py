import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Fibril import Fibril
from Morphology import Morphology

class ReducedFibril:

    def __init__(self, f: Fibril):
        
        self.fibril_indices = np.array(f.voxel_mesh.vertices, dtype=int)
        self.direction = f.direction

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

        self.fibrils = []

        for fibril in m.fibrils:
            self.fibrils.append(ReducedFibril(fibril))

        
    def generate_material_matricies(self):

        mat1_Vfrac = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        mat1_S     = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        mat1_theta = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        mat1_psi   = np.zeros((self.z_dim, self.y_dim, self.x_dim))

        mat2_Vfrac = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        mat2_S     = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        mat2_theta = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        mat2_psi   = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        
        for fibril in self.fibrils:
            fibril_indices = fibril.fibril_indices
            fibril.set_fibril_orientation()
            for index in fibril_indices:
                # Convert XYZ to ZYX convention
                index = np.flip(index)
                if index[0] < self.z_dim and index[1] < self.y_dim and index[2] < self.x_dim:
                    mat1_Vfrac[tuple(index)] = 1
                    mat1_S[tuple(index)]     = 1
                    mat1_theta[tuple(index)] = fibril.orientation_theta
                    mat1_psi[tuple(index)]   = fibril.orientation_psi
        mat2_Vfrac = 1 - mat1_Vfrac

        return mat1_Vfrac, mat1_S, mat1_theta, mat1_psi, mat2_Vfrac, mat2_S, mat2_theta, mat2_psi
        