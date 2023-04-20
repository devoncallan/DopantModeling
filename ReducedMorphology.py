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

        # mat1_Vfrac = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # mat1_S     = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # mat1_theta = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # mat1_psi   = np.zeros((self.z_dim, self.y_dim, self.x_dim))

        # mat2_Vfrac = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # mat2_S     = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # mat2_theta = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        # mat2_psi   = np.zeros((self.z_dim, self.y_dim, self.x_dim))
        
        mat_Vfrac       = np.zeros((self.num_materials,self.z_dim,self.y_dim,self.x_dim))
        mat_S           = np.zeros((self.num_materials,self.z_dim,self.y_dim,self.x_dim))
        mat_theta       = np.zeros((self.num_materials,self.z_dim,self.y_dim,self.x_dim))
        mat_psi         = np.zeros((self.num_materials,self.z_dim,self.y_dim,self.x_dim))
        mat_orientation = np.zeros((self.num_materials,self.z_dim,self.y_dim,self.x_dim, 3))

        # Assumes mat1 is the primary fibril material
        for fibril in self.fibrils:
            fibril_indices = fibril.fibril_indices
            fibril.set_fibril_orientation()
            for index in fibril_indices:
                # Convert XYZ to ZYX convention
                index = np.flip(index)
                if index[0] < self.z_dim and index[1] < self.y_dim and index[2] < self.x_dim:
                    mat_Vfrac[tuple(0,index)] = 1
                    mat_S[tuple(0,index)]     = 1
                    mat_theta[tuple(0,index)] = fibril.orientation_theta
                    mat_psi[tuple(0,index)]   = fibril.orientation_psi

                    
        mat_Vfrac[1,:,:,:] = 1 - mat_Vfrac[0,:,:,:]
        # Matrices have indeces of (mat#-1, z, y, x)
        return mat_Vfrac, mat_S, mat_theta, mat_psi
        
