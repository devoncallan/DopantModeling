import numpy as np

class MorphologyData:
    
    def __init__(self, dims, num_materials, pitch_nm):
        
        self.dims = dims
        self.x_dim, self.y_dim, self.z_dim = dims
        self.num_materials = num_materials
        self.pitch_nm = pitch_nm
        
        # Initialize material matricies
        self.mat_Vfrac = np.zeros((self.num_materials, self.z_dim, self.y_dim, self.x_dim))
        self.mat_S     = np.zeros((self.num_materials, self.z_dim, self.y_dim, self.x_dim))
        self.mat_theta = np.zeros((self.num_materials, self.z_dim, self.y_dim, self.x_dim))
        self.mat_psi   = np.zeros((self.num_materials, self.z_dim, self.y_dim, self.x_dim))

    def get_data(self):
        data = []
        for mat in range(self.num_materials):
            data.append([
                self.mat_Vfrac[mat], self.mat_S[mat], self.mat_theta[mat], self.mat_psi[mat]
            ])
        return data