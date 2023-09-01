import numpy as np
import matplotlib.pyplot as plt
import opensimplex as simplex
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


from ReducedMorphology import ReducedMorphology
from FieldGeneration import generate_field_with_PSD

# dopant_method:
    # all dopant methods:
        # 0: no dopant 
    # random dopant:
        # 1: uniform random replacing p3ht 
        # 2: Dopant only in amorph matrix
        # 3: Dopant only in fibrils (mainly for f4tcnq, tfsi likely won't do this)
        # 4: Uniformly doped to the dopant frac, all subtracted from P3HT
        # 5: Dope preferentially towards fibrils or matrix
            # crystal_dope_frac: fraction of total dopant in crystals
    # uniform dopant
        # 1: uniform random replacing p3ht 
        # 2: Dopant only in amorph matrix
        # 3: Dopant only in fibrils (mainly for f4tcnq, tfsi likely won't do this)
    # preferential_dopant
        # != 0: add dopant
            # crystal_dope_frac: fraction of total dopant in crystals
        
class PostProcessor:
    VACUUM_ID = 0
    CRYSTAL_ID = 1
    AMORPH_ID = 2
    DOPANT_ID = 3

    def __init__(self, num_materials=4, mol_weight=None, density=None,
                 dope_case=1, dopant_method='random', dopant_orientation=None, dopant_vol_frac=0.0825, crystal_dope_frac=0.5,
                 core_shell_morphology=True, gaussian_std=3, fibril_shell_cutoff=0.2, 
                 surface_roughness=False, height_feature=3, max_valley_nm=46, 
                 amorph_matrix_Vfrac=0.9, amorphous_orientation=True):
        self.num_materials = num_materials
        self.mol_weight = mol_weight
        self.density = density
        self.dope_case = dope_case
        self.dopant_orientation = dopant_orientation
        self.dopant_method = dopant_method
        self.dopant_vol_frac = dopant_vol_frac
        self.crystal_dope_frac = crystal_dope_frac
        self.core_shell_morphology = core_shell_morphology
        self.gaussian_std = gaussian_std
        self.fibril_shell_cutoff = fibril_shell_cutoff
        self.surface_roughness = surface_roughness
        self.height_feature = height_feature
        self.max_valley_nm = max_valley_nm
        self.amorph_matrix_Vfrac = amorph_matrix_Vfrac
        self.amorphous_orientation = amorphous_orientation

    def generate_material_matrices(self, rm: ReducedMorphology):
        mat_Vfrac = np.zeros((self.num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
        mat_S     = np.zeros((self.num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
        mat_theta = np.zeros((self.num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
        mat_psi   = np.zeros((self.num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    
        # Initialize matrices with fibrils
        for fibril in tqdm(rm.fibrils, desc='Iterating over fibrils'): 
            fibril_indices = fibril.fibril_indices
            fibril.set_fibril_orientation()
            for index in fibril_indices:
                # Convert XYZ to ZYX convention
                index = np.flip(index)
                if index[0] < rm.z_dim and index[1] < rm.y_dim and index[2] < rm.x_dim:
                    mat_Vfrac[self.CRYSTAL_ID][tuple(index)] = 1
                    mat_S[self.CRYSTAL_ID][tuple(index)]     = 1
                    mat_theta[self.CRYSTAL_ID][tuple(index)] = fibril.orientation_theta
                    mat_psi[self.CRYSTAL_ID][tuple(index)]   = fibril.orientation_psi
    
        # Add core-shell
        if self.core_shell_morphology:
            print("Adding core-shell structure...")
            mat_Vfrac, mat_S, mat_theta, mat_psi = self.add_fibril_shell(mat_Vfrac, mat_S, mat_theta, mat_psi)
    
        # Add surface roughness
        if self.surface_roughness:
            print("Adding surface roughness...")
            mat_Vfrac, mat_S, mat_theta, mat_psi = self.add_surface_roughness(rm, mat_Vfrac, mat_S, mat_theta, mat_psi)
        else:
            print("Handling amorphous material without surface roughness...")
            amorph_mask = np.where(mat_Vfrac[self.CRYSTAL_ID] != 1)
            amorph_Vfrac = mat_Vfrac[self.AMORPH_ID].copy()
            amorph_Vfrac[amorph_mask] += self.amorph_matrix_Vfrac
            amorph_Vfrac[amorph_mask] = np.clip(amorph_Vfrac[amorph_mask], 0, 1)
            mat_Vfrac[self.AMORPH_ID] = amorph_Vfrac
    
        # Add amorphous orientation
        if self.amorphous_orientation:
            print("Setting amorphous orientation...")
            mat_Vfrac, mat_S, mat_theta, mat_psi = self.set_amorphous_orientation(rm, mat_Vfrac, mat_S, mat_theta, mat_psi)
    
        # Add dopant based on selected method
        if self.dope_case != 0:
            if self.dopant_method == 'random':
                print("Adding random dopant...")
                mat_Vfrac = self.add_dopant(mat_Vfrac)
            elif self.dopant_method == 'uniform':
                print("Adding uniform dopant...")
                mat_Vfrac = self.add_uniform_dopant(mat_Vfrac)
            elif self.dopant_method == 'preferential':
                print("Adding preferential dopant...")
                mat_Vfrac = self.add_preferential_dopant(rm, mat_Vfrac)
            else:
                raise ValueError(f"Invalid dopant method: {self.dopant_method}")
                
        # Add dopant orientation
        if self.dopant_orientation != None:
            print("Setting dopant orientation...")
            mat_Vfrac, mat_S, mat_theta, mat_psi = self.set_dopant_orientation(rm, mat_Vfrac, mat_S, mat_theta, mat_psi)
    
        print("Material matrices generation completed.")
        
        # Matrices have indices of (mat#-1, z, y, x)
        return mat_Vfrac, mat_S, mat_theta, mat_psi

    def add_fibril_shell(self, mat_Vfrac, mat_S, mat_theta, mat_psi):
        fibril_core_Vfrac = mat_Vfrac[self.CRYSTAL_ID].copy()
        fibril_core_mask  = np.where(fibril_core_Vfrac == 1)
    
        fibril_shell_Vfrac = gaussian_filter(fibril_core_Vfrac, self.gaussian_std)
        fibril_shell_Vfrac[fibril_core_mask]  = 0
    
        fibril_shell_mask  = np.where(fibril_shell_Vfrac >= self.fibril_shell_cutoff)
    
        fibril_shell_Vfrac = np.zeros_like(fibril_shell_Vfrac)
        fibril_shell_Vfrac[fibril_shell_mask] = 1
    
        mat_Vfrac[self.AMORPH_ID] = fibril_shell_Vfrac
    
        return mat_Vfrac, mat_S, mat_theta, mat_psi

    def add_surface_roughness(self, rm, mat_Vfrac, mat_S, mat_theta, mat_psi):
        feature_size = int(rm.x_dim / self.height_feature)
        max_valley = int(self.max_valley_nm / rm.pitch_nm)
    
        for x in range(rm.x_dim):
            for y in range(rm.y_dim):
                #rounded gradient from 0 to max_val with average feature width of feature_size
                height = int(np.round(max_valley / 2 * (simplex.noise2(x / feature_size, y / feature_size) + 1))) 
                max_z_dim = round(rm.z_dim - height)
                for z in range(max_z_dim):
                    current_Vfrac = mat_Vfrac[self.AMORPH_ID, z, y, x] + mat_Vfrac[self.CRYSTAL_ID, z, y, x]
                    if current_Vfrac < 1:
                        mat_Vfrac[self.AMORPH_ID, z, y, x] += self.amorph_matrix_Vfrac
        mat_Vfrac[self.AMORPH_ID] = np.clip(mat_Vfrac[self.AMORPH_ID], 0, 1)
    
        return mat_Vfrac, mat_S, mat_theta, mat_psi

    def calc_roughness(self, mat_Vfrac, pitch):
        #https://www.keyence.com/ss/products/microscope/roughness/surface/sq-root-mean-square-height.jsp
        vac_Vfrac = mat_Vfrac[self.VACUUM_ID]
        vac_Vfrac[vac_Vfrac != 1] = 0
        rms_surface = 0
        for x in range(vac_Vfrac.shape[2]):
            for y in range(vac_Vfrac.shape[1]):
                try:
                    current_height = min(np.nonzero(vac_Vfrac[:, y, x])[0])
                except ValueError:
                    current_height = vac_Vfrac.shape[0]
                rms_surface += pitch * pitch * ((pitch * current_height) ** 2)
        area = pitch * len(vac_Vfrac[2]) * pitch * len(vac_Vfrac[1])
        rms_surface = np.sqrt(rms_surface / area)
        return rms_surface
    
    def add_dopant(self, mat_Vfrac):
        if self.dope_method == 0:
            # Fill with vacuum
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID]
        elif self.dope_method == 1: #random everywhere
            amorph_dopant  = mat_Vfrac[self.AMORPH_ID] * np.random.random_sample(mat_Vfrac[self.AMORPH_ID].shape)
            crystal_dopant = mat_Vfrac[self.CRYSTAL_ID] * np.random.random_sample(mat_Vfrac[self.CRYSTAL_ID].shape)
            # Normalize
            norm_factor = self.dopant_vol_frac / ((amorph_dopant + crystal_dopant).mean())
            amorph_dopant = amorph_dopant*norm_factor
            crystal_dopant = crystal_dopant*norm_factor
            mat_Vfrac[self.DOPANT_ID]  = crystal_dopant+amorph_dopant
            mat_Vfrac[self.CRYSTAL_ID] = mat_Vfrac[self.CRYSTAL_ID] - crystal_dopant
            mat_Vfrac[self.AMORPH_ID]  = mat_Vfrac[self.AMORPH_ID] - amorph_dopant
            mat_Vfrac[self.VACUUM_ID]  = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]
        elif self.dope_method == 2: # random matrix only
            amorph_dopant = mat_Vfrac[self.AMORPH_ID]* np.random.random_sample(mat_Vfrac[self.AMORPH_ID].shape)
            norm_factor = self.dopant_vol_frac / (amorph_dopant/self.amorph_matrix_Vfrac).mean()
            amorph_dopant = amorph_dopant*norm_factor
            mat_Vfrac[self.DOPANT_ID] = amorph_dopant
            mat_Vfrac[self.AMORPH_ID] = mat_Vfrac[self.AMORPH_ID] - amorph_dopant
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]
        elif self.dope_method == 3: # random fibrils only
            crystal_dopant = mat_Vfrac[self.CRYSTAL_ID]* np.random.random_sample(mat_Vfrac[self.CRYSTAL_ID].shape)
            norm_factor = self.dopant_vol_frac / crystal_dopant.mean()
            crystal_dopant = crystal_dopant*norm_factor
            mat_Vfrac[self.DOPANT_ID]  = crystal_dopant
            mat_Vfrac[self.CRYSTAL_ID] = mat_Vfrac[self.CRYSTAL_ID] - crystal_dopant
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]
        elif self.dope_method == 4: #Uniform dopant
            # Making dopant:
            mat_Vfrac[self.DOPANT_ID] = (mat_Vfrac[self.CRYSTAL_ID] + mat_Vfrac[self.AMORPH_ID])*self.dopant_vol_frac
            # Subtracting dopant:
            mat_Vfrac[self.CRYSTAL_ID] = mat_Vfrac[self.CRYSTAL_ID]*(1-self.dopant_vol_frac)
            mat_Vfrac[self.AMORPH_ID] = mat_Vfrac[self.AMORPH_ID]*(1-self.dopant_vol_frac)
            # Vacuum remaining:
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]
        elif self.dope_method == 5: # preferential random doping
            amorph_dopant = self.dope_method * mat_Vfrac[self.AMORPH_ID] * (self.dope_method*np.random.random_sample(mat_Vfrac[self.AMORPH_ID].shape))
            crystal_dopant = self.crystal_dope_frac * mat_Vfrac[self.CRYSTAL_ID] * (self.crystal_dope_frac*np.random.random_sample(mat_Vfrac[self.CRYSTAL_ID].shape))
            # Normalize
            norm_factor = self.dopant_vol_frac / ((amorph_dopant + crystal_dopant).mean())
            amorph_dopant = amorph_dopant*norm_factor
            crystal_dopant = crystal_dopant*norm_factor
            mat_Vfrac[self.DOPANT_ID] = crystal_dopant+amorph_dopant
            mat_Vfrac[self.CRYSTAL_ID] = mat_Vfrac[self.CRYSTAL_ID] - crystal_dopant
            mat_Vfrac[self.AMORPH_ID] = mat_Vfrac[self.AMORPH_ID] - amorph_dopant
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]
        return mat_Vfrac

    def add_uniform_dopant(self, mat_Vfrac):
        crystalline_mol_fraction, amorphous_mol_fraction, _ = self.analyze_mol_fractions(mat_Vfrac)
        total_mol_fraction = crystalline_mol_fraction + amorphous_mol_fraction
        x = crystalline_mol_fraction / total_mol_fraction if total_mol_fraction > 0 else 0
        
        approx_dopant_vol_frac = self.dopant_vol_frac / (x * (1 - self.dopant_vol_frac))
    
        if self.dope_case == 0:  # Fill with vacuum
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID]
    
        if self.dope_case == 1:  # Uniform everywhere
            total_material = mat_Vfrac[self.CRYSTAL_ID] + mat_Vfrac[self.AMORPH_ID]
            dopant_vol_fraction = total_material * self.dopant_vol_frac
            mat_Vfrac[self.DOPANT_ID] = dopant_vol_fraction
            mat_Vfrac[self.CRYSTAL_ID] -= dopant_vol_fraction * mat_Vfrac[self.CRYSTAL_ID] / total_material
            mat_Vfrac[self.AMORPH_ID] -= dopant_vol_fraction * mat_Vfrac[self.AMORPH_ID] / total_material
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]

        elif self.dope_case == 2:  # Uniform in matrix only
            dopant_vol_fraction = mat_Vfrac[self.AMORPH_ID] * approx_dopant_vol_frac
            mat_Vfrac[self.DOPANT_ID] = dopant_vol_fraction
            mat_Vfrac[self.AMORPH_ID] -= dopant_vol_fraction
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]
                
        elif self.dope_case == 3:  # Uniform in fibrils only
            dopant_vol_fraction = mat_Vfrac[self.CRYSTAL_ID] * approx_dopant_vol_frac
            mat_Vfrac[self.DOPANT_ID] = dopant_vol_fraction
            mat_Vfrac[self.CRYSTAL_ID] -= dopant_vol_fraction
            mat_Vfrac[self.VACUUM_ID] = 1 - mat_Vfrac[self.CRYSTAL_ID] - mat_Vfrac[self.AMORPH_ID] - mat_Vfrac[self.DOPANT_ID]

        return mat_Vfrac
    
    def euler_to_cartesian(self, theta, psi):
        z = np.sin(theta) * np.cos(psi)
        y = np.sin(theta) * np.sin(psi)
        x = np.cos(theta)
        return np.array([z, y, x])
    
    def cartesian_to_euler(self, orientation):
        orientation /= np.linalg.norm(orientation)
        theta = np.arccos(orientation[2])
        psi = np.arctan2(orientation[1], orientation[0])
        return theta, psi
    
    def add_preferential_dopant(self, rm, mat_Vfrac):
        """
        Adds dopants to the semicrystalline system based on specified volume fractions and updates mat_Vfrac.
        
        Parameters:
        -----------
        mat_Vfrac : numpy array
            An array containing the initial volume fractions of different components.
            Assumed to have the following indices: CRYSTAL_ID, AMORPH_ID, DOPANT_ID, VACUUM_ID.
            
        self.dopant_vol_frac : float
            The desired overall volume fraction of dopants in the system.
            
        self.crystal_dope_frac : float
            The desired fraction of the total dopant volume to be located in the crystalline region.
        
        Returns:
        --------
        mat_Vfrac : numpy array
            An updated array containing the new volume fractions of different components after doping.
            
        Notes:
        ------
        1. Initial volume fractions of crystalline (v_c0) and amorphous (v_a0) regions are calculated.
        2. The complementary fraction of dopant in amorphous region is calculated as x_a = 1 - self.crystal_dope_frac.
        3. The amount of dopant to be added, f, is calculated as:
        
           f = (self.dopant_vol_frac * (v_a0 + v_c0)) / (v_a0 * x_a + v_c0 * self.crystal_dope_frac)
           
        4. Volume fractions in the crystalline and amorphous regions are then updated based on this value of f.
        """
        v_c0_total, v_a0_total, _ = self.analyze_vol_fractions(mat_Vfrac)
        x_a_total = 1 - self.crystal_dope_frac
        f_global = (self.dopant_vol_frac * (v_a0_total + v_c0_total)) / (v_a0_total * x_a_total + v_c0_total * self.crystal_dope_frac)
        
        error_flag = False
        
        for z in tqdm(range(rm.z_dim), desc="Progress", total=rm.z_dim):
            for y in range(rm.y_dim):
                for x in range(rm.x_dim):
                    v_c0 = mat_Vfrac[self.CRYSTAL_ID, z, y, x]
                    v_a0 = mat_Vfrac[self.AMORPH_ID, z, y, x]
                    x_a = 1 - self.crystal_dope_frac
        
                    if self.crystal_dope_frac * v_c0 + x_a * v_a0 == 0:
                        continue
        
                    f = f_global * ((v_a0 + v_c0) / (v_a0_total + v_c0_total))
        
                    add_dopant = f * (self.crystal_dope_frac * v_c0 + x_a * v_a0)
                    remove_crys = f * self.crystal_dope_frac * v_c0
                    remove_amorph = f * x_a * v_a0
                    
                    # Ensure 1% of the original P3HT remains, so hat there are no negative volume fractions
                    max_volume = 0.99 * (v_c0 + v_a0) 
    
                    if add_dopant > max_volume:
                        add_dopant = max_volume
                        error_flag = True
    
                    max_crys_volume = 0.99 * v_c0
                    if remove_crys > max_crys_volume:
                        remove_crys = max_crys_volume
                        error_flag = True
    
                    max_amorph_volume = 0.99 * v_a0
                    if remove_amorph > max_amorph_volume:
                        remove_amorph = max_amorph_volume
                        error_flag = True
                        
                    # Update the volume fractions
                    mat_Vfrac[self.DOPANT_ID, z, y, x] += add_dopant
                    mat_Vfrac[self.CRYSTAL_ID, z, y, x] -= remove_crys
                    mat_Vfrac[self.AMORPH_ID, z, y, x] -= remove_amorph
                    mat_Vfrac[self.VACUUM_ID, z, y, x] = 1 - (mat_Vfrac[self.CRYSTAL_ID, z, y, x] +
                                                              mat_Vfrac[self.AMORPH_ID, z, y, x] +
                                                              mat_Vfrac[self.DOPANT_ID, z, y, x])
        
        if error_flag:
            print("Warning: There was not enough of the corresponding material to achieve the desired dopant volume fraction at some voxels.")
        
        return mat_Vfrac
    
    def set_dopant_orientation(self, rm, mat_Vfrac, mat_S, mat_theta, mat_psi):
        for z in tqdm(range(rm.z_dim), desc="Progress", total=rm.z_dim):
            for y in range(rm.y_dim):
                for x in range(rm.x_dim):
                    if mat_Vfrac[self.DOPANT_ID, z, y, x] > 0:
                        amorph_frac = mat_Vfrac[self.AMORPH_ID, z, y, x]
                        
                        # Generate normalized random 3D vector
                        random_orientation = np.random.normal(size=3)
                        random_orientation /= np.linalg.norm(random_orientation)
                        
                        crystal_frac = mat_Vfrac[self.CRYSTAL_ID, z, y, x]
                        crystal_theta = mat_theta[self.CRYSTAL_ID, z, y, x]
                        crystal_psi = mat_psi[self.CRYSTAL_ID, z, y, x]
                        crystal_orientation = self.euler_to_cartesian(crystal_theta, crystal_psi)
        
                        if self.dopant_orientation == 'parallel':
                            pass
                        elif self.dopant_orientation == 'perpendicular':
                            #orthogonal_vector = np.cross(crystal_orientation, [1, 0, 0])
                            #orthogonal_vector /= np.linalg.norm(orthogonal_vector)  # Normalize
                            #crystal_theta, crystal_psi = self.cartesian_to_euler(orthogonal_vector)
                            
                            crystal_theta = (crystal_theta + np.pi/2) % np.pi
                            crystal_psi = (crystal_psi + np.pi/2) % (2 * np.pi)
                            crystal_orientation = self.euler_to_cartesian(crystal_theta, crystal_psi)

                        elif self.dopant_orientation == 'isotropic':
                            dims = (rm.z_dim, rm.y_dim, rm.x_dim)
                            params_psi = {'k': 1, 'std': 1.0, 'dims': dims, 'max_value': 2 * np.pi}
                            params_u = {'k': 1, 'std': 1.0, 'dims': dims, 'max_value': 2}  # u = cos(theta), uniformly distributed in [-1, 1]
                            
                            random_field_psi = generate_field_with_PSD(**params_psi)
                            random_field_u = generate_field_with_PSD(**params_u)
                            
                            self.plot_field(random_field_psi, "psi")
                            self.plot_field(random_field_u, "u")
                            
                            # Compute z, y, x from theta and u
                            z = random_field_u
                            y = np.sqrt(1 - random_field_u ** 2) * np.sin(random_field_psi)
                            x = np.sqrt(1 - random_field_u ** 2) * np.cos(random_field_psi)
                            
                            # Convert to Euler angles
                            theta, psi = self.cartesian_to_euler(np.array([z, y, x]))
                            
                            dopant_mask = np.where(mat_Vfrac[self.DOPANT_ID] > 0)
                            mat_theta[self.DOPANT_ID][dopant_mask] = theta[dopant_mask]
                            mat_psi[self.DOPANT_ID][dopant_mask] = psi[dopant_mask]

        return mat_Vfrac, mat_S, mat_theta, mat_psi
    
    def set_amorphous_orientation(self, rm, mat_Vfrac, mat_S, mat_theta, mat_psi):
        dims = (rm.z_dim, rm.y_dim, rm.x_dim)
        params_psi = {'k': 1, 'std': 1.0, 'dims': dims, 'max_value': 2 * np.pi}
        params_u = {'k': 1, 'std': 1.0, 'dims': dims, 'max_value': 2}  # u = cos(theta), uniformly distributed in [-1, 1]
        
        random_field_psi = generate_field_with_PSD(**params_psi)
        random_field_u = generate_field_with_PSD(**params_u)
        
        self.plot_field(random_field_psi, "psi")
        self.plot_field(random_field_u, "u")
        
        # Compute z, y, x from theta and u
        z = random_field_u
        y = np.sqrt(1 - random_field_u ** 2) * np.sin(random_field_psi)
        x = np.sqrt(1 - random_field_u ** 2) * np.cos(random_field_psi)
        
        # Convert to Euler angles
        theta, psi = self.cartesian_to_euler(np.array([z, y, x]))
        
        amorph_mask = np.where(mat_Vfrac[self.AMORPH_ID] > 0)
        mat_theta[self.AMORPH_ID][amorph_mask] = theta[amorph_mask]
        mat_psi[self.AMORPH_ID][amorph_mask] = psi[amorph_mask]
        
        return mat_Vfrac, mat_S, mat_theta, mat_psi
    
    def plot_field(self, field_ref, field_name):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
        # If field_ref is 3D, take a 2D slice for plotting
        field_data_to_plot = field_ref[0, :, :] if len(field_ref.shape) == 3 else field_ref
    
        # Plot the field
        im = ax[0].imshow(field_data_to_plot, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax[0], label=f'{field_name} Value')
        ax[0].set_title(f'Generated {field_name} Field')
    
        # Plot histogram of field values
        ax[1].hist(field_data_to_plot.flatten(), bins=50, color='blue', edgecolor='black')
        ax[1].set_title(f'Histogram of {field_name} Values')
        ax[1].set_xlabel(f'{field_name} Value')
        ax[1].set_ylabel('Frequency')
    
        plt.tight_layout()
        plt.show()

    def analyze_mol_fractions(self, mat_Vfrac):
        crystalline_volume = np.sum(mat_Vfrac[self.CRYSTAL_ID])
        amorphous_volume = np.sum(mat_Vfrac[self.AMORPH_ID])
        dopant_volume = np.sum(mat_Vfrac[self.DOPANT_ID]) if self.DOPANT_ID < len(mat_Vfrac) else 0

        # Calculate mole fractions
        crystalline_moles = (crystalline_volume * self.density[self.CRYSTAL_ID]) / self.mol_weight[self.CRYSTAL_ID]
        amorphous_moles = (amorphous_volume * self.density[self.AMORPH_ID]) / self.mol_weight[self.AMORPH_ID]
        dopant_moles = (dopant_volume * self.density[self.DOPANT_ID]) / self.mol_weight[self.DOPANT_ID] if dopant_volume > 0 else 0
        
        total_moles = crystalline_moles + amorphous_moles + dopant_moles

        crystalline_mol_fraction = crystalline_moles / total_moles
        amorphous_mol_fraction = amorphous_moles / total_moles    
        dopant_mol_fraction = dopant_moles / total_moles if dopant_moles > 0 else 0

        return crystalline_mol_fraction, amorphous_mol_fraction, dopant_mol_fraction
    
    def analyze_vol_fractions(self, mat_Vfrac):
        # Sum the volume of each component
        crystalline_volume = np.sum(mat_Vfrac[self.CRYSTAL_ID])
        amorphous_volume = np.sum(mat_Vfrac[self.AMORPH_ID])
        dopant_volume = np.sum(mat_Vfrac[self.DOPANT_ID]) if self.DOPANT_ID < len(mat_Vfrac) else 0

        # Calculate the total occupied volume
        total_volume = crystalline_volume + amorphous_volume + dopant_volume

        # Calculate volume fractions
        crystalline_vol_fraction = crystalline_volume / total_volume
        amorphous_vol_fraction = amorphous_volume / total_volume
        dopant_vol_fraction = dopant_volume / total_volume if dopant_volume > 0 else 0

        return crystalline_vol_fraction, amorphous_vol_fraction, dopant_vol_fraction

    def save_parameters(self, filename, rm, mat_Vfrac, mol_weight=None, density=None, material_dict=None, notes=None):
        crystalline_mol_fraction, amorphous_mol_fraction, dopant_mol_fraction = self.analyze_mol_fractions(mat_Vfrac)
        with open("Parameters_" + filename + ".txt", "w") as f:
            f.write(filename + "\n")
            if notes:
                f.write("Notes:\n")
                f.write(notes + "\n\n")
            
            # Materials Information
            f.write("Materials Information:\n")
            f.write(f"Number of Materials: {self.num_materials}\n")
            if material_dict:
                f.write(f"Materials: {material_dict}\n")
            if mol_weight:
                f.write(f"Molecular Weights: {mol_weight}\n")
            if density:
                f.write(f"Density: {density}\n")
            
            # Box dimensions
            f.write("\nBox dimensions:\n")
            f.write(f"x: {rm.x_dim_nm} nm ({rm.x_dim} voxels)\n")
            f.write(f"y: {rm.y_dim_nm} nm ({rm.y_dim} voxels)\n")
            f.write(f"z: {rm.z_dim_nm} nm ({rm.z_dim} voxels)\n")
            f.write(f"Pitch: {rm.pitch_nm} nm\n")
            
            # Fibril description
            f.write("\nFibril description:\n")
            f.write(f"Average radius: {rm.radius_nm_avg} nm\n")
            f.write(f"Radius std deviation: {rm.radius_nm_std} nm\n")
            f.write(f"Length range: [{rm.min_fibril_length_nm}, {rm.max_fibril_length_nm}]\n")
            f.write(f"Number of generated fibrils: {rm.num_fibrils}\n")
            
            # Simulation type and parameters
            f.write("\nSimulation type and parameters:\n")
            f.write(f"Amorphous matrix total volume fraction: {self.amorph_matrix_Vfrac}\n")
            f.write(f"Surface roughness: {self.surface_roughness}\n")
            
            # Mole Fractions
            f.write("\nMole Fractions:\n")
            f.write(f"Crystalline Mole Fraction: {crystalline_mol_fraction}\n")
            f.write(f"Amorphous Mole Fraction: {amorphous_mol_fraction}\n")
            f.write(f"Dopant Mole Fraction: {dopant_mol_fraction if dopant_mol_fraction > 0 else 'No dopant present'}\n")
            
            # Doping Details
            f.write(f"\nDoping of the system: {bool(self.dope_case)}\n")
            if bool(self.dope_case):
                if self.dopant_method != 'preferential':
                    dope_message = ["", "Dopant distributed throughout randomly", "Dopant distributed through amorphous matrix only", "Dopant distributed through fibrils only"]
                    f.write(f"    Doping Method: {dope_message[self.dope_case]}\n")
                    f.write(f"    Dopant total volume fraction normalized to: {self.dopant_vol_frac}\n")
                    f.write(f"    Dopant orientation (within fibrils, relative to fibril long axis): {self.dopant_orientation}\n")
                else:
                    f.write(f"    Dopant total volume fraction normalized to: {self.dopant_vol_frac}\n")
                    f.write(f"    Preferential Dopant Method Details:\n")
                    f.write(f"        Fraction of dopant in crystalline regions: {self.crystal_dope_frac}\n")
                    f.write(f"        Fraction of dopant in amorphous regions: {1 - self.crystal_dope_frac}\n")
                    
            # Other details
            f.write("\nAdditional Parameters:\n")
            if self.surface_roughness:
                f.write(f"    Height of features: {self.max_valley_nm} nm\n")
                f.write(f"    Width of features: 1/{self.height_feature} of box, {rm.x_dim_nm/self.height_feature} nm\n")
            f.write(f"Core/shell morphology: {self.core_shell_morphology}\n")
            if self.core_shell_morphology:
                f.write(f"    Shell Gaussian std: {self.gaussian_std}\n")
                f.write(f"    Shell cutoff: {self.fibril_shell_cutoff}\n")