import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import opensimplex as simplex

from ReducedMorphology import ReducedFibril
from ReducedMorphology import ReducedMorphology


VACUUM_ID  = 0 # Vacuum
CRYSTAL_ID = 1 # Crystalline P3HT
AMORPH_ID  = 2 # Amorphous P3HT
DOPANT_ID  = 3 # Dopant (optional)

### Post processing parameters
num_materials = 3

# Core-shell parameters
core_shell_morphology = True
gaussian_std = 3
fibril_shell_cutoff = 0.2

# Surface roughness parameters
surface_roughness = True
height_feature = 3
max_valley_nm = 32
amorph_matrix_Vfrac = 0.9

def generate_material_matricies(rm: ReducedMorphology):
    mat_Vfrac = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    mat_S     = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    mat_theta = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    mat_psi   = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))

    # Assumes mat1 is the primary fibril material
    for fibril in rm.fibrils:
        fibril_indices = fibril.fibril_indices
        fibril.set_fibril_orientation()
        for index in fibril_indices:
            # Convert XYZ to ZYX convention
            index = np.flip(index)
            if index[0] < rm.z_dim and index[1] < rm.y_dim and index[2] < rm.x_dim:
                mat_Vfrac[CRYSTAL_ID][tuple(index)] = 1
                mat_S[CRYSTAL_ID][tuple(index)]     = 1
                mat_theta[CRYSTAL_ID][tuple(index)] = fibril.orientation_theta
                mat_psi[CRYSTAL_ID][tuple(index)]   = fibril.orientation_psi


    if core_shell_morphology:
        mat_Vfrac, mat_S, mat_theta, mat_psi = add_fibril_shell(mat_Vfrac, mat_S, mat_theta, mat_psi)
    
    if surface_roughness:
        mat_Vfrac, mat_S, mat_theta, mat_psi = add_surface_roughness(rm, mat_Vfrac, mat_S, mat_theta, mat_psi)
    else:
        amorph_mask = np.where(mat_Vfrac[CRYSTAL_ID] != 1)
        amorph_Vfrac = mat_Vfrac[AMORPH_ID].copy()
        amorph_Vfrac[amorph_mask] += amorph_matrix_Vfrac
        amorph_Vfrac[amorph_mask] = np.clip(amorph_Vfrac[amorph_mask], 0, 1)
        mat_Vfrac[AMORPH_ID] = amorph_Vfrac

    mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID]

    # Matrices have indeces of (mat#-1, z, y, x)
    return mat_Vfrac, mat_S, mat_theta, mat_psi

def add_fibril_shell(mat_Vfrac, mat_S, mat_theta, mat_psi):

    fibril_core_Vfrac = mat_Vfrac[CRYSTAL_ID].copy()
    fibril_core_mask  = np.where(fibril_core_Vfrac == 1)

    fibril_shell_Vfrac = gaussian_filter(fibril_core_Vfrac, gaussian_std)
    fibril_shell_Vfrac[fibril_core_mask]  = 0

    fibril_shell_mask  = np.where(fibril_shell_Vfrac >= fibril_shell_cutoff)

    fibril_shell_Vfrac = np.zeros_like(fibril_shell_Vfrac)
    fibril_shell_Vfrac[fibril_shell_mask] = 1

    mat_Vfrac[AMORPH_ID] = fibril_shell_Vfrac

    return mat_Vfrac, mat_S, mat_theta, mat_psi

def add_surface_roughness(rm: ReducedMorphology, mat_Vfrac, mat_S, mat_theta, mat_psi):

    feature_size = int(rm.x_dim / height_feature)
    max_valley = int(max_valley_nm / rm.pitch_nm)
    
    for x in range(rm.x_dim):
        for y in range(rm.y_dim):
            #rounded gradient from 0 to max_val with average feature width of feature_size
            height = int(np.round(max_valley/2 * (simplex.noise2(x/feature_size, y/feature_size)+1))) 
            max_z_dim = round(rm.z_dim - height)
            for z in range(max_z_dim):
                current_Vfrac = mat_Vfrac[AMORPH_ID,z,y,x] + mat_Vfrac[CRYSTAL_ID,z,y,x]
                if current_Vfrac < 1:
                    mat_Vfrac[AMORPH_ID,z,y,x] += amorph_matrix_Vfrac
    mat_Vfrac[AMORPH_ID] = np.clip(mat_Vfrac[AMORPH_ID], 0, 1)

    return mat_Vfrac, mat_S, mat_theta, mat_psi