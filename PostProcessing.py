import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import opensimplex as simplex
import os

from ReducedMorphology import ReducedFibril
from ReducedMorphology import ReducedMorphology


VACUUM_ID  = 0 # Vacuum
CRYSTAL_ID = 1 # Crystalline P3HT
AMORPH_ID  = 2 # Amorphous P3HT
DOPANT_ID  = 3 # Dopant (optional)

### Post processing parameters
num_materials = 4
dope_type = 1 # 0: no dopant 
              # 1: uniform random replacing p3ht 
              # 2: Dopant only in amorph matrix
              # 3: Dopant only in fibrils (mainly for f4tcnq, tfsi likely won't do this)
              # 4: Uniformly doped to the dopant frac, all subtracted from P3HT
dopant_frac = 0.0825 #approx total vfrac dopant for normalization

# Core-shell parameters
core_shell_morphology = False
gaussian_std = 3
fibril_shell_cutoff = 0.2

# Surface roughness parameters
surface_roughness = True
height_feature = 3
max_valley_nm = 46
amorph_matrix_Vfrac = 0.9

def generate_material_matricies(rm: ReducedMorphology):
    mat_Vfrac = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    mat_S     = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    mat_theta = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))
    mat_psi   = np.zeros((num_materials, rm.z_dim, rm.y_dim, rm.x_dim))

    # Initialize matrices with fibrils
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

    # Add core-shell
    if core_shell_morphology:
        mat_Vfrac, mat_S, mat_theta, mat_psi = add_fibril_shell(mat_Vfrac, mat_S, mat_theta, mat_psi)

    # Add surface roughness
    if surface_roughness:
        mat_Vfrac, mat_S, mat_theta, mat_psi = add_surface_roughness(rm, mat_Vfrac, mat_S, mat_theta, mat_psi)
    else:
        amorph_mask = np.where(mat_Vfrac[CRYSTAL_ID] != 1)
        amorph_Vfrac = mat_Vfrac[AMORPH_ID].copy()
        amorph_Vfrac[amorph_mask] += amorph_matrix_Vfrac
        amorph_Vfrac[amorph_mask] = np.clip(amorph_Vfrac[amorph_mask], 0, 1)
        mat_Vfrac[AMORPH_ID] = amorph_Vfrac

    # Add dopant:
    mat_Vfrac = add_dopant(mat_Vfrac, dope_type)
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
def calc_roughness(mat_Vfrac, pitch):
    #https://www.keyence.com/ss/products/microscope/roughness/surface/sq-root-mean-square-height.jsp
    vac_Vfrac = mat_Vfrac[VACUUM_ID]
    vac_Vfrac[vac_Vfrac != 1] = 0
    rms_surface = 0
    for x in range(vac_Vfrac.shape[2]):
        for y in range(vac_Vfrac.shape[1]):
            try:
                current_height = min(np.nonzero(vac_Vfrac[:,y,x])[0])
            except ValueError:
                current_height = vac_Vfrac.shape[0]
            rms_surface += pitch*pitch*((pitch*current_height)**2)
    area = pitch*len(vac_Vfrac[2]) * pitch*len(vac_Vfrac[1])
    rms_surface = np.sqrt(rms_surface/area)
    return rms_surface
    
def add_dopant(mat_Vfrac,dope_method, partMat=0.5, partFib=0.5):
    if dope_method == 0:
        # Fill with vacuum
        mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID]
    elif dope_method == 1: #random everywhere
        amorph_dopant = mat_Vfrac[AMORPH_ID] * np.random.random_sample(mat_Vfrac[AMORPH_ID].shape)
        crystal_dopant = mat_Vfrac[CRYSTAL_ID] * np.random.random_sample(mat_Vfrac[CRYSTAL_ID].shape)
        # Normalize
        norm_factor = dopant_frac / ((amorph_dopant + crystal_dopant).mean())
        amorph_dopant = amorph_dopant*norm_factor
        crystal_dopant = crystal_dopant*norm_factor
        mat_Vfrac[DOPANT_ID] = crystal_dopant+amorph_dopant
        mat_Vfrac[CRYSTAL_ID] = mat_Vfrac[CRYSTAL_ID] - crystal_dopant
        mat_Vfrac[AMORPH_ID] = mat_Vfrac[AMORPH_ID] - amorph_dopant
        mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID] - mat_Vfrac[DOPANT_ID]
    elif dope_method == 2: # random matrix only
        amorph_dopant = mat_Vfrac[AMORPH_ID]* np.random.random_sample(mat_Vfrac[AMORPH_ID].shape)
        norm_factor = dopant_frac / (amorph_dopant/amorph_matrix_Vfrac).mean()
        amorph_dopant = amorph_dopant*norm_factor
        mat_Vfrac[DOPANT_ID] = amorph_dopant
        mat_Vfrac[AMORPH_ID] = mat_Vfrac[AMORPH_ID] - amorph_dopant
        mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID] - mat_Vfrac[DOPANT_ID]
    elif dope_method == 3: # random fibrils only
        crystal_dopant = mat_Vfrac[CRYSTAL_ID]* np.random.random_sample(mat_Vfrac[CRYSTAL_ID].shape)
        norm_factor = dopant_frac / crystal_dopant.mean()
        crystal_dopant = crystal_dopant*norm_factor
        mat_Vfrac[DOPANT_ID] = crystal_dopant
        mat_Vfrac[CRYSTAL_ID] = mat_Vfrac[CRYSTAL_ID] - crystal_dopant
        mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID] - mat_Vfrac[DOPANT_ID]
    elif dope_method == 4: #Uniform dopant
        # Making dopant:
        mat_Vfrac[DOPANT_ID] = (mat_Vfrac[CRYSTAL_ID] + mat_Vfrac[AMORPH_ID])*dopant_frac
        # Subtracting dopant:
        mat_Vfrac[CRYSTAL_ID] = mat_Vfrac[CRYSTAL_ID]*(1-dopant_frac)
        mat_Vfrac[AMORPH_ID] = mat_Vfrac[AMORPH_ID]*(1-dopant_frac)
        # Vacuum remaining:
        mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID] - mat_Vfrac[DOPANT_ID]
    elif dope_method == 5: # preferential random doping
        amorph_dopant = mat_Vfrac[AMORPH_ID] * (partMat*np.random.random_sample(mat_Vfrac[AMORPH_ID].shape))
        crystal_dopant = mat_Vfrac[CRYSTAL_ID] * (partFib*np.random.random_sample(mat_Vfrac[CRYSTAL_ID].shape))
        # Normalize
        norm_factor = dopant_frac / ((amorph_dopant + crystal_dopant).mean())
        amorph_dopant = amorph_dopant*norm_factor
        crystal_dopant = crystal_dopant*norm_factor
        mat_Vfrac[DOPANT_ID] = crystal_dopant+amorph_dopant
        mat_Vfrac[CRYSTAL_ID] = mat_Vfrac[CRYSTAL_ID] - crystal_dopant
        mat_Vfrac[AMORPH_ID] = mat_Vfrac[AMORPH_ID] - amorph_dopant
        mat_Vfrac[VACUUM_ID] = 1 - mat_Vfrac[CRYSTAL_ID] - mat_Vfrac[AMORPH_ID] - mat_Vfrac[DOPANT_ID]
    return mat_Vfrac

def save_parameters(filename: str, rm: ReducedMorphology, morph_filename:str, notes: str=None):
    with open("Parameters_" + filename + ".txt", "w") as f:
        f.write(filename + "\n")
        f.write(f'Morphology file used: {morph_filename}')
        f.write(notes + "\n")
        f.write("Box dimensions: \n")
        f.write(f"x: {rm.x_dim_nm} nm ({rm.x_dim} voxels)\n")
        f.write(f"y: {rm.y_dim_nm} nm ({rm.y_dim} voxels)\n")
        f.write(f"z: {rm.z_dim_nm} nm ({rm.z_dim} voxels)\n")
        f.write(f"pitch: {rm.pitch_nm} nm\n\n")
        f.write(f"Fibril description: \n")
        f.write(f"Average radius: {rm.radius_nm_avg} nm\n")
        f.write(f"Radius std: {rm.radius_nm_std} nm\n")
        f.write(f"Length range: [{rm.min_fibril_length_nm},{rm.max_fibril_length_nm}]\n")
        f.write(f"Number of generated fibrils: {rm.num_fibrils}\n\n")
        f.write("Simulation type and parameters:\n")
        f.write(f"Amorphous matrix total volume fraction: {amorph_matrix_Vfrac}\n")
        f.write(f"Surface roughness?: {surface_roughness}\n")
        if surface_roughness:
            f.write(f"    Height of features: {max_valley_nm} nm\n")
            f.write(f"    Width of features: 1/{height_feature} of box, {rm.x_dim_nm/height_feature} nm\n")
        f.write(f"Core/shell morphology?: {core_shell_morphology}\n")
        if core_shell_morphology:
            f.write(f"    Shell gaussian std: {gaussian_std}\n")
            f.write(f"    Shell cutoff: {fibril_shell_cutoff}\n")
        f.write(f"Doping of the system: {bool(dope_type)}")
        if bool(dope_type):
            dope_message = ["","Dopant distributed throughout randomly","Dopant distributed through amorphous matrix only","Dopant distributed through fibrils only"]
            f.write(f"    {dope_message[dope_type]}")
            f.write(f"    Dopant total volume fraction normalized to {dopant_frac}")