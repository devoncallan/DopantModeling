import os
import sys
import numpy as np


#######################
### File Parameters ###
#######################

# Get the experiment series folder
DEFAULT_MORPH_FILE = 'Morphology.hdf5'
base_path = os.path.dirname(os.path.abspath(__file__))
# main_dir = find_parent_of_subdir(base_path, "experiments")

# src_dir = os.path.join(main_dir, 'src')
# data_dir = os.path.join(main_dir, 'data')
# exp_dir = os.path.join(main_dir, 'experiments')

# sys.path.append(main_dir)
# sys.path.append(src_dir)

# sys.path.append('/home/devon/Documents/Github/NRSS/')
# sys.path.append('/home/devon/Documents/Github/PyHyperScattering/src/')
# sys.path.append('/home/devon/Documents/Github/Fibril/')
# sys.path.append('/home/devon/Documents/Github/Fibril/src/')


from Morphology.FibrilGenerator import Materials
from Morphology.FibrilPostProcessor import FibrilPostProcessor, MaterialParams, SurfaceRoughnessParams, CoreShellParams, MatrixParams, DopantParams
from Morphology.FibrilPostProcessor import DopantOrientation

from Morphology.FibrilGenerator import FibrilGenerator, FibrilSizeParams, FibrilGrowthParams, FibrilOrientationParams
from Morphology.FibrilGenerator import FibrilOrientation, FibrilDistribution
from Morphology.MorphologyData import MorphologyData

from Common.files import make_output_dir, move, find_parent_of_subdir



####################################
### FIBRIL GENERATION PARAMETERS ###
####################################

# Morphology dimensions
x_dim_nm = 1024
y_dim_nm = x_dim_nm
z_dim_nm = 512
pitch_nm = 2

# Fibril Size Params
radius_nm_avg = 15
radius_nm_std = 3
min_fibril_length_nm = 100
max_fibril_length_nm = 400

# Fibril Growth Params
max_num_fibrils = 1250
fibril_distribution = FibrilDistribution.PDS
c2c_dist_nm = 45
symmetrical_growth = False
periodic_bc = False

# Fibril Orientation Params
fibril_orientation = FibrilOrientation.GRF_SAMPLE_FLAT
theta_distribution_csv = None
k = 1./25
std = 1./125


##################################
### POST PROCESSING PARAMETERS ###
##################################

# Material Params
num_materials = 3
mw = {
    Materials.CRYSTAL_ID: 166.2842, # Molecular weight of crystalline P3HT repeat
    Materials.AMORPH_ID: 166.2842,  # Molecular weight of amorphous P3HT repeat
    Materials.DOPANT_ID: 276.15     # Molecular weight of TFSI- = 280.14, Molecular weight of F4TCNQ = 276.15
}
density = {
    Materials.CRYSTAL_ID: 1.1, # Density of crystalline P3HT
    Materials.AMORPH_ID: 1.1,  # Density of amorphous P3HT
    Materials.DOPANT_ID: 1.1   # Density of dopant in P3HT
}

# Core Shell Params
use_core_shell_params = True
gaussian_std = 3
fibril_shell_cutoff = 0.2

# Matrix Params
amorph_matrix_Vfrac = 0.9
amorph_orientation = True

# Surface Roughness Params
use_surface_roughness_params = False
height_feature = 3
max_valley_nm = 43

# Dopant Params
use_dopant_params = False
dopant_vol_frac = 0.0
crystal_dopant_frac = 1.0
dopant_orientation = DopantOrientation.ISOTROPIC

#####################################
### CYRSOXS SIMULATION PARAMETERS ###
#####################################

energies1 = np.round(np.arange(280., 286., 0.5),1)
energies2 = np.round(np.arange(286., 288., 0.2),1)
energies3 = np.round(np.arange(288., 291.5, 0.5),1)
energies = np.concatenate([energies1, energies2, energies3])
energies = np.round(np.arange(280., 281., 1),1)

material_dict = {
    'Material1': 'vacuum', 
    'Material2': '/home/devon/Documents/Github/Fibril/data/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge_isotropic.txt', 
    'Material3': '/home/devon/Documents/Github/Fibril/data/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge_isotropic.txt',
    'Material4': 'vacuum'
}
energy_dict = {
    'Energy': 6,
    'DeltaPerp': 3,
    'BetaPerp': 1,
    'DeltaPara': 2,
    'BetaPara': 0 
}

################################
### VISUALIZATION PARAMETERS ###
################################

q_min = 0.1
q_max = 0.9
q_range = (q_min, q_max)

E_min = 280
E_max = 290
E_range = (E_min, E_max)

I_min = 1e-1
I_max = 1e+7
I_range = (I_min, I_max)
