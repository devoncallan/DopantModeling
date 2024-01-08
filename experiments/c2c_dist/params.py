import os
import re
import sys
import numpy as np
from scipy.interpolate import interp1d

# Ensure all paths are correctly set
sys.path.append('/home/php/DopantModeling-dev/')

from src.Morphology.Fibril.FibrilGenerator import Materials
from src.Morphology.MorphologyData import MorphologyData
from src.Morphology.Fibril.FibrilPostProcessor import FibrilPostProcessor, MaterialParams, SurfaceRoughnessParams, CoreShellParams, MatrixParams, DopantParams, DopantOrientation
from src.Morphology.Fibril.FibrilGenerator import FibrilGenerator, FibrilSizeParams, FibrilGrowthParams, FibrilOrientationParams, FibrilOrientation, FibrilDistribution
from src.Common.files import make_output_dir, move, find_parent_of_subdir

#######################
### File Parameters ###
#######################

# Get the experiment series folder
DEFAULT_MORPH_FILE = 'Morphology.hdf5'
base_path = os.path.dirname(os.path.abspath(__file__))

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
symmetrical_growth = True
periodic_bc = True

# Fibril Orientation Params
fibril_orientation = FibrilOrientation.GRF_SAMPLE_ALL
theta_distribution_csv = '/home/php/DopantModeling-dev/data/theta_distributions/avg_0p1_theta_distribution.csv'
k = 1./25
std = 1./125

##################################
### POST PROCESSING PARAMETERS ###
##################################

# Material Params
num_materials = 4

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
amorph_matrix_Vfrac = 0.90
amorph_orientation = True

# Surface Roughness Params
use_surface_roughness_params = False
height_feature = 3
max_valley_nm = 43

# Dopant Params
use_dopant_params = True
uniform_doping = False
dopant = 'F4TCNQ'
dopant_vol_frac = 0.1
crystal_dopant_frac = 0.5
dopant_orientation = DopantOrientation.ISOTROPIC

#####################################
### CYRSOXS SIMULATION PARAMETERS ###
#####################################

edge_to_find = 'C 1s'  # Default value, change as required

energies1 = np.round(np.arange(280., 286., 0.5),1)
energies2 = np.round(np.arange(286., 288., 0.2),1)
energies3 = np.round(np.arange(288., 291.5, 0.5),1)
energies = np.concatenate([energies1, energies2, energies3])
energies = np.round(np.arange(280., 281., 1),1)

material_dict = {
    'Material1': 'vacuum', 
    'Material2': '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge.txt', 
    'Material3': '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge_isotropic.txt',
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

#####################################
### DOPANT INTERPOLATION FUNCTIONS ###
#####################################

def interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.019, 0.053, 0.049, 0.067, 0.096])
    f = interp1d(x, y, fill_value="extrapolate")
    return f(crystalline_mol_frac)

def interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.0242, 0.0242, 0.0242, 0.0242, 0.0242])
    f = interp1d(x, y, fill_value="extrapolate")
    return f(crystalline_mol_frac)

###################################
### DOPANT CONFIGURATION LOGIC  ###
###################################

def configure_dopant_parameters(p, crystalline_mol_frac, edge_to_find):
    if p.dopant == 'F4TCNQ':
        p.mw[Materials.DOPANT_ID] = 276.15
        p.dopant_vol_frac = interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac)
        if p.DopantOrientation == DopantOrientation.ISOTROPIC:
            if edge_to_find == 'C 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/F4TCNQ_C_Kedge_isotropic.txt'
            elif edge_to_find == 'N 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/F4TCNQ_N_Kedge_isotropic.txt'
            elif edge_to_find == 'F 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/F4TCNQ_F_Kedge_isotropic.txt'
        else:
            if edge_to_find == 'C 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/F4TCNQ_C_Kedge.txt'
            elif edge_to_find == 'N 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/F4TCNQ_N_Kedge.txt'
            elif edge_to_find == 'F 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/F4TCNQ_F_Kedge.txt'
    elif p.dopant == 'TFSI':
        p.mw[Materials.DOPANT_ID] = 280.14
        p.dopant_vol_frac = interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac)
        if p.DopantOrientation == DopantOrientation.ISOTROPIC:
            if edge_to_find == 'C 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/TFSI_C_Kedge_isotropic.txt'
            elif edge_to_find == 'N 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/TFSI_N_Kedge_isotropic.txt'
            elif edge_to_find == 'F 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/TFSI_F_Kedge_isotropic.txt'
        else:
            if edge_to_find == 'C 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/TFSI_C_Kedge.txt'
            elif edge_to_find == 'N 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/TFSI_N_Kedge.txt'
            elif edge_to_find == 'F 1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/xspectra_refractive_indices/TFSI_F_Kedge.txt'
    elif p.dopant == None:
        p.dopant_vol_frac = 0