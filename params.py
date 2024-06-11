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
x_dim_nm = 256
y_dim_nm = x_dim_nm
z_dim_nm = 128
pitch_nm = 2

# Fibril Size Params
radius_nm_avg = 15
radius_nm_std = 3
min_fibril_length_nm = 100
max_fibril_length_nm = 400

# Fibril Growth Params
max_num_fibrils = 10000
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
height_feature = 0
max_valley_nm = 0

# Dopant Params
uniform_doping = False
dopant = 'Reduced_F4TCNQ'
dopant_vol_frac = 0
crystal_dopant_frac = 0.5
dopant_orientation = DopantOrientation.PERPENDICULAR

#####################################
### CYRSOXS SIMULATION PARAMETERS ###
#####################################

edge_to_find = 'C_1s'  # Default value, change as required

energies1 = np.round(np.arange(280., 286., 0.5),1)
energies2 = np.round(np.arange(286., 288., 0.2),1)
energies3 = np.round(np.arange(288., 291.5, 0.5),1)
energies = np.concatenate([energies1, energies2, energies3])
energies = np.round(np.arange(280., 281., 1),1)

material_dict = {
    'Material1': 'vacuum', 
    'Material2': 'vacuum',
    'Material3': 'vacuum',
    'Material4': 'vacuum'
}

energy_dict = {
    'Energy': 6,
    'DeltaPerp': 3,
    'BetaPerp': 1,
    'DeltaPara': 2,
    'BetaPara': 0 
}

######################################
### DOPANT INTERPOLATION FUNCTIONS ###
######################################

def interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.019, 0.053, 0.049, 0.067, 0.096])
    f = interp1d(x, y, fill_value="extrapolate")
    return max(f(crystalline_mol_frac), 0)

def interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.0242, 0.0242, 0.0242, 0.0242, 0.0242])
    f = interp1d(x, y, fill_value="extrapolate")
    return max(f(crystalline_mol_frac), 0)

###################################
### ENERGY CONFIGURATION LOGIC  ###
###################################
def configure_energies(p):
    if p.edge_to_find == 'C_1s':
        p.energies = [270., 272., 274., 276., 278., 280., 282., 282.25, 282.5, 282.75, 
                      283., 283.25, 283.5, 283.75, 284., 284.25, 284.5, 284.75, 285., 285.25, 
                      285.5, 285.75, 286., 286.5, 287., 287.5, 288., 288.5, 289., 289.5, 
                      290., 290.5, 291., 291.5, 292., 293., 294., 295., 296., 297., 
                      298., 299., 300., 301., 302., 303., 304., 305., 306., 310., 
                      314., 318., 320., 330., 340.]
    elif p.edge_to_find == 'N_1s':
        p.energies = [385., 386., 387., 388., 389., 390., 391., 392., 393., 394., 395., 396., 
                      397., 397.2, 397.4, 397.6, 397.8, 398., 398.2, 398.4, 398.6, 398.8, 399., 399.2, 
                      399.4, 399.6, 399.8, 400., 400.2, 400.4, 400.6, 400.8, 401., 402., 403., 404., 
                      405., 406., 407., 408., 409., 410., 412., 414., 416., 418., 420., 422., 
                      424., 426., 428.]
    elif p.edge_to_find == 'F_1s':
        p.energies = [670., 671., 672., 673., 674., 675., 676., 677., 678., 679., 680., 681., 682., 683., 
                      684., 685., 686., 687., 688., 689., 690., 691., 692., 693., 694., 695., 696., 697., 
                      698., 699., 700., 701., 702., 703., 704., 705., 706., 707., 708., 709.]

#############################
### POLYMER NEXAFS LOGIC  ###
#############################

def configure_polymer_NEXAFS(p):
    if p.edge_to_find == 'C_1s':
        p.material_dict['Material2'] = '/home/php/DopantModeling-dev/data/refractive_indices/P3HT_database_C_Kedge.txt'
        p.material_dict['Material3'] = '/home/php/DopantModeling-dev/data/refractive_indices/P3HT_database_C_Kedge_isotropic.txt'
    elif p.edge_to_find == 'N_1s':
        p.material_dict['Material2'] = '/home/php/DopantModeling-dev/data/refractive_indices/P3HT_database_N_Kedge.txt'
        p.material_dict['Material3'] = '/home/php/DopantModeling-dev/data/refractive_indices/P3HT_database_N_Kedge_isotropic.txt'
    elif p.edge_to_find == 'F_1s':
        p.material_dict['Material2'] = '/home/php/DopantModeling-dev/data/refractive_indices/P3HT_database_F_Kedge.txt'
        p.material_dict['Material3'] = '/home/php/DopantModeling-dev/data/refractive_indices/P3HT_database_F_Kedge_isotropic.txt'

###################################
### DOPANT CONFIGURATION LOGIC  ###
###################################

def configure_dopant_parameters(p, crystalline_mol_frac):
    if p.dopant == 'F4TCNQ':
        p.mw[Materials.DOPANT_ID] = 276.15
        p.dopant_vol_frac = interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac)
        if p.dopant_orientation == DopantOrientation.ISOTROPIC:
            if p.edge_to_find == 'C_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/F4TCNQ_C_Kedge_isotropic.txt'
            elif p.edge_to_find == 'N_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/F4TCNQ_N_Kedge_isotropic.txt'
            elif p.edge_to_find == 'F_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/F4TCNQ_F_Kedge_isotropic.txt'
        else:
            if p.edge_to_find == 'C_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/F4TCNQ_C_Kedge.txt'
            elif p.edge_to_find == 'N_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/F4TCNQ_N_Kedge.txt'
            elif p.edge_to_find == 'F_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/F4TCNQ_F_Kedge.txt'
    elif p.dopant == 'Reduced_F4TCNQ':
        p.mw[Materials.DOPANT_ID] = 276.15
        p.dopant_vol_frac = interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac)
        if p.dopant_orientation == DopantOrientation.ISOTROPIC:
            if p.edge_to_find == 'C_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/Reduced_F4TCNQ_C_Kedge_isotropic.txt'
            elif p.edge_to_find == 'N_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/Reduced_F4TCNQ_N_Kedge_isotropic.txt'
            elif p.edge_to_find == 'F_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/Reduced_F4TCNQ_F_Kedge_isotropic.txt'
        else:
            if p.edge_to_find == 'C_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/Reduced_F4TCNQ_C_Kedge.txt'
            elif p.edge_to_find == 'N_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/Reduced_F4TCNQ_N_Kedge.txt'
            elif p.edge_to_find == 'F_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/Reduced_F4TCNQ_F_Kedge.txt'
    elif p.dopant == 'TFSI':
        p.mw[Materials.DOPANT_ID] = 280.14
        p.dopant_vol_frac = interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac)
        if p.dopant_orientation == DopantOrientation.ISOTROPIC:
            if p.edge_to_find == 'C_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/TFSI_2_C_Kedge_isotropic.txt'
            elif p.edge_to_find == 'N_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/TFSI_2_N_Kedge_isotropic.txt'
            elif p.edge_to_find == 'F_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/TFSI_2_F_Kedge_isotropic.txt'
        else:
            if p.edge_to_find == 'C_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/TFSI_2_C_Kedge.txt'
            elif p.edge_to_find == 'N_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/TFSI_2_N_Kedge.txt'
            elif p.edge_to_find == 'F_1s':
                p.material_dict['Material4'] = '/home/php/DopantModeling-dev/data/refractive_indices/TFSI_2_F_Kedge.txt'
    elif p.dopant == None or p.dopant == 'Undoped':
        p.dopant_vol_frac = 0
        p.material_dict['Material4'] = 'vacuum'