import sys
sys.path.append('/home/devon/Documents/Github/NRSS/')
sys.path.append('/home/devon/Documents/Github/PyHyperScattering/src/')
sys.path.append('/home/devon/Documents/Github/Fibril/src/')


from Morphology.FibrilGenerator import Materials
from Morphology.FibrilPostProcessor import FibrilPostProcessor, MaterialParams, SurfaceRoughnessParams, CoreShellParams, MatrixParams, DopantParams

from src.Morphology.FibrilPostProcessor import DopantOrientation

from src.Morphology.FibrilGenerator import FibrilGenerator, FibrilSizeParams, FibrilGrowthParams, FibrilOrientationParams
from src.Morphology.FibrilGenerator import FibrilOrientation, FibrilDistribution
from src.Morphology.MorphologyData import MorphologyData


from NRSS.writer import write_materials, write_hdf5, write_config
from NRSS.checkH5 import checkH5

import numpy as np
import subprocess

import pathlib
from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

import matplotlib.pyplot as plt

from src.Common.files import make_output_dir, move

DEFAULT_MORPH_FILE = 'Morphology.hdf5'

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


def fill_model() -> FibrilGenerator:

    # Initialize fibril generator
    fibgen = FibrilGenerator(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)

    # Define fibril generator parameters
    fibril_size_params = FibrilSizeParams(
        radius_nm_avg=radius_nm_avg, 
        radius_nm_std=radius_nm_std,
        min_fibril_length_nm=min_fibril_length_nm, 
        max_fibril_length_nm=max_fibril_length_nm
    )

    fibril_growth_params = FibrilGrowthParams(
        max_num_fibrils=max_num_fibrils, 
        fibril_distribution=fibril_distribution,
        c2c_dist_nm=c2c_dist_nm, 
        symmetrical_growth=symmetrical_growth, 
        periodic_bc=periodic_bc
    )

    fibril_orientation_params = FibrilOrientationParams(
        fibril_orientation=FibrilOrientation.GRF_SAMPLE_FLAT,
        k=k,
        std=std
        # k=1./(fibgen.x_dim*np.pi*f), 
        # std=1./(fibgen.x_dim*np.pi*f)
    )

    # Set fibril generator parameters
    fibgen.set_model_parameters(
        fibril_size_params=fibril_size_params, 
        fibril_growth_params=fibril_growth_params,
        fibril_orientation_params=fibril_orientation_params
    )

    # Fill model with fibrils
    fibgen.fill_model()

    return fibgen

def post_process_model(fibgen: FibrilGenerator) -> MorphologyData:

    material_params = MaterialParams(
        num_materials=num_materials, 
        mw=mw, 
        density=density
    )

    core_shell_params = CoreShellParams(
        gaussian_std=gaussian_std, 
        fibril_shell_cutoff=fibril_shell_cutoff)

    matrix_params = MatrixParams(
        amorph_matrix_Vfrac=amorph_matrix_Vfrac, 
        amorph_orientation=amorph_orientation)

    surface_roughness_params = None
    if use_surface_roughness_params:
        surface_roughness_params = SurfaceRoughnessParams(
            height_feature=height_feature,
            max_valley_nm=max_valley_nm)
    
    dopant_params = None
    if use_dopant_params:
        dopant_params = DopantParams(
            dopant_vol_frac=dopant_vol_frac, 
            crystal_dopant_frac=crystal_dopant_frac,
            dopant_orientation=DopantOrientation.ISOTROPIC)

    post_processor = FibrilPostProcessor(
        material_params=material_params,
        matrix_params=matrix_params,
        surface_roughness_params=surface_roughness_params,
        core_shell_params=core_shell_params,
        dopant_params=dopant_params)
    
    data = fibgen.create_morphology_data()
    data = post_processor.process(data)
    post_processor.save_parameters(data, fibgen)
    write_hdf5(data.get_data()[0:num_materials], float(pitch_nm), DEFAULT_MORPH_FILE)

    return data
    
    
def run_cyrsoxs(save_dir=''):

    
    write_materials(energies, material_dict, energy_dict, num_materials)
    write_config(list(energies), [0.0, 1.0, 360.0], CaseType=0, MorphologyType=0)

    subprocess.run(['CyRSoXS', 'Morphology.hdf5'])

    if save_dir == '':
        return
    
    make_output_dir(save_dir)
    
    move(src='config.txt', dest_dir=save_dir)
    move(src='CyRSoXS.log', dest_dir=save_dir)
    move(src='parameters.txt', dest_dir=save_dir)
    move(src='HDF5', dest_dir=save_dir)
    for i in range(num_materials):
        move(src=f'Material{i+1}.txt', dest_dir=save_dir)
       
    

def get_para_perp_AR(raw_data):

    integ = WPIntegrator()
    integ_data = integ.integrateImageStack(raw_data)

    para = integ_data.rsoxs.slice_chi(90, chi_width = 45).sel(q = slice(min_q, max_q)) + \
        integ_data.rsoxs.slice_chi(-90, chi_width = 45).sel(q = slice(min_q, max_q))
    perp = integ_data.rsoxs.slice_chi(0, chi_width = 45).sel(q = slice(min_q, max_q)) + \
        integ_data.rsoxs.slice_chi(180, chi_width = 45).sel(q = slice(min_q, max_q))
    AR = (para - perp)/(para + perp)

    return para, perp, AR


# def plot_raw_data_2D(raw_data, q_range=(0.1, 0.9), I_range=(1e-1, 1e7), save_dir=''):

#     q_min, q_max = q_range
#     I_min, I_max = I_range

#     fig, axs = plt.subplots(2,3,figsize=(10,6),dpi=140,constrained_layout=True)
#     axs = axs.flatten()

#     raw_data.sel(energy=280).plot(x='qx',y='qy',norm=LogNorm(I_min,I_max),cmap='terrain',ax=axs[0],add_colorbar=False)
#     raw_data.sel(energy=282).plot(x='qx',y='qy',norm=LogNorm(I_min,I_max),cmap='terrain',ax=axs[1],add_colorbar=False)
#     raw_data.sel(energy=284).plot(x='qx',y='qy',norm=LogNorm(I_min,I_max),cmap='terrain',ax=axs[2])
#     raw_data.sel(energy=286).plot(x='qx',y='qy',norm=LogNorm(I_min,I_max),cmap='terrain',ax=axs[3],add_colorbar=False)
#     raw_data.sel(energy=288).plot(x='qx',y='qy',norm=LogNorm(I_min,I_max),cmap='terrain',ax=axs[4],add_colorbar=False)
#     raw_data.sel(energy=290).plot(x='qx',y='qy',norm=LogNorm(I_min,I_max),cmap='terrain',ax=axs[5])

#     [{
#         ax.set_xlim(-min_q,max_q),
#         ax.set_ylim(-min_q,max_q),
#         ax.set_xlabel('$q_x$ (nm$^{-1}$)'),
#         ax.set_ylabel('$q_y$ (nm$^{-1}$)')} 
#     for ax in axs]
#     plt.show()


# def animate_raw_data_1D(raw_data)


def load_cyrsoxs(base_path):

    load = cyrsoxsLoader()
    raw_data = load.loadDirectory(base_path, PhysSize=pitch_nm)

    return raw_data

def analyze_cyrsoxs(raw_data):

    para, perp, AR = get_para_perp_AR(raw_data)

    return



##########################
### CYRSOXS PARAMETERS ###
##########################

energies1 = np.round(np.arange(280., 286., 0.5),1)
energies2 = np.round(np.arange(286., 288., 0.2),1)
energies3 = np.round(np.arange(288., 291.5, 0.5),1)
energies = np.concatenate([energies1, energies2, energies3])

material_dict = {
    'Material1': 'vacuum', 
    'Material2': 'data/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge.txt', 
    'Material3': 'data/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge_isotropic.txt',
    'Material4': 'vacuum'
}
energy_dict = {
    'Energy': 6,
    'DeltaPerp': 3,
    'BetaPerp': 1,
    'DeltaPara': 2,
    'BetaPara': 0 
}

min_q = 0.1
max_q = 0.9

min_E = 280
max_E = 290

min_I = 1e-1
max_I = 1e+7

MOPRH_FILE = 'Morphology.hdf5'

dest_dir = 'test/'


# for i in range(10):

# fibgen = fill_model()
# data = post_process_model(fibgen)

run_cyrsoxs(data, save_dir='experiments/num_fibrils/')

# raw_data = load_cyrsoxs(base_path)

# para, perp, AR = get_para_perp_AR(raw_data)

    



