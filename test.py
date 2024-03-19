import numpy as np
import matplotlib.pyplot as plt
import napari 
import sys
from tqdm import tqdm
from src.Morphology.Fibril.FibrilGenerator import FibrilGenerator, FibrilSizeParams, FibrilGrowthParams, FibrilOrientationParams
from src.Morphology.Fibril.FibrilGenerator import FibrilOrientation, FibrilDistribution, Materials
from src.Morphology.Fibril.FibrilPostProcessor import FibrilPostProcessor, MaterialParams, SurfaceRoughnessParams, CoreShellParams, MatrixParams, DopantParams
from src.Morphology.Fibril.FibrilPostProcessor import DopantOrientation, analyze_mol_fractions, analyze_vol_fractions

x_dim_nm = 1024
y_dim_nm = x_dim_nm
z_dim_nm = 512
pitch_nm = 2

# Initialize fibril generator
fibgen = FibrilGenerator(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)

# Define fibril generator parameters
fibril_size_params = FibrilSizeParams(
    radius_nm_avg=15, 
    radius_nm_std=3,
    min_fibril_length_nm=100, 
    max_fibril_length_nm=400
)

fibril_growth_params = FibrilGrowthParams(
    max_num_fibrils=1250, 
    fibril_distribution=FibrilDistribution.PDS,
    c2c_dist_nm=45, 
    symmetrical_growth=True, 
    periodic_bc=True
)

f=0.05
fibril_orientation_params = FibrilOrientationParams(
    fibril_orientation=FibrilOrientation.GRF_SAMPLE_ALL,
    theta_distribution_csv=r'/home/php/DopantModeling-dev/data/theta_distributions/avg_0p1_theta_distribution.csv',
    k=1./25,
    std=1./125
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

mw = {
    Materials.CRYSTAL_ID: 166.2842, # Molecular weight of crystalline P3HT repeat
    Materials.AMORPH_ID: 166.2842,  # Molecular weight of amorphous P3HT repeat
    Materials.DOPANT_ID: 276.15     # Molecular weight of TFSI- = 280.14, Molecular weight of F4TCNQ = 276.15
}

density = {
    Materials.CRYSTAL_ID: 1.1, # Density of crystalline P3HT
    Materials.AMORPH_ID: 1.1*0.9,  # Density of amorphous P3HT
    Materials.DOPANT_ID: 1.1   # Density of dopant in P3HT
}

material_params = MaterialParams(
    num_materials=4, mw=mw, density=density)

core_shell_params = CoreShellParams(
    gaussian_std=3, fibril_shell_cutoff=0.2)

matrix_params = MatrixParams(
    amorph_orientation=True)

# surface_roughness_params = SurfaceRoughnessParams(
#     height_feature=3,max_valley_nm=43)
surface_roughness_params = None

dopant_params = DopantParams(
    dopant_vol_frac=0.09, crystal_dopant_frac=0,
    uniform_doping = True, # if True, automatically calculates crystal_dopant_frac needed
    dopant_orientation=DopantOrientation.ISOTROPIC)

post_processor = FibrilPostProcessor(
    material_params=material_params,
    matrix_params=matrix_params,
    surface_roughness_params=surface_roughness_params,
    core_shell_params=core_shell_params,
    dopant_params=dopant_params)

data = fibgen.create_morphology_data()
data = post_processor.process(data)
post_processor.save_parameters(data, fibgen, 'Fibril')