from src.Morphology.FibrilGenerator import FibrilGenerator, FibrilSizeParams, FibrilGrowthParams, FibrilOrientationParams
from src.Morphology.FibrilGenerator import FibrilOrientation, FibrilDistribution
from tqdm import tqdm
import napari 
import numpy as np


# Simulation volume dimensions
x_dim_nm = 1024
y_dim_nm = x_dim_nm
z_dim_nm = 256
pitch_nm = 2

max_dim = int(x_dim_nm / pitch_nm)

# Define fibril generator parameters
fibril_size_params = FibrilSizeParams(
    radius_nm_avg=15,
    radius_nm_std=3,
    min_fibril_length_nm=50,
    max_fibril_length_nm=400
)

fibril_growth_params = FibrilGrowthParams(
    max_num_fibrils=200,
    fibril_distribution=FibrilDistribution.PDS,
    c2c_dist_nm=45,
    symmetrical_growth=False,
    periodic_bc=False
)

f=0.3
fibril_orientation_params = FibrilOrientationParams(
    fibril_orientation=FibrilOrientation.GRF_SAMPLE_FLAT,
    theta_distribution_csv=None,
    k=1./(max_dim*np.pi*f),
    std=1./(max_dim*np.pi*f)
)

# Initialize fibril generator
fibgen = FibrilGenerator(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)







# Set fibril generator parameters
fibgen.set_model_parameters(
    fibril_size_params=fibril_size_params,
    fibril_growth_params=fibril_growth_params,
    fibril_orientation_params=fibril_orientation_params
)

# Fill model with fibrils
fibgen.fill_model()

viewer = napari.Viewer(ndisplay=3)
layer = viewer.add_image(fibgen.box)
layer.blending = 'additive'
layer.colormap = 'twilight_shifted'
layer.rendering = 'attenuated_mip'
napari.run()



from src.Morphology.FibrilGenerator import Materials
from src.Morphology.FibrilPostProcessor import FibrilPostProcessor, MaterialParams, SurfaceRoughnessParams, CoreShellParams, MatrixParams, DopantParams

from src.Morphology.FibrilPostProcessor import DopantOrientation

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

material_params = MaterialParams(
    num_materials=4, mw=mw, density=density)

core_shell_params = CoreShellParams(
    gaussian_std=3, fibril_shell_cutoff=0.2)

matrix_params = MatrixParams(
    amorph_matrix_Vfrac=0.9,
    amorph_orientation=False)

surface_roughness_params = SurfaceRoughnessParams(
    height_feature=3,
    max_valley_nm=43)

dopant_params = DopantParams(
    dopant_vol_frac=0.0,
    crystal_dopant_frac=1.0,
    dopant_orientation=DopantOrientation.ISOTROPIC)

post_processor = FibrilPostProcessor(
    material_params=material_params,
    matrix_params=matrix_params,
    surface_roughness_params=surface_roughness_params,
    core_shell_params=core_shell_params,
    dopant_params=dopant_params)