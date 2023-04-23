#%% Importing

from scipy.interpolate import interp1d
from scipy.ndimage import rotate
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import trimesh
import random
import pickle
from NRSS.writer import write_materials, write_hdf5, write_config, write_slurm
from NRSS.checkH5 import checkH5

from Morphology import Morphology
from Fibril import Fibril
from ReducedMorphology import ReducedMorphology

import sys
import pathlib


import subprocess
import h5py

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator


import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.colors import LogNorm

import numpy as np
import io
from PIL import Image
import PIL

#%% System Setup
sys.path.append('/home/maxgruschka/NRSS/')

#%% Create the morphology
# Declare model box size in nm (x,y,z)
x_dim_nm  = 1028
y_dim_nm  = 1028
z_dim_nm  = 128
pitch_nm = 2 # Dimension of voxel in nm
# uniformDopant_Flag = True
amorph_mat_vfrac = 0.9

# Initialize morphology
# Chosen Parameters:
num_materials = 2
r_nm_avg = 12
r_nm_std = 3
num_fibrils = 50
fib_length_nm_range = [100,400]
energies = np.round(np.arange(280., 300., 0.5),1)


morphology = Morphology(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm, num_materials)
morphology.set_model_parameters(radius_nm_avg = r_nm_avg,
                                radius_nm_std = r_nm_std,
                                max_num_fibrils = num_fibrils,
                                fibril_length_range_nm = fib_length_nm_range)

morphology.fill_model()

# May not show if the morphology is too large (too many fibrils)
# scene = morphology.get_scene(show_bounding_box=True)
# scene.show()
morphology.voxelize_model()
# sceneVoxel = morphology.get_scene(show_bounding_box=True,show_voxelized=True)

rmorphology = ReducedMorphology(morphology)
# Generate material matricies
mat_Vfrac, mat_S, mat_theta, mat_psi = rmorphology.generate_material_matricies()


# Change material 1 matricies

mat_Vfrac[0] = amorph_mat_vfrac*np.ones_like(mat_Vfrac[0]) + (1-amorph_mat_vfrac)*mat_Vfrac[0] # creates a matrix with Vfrac of mat1 = f2 everywhere, but 1 where fibrils are
mat_Vfrac[0] = gaussian_filter(mat_Vfrac[0],  1) # Apply a gaussian filter to the above
mat_S[0]     = gaussian_filter(0.75*mat_S[0], 1) # Apply Gaussian filter to array with 0.75 wherever the fibrils are

# mat1_theta = gaussian_filter(mat1_theta,  0) # I don't think we want to adjust the angles like this- 
# mat1_psi   = gaussian_filter(mat1_psi,    0) # this points them towards z = 0, which doens't make sense

# Change material 2 matricies
mat_Vfrac[1] = 1 - mat_Vfrac[0]
mat_S[1]     = mat_S[1]
mat_theta[1] = mat_theta[1]
mat_psi[1]   = mat_psi[1]

write_hdf5([[mat_Vfrac[0,:,:,:], mat_S[0,:,:,:], mat_theta[0,:,:,:], mat_psi[0,:,:,:]], 
            [mat_Vfrac[1,:,:,:], mat_S[1,:,:,:], mat_theta[1,:,:,:], mat_psi[1,:,:,:]]],
            float(pitch_nm), 'Fibril.hdf5')
checkH5('Fibril.hdf5', z_slice=25, runquiet=True)
material_dict = {'Material1':'/home/maxgruschka/DopantModeling/P3HT.txt','Material2':'vacuum'}
energy_dict = {'Energy':6,'DeltaPerp':3, 'BetaPerp':1, 'DeltaPara':2, 'BetaPara':0}  
write_materials(energies, material_dict, energy_dict, 2)
write_config(list(energies), [0.0, 1.0, 360.0], CaseType=0, MorphologyType=0)