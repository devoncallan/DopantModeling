#%% Importing

from scipy.interpolate import interp1d
from scipy.ndimage import rotate
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import trimesh
# import pyembree
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
x_dim_nm  = 256
y_dim_nm  = 256
z_dim_nm  = 32
pitch_nm = 2 # Dimension of voxel in nm

# Initialize morphology
morphology = Morphology(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm, 2)
morphology.set_model_parameters(radius_nm_avg = 3,
                                radius_nm_std = 1,
                                max_num_fibrils = 24,
                                fibril_length_range_nm = [25, 100])

morphology.fill_model()
# May not show if the morphology is too large (too many fibrils)
scene = morphology.get_scene(show_bounding_box=True)
scene.show()
morphology.voxelize_model()
sceneVoxel = morphology.get_scene(show_bounding_box=True,show_voxelized=True)

rmorphology = ReducedMorphology(morphology)
# Generate material matricies
mat1_Vfrac, mat1_S, mat1_theta, mat1_psi, mat2_Vfrac, mat2_S, mat2_theta, mat2_psi = rm.generate_material_matricies()

# Change material 1 matricies
f2 = 0.90
mat1_Vfrac = f2*np.ones_like(mat1_Vfrac) + (1-f2)*mat1_Vfrac # creates a matrix with Vfrac of mat1 = f2 everywhere, but 1 where fibrils are
mat1_Vfrac = gaussian_filter(mat1_Vfrac,  1) # Apply a gaussian filter to the above
mat1_S     = gaussian_filter(0.75*mat1_S, 1) # Apply Gaussian filter to array with 0.75 wherever the fibrils are

# mat1_theta = gaussian_filter(mat1_theta,  0) # I don't think we want to adjust the angles like this- 
# mat1_psi   = gaussian_filter(mat1_psi,    0) # this points them towards z = 0, which doens't make sense

# Change material 2 matricies
mat2_Vfrac = 1 - mat1_Vfrac
mat2_S     = mat2_S
mat2_theta = mat2_theta
mat2_psi   = mat2_psi