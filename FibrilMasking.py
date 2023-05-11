#%% Importing

from scipy.interpolate import interp1d
from scipy.ndimage import rotate, gaussian_filter
from sklearn.preprocessing import normalize
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
import opensimplex as simplex

from PostProcessing import *
#%% System Setup
sys.path.append('/home/maxgruschka/NRSS/')

#%% Create the morphology/run parameters
# Declare model box size in nm (x,y,z)
gen_new_morph = False
morph_filename = "512x512x150nm_pitch2nm_rad8nm_std2nm_150fib_100-400nm.pickle"
energies = np.round(np.arange(283., 300.,1),1) # Energies for CyRSoXs (init,fin,step) (eV)
if gen_new_morph:
    x_dim_nm  = 512
    y_dim_nm  = 512
    z_dim_nm  = 150
    pitch_nm = 2 # Dimension of voxel in nm

    # Initialize morphology
    # Chosen Parameters:
    r_nm_avg = 8
    r_nm_std = 2
    num_fibrils = 150
    fib_length_nm_range = [100,400]
    

    morphology = Morphology(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)
    morphology.set_model_parameters(radius_nm_avg = r_nm_avg,
                                    radius_nm_std = r_nm_std,
                                    max_num_fibrils = num_fibrils,
                                    fibril_length_range_nm = fib_length_nm_range)

    morphology.fill_model()
    print("filled")
    morphology.voxelize_model()
    # sceneVoxel = morphology.get_scene(show_bounding_box=True,show_voxelized=True)
    print("voxelized")
    rmorphology = ReducedMorphology(morphology)
    print("reduced")
    # Generate material matricies
    rmorphology.pickle()
else:
    with open(morph_filename, "rb") as f:
        rmorphology = pickle.load(f)

mat_Vfrac, mat_S, mat_theta, mat_psi = generate_material_matricies(rmorphology)

# TODO: Re-implement roughness calculations
# surface = np.zeros((morphology.x_dim,morphology.y_dim))
# for x in range(morphology.x_dim):
#     for y in range(morphology.y_dim):
#         surface[x,y] = max(np.nonzero(mat_Vfrac[0][:,y,x])[0])
# #Roughness calcs: https://www.olympus-ims.com/en/metrology/surface-roughness-measurement-portal/parameters/#!cms[focus]=009
# rms_Surface = np.sqrt(1/(morphology.x_dim*morphology.y_dim) * ((surface*surface).sum())) #Rq
# skewness = 1/(rms_Surface**3) * (1/(morphology.x_dim*morphology.y_dim) * ((surface*surface*surface).sum()))
# print(f"R_q = {rms_Surface}\n Skewness = {skewness}")

write_hdf5([[mat_Vfrac[VACUUM_ID], mat_S[VACUUM_ID], mat_theta[VACUUM_ID], mat_psi[VACUUM_ID]],[mat_Vfrac[CRYSTAL_ID], mat_S[CRYSTAL_ID], mat_theta[CRYSTAL_ID], mat_psi[CRYSTAL_ID]],[mat_Vfrac[AMORPH_ID], mat_S[AMORPH_ID], mat_theta[AMORPH_ID], mat_psi[AMORPH_ID]],[mat_Vfrac[DOPANT_ID],mat_S[DOPANT_ID], mat_theta[DOPANT_ID], mat_psi[DOPANT_ID]]],
            float(rmorphology.pitch_nm), 'Fibril.hdf5')
checkH5('Fibril.hdf5', z_slice=0, runquiet=False) # I think it'll save the figs to the working dir for Max's pod setup - had to change an NRSS file

material_dict = {'Material1':'vacuum','Material2':'/home/maxgruschka/DopantModeling/P3HT.txt','Material3':'/home/maxgruschka/DopantModeling/P3HT_amorph.txt', 'Material4':'/home/maxgruschka/DopantModeling/TFSI_Na_C_Matfile.txt'}
energy_dict = {'Energy':6,'DeltaPerp':3, 'BetaPerp':1, 'DeltaPara':2, 'BetaPara':0}  
write_materials(energies, material_dict, energy_dict, num_materials)
write_config(list(energies), [0.0, 0.5, 360.0], CaseType=0, MorphologyType=0)


# Run the Cyrsoxs:
subprocess.run(["CyRSoXS","Fibril.hdf5"])
#%% Graphing
basePath = pathlib.Path('.').absolute()
h5path = pathlib.Path(basePath,'HDF5')
h5list = list(sorted(h5path.glob('*h5')))

def print_key(f, key):
    try:
        keys2 = f[key].keys()
        for key2 in keys2:
            new_key = key + '/' + key2
            print_key(f, new_key)
    except AttributeError:
        print(key)

with h5py.File(h5list[0],'r') as f:
    for key in f.keys():
        print_key(f, key)

load = cyrsoxsLoader()
integ = WPIntegrator(force_np_backend=True) # avoiding gpu backend for this tutorial
raw = load.loadDirectory(basePath)
remeshed = integ.integrateImageStack(raw)

fig, ax = plt.subplots(1,3,figsize=(10,3),dpi=140,constrained_layout=True)
raw.sel(energy=283).plot(norm=LogNorm(1e-2,1e7),cmap='terrain',ax=ax[0],add_colorbar=False)
raw.sel(energy=290).plot(norm=LogNorm(1e-2,1e7),cmap='terrain',ax=ax[1],add_colorbar=False)
raw.sel(energy=299).plot(norm=LogNorm(1e-2,1e7),cmap='terrain',ax=ax[2])

[{axes.set_xlim(-0.4,0.4),axes.set_ylim(-0.4,0.4)} for axes in ax]
plt.savefig("Scattering_3x_E-vals.png")

fig2,ax2 = plt.subplots(figsize=(3,3), dpi=140, constrained_layout=True)
# calculate the anisotropy metric
A = remeshed.rsoxs.AR(chi_width=20)

A.plot(x='q',cmap='bwr_r', vmin=-0.45, vmax=0.45,ax=ax2)
ax2.set_xlim(0.01,0.1)
fig2.savefig("Anisotropy_low-q.png")

fig3,ax3 = plt.subplots(figsize=(3,3), dpi=140, constrained_layout=True)
A.plot(x='q',cmap='bwr_r',ax=ax3)
ax3.set_xlim(.01,1)
ax3.set_ylim(283,295)
ax3.set_xscale('log')
fig3.savefig("Anisotropy_high-q.png")

save_parameters("Save-and-dope-trial",rmorphology, notes = "I am hoping to be able to use this setup to record experimental results for in-silico experiments, in a method that feels similar to benchtop - plan, run, record, analyze, document")