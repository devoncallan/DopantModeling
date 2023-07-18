#%% Importing

from scipy.interpolate import interp1d
from scipy.ndimage import rotate, gaussian_filter
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import trimesh
import pickle
from NRSS.writer import write_materials, write_hdf5, write_config, write_slurm
from NRSS.checkH5 import checkH5

from Morphology import Morphology
from ReducedMorphology import ReducedMorphology

import sys
import pathlib

import subprocess
import h5py

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator


import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.colors import LogNorm

import os
import glob

import PostProcessing
#%% System Setup
sys.path.append('/home/maxgruschka/NRSS/')

#%% Create the morphology/run parameters
# Declare model box size in nm (x,y,z)
gen_new_morph = False

morphs12nmKnown = ['/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/512x512x150nm_pitch2nm_rad8nm_std2nm_150fib_100-400nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x256nm_pitch2nm_rad12nm_std3nm_400fib_100-400nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x256nm_pitch2nm_rad12nm_std3nm_400fib_100-500nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x256nm_pitch2nm_rad12nm_std3nm_500fib_100-400nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x500nm_pitch2nm_rad12nm_std3nm_1000fib_100-500nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x256nm_pitch2nm_rad12nm_std3nm_100fib_100-400nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x256nm_pitch2nm_rad12nm_std3nm_300fib_100-400nm.pickle', 
                     "/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/1024x1024x256nm_pitch2nm_rad12nm_std3nm_50fib_100-400nm.pickle",
                     '/home/maxgruschka/gpuTest/Morphologies/12_nmFibs/RandFibs1024x1024x256nm_pitch2nm_rad12nm_std3nm_400fib_100-400nm.pickle']

morphs15nmKnown = ['/home/maxgruschka/gpuTest/Morphologies/15_nmFibs/1024x1024x256nm_pitch2nm_rad15nm_std3nm_100fib_100-400nm.pickle',
                     '/home/maxgruschka/gpuTest/Morphologies/15_nmFibs/1024x1024x256nm_pitch2nm_rad15nm_std3nm_300fib_100-400nm.pickle', 
                     '/home/maxgruschka/gpuTest/Morphologies/15_nmFibs/1024x1024x256nm_pitch2nm_rad15nm_std3nm_400fib_100-400nm.pickle', 
                     '/home/maxgruschka/gpuTest/Morphologies/15_nmFibs/1024x1024x256nm_pitch2nm_rad15nm_std3nm_500fib_100-400nm.pickle']

runNote = 'Comparison of NaTFSI anisotropy to aligned/antialigned F4TCNQ'

morph_filename = morphs12nmKnown[1]
# dopantFile = '/home/maxgruschka/DopantModeling/F4TCNQ_Reor_C.txt'
dopantFile = '/home/maxgruschka/DopantModeling/TFSINa_C.txt'
# dopantAlignments = [[True, False, True], [True, True, False], [True, False, False]]
dopantAlignments = [[False,False,False]]
# morph_filename = morphs15nmKnown[]
energies1 = np.round(np.arange(280., 286., 0.5),1)
energies2 = np.round(np.arange(286., 288., 0.2),1)
energies3 = np.round(np.arange(288., 295., 0.5),1)
energies = np.concatenate([energies1, energies2, energies3])

dope_types = [0]
dopant_frac = 0.0825
core_shell_morphologies = [True]
gaussian_std = 3
fibril_shell_cutoff = 0.2
# Surface roughness parameters
surface_roughnesses = [True]
height_features = [3]
max_valley_nms = [45]
amorph_matrix_Vfrac = 0.9

if gen_new_morph:
    x_dim_nm  = 1024
    y_dim_nm  = 1024
    z_dim_nm  = 256
    pitch_nm = 2 # Dimension of voxel in nm

    # Initialize morphology
    # Chosen Parameters:
    r_nm_avg = 12
    r_nm_std = 3
    num_fibrils = 400
    fib_length_nm_range = [100,400]
    

    morphology = Morphology(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)
    morphology.set_model_parameters(radius_nm_avg = r_nm_avg,
                                    radius_nm_std = r_nm_std,
                                    max_num_fibrils = num_fibrils,
                                    fibril_length_range_nm = fib_length_nm_range, rand_orientation=True)
    
    morphology.fill_model()
    print("filled")
    morphology.voxelize_model()
    # sceneVoxel = morphology.get_scene(show_bounding_box=True,show_voxelized=True)
    print("voxelized")
    rmorphology = ReducedMorphology(morphology)
    del morphology
    print("reduced")
    # Generate material matricies
    rmorphology.pickle()
    morph_filename = glob.glob('*.pickle')
else:
    with open(morph_filename, "rb") as f:
        rmorphology = pickle.load(f)

parent_dir = os.getcwd()
for dopantAlignment in dopantAlignments:
    for dope_type in dope_types:
        for core_shell_morphology in core_shell_morphologies:
            for surface_roughness in surface_roughnesses:
                for height_feature in height_features:
                    for max_valley_nm in max_valley_nms:
                        path = os.path.join(parent_dir,f"TFSI_Align{dopantAlignment[1]}_Anti{dopantAlignment[2]}")
                        try:
                            os.mkdir(path)
                        except:
                            print("No new folder needed")
                        os.chdir(path)
                        PostProcessing.dope_type = dope_type
                        PostProcessing.surface_roughness = surface_roughness
                        PostProcessing.height_feature = height_feature
                        PostProcessing.max_valley_nm = max_valley_nm
                        PostProcessing.core_shell_morphology = core_shell_morphology
                        PostProcessing.gaussian_std = gaussian_std
                        PostProcessing.dopant_orientation = dopantAlignment[0]
                        PostProcessing.dopant_aligned = dopantAlignment[1]
                        PostProcessing.dopant_antialigned = dopantAlignment[2]
                        mat_Vfrac, mat_S, mat_theta, mat_psi = PostProcessing.generate_material_matricies(rmorphology)
                        
                        write_hdf5([[mat_Vfrac[PostProcessing.VACUUM_ID], mat_S[PostProcessing.VACUUM_ID], mat_theta[PostProcessing.VACUUM_ID], mat_psi[PostProcessing.VACUUM_ID]],[mat_Vfrac[PostProcessing.CRYSTAL_ID], mat_S[PostProcessing.CRYSTAL_ID], mat_theta[PostProcessing.CRYSTAL_ID], mat_psi[PostProcessing.CRYSTAL_ID]],[mat_Vfrac[PostProcessing.AMORPH_ID], mat_S[PostProcessing.AMORPH_ID], mat_theta[PostProcessing.AMORPH_ID], mat_psi[PostProcessing.AMORPH_ID]],[mat_Vfrac[PostProcessing.DOPANT_ID],mat_S[PostProcessing.DOPANT_ID], mat_theta[PostProcessing.DOPANT_ID], mat_psi[PostProcessing.DOPANT_ID]]],
                                    float(rmorphology.pitch_nm), 'Fibril.hdf5')
                        checkH5('Fibril.hdf5', z_slice=0, runquiet=False) # I think it'll save the figs to the working dir for Max's pod setup - had to change an NRSS file
                        
                        material_dict = {'Material1':'vacuum','Material2':'/home/maxgruschka/DopantModeling/P3HT.txt','Material3':'/home/maxgruschka/DopantModeling/P3HT.txt', 'Material4':dopantFile}
                        energy_dict = {'Energy':6,'DeltaPerp':3, 'BetaPerp':1, 'DeltaPara':2, 'BetaPara':0}  
                        write_materials(energies, material_dict, energy_dict, PostProcessing.num_materials)
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
                        integ = WPIntegrator(force_np_backend=False) # avoiding gpu backend for this tutorial
                        raw = load.loadDirectory(basePath)
                        remeshed = integ.integrateImageStack(raw)
                        
                        fig, ax = plt.subplots(1,3,figsize=(10,3),dpi=140,constrained_layout=True)
                        raw.sel(energy=280).plot(norm=LogNorm(1e-2,1e7),cmap='terrain',ax=ax[0],add_colorbar=False)
                        raw.sel(energy=290).plot(norm=LogNorm(1e-2,1e7),cmap='terrain',ax=ax[1],add_colorbar=False)
                        raw.sel(energy=294).plot(norm=LogNorm(1e-2,1e7),cmap='terrain',ax=ax[2])
                        
                        [{axes.set_xlim(-0.4,0.4),axes.set_ylim(-0.4,0.4)} for axes in ax]
                        plt.savefig("Scattering_3x_E-vals.png")
                        
                        fig2,ax2 = plt.subplots(figsize=(4,3), dpi=140, constrained_layout=True)
                        # calculate the anisotropy metric
                        A = remeshed.rsoxs.AR(chi_width=20)
                        
                        A.plot(x='q',cmap='bwr_r', vmin=-0.45, vmax=0.45,ax=ax2)
                        ax2.set_xlim(0.01,0.1)
                        fig2.savefig("Anisotropy_low-q.png")
                        
                        fig3,ax3 = plt.subplots(figsize=(0.55*8,0.5*6), dpi=140, constrained_layout=True)
                        # cbar_ax = fig3.add_axes([0.7, 0.2, 0.02, 0.62],label='Anisotropy')
                        im_cbar = A.plot(x='q',cmap = 'RdBu', vmin=-0.35, vmax=0.35, ax=ax3, add_colorbar=False)
                        cbar = fig3.colorbar(im_cbar, ax=ax3,label='Anisotropy')

                        ax3.set_xlim(0.1,0.9)
                        ax3.set_xticks([0.3,0.5,0.7])
                        ax3.set_xticklabels(['0.03','0.05','0.07'])
                        ax3.xaxis.set_tick_params(which='major',size=5, width=2, direction='inout',top='on')
                        ax3.yaxis.set_tick_params(which='major', size=5, width=2, direction='inout', right='on')
                        ax3.set_xlabel('$\it{q}$ ($\mathrm{\AA}^{-1}$)')
                        ax3.set_ylim(280,292)
                        ax3.set_yticks([283,286,289])
                        ax3.set_ylabel('Energy (eV)')
                        fig3.savefig("CleanAnisotropy_high-q.png")
                        print(f"Roughness = {PostProcessing.calc_roughness(mat_Vfrac,rmorphology.pitch_nm)}")
                        crysVfrac, amorphVfrac, dopeVfrac = PostProcessing.calc_volFracs(mat_Vfrac)
                        PostProcessing.save_parameters("Save-and-dope-trial",rmorphology, morph_filename, notes = runNote+f"\nroughness={PostProcessing.calc_roughness(mat_Vfrac,rmorphology.pitch_nm)} \n Vol Fractions of Occupied:\n \t Crystal: {crysVfrac},\n \t Amorphous: {amorphVfrac},\n \t Dopant: {dopeVfrac}\n Dopant File: {dopantFile}")
                        load = None
                        integ = None
                        os.chdir("..")
                        plt.close('all')
