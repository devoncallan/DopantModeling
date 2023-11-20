import pickle
import os
import sys
import shutil
import subprocess

# Append Paths
sys.path.append('/home/php/NRSS/')
sys.path.append('/home/php/DopantModeling/')

from Morphology import Morphology
from ReducedMorphology import ReducedMorphology
from PostProcessor import PostProcessor

# Declare model box size in nm (x,y,z)
x_dim_nm  = 1024
y_dim_nm  = 1024
z_dim_nm  = 512
pitch_nm = 2 # Dimension of voxel in nm

VACUUM_ID, CRYSTAL_ID, AMORPH_ID, DOPANT_ID = 0, 1, 2, 3

mol_weight = {
    CRYSTAL_ID: 166.2842, # Molecular weight of crystalline P3HT repeat
    AMORPH_ID: 166.2842,  # Molecular weight of amorphous P3HT repeat
    DOPANT_ID: 280.14     # Molecular weight of TFSI- = 280.14, Molecular weight of F4TCNQ = 276.15
}

density = {
    CRYSTAL_ID: 1.1, # Density of crystalline P3HT
    AMORPH_ID: 1.1,  # Density of amorphous P3HT
    DOPANT_ID: 1.1   # Density of dopant in P3HT
}

# Loop through a range of fibril numbers
for num_fibrils in range(250, 100001, 250):
    # Initialize morphology
    morphology = Morphology(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)
    morphology.set_model_parameters(radius_nm_avg = 15,
                                    radius_nm_std = 3,
                                    max_num_fibrils = num_fibrils,
                                    fibril_length_range_nm = [50, 200],
                                    rand_orientation = 3, # sample psi and theta using gaussian random fields
                                    theta_distribution_csv = r'/home/php/DopantModeling/theta_distributions/avg_0p1_theta_distribution.csv',
				    k = 1/2.5,
                                    std = 1/12.5)
    
    morphology.fill_model(timeout = 360, plot_histogram=False)
    morphology.voxelize_model()

    rm = ReducedMorphology(morphology)

    rm.pickle()
    print(f"Simulation and pickling completed for {num_fibrils} fibrils.")

    most_recent_file = None
    most_recent_mtime = 0
    
    # Iterate through all subdirectories and find the most recently modified .pickle file
    for root, dirs, files in os.walk('.'):
        for filename in files:
            if filename.endswith('.pickle') and 'HDF5' not in os.listdir(root):
                full_path = os.path.abspath(os.path.join(root, filename))
                mtime = os.path.getmtime(full_path)
                if mtime > most_recent_mtime:
                    most_recent_mtime = mtime
                    most_recent_file = full_path
    
    # Load the most recently modified .pickle file
    rm = None
    if most_recent_file:
        print(f"Loading the most recently modified file: {most_recent_file}")
        with open(most_recent_file, 'rb') as f:
            rm = pickle.load(f)
    else:
        print("No pickle files found.")
        
    # PostProcessor Setup
    post_processor = PostProcessor(
        num_materials=4, mol_weight=mol_weight, density=density,
        dope_case=0, dopant_method='preferential', dopant_orientation='parallel', 
        dopant_vol_frac=0.0, crystal_dope_frac = 0.5,
        core_shell_morphology=True, gaussian_std=3, fibril_shell_cutoff=0.2, 
        surface_roughness=False, height_feature=3, max_valley_nm=46, 
        amorph_matrix_Vfrac=1, amorphous_orientation=False)
    
    mat_Vfrac, mat_S, mat_theta, mat_psi = post_processor.generate_material_matrices(rm)
    crystalline_mol_fraction, amorphous_mol_fraction, dopant_mol_fraction = post_processor.analyze_mol_fractions(mat_Vfrac)
     
    post_processor.save_parameters(
        os.path.basename(os.path.splitext(most_recent_file)[0]),
        rm,
        mat_Vfrac,
        mol_weight,
        density)