import pickle
import os
import sys
import shutil
import subprocess

# Append Paths
sys.path.append('/home/php/NRSS/')
sys.path.append('/home/php/DopantModeling/')

from NRSS.writer import write_materials, write_hdf5, write_config
from PostProcessor import PostProcessor

# Edge you want to find
edge_to_find = 'C 1s'

# Material Definitions
material_dict = {
    'Material1': 'vacuum', 
    'Material2': '/home/php/DopantModeling/xspectra_refractive_indices/P3HT_database_C_Kedge.txt', 
    'Material3': '/home/php/DopantModeling/xspectra_refractive_indices/P3HT_database_C_Kedge_isotropic.txt',
    'Material4': '/home/php/DopantModeling/xspectra_refractive_indices/TFSI_2_C_Kedge.txt'
}

energy_dict = {'Energy': 6, 'DeltaPerp': 3, 'BetaPerp': 1, 'DeltaPara': 2, 'BetaPara': 0}

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

# PostProcessor Setup
post_processor = PostProcessor(
    num_materials=4, mol_weight=mol_weight, density=density,
    dope_case=0, dopant_method='uniform', dopant_orientation=None, dopant_frac=0.0825, 
    core_shell_morphology=True, gaussian_std=3, fibril_shell_cutoff=0.2, 
    surface_roughness=True, height_feature=3,max_valley_nm=46, 
    amorph_matrix_Vfrac=1, amorphous_orientation=True)

# File I/O
pickle_save_name = '/home/php/CyRSoXS/PyHyperScattering_Batch_SST1_JupyterHub.pkl'

# Load Data
with open(pickle_save_name, 'rb') as file:
    data = pickle.load(file)
data_dict = {'scan_ids': data[0], 'sample_names': data[1], 'edges': data[2], 'ARs': data[3], 'paras': data[4], 'perps': data[5], 'circs': data[6], 'FY_NEXAFSs': data[7], 'Iq2s': data[8], 'ISIs': data[9]}
indices = [index for index, edge in enumerate(data_dict['edges']) if edge == edge_to_find]
energies = [data_dict['ARs'][index].energy for index in indices]
energies = energies[0].values[1:]

def process_pickle_file(filename, post_processor):
    print(f"Starting processing of pickle file: {filename}")

    hdf5_filename = os.path.splitext(filename)[0] + '.hdf5'
    description_filename = os.path.splitext(filename)[0]

    # Check if the corresponding .hdf5 file exists
    if not os.path.exists(hdf5_filename):
        print(f"Processing HDF5 file: {hdf5_filename}")
        with open(filename, 'rb') as f:
            rm = pickle.load(f)

        mat_Vfrac, mat_S, mat_theta, mat_psi = post_processor.generate_material_matrices(rm)
        crystalline_mol_fraction, amorphous_mol_fraction, dopant_mol_fraction = post_processor.analyze_mol_fractions(mat_Vfrac)

        post_processor.save_parameters(
            description_filename,
            rm,
            mat_Vfrac,
            mol_weight,
            density)

        max_index = len(mat_Vfrac)
        data_to_write = [[mat_Vfrac[i], mat_S[i], mat_theta[i], mat_psi[i]] for i in range(max_index)]
        phys_size = 2.0
        write_hdf5(data_to_write, phys_size, hdf5_filename)
        print(f"Wrote HDF5 file: {hdf5_filename}")
    else:
        print(f"{hdf5_filename} already exists, skipping generation...")

    write_materials(energies, material_dict, energy_dict, 4)
    write_config(list(energies), [0.0, 1.0, 360.0], CaseType=0, MorphologyType=0)
    print(f"Running CyRSoXS with file: {hdf5_filename}")
    subprocess.run(['CyRSoXS', hdf5_filename])

    print(f"Finished processing pickle file: {filename}")

# Gather a list of all .pickle files, excluding those in directories containing 'HDF5'
pickle_files = []
for root, dirs, files in os.walk('.'):
    for filename in files:
        if filename.endswith('.pickle') and 'HDF5' not in os.listdir(root):
            full_path = os.path.abspath(os.path.join(root, filename))
            pickle_files.append(full_path)

# Process the gathered .pickle files
for full_path in pickle_files:
    root, filename = os.path.split(full_path)
    print(f"Processing file: {filename}")

    base_filename = os.path.splitext(filename)[0]
    if os.path.basename(root) == base_filename:
        print(f"{filename} is already in a directory with a matching name")
        os.chdir(root)
        print(f"Changed working directory to: {root}")
        process_pickle_file(filename, post_processor)
        continue

    new_directory = os.path.join(root, base_filename)
    os.makedirs(new_directory, exist_ok=True)
    print(f"Created directory: {new_directory}")
    new_path = os.path.abspath(os.path.join(new_directory, filename))
    shutil.move(full_path, new_path)
    print(f"Moved {filename} to {new_path}")

    os.chdir(new_directory)
    print(f"Changed working directory to: {new_directory}")
    process_pickle_file(filename, post_processor)