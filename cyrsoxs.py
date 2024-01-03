import pickle
import re
import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import random

# Append Paths
sys.path.append('/home/php/NRSS/')
sys.path.append('/home/php/DopantModeling/')

from NRSS.writer import write_materials, write_hdf5, write_config
from PostProcessor import PostProcessor

def generate_ticks(start, end, num_ticks, rounding_order):
    tick_spacing = (end - start) / (num_ticks - 1)
    tick_vals = [start + i * tick_spacing for i in range(num_ticks)]
    rounded_tick_vals = [round(val / rounding_order) * rounding_order for val in tick_vals]
    
    return rounded_tick_vals

def update_mol_weight_for_dopant(root, mol_weight):
    if 'TFSI' in root:
        mol_weight[DOPANT_ID] = 280.14  # Molecular weight of TFSI-
    elif 'F4TCNQ' in root:
        mol_weight[DOPANT_ID] = 276.15  # Molecular weight of F4TCNQ
    else:
        print(f"No known dopant type found in directory: {root}")
        mol_weight[DOPANT_ID] = 280.14  # Molecular weight of TFSI-
    return mol_weight

def read_crystalline_mol_frac_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'Crystalline Mole Fraction:' in line:
                return float(re.findall("\d+\.\d+", line)[0])
    return None

def interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.019, 0.053, 0.049, 0.067, 0.096])
    f = interp1d(x, y, fill_value="extrapolate")
    return f(crystalline_mol_frac)

def interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.0242, 0.0242, 0.0242, 0.0242, 0.0242])
    f = interp1d(x, y, fill_value="extrapolate")
    return f(crystalline_mol_frac)

def generate_material_dict(root):
    base_path = '/home/php/DopantModeling/xspectra_refractive_indices/'
    material_dict = {
        'Material1': 'vacuum',
        'Material2': '/home/php/DopantModeling/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge.txt',
        'Material3': '/home/php/DopantModeling/xspectra_refractive_indices/interp_P3HT_database_kkcalc_merge_isotropic.txt'
    }

    if 'TFSI' in root:
        material_name = 'TFSI_2_'
    elif 'F4TCNQ' in root:
        material_name = 'Reduced_F4TCNQ_'
    else:
        print(f"No known dopant type found in directory: {root}")
        material_name = 'TFSI_2_'

    for element in ['_C_', '_N_', '_F_']:
        if element in root:
            material_name += element.strip('_') + '_'
            break

    material_name += 'Kedge'

    if 'Para' not in root and 'Perp' not in root:
        material_name += '_isotropic'

    material_dict['Material4'] = base_path + material_name + '.txt'

    return material_dict

def process_pickle_file(filename, post_processor):
    print(f"Starting processing of pickle file: {filename}")

    hdf5_filename = os.path.splitext(filename)[0] + '.hdf5'
    description_filename = os.path.splitext(filename)[0]

    # Check if the corresponding .hdf5 file exists or if force_hdf5 is True
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
    process = subprocess.run(['CyRSoXS', hdf5_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if 'HDF5-DIAG' in process.stderr:
        print(f"HDF5 error detected during subprocess execution for {filename}:")
        print(process.stderr)
        raise Exception("HDF5 Error")

    print(f"Finished processing pickle file: {filename}")
    
def find_parameters_file(root, base_filename):
    file_name = "Parameters_" + base_filename + ".txt"
    file_path = os.path.join(root, file_name)
    if not os.path.isfile(file_path):
        # Try parent directory
        parent_dir = os.path.dirname(root)
        file_path = os.path.join(parent_dir, file_name)
        if not os.path.isfile(file_path):
            return None
    return file_path

energy_dict = {'Energy': 6, 'DeltaPerp': 3, 'BetaPerp': 1, 'DeltaPara': 2, 'BetaPara': 0}

VACUUM_ID, CRYSTAL_ID, AMORPH_ID, DOPANT_ID = 0, 1, 2, 3

default_mol_weight = {
    CRYSTAL_ID: 166.2842, # Molecular weight of crystalline P3HT repeat
    AMORPH_ID: 166.2842,  # Molecular weight of amorphous P3HT repeat
    DOPANT_ID: 276.15     # Molecular weight of TFSI- = 280.14, Molecular weight of F4TCNQ = 276.15
}

density = {
    CRYSTAL_ID: 1.1, # Density of crystalline P3HT
    AMORPH_ID: 1.1,  # Density of amorphous P3HT
    DOPANT_ID: 1.1   # Density of dopant in P3HT
}

# File I/O
pickle_save_name = '/home/php/CyRSoXS_sims/PyHyperScattering_Batch_SST1_JupyterHub.pkl'

# Load Data
with open(pickle_save_name, 'rb') as file:
    data = pickle.load(file)
data_dict = {'scan_ids': data[0], 'sample_names': data[1], 'edges': data[2], 'ARs': data[3], 'paras': data[4], 'perps': data[5], 'circs': data[6], 'FY_NEXAFSs': data[7], 'Iq2s': data[8], 'ISIs': data[9]}

# Gather a list of all .pickle files
pickle_files = []
for root, dirs, files in os.walk('.'):
    for filename in files:
        if filename.endswith('.pickle'):
            full_path = os.path.abspath(os.path.join(root, filename))
            pickle_files.append(full_path)

# Shuffle the list of .pickle files
random.shuffle(pickle_files)

# Initialize a dictionary to keep track of files that encountered errors
error_files = {}

# Process the gathered .pickle files
i = 0
while i < len(pickle_files):
    full_path = pickle_files[i]
    root, filename = os.path.split(full_path)
    base_filename = os.path.splitext(filename)[0]

    print(f"\nProcessing file: {filename} in directory: {root}")

    # Skip if HDF5 file or directory exists
    hdf5_filename = os.path.splitext(filename)[0] + '.hdf5'
    if os.path.exists(hdf5_filename) or 'HDF5' in os.listdir(root):
        print(f"Skipped processing {filename} as 'HDF5' directory or .hdf5 file exists in {root}")
        i += 1
        continue

    try:
        params_file_path = find_parameters_file(root, base_filename)
        if params_file_path is None:
            raise FileNotFoundError(f"Parameters file not found for {filename}")

        crystalline_mol_frac = read_crystalline_mol_frac_from_file(params_file_path)
    
        # Path-dependent configurations
        edge_to_find = 'C 1s' if 'C_K_Edge' in root else 'N 1s' if 'N_K_Edge' in root else 'F 1s' if 'F_K_Edge' in root else 'C 1s'
        indices = [index for index, edge in enumerate(data_dict['edges']) if edge == edge_to_find]
        energies = [data_dict['ARs'][index].energy for index in indices]
        energies = energies[0].values[1:]
        dopant_vol_frac = 0.0 if 'Undoped' in root else interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac) if 'TFSI' in root else interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac) if 'F4TCNQ' in root else 0.0
        mol_weight = update_mol_weight_for_dopant(root, default_mol_weight.copy())
        
        material_dict = generate_material_dict(root)
        print(f"Generated material_dict: {material_dict}")
        
        dopant_orientation = 'perpendicular' if 'Perp' in root else 'parallel' if 'Para' in root else 'isotropic'
        print(f"Set dopant orientation to: {dopant_orientation}")
        
        crystal_dope_frac = 1 if 'Fibril' in root else 0 if 'Matrix' in root else 0.5
        print(f"Set crystal dope fraction to: {crystal_dope_frac}")
            
        # Initialize the PostProcessor with the new dopant_vol_frac
        post_processor = PostProcessor(
            num_materials=4, mol_weight=mol_weight, density=density,
            dope_case=1, dopant_method='preferential', dopant_orientation=dopant_orientation, 
            dopant_vol_frac=dopant_vol_frac, crystal_dope_frac=crystal_dope_frac,
            core_shell_morphology=True, gaussian_std=3, fibril_shell_cutoff=0.2, 
            surface_roughness=False, height_feature=3, max_valley_nm=46, 
            amorph_matrix_Vfrac=0.9, amorphous_orientation=True)
    
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
    
        print(f"Finished processing pickle file: {filename}")

    except Exception as e:  
        print(f"HDF5 FileIException encountered for file {filename}: {e}")
        if filename not in error_files:
            error_files[filename] = 1
            print(f"First error encounter for {filename}, retrying at the end of the list.")
            pickle_files.append(full_path)
        elif error_files[filename] == 1:
            error_files[filename] += 1
            print(f"Second error encounter for {filename}, regenerating HDF5 file.")
            
            # Delete existing HDF5 file if it exists
            hdf5_filename = os.path.splitext(filename)[0] + '.hdf5'
            if os.path.exists(hdf5_filename):
                print(f"Deleting existing HDF5 file: {hdf5_filename}")
                os.remove(hdf5_filename)
    
            base_filename = os.path.splitext(filename)[0]
            
            params_file_path = os.path.join(root, "Parameters_" + base_filename + ".txt")
            crystalline_mol_frac = read_crystalline_mol_frac_from_file(params_file_path)
        
            # Path-dependent configurations
            edge_to_find = 'C 1s' if 'C_K_Edge' in root else 'N 1s' if 'N_K_Edge' in root else 'F 1s' if 'F_K_Edge' in root else 'C 1s'
            indices = [index for index, edge in enumerate(data_dict['edges']) if edge == edge_to_find]
            energies = [data_dict['ARs'][index].energy for index in indices]
            energies = energies[0].values[1:]
            dopant_vol_frac = 0.0 if 'Undoped' in root else interpolate_dopant_vol_frac_TFSI(crystalline_mol_frac) if 'TFSI' in root else interpolate_dopant_vol_frac_F4TCNQ(crystalline_mol_frac) if 'F4TCNQ' in root else 0.0
            mol_weight = update_mol_weight_for_dopant(root, default_mol_weight.copy())
            
            material_dict = generate_material_dict(root)
            print(f"Generated material_dict: {material_dict}")
            
            dopant_orientation = 'perpendicular' if 'Perp' in root else 'parallel' if 'Para' in root else 'isotropic'
            print(f"Set dopant orientation to: {dopant_orientation}")
            
            crystal_dope_frac = 1 if 'Fibril' in root else 0 if 'Matrix' in root else 0.5
            print(f"Set crystal dope fraction to: {crystal_dope_frac}")
                
            # Initialize the PostProcessor with the new dopant_vol_frac
            post_processor = PostProcessor(
                num_materials=4, mol_weight=mol_weight, density=density,
                dope_case=1, dopant_method='preferential', dopant_orientation=dopant_orientation, 
                dopant_vol_frac=dopant_vol_frac, crystal_dope_frac=crystal_dope_frac,
                core_shell_morphology=True, gaussian_std=3, fibril_shell_cutoff=0.2, 
                surface_roughness=False, height_feature=3, max_valley_nm=46, 
                amorph_matrix_Vfrac=0.9, amorphous_orientation=True)
        
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
        
            print(f"Finished processing pickle file: {filename}")
        else:
            print(f"Multiple errors encountered for {filename}, skipping further processing.")
            error_files[filename] += 1

    finally:
        # Close any open Matplotlib figures
        plt.close('all')

    i += 1  # Move to the next file in the list