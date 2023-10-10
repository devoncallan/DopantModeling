import pickle
import re
import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import ticker
import PyHyperScattering
from matplotlib import font_manager

# Append Paths
sys.path.append('/home/php/NRSS/')
sys.path.append('/home/php/DopantModeling/')

from NRSS.writer import write_materials, write_hdf5, write_config
from PostProcessor import PostProcessor

def read_crystalline_mol_frac_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'Crystalline Mole Fraction:' in line:
                return float(re.findall("\d+\.\d+", line)[0])
    return None

def interpolate_dopant_vol_frac(crystalline_mol_frac):
    x = np.array([0.17, 0.22, 0.27, 0.29, 0.37])
    y = np.array([0.0242, 0.0242, 0.0242, 0.0242, 0.0242])
    f = interp1d(x, y, fill_value="extrapolate")
    return f(crystalline_mol_frac)

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

# Edge you want to find
edge_to_find = 'C 1s'

# Material Definitions
material_dict = {
    'Material1': 'vacuum', 
    'Material2': '/home/php/DopantModeling/xspectra_refractive_indices/P3HT_database_C_Kedge.txt', 
    'Material3': '/home/php/DopantModeling/xspectra_refractive_indices/P3HT_database_C_Kedge_isotropic.txt',
    'Material4': '/home/php/DopantModeling/xspectra_refractive_indices/Reduced_F4TCNQ_C_Kedge.txt'
}

energy_dict = {'Energy': 6, 'DeltaPerp': 3, 'BetaPerp': 1, 'DeltaPara': 2, 'BetaPara': 0}

VACUUM_ID, CRYSTAL_ID, AMORPH_ID, DOPANT_ID = 0, 1, 2, 3

mol_weight = {
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
pickle_save_name = '/home/php/CyRSoXS/PyHyperScattering_Batch_SST1_JupyterHub.pkl'

# Load Data
with open(pickle_save_name, 'rb') as file:
    data = pickle.load(file)
data_dict = {'scan_ids': data[0], 'sample_names': data[1], 'edges': data[2], 'ARs': data[3], 'paras': data[4], 'perps': data[5], 'circs': data[6], 'FY_NEXAFSs': data[7], 'Iq2s': data[8], 'ISIs': data[9]}
indices = [index for index, edge in enumerate(data_dict['edges']) if edge == edge_to_find]
energies = [data_dict['ARs'][index].energy for index in indices]
energies = energies[0].values[1:]

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
    
    params_file_path = os.path.join(root, "Parameters_" + base_filename + ".txt")
    crystalline_mol_frac = read_crystalline_mol_frac_from_file(params_file_path)

    dopant_vol_frac = interpolate_dopant_vol_frac(crystalline_mol_frac)
    
    # Initialize the PostProcessor with the new dopant_vol_frac
    post_processor = PostProcessor(
        num_materials=4, mol_weight=mol_weight, density=density,
        dope_case=1, dopant_method='preferential', dopant_orientation='perpendicular', 
        dopant_vol_frac=dopant_vol_frac, crystal_dope_frac = 0,
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
    
    file_loader = PyHyperScattering.load.cyrsoxsLoader()
    
    scan = file_loader.loadDirectory('.')
    
    integ = PyHyperScattering.integrate.WPIntegrator()
    integrated_data = integ.integrateImageStack(scan)
    integrated_data = integrated_data.sel(energy=slice(280, 292), q=slice(0, 0.1))
    
    # Add your start_q, end_q, start_en, end_en values
    start_q = 0  # Replace with your actual start value for q
    end_q = 0.1  # Replace with your actual end value for q
    start_en = 280  # Replace with your actual start value for energy
    end_en = 292  # Replace with your actual end value for energy
    
    # Continue with your existing plotting code
    para = integrated_data.rsoxs.slice_chi(0, chi_width=45).sel(q=slice(start_q, end_q))
    perp = integrated_data.rsoxs.slice_chi(90, chi_width=45).sel(q=slice(start_q, end_q))
    
    AR = (para - perp) / (para + perp)
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    
    im_cbar = AR.sel(energy = slice(start_en, end_en)).plot(
        x = 'q', y = 'energy',
        ax = ax,
        #vmin = -np.nanpercentile(AR,95),
        #vmax = np.nanpercentile(AR,95),
        cmap = 'RdBu',
        add_colorbar = False)
    
    cbar_ax = fig.add_axes([0.975, 0.145, 0.03, 0.715])
    cbar = fig.colorbar(im_cbar, cax = cbar_ax)
    
    # plt.setp(cbar.ax.get_yticklabels(), ha = "right")
    # cbar.ax.tick_params(pad = 37.5)
    cbar.ax.set_ylabel('Anisotropy')
    
    fig.suptitle(f'Simulation C 1s Edge', fontsize = 20, y = 1, x = 0.5)
    ax.set_xlabel(r'$\it{q}$ (A$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlim([start_q, end_q])
    ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09], ['', '0.03', '0.05', '0.07', '0.09'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.set_ylim([start_en, end_en])
    
    ax.xaxis.set_tick_params(which = 'major', size = 5, width = 2, direction = 'in', top = 'on')
    ax.xaxis.set_tick_params(which = 'minor', size = 2.5, width = 2, direction = 'in', top = 'on')
    ax.yaxis.set_tick_params(which = 'major', size = 5, width = 2, direction = 'in', right = 'on')
    ax.yaxis.set_tick_params(which = 'minor', size = 2.5, width = 2, direction = 'in', right = 'on')
    
    plt.savefig("AR_plot.tiff", bbox_inches = 'tight', pad_inches = 0.2, dpi = 150, transparent = False)