import os
import sys
import numpy as np
import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import NRSS
from NRSS.checkH5 import checkH5
import params as p
from params import interpolate_dopant_vol_frac_TFSI, interpolate_dopant_vol_frac_F4TCNQ

# Ensure all paths are correctly set
sys.path.append('/home/php/DopantModeling-dev/')

from src.Morphology.Fibril.FibrilGenerator import generate_morphology
from src.Morphology.Fibril.FibrilPostProcessor import process_morphology, read_crystalline_mol_frac_from_file
import src.Simulation.cyrsoxs as cyrsoxs 

# Additional setup if needed
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# Load experimental data for reference
pickle_save_name = '/home/php/CyRSoXS_sims/PyHyperScattering_Batch_SST1_JupyterHub.pkl'
with open(pickle_save_name, 'rb') as file:
    experimental_data = pickle.load(file)

data_dict = {
    'scan_ids': experimental_data[0],
    'sample_names': experimental_data[1],
    'edges': experimental_data[2],
    'ARs': experimental_data[3],
    'paras': experimental_data[4],
    'perps': experimental_data[5],
    'circs': experimental_data[6],
    'FY_NEXAFSs': experimental_data[7],
    'Iq2s': experimental_data[8],
    'ISIs': experimental_data[9]
}

##########################################
### DEFINE EXPERIMENT SWEEP PARAMETERS ###
##########################################

exp1 = {
    "100": 100,
    "90": 90,
    "80": 80,
    "70": 70,
    "60": 60,
    "50": 50,
    "40": 40,
    "30": 30,
    "20": 20,
    "10": 10,
}

# SINGLE VARIABLE SWEEP
for exp1_name, exp1_val in exp1.items():
    dir_name = f'{exp1_name}_nm'
    save_dir = cyrsoxs.make_output_dir(p.base_path, dir_name)
    
    # Change to the new directory
    os.chdir(save_dir)

    # Adjust input parameters based on experiment values
    p.c2c_dist_nm = exp1_val
    
    p.x_dim_nm = 256
    p.y_dim_nm = p.x_dim_nm
    p.z_dim_nm = 128

    # Generate morphology
    fibgen = generate_morphology(p)
    
    # Find the appropriate energies to simulate at based on experimental energies:
    edge_to_find = p.edge_to_find  # Assuming this is defined in your params file
    indices = [index for index, edge in enumerate(data_dict['edges']) if edge == edge_to_find]
    energies = [data_dict['ARs'][index].energy for index in indices]
    p.energies = energies[0].values[1:]
    
    # Process undoped morphology
    p.use_dopant_params = False
    data, post_processor = process_morphology(fibgen, p)
    post_processor.save_parameters(data, fibgen, p, 'undoped')
    
    # Read crystalline mole fraction from file
    crystalline_mol_frac = read_crystalline_mol_frac_from_file('./parameters_undoped.txt')
    
    # Find the appropriate energies to simulate at based on experimental energies
    p.configure_dopant_parameters(p, crystalline_mol_frac, p.edge_to_find)

    # Process doped morphology with existing data
    p.use_dopant_params = True
    data, _ = process_morphology(fibgen, p, existing_data=data, just_add_dopants=True)
    post_processor.save_parameters(data, fibgen, p, 'doped')
    
    # Prepare and run simulation
    cyrsoxs.create_hdf5(data, p)
    cyrsoxs.create_inputs(p)
    
    # Generate figures
    checkH5(p.DEFAULT_MORPH_FILE)
    
    # Retrieve figure numbers and save figures
    fig_nums = plt.get_fignums()
    fig_nums.sort()
    for fig_num in fig_nums[-4:]:
        fig = plt.figure(fig_num)
        filename = os.path.join(save_dir, f"mat{fig_num}_checkH5.png")
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.2, dpi=150, transparent=True)
        plt.close(fig)

    # Execute CyRSoXS simulation
    cyrsoxs.run(p, save_dir='.')
    
    # Change back to the base directory
    os.chdir(p.base_path)

# exp2 = {
#     "1/100": 1./100,
#     "1/10": 1./10,
#     "1": 1
# }

# # DOUBLE VARIABLE SWEEP (SINGLE LAYER)
# for exp1_name, exp1_val in exp1.items():
#     for exp2_name, exp2_val in exp2.items():

#         dir_name = f'{exp1_name}_nm_k={exp2_name}'
#         save_dir = make_output_dir(p.base_path, dir_name)

#         # Adjust input parameters based on experiment values
#         p.c2c_dist_nm = exp1_val
#         p.k = exp2_val

#         # Build morphology 
#         fibgen = generate_morphology(p)
#         data = process_morphology(fibgen, p)
#         cyrsoxs.create_inputs(p, data)

#         # Simulate scattering
#         cyrsoxs.run(p.DEFAULT_MORPH_FILE, save_dir=save_dir)


# # DOUBLE VARIABLE SWEEP (NESTED DIRECTORY)
# for exp1_name, exp1_val in exp1.items():

#     dir_name = f'{exp1_name}_nm'
#     save_dir = make_output_dir(p.base_path, dir_name)

#     for exp2_name, exp2_val in exp2.items():

#         dir_name = f'k={exp2_name}'
#         save_dir = make_output_dir(save_dir, dir_name)

#         # Adjust input parameters based on experiment values
#         p.c2c_dist_nm = exp1_val
#         p.k = exp2_val

#         # Build morphology 
#         fibgen = generate_morphology(p)
#         data = process_morphology(fibgen, p)
#         cyrsoxs.create_inputs(p, data)

#         # Simulate scattering
#         cyrsoxs.run(p.DEFAULT_MORPH_FILE, save_dir=save_dir)
