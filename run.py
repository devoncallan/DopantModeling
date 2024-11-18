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
from params import configure_energies, configure_dopant_parameters, configure_polymer_NEXAFS

# Ensure all paths are correctly set
sys.path.append('/home/php/DopantModeling-dev/')

from src.Morphology.Fibril.FibrilGenerator import generate_morphology
from src.Morphology.Fibril.FibrilPostProcessor import process_morphology, read_crystalline_mol_frac_from_file, DopantOrientation
from src.Common.files import delete_path
import src.Simulation.cyrsoxs as cyrsoxs 

# Additional setup if needed
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

##########################################
### DEFINE EXPERIMENT SWEEP PARAMETERS ###
##########################################

exp1 = {
    "C_1s": 'C_1s',
    "N_1s": 'N_1s',
    "F_1s": 'F_1s',
}

exp2 = {
    "10": 10,
    "30": 30,
    "40": 40,
    "50": 50,
    "100": 100,
    "90": 90,
    "80": 80,
    "70": 70,
    "60": 60,
    "20": 20,
    "5": 5
}


exp3 = {
    "F4TCNQ": 'F4TCNQ',
    "Reduced_F4TCNQ": 'Reduced_F4TCNQ',
    "TFSI": 'TFSI',
    "Undoped": 'Undoped'
}

exp4 = {
    "Uniform_Doping": None,
    "Crystallite_Doping": 1,
    "Amorphous_Doping": 0,
}

exp5 = {
    "Perpendicular": DopantOrientation.PERPENDICULAR,
    "Parallel": DopantOrientation.PARALLEL,
    "Isotropic": DopantOrientation.ISOTROPIC,
}

rerun_simulations = False

p.edge_to_find = 'C_1s'

for exp2_name, exp2_val in exp2.items():
    print(f"Starting simulations with c2c_dist_nm: {exp2_val}")
    dir_name2 = f'{exp2_name}_c2c'
    save_dir2 = cyrsoxs.make_output_dir(p.base_path, dir_name2)
    os.chdir(save_dir2)

    for exp3_name, exp3_val in exp3.items():
        print(f"  Setting dopant to: {exp3_val}")
        dir_name3 = f'{exp3_name}'
        save_dir3 = cyrsoxs.make_output_dir(save_dir2, dir_name3)
        os.chdir(save_dir3)
        
        for exp4_name, exp4_val in exp4.items():
            print(f"    Configuring fraction of dopant in crystallites: {exp4_name}")
            dir_name4 = f'{exp4_name}'
            save_dir4 = cyrsoxs.make_output_dir(save_dir3, dir_name4)
            os.chdir(save_dir4)
            
            for exp5_name, exp5_val in exp5.items():
                print(f"      Configuring dopant orientation: {exp5_name}")
                dir_name5 = f'Dopant_{exp5_name}'
                save_dir5 = cyrsoxs.make_output_dir(save_dir4, dir_name5)
                os.chdir(save_dir5)

                # Check if 'HDF5' directory exists
                if os.path.exists('HDF5'):
                    if not rerun_simulations:
                        print(f"      Skipping {save_dir4} as 'HDF5' directory already exists.")
                        continue
    
                # Adjust input parameters based on experiment values
                p.c2c_dist_nm = exp2_val
                p.dopant = exp3_val
                p.crystal_dopant_frac = exp4_val
                if exp4_val == None:
                    p.uniform_doping = True
                else:
                    p.uniform_doping = False
                p.dopant_orientation = exp5_val
                
                # Configure polymer NEXAFS
                configure_polymer_NEXAFS(p)
    
                # Generate morphology
                fibgen = generate_morphology(p)
                
                # Configure simulation energies
                configure_energies(p)
                
                # Process undoped morphology
                p.dopant = None
                p.material_dict['Material4'] = 'vacuum'
                data, post_processor = process_morphology(fibgen, p)
                post_processor.save_parameters(data, fibgen, p, 'undoped')
    
                # Read crystalline mole fraction from file
                crystalline_mol_frac = read_crystalline_mol_frac_from_file('./parameters_undoped.txt')
                   
                # Configure dopant settings
                p.dopant = exp3_val
                configure_dopant_parameters(p, crystalline_mol_frac)
                   
                # Process doped morphology with existing data
                data, post_processor = process_morphology(fibgen, p, existing_data=data, just_add_dopants=True)
                post_processor.save_parameters(data, fibgen, p, 'doped')
    
                # Prepare and run simulation
                cyrsoxs.create_hdf5(data, p)
                cyrsoxs.create_inputs(p)
                
                # Close all remaining figures
                plt.close('all')
                
                # Generate figures
                checkH5(p.DEFAULT_MORPH_FILE)
                
                # Retrieve figure numbers and save figures
                fig_nums = plt.get_fignums()
                fig_nums.sort()
                for fig_num in fig_nums[-4:]:
                    fig = plt.figure(fig_num)
                    filename = os.path.join(save_dir5, f"mat{fig_num}_checkH5.png")
                    fig.savefig(filename, bbox_inches='tight', pad_inches=0.2, dpi=150, transparent=True)                    
    
                # Execute CyRSoXS simulation
                cyrsoxs.run(p, save_dir='.')
                
                delete_path(p.DEFAULT_MORPH_FILE)
                
                # Change back to the previous directory
                os.chdir(save_dir4)
            
            # Change back to the previous directory
            os.chdir(save_dir3)

        # Change back to the previous directory
        os.chdir(save_dir2)

    # Return to the base path at the end of each loop iteration
    os.chdir(p.base_path)