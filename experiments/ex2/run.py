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
from params import configure_dopant_parameters

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

exp2 = {
    "256": 256,
}


exp3 = {
    "Undoped": 'Undoped'
}

exp4 = {
    "Uniform_Doping": None,
}

exp5 = {
    "Isotropic": DopantOrientation.ISOTROPIC,
}

rerun_simulations = False

p.edge_to_find = 'C 1s'

# NESTED SWEEP
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
    
                # Generate morphology
                fibgen = generate_morphology(p)
                
                # Find the appropriate energies to simulate at based on the specific elementel edge:
                if p.edge_to_find == 'C 1s':
                    p.energies = [270.  , 272.  , 274.  , 276.  , 278.  , 280.  , 282.  , 282.25, 282.5 ,
                                  282.75, 283.  , 283.25, 283.5 , 283.75, 284.  , 284.25, 284.5 , 284.75,
                                  285.  , 285.25, 285.5 , 285.75, 286.  , 286.5 , 287.  , 287.5 , 288.  ,
                                  288.5 , 289.  , 289.5 , 290.  , 290.5 , 291.  , 291.5 , 292.  , 293.  ,
                                  294.  , 295.  , 296.  , 297.  , 298.  , 299.  , 300.  , 301.  , 302.  ,
                                  303.  , 304.  , 305.  , 306.  , 310.  , 314.  , 318.  , 320.  , 330.  ,
                                  340.  ]
                elif p.edge_to_find == 'N 1s':
                    p.energies = [385. , 386. , 387. , 388. , 389. , 390. , 391. , 392. , 393. , 394. ,
                                  395. , 396. , 397. , 397.2, 397.4, 397.6, 397.8, 398. , 398.2, 398.4,
                                  398.6, 398.8, 399. , 399.2, 399.4, 399.6, 399.8, 400. , 400.2, 400.4,
                                  400.6, 400.8, 401. , 402. , 403. , 404. , 405. , 406. , 407. , 408. ,
                                  409. , 410. , 412. , 414. , 416. , 418. , 420. , 422. , 424. , 426. ,
                                  428. ]
                elif p.edge_to_find == 'F 1s':
                    p.energies = [670., 671., 672., 673., 674., 675., 676., 677., 678., 679., 680., 681.,
                                  682., 683., 684., 685., 686., 687., 688., 689., 690., 691., 692., 693.,
                                  694., 695., 696., 697., 698., 699., 700., 701., 702., 703., 704., 705.,
                                  706., 707., 708., 709.]
                    
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