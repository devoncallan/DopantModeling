import os
import sys
import pathlib
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

####################################################################
### NOTE: CHANGE PATHS FOR NRSS, PYHYPER, AND CURRENT REPOSITORY ###
####################################################################

sys.path.append('/home/devon/Documents/Github/NRSS/')
sys.path.append('/home/devon/Documents/Github/PyHyperScattering/src/')
sys.path.append('/home/devon/Documents/Github/Fibril/')
sys.path.append('/home/devon/Documents/Github/Fibril/src/')

from Common.files import make_output_dir, move
import experiments.c2c_dist.params as p
from Morphology.FibrilGenerator import generate_morphology
from Morphology.FibrilPostProcessor import process_morphology
import Simulation.cyrsoxs as cyrsoxs 

##########################################
### DEFINE EXPERIMENT SWEEP PARAMETERS ###
##########################################

exp1 = {
    "45": 45,
    "60": 60,
    "75": 75,
    "90": 90
}

# SINGLE VARIABLE SWEEP
for exp1_name, exp1_val in exp1.items():

    dir_name = f'{exp1_name}_nm'
    save_dir = make_output_dir(p.base_path, dir_name)

    # Adjust input parameters based on experiment values
    p.c2c_dist_nm = exp1_val

    # Build morphology 
    fibgen = generate_morphology(p)
    data = process_morphology(fibgen, p)
    cyrsoxs.create_inputs(p, data)

    # Simulate scattering
    cyrsoxs.run(p.DEFAULT_MORPH_FILE, save_dir=save_dir)

exp2 = {
    "1/100": 1./100,
    "1/10": 1./10,
    "1": 1
}

# DOUBLE VARIABLE SWEEP (SINGLE LAYER)
for exp1_name, exp1_val in exp1.items():
    for exp2_name, exp2_val in exp2.items():

        dir_name = f'{exp1_name}_nm_k={exp2_name}'
        save_dir = make_output_dir(p.base_path, dir_name)

        # Adjust input parameters based on experiment values
        p.c2c_dist_nm = exp1_val
        p.k = exp2_val

        # Build morphology 
        fibgen = generate_morphology(p)
        data = process_morphology(fibgen, p)
        cyrsoxs.create_inputs(p, data)

        # Simulate scattering
        cyrsoxs.run(p.DEFAULT_MORPH_FILE, save_dir=save_dir)


# DOUBLE VARIABLE SWEEP (NESTED DIRECTORY)
for exp1_name, exp1_val in exp1.items():

    dir_name = f'{exp1_name}_nm'
    save_dir = make_output_dir(p.base_path, dir_name)

    for exp2_name, exp2_val in exp2.items():

        dir_name = f'k={exp2_name}'
        save_dir = make_output_dir(save_dir, dir_name)

        # Adjust input parameters based on experiment values
        p.c2c_dist_nm = exp1_val
        p.k = exp2_val

        # Build morphology 
        fibgen = generate_morphology(p)
        data = process_morphology(fibgen, p)
        cyrsoxs.create_inputs(p, data)

        # Simulate scattering
        cyrsoxs.run(p.DEFAULT_MORPH_FILE, save_dir=save_dir)
