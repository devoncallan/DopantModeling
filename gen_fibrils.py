import pickle
import os
import sys
import shutil
import subprocess

# Append Paths
sys.path.append('/home/php/NRSS/')
sys.path.append('/home/php/DopantModeling/')

# Append Paths
sys.path.append('/home/php/NRSS/')
sys.path.append('/home/php/DopantModeling/')

from Morphology import Morphology
from ReducedMorphology import ReducedMorphology

# Declare model box size in nm (x,y,z)
x_dim_nm  = 1024
y_dim_nm  = 1024
z_dim_nm  = 256
pitch_nm = 2 # Dimension of voxel in nm

# Loop through a range of fibril numbers
for num_fibrils in range(500, 2001, 250):
    # Initialize morphology
    morphology = Morphology(x_dim_nm, y_dim_nm, z_dim_nm, pitch_nm)
    morphology.set_model_parameters(radius_nm_avg = 15,
                                    radius_nm_std = 3,
                                    max_num_fibrils=num_fibrils,
                                    fibril_length_range_nm = [30, 200])
    
    print(f"Simulating with {num_fibrils} fibrils...")
    morphology.fill_model(timeout=60)
    morphology.voxelize_model()

    rm = ReducedMorphology(morphology)

    rm.pickle()
    print(f"Simulation and pickling completed for {num_fibrils} fibrils.")