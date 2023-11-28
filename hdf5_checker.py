import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
sys.path.append('/home/php/NRSS/')
from NRSS.checkH5 import checkH5

def find_unprocessed_hdf5_directories(root_dir):
    unprocessed_directories = []

    for subdir, _, _ in os.walk(root_dir):
        os.chdir(subdir)
        hdf5_files = glob.glob('*.hdf5')

        # Check if any HDF5 files in this directory have been processed
        if not any(glob.glob(os.path.splitext(file)[0] + "_mat*_checkH5.png") for file in hdf5_files):
            unprocessed_directories.append(subdir)

    return unprocessed_directories

def process_hdf5_files_in_directory(directory):
    os.chdir(directory)
    hdf5_files = glob.glob('*.hdf5')

    for file in hdf5_files:
        # Processing logic (similar to the original code)
        output_pattern = os.path.splitext(file)[0] + "_mat*_checkH5.png"
        if glob.glob(output_pattern):
            tqdm.write(f"Skipping already processed file: {os.path.join(directory, file)}")
            continue

        tqdm.write(f"Processing file: {os.path.join(directory, file)}")

        try:
            checkH5(file, z_slice=128, plotstyle='light')
        except Exception as e:
            tqdm.write(f"Error processing {file}: {e}")
            continue

        fig_nums = plt.get_fignums()
        for fig_num in fig_nums:
            fig = plt.figure(fig_num)
            filename = f"{os.path.splitext(file)[0]}_mat{fig_num}_checkH5.png"
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.2, dpi=150, transparent=False)
            plt.close(fig)

# Get the directory where the script is running
current_directory = os.getcwd()
unprocessed_dirs = find_unprocessed_hdf5_directories(current_directory)

# Use tqdm for the loop of unprocessed directories
for directory in tqdm(unprocessed_dirs, desc="Processing Directories"):
    process_hdf5_files_in_directory(directory)