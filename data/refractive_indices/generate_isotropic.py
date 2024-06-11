# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:53:44 2023

@author: Phong
"""

import os
import pandas as pd

def process_files(folder_path):
    """
    Process all text files in the given folder. For each file, it reads the data, replaces 'beta_para', 'beta_perp',
    'delta_para', and 'delta_perp' columns with 'beta' and 'delta' values respectively, and saves the modified
    file with '_isotropic' appended to the original file name.

    :param folder_path: Path to the folder containing the text files.
    """
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            # Read the file into a DataFrame
            data = pd.read_csv(file_path, delim_whitespace=True)

            # Replace the required columns
            data['beta_para'] = data['beta']
            data['beta_perp'] = data['beta']
            data['delta_para'] = data['delta']
            data['delta_perp'] = data['delta']

            # Save the modified DataFrame
            new_file_name = os.path.splitext(file)[0] + '_isotropic.txt'
            new_file_path = os.path.join(folder_path, new_file_name)
            data.to_csv(new_file_path, sep='\t', index=False, float_format='%.18f')

# Example usage
folder_path = '.'
process_files(folder_path)

"""
Note: Replace 'path_to_your_folder' with the actual path to your folder containing the text files.
Run this script in your Python environment. It will process all '.txt' files in the specified folder.
"""
