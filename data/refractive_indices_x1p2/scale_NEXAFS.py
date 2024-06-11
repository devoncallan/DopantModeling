# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:00:12 2024

@author: Phong
"""

import os
import pandas as pd

def process_files(folder_path, scaling_factor=1.2):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the data from the file
            data = pd.read_csv(file_path, delim_whitespace=True)
            
            # Scale the values by the scaling factor, except for the 'energy' column
            columns_to_scale = data.columns.drop('energy')
            data[columns_to_scale] = data[columns_to_scale].applymap(lambda x: x * scaling_factor)
            
            # Save the modified data back to the file
            data.to_csv(file_path, sep='\t', index=False)
            print(f"Processed {filename}")

# Example usage
folder_path = '.'  # Replace 'path_to_your_folder' with the actual path
process_files(folder_path, scaling_factor=1.2)

"""
Note: Replace 'path_to_your_folder' with the actual path to your folder containing the text files.
Run this script in your Python environment. It will process all '.txt' files in the specified folder.
"""