import os
import pandas as pd

def process_files(folder_path, scaling_factor=1.2):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            data = pd.read_csv(file_path, delim_whitespace=True)
            
            columns_to_scale = data.columns.drop('energy')
            data[columns_to_scale] = data[columns_to_scale].applymap(lambda x: x * scaling_factor)
            
            data.to_csv(file_path, sep='\t', index=False)
            print(f"Processed {filename}")

folder_path = '.'
process_files(folder_path, scaling_factor=1.2)