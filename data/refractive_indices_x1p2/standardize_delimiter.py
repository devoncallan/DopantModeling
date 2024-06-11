# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:57:26 2023

@author: Phong
"""

import os
import pandas as pd

def standardize_delimiter(folder_path):
    """
    Standardize the delimiter of all text files in the given folder to tabs. Skips empty or improperly formatted files.

    :param folder_path: Path to the folder containing the text files.
    """
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            try:
                # Attempt to read the file with a generic delimiter
                data = pd.read_csv(file_path, delim_whitespace=True)

                # Save the DataFrame with tab as the delimiter if data is successfully read
                new_file_path = os.path.join(folder_path, file)
                data.to_csv(new_file_path, sep='\t', index=False)
            except pd.errors.EmptyDataError:
                print(f"Skipped empty or improperly formatted file: {file}")

standardize_delimiter('.')