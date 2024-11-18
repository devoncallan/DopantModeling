import os
import pandas as pd

def standardize_delimiter(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            try:
                data = pd.read_csv(file_path, delim_whitespace=True)

                new_file_path = os.path.join(folder_path, file)
                data.to_csv(new_file_path, sep='\t', index=False)
            except pd.errors.EmptyDataError:
                print(f"Skipped empty or improperly formatted file: {file}")

standardize_delimiter('.')