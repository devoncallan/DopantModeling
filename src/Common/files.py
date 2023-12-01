import shutil
import os

def make_output_dir(base_save_dir:str, dir_name:str):
    save_dir = os.path.join(base_save_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def delete_path(path):
    """
    Delete a file or directory at the given path.

    Parameters:
    path (str): Path of the file or directory to be deleted.
    """
    # Check if the path exists
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        return

    try:
        # If path is a file, delete the file
        if os.path.isfile(path):
            os.remove(path)
            print(f"File {path} has been deleted.")

        # If path is a directory, delete the directory
        elif os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
                print(f"Empty directory {path} has been deleted.")
            else:
                shutil.rmtree(path)
                print(f"Directory {path} and all its contents have been deleted.")
    except Exception as e:
        print(f"Error occurred while deleting: {e}")


def move(src:str, dest_dir:str):
    """
    Move a directory from src to dest.

    Parameters:
    src (str): Source directory/file path.
    dest_dir (str): Destination directory path.
    """
    # Check if the source directory exists
    if not os.path.exists(src):
        print(f"The source directory {src} does not exist.")
        return

    # Move the directory
    try:
        print(f"Directory moved from {src} to {dest_dir}")
        return shutil.move(src, dest_dir)
    except Exception as e:
        print(f"Error occurred while moving the directory: {e}")

def copy(src:str, dest_dir:str):
    """
    Copy a file or directory from src to dest.

    Parameters:
    src (str): Source file or directory path.
    dest_dir (str): Destination directory path.
    """
    # Check if the source exists
    if not os.path.exists(src):
        print(f"The source path {src} does not exist.")
        return

    # Prepare the destination path
    dest = os.path.join(dest_dir, os.path.basename(src))

    try:
        # If source is a file, copy the file
        if os.path.isfile(src):
            shutil.copy(src, dest)
            print(f"File copied from {src} to {dest}")

        # If source is a directory, copy the directory
        elif os.path.isdir(src):
            shutil.copytree(src, dest)
            print(f"Directory copied from {src} to {dest}")
    except Exception as e:
        print(f"Error occurred while copying: {e}")

def find_parent_of_subdir(full_path, subdir_name):
    """
    Find the parent directory of a specified subdirectory in a given path.

    Parameters:
    full_path (str): The full path.
    subdir_name (str): The name of the subdirectory to search for.

    Returns:
    str: The path of the parent directory of the specified subdirectory, or None if not found.
    """
    # Split the path into components
    path_components = full_path.split(os.sep)

    # Iterate over the components and find the subdirectory
    for i in range(len(path_components)):
        if path_components[i] == subdir_name:
            # Return the path up to the parent directory of the subdirectory
            return os.sep.join(path_components[:i])

    print(f"Subdirectory '{subdir_name}' not found in '{full_path}'")
    return None


