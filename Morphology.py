import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
from pathlib import Path
import io
import matplotlib.animation as animation
import trimesh
import copy
import threading
import pandas as pd
import scipy
import warnings
from scipy.optimize import curve_fit
from FyeldGenerator import generate_field
from Fibril import Fibril
from tqdm import tqdm
from IPython.display import display, clear_output
from FieldGeneration import generate_field_with_PSD

class Morphology:
    """
    Dimensions:
    x_dim_nm - 
    y_dim_nm - 
    z_dim_nm - z dimension of morphology box in nm
    pitch_nm - 
    """
    def __init__(self, x_dim_nm: float, y_dim_nm: float, z_dim_nm: float, pitch_nm: float):
        """Initialize morphology object

        Args:
            x_dim_nm (float): x dimension of morphology box in nm
            y_dim_nm (float): y dimension of morphology box in nm
            z_dim_nm (float): z dimension of morphology box in nm
            pitch_nm (float): size of voxels in morphology in nm
        """
        # Dimensions of morphology in nm
        self.x_dim_nm = x_dim_nm
        self.y_dim_nm = y_dim_nm
        self.z_dim_nm = z_dim_nm
        
        # Dimensions of morphology in voxels
        self.pitch_nm = pitch_nm # Dimension of voxel in nm
        self.x_dim = int(round(x_dim_nm / pitch_nm))
        self.y_dim = int(round(y_dim_nm / pitch_nm))
        self.z_dim = int(round(z_dim_nm / pitch_nm))
        self.box_volume = self.x_dim * self.y_dim * self.z_dim
        self.dims  = np.array([self.x_dim, self.y_dim, self.z_dim])
        
        self.initialize_box()
        
        self.fibrils = []        

    def initialize_box(self):
        """Initialize voxelized box and bounding box
        """
        
        # Initialize morphology box as numpy array
        # Each xyz coordinate contains (material, x_orientation, y_orientation, z_orientation)
        self.box = np.zeros((self.x_dim, self.y_dim, self.z_dim, 4))

        # Set up bounding box mesh to detect out of bounds collisions
        self.bounding_box = trimesh.primitives.Box(extents=[self.x_dim, self.y_dim, self.z_dim])
        
        # Translate bounding box to center
        center = [self.x_dim/2, self.y_dim/2, self.z_dim/2]
        translation_matrix = trimesh.transformations.translation_matrix(center)
        self.bounding_box.apply_transform(translation_matrix)
        
        # Create bounding box path for visualization
        self.bounding_path = self.bounding_box.as_outline()
        self.bounding_path.colors = [[100,100,100]]

    # Check if any points are within the mesh  
    def check_mesh_contains_any(self, mesh, points):
        return np.any(mesh.contains(points))

    # Checks if all points are within the mesh
    def check_mesh_contains_all(self, mesh, points):
        return np.all(mesh.contains(points))
    
    # Check that the vertices of a given mesh are within the bounding box
    def check_mesh_within_bounding_box(self, mesh):
        return self.check_mesh_contains_all(self.bounding_box, mesh.vertices)

    def is_initial_mesh_valid(self, initial_mesh):
        
        def check_initial_mesh_intersection_with_fibrils(initial_mesh):
            points = initial_mesh.vertices

            # Sample surface points of fibril
            surface, face_index = trimesh.sample.sample_surface_even(initial_mesh, 150)
            points = np.concatenate((points, np.array(surface)))
            for fibril in self.fibrils:
                if self.check_mesh_contains_any(fibril.intersection_mesh, points):
                    return True
            return False
        
        return self.check_mesh_within_bounding_box(initial_mesh) and not check_initial_mesh_intersection_with_fibrils(initial_mesh)

    def is_mesh_valid(self, mesh):

        def check_mesh_intersection_with_fibrils(mesh):
            points = mesh.vertices
            for fibril in self.fibrils:
                if self.check_mesh_contains_any(fibril.intersection_mesh, points):
                    return True
            return False
        
        return self.check_mesh_within_bounding_box(mesh) and not check_mesh_intersection_with_fibrils(mesh)
    
    # Define Gaussian function to fit to data
    def gaussian(self, x, sigma, scale):
        return scale * (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - np.pi/2) / sigma)**2)

    def set_model_parameters(self, radius_nm_avg: float, radius_nm_std: float, max_num_fibrils: int,
                             fibril_length_range_nm: list, rand_orientation: int=0,
                             theta_distribution_csv=None, k = 1, std = 1, theta_sigma = 30):
        self.radius_nm_avg = radius_nm_avg
        self.radius_nm_std = radius_nm_std
        self.radius_avg = radius_nm_avg / self.pitch_nm
        self.radius_std = radius_nm_std / self.pitch_nm
        self.max_num_fibrils = max_num_fibrils
        self.min_fibril_length_nm = min(fibril_length_range_nm)
        self.max_fibril_length_nm = max(fibril_length_range_nm)
        self.min_fibril_length = self.min_fibril_length_nm / self.pitch_nm
        self.max_fibril_length = self.max_fibril_length_nm / self.pitch_nm
        self.rand_orientation = rand_orientation
        self.k = k
        self.std = std
        self.theta_sigma = theta_sigma
        self.theta_distribution_csv = theta_distribution_csv
            
    def get_random_point(self):
        return self.dims * np.random.rand(3)
    
    def get_random_direction(self, point):
        # Ensure the point is a valid index
        if len(point) != 3:
            raise ValueError("Expected point to be a 3-element array for indexing into 3D reference arrays.")

        if self.rand_orientation==0:
            theta = np.random.normal(90,30)/180 * np.pi
            psi = np.random.uniform(0,np.pi)
        elif self.rand_orientation==1:
            theta = (90 + np.random.normal(0, 1.0))/180 * np.pi
            psi   = np.random.uniform(0, np.pi)
        elif self.rand_orientation==2:
            # use random theta close to/in plane of film
            theta = np.random.normal(90,30)/180 * np.pi
            # use arrays to 'lookup' the pregenerated angles at the point
            psi = self.psi_ref[tuple(point.astype(int))]
        elif self.rand_orientation==3:
            # use arrays to 'lookup' the pregenerated angles at the point
            theta = self.theta_ref[tuple(point.astype(int))]
           # print(theta)
            psi = self.psi_ref[tuple(point.astype(int))]
           # print(psi)
           
        direction = np.asarray([np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta)])
        return direction / np.linalg.norm(direction)
            
    def generate_psi_field(self, normalization_type='cdf', new_mean=None, new_std=None):
        self.psi_ref = generate_field_with_PSD(self.k, self.std, self.dims, 2 * np.pi, normalization_type=normalization_type, new_mean=new_mean, new_std=new_std)
            
    def generate_theta_field(self, normalization_type='psd', new_mean=np.pi/2, new_std=np.pi/5):
        self.theta_ref = generate_field_with_PSD(self.k, self.std, self.dims, np.pi, normalization_type=normalization_type, new_mean=new_mean, new_std=new_std)

    def plot_field(self, field_type='psi'):
        if field_type == 'psi':
            field_ref = self.psi_ref
            field_name = 'Psi'
        elif field_type == 'theta':
            field_ref = self.theta_ref
            field_name = 'Theta'
        else:
            warnings.warn(f"Invalid field_type '{field_type}'. Use either 'theta' or 'psi'.")
            return

        fig, ax = plt.subplots(1, 3, figsize=(3.5*3, 3),dpi=100)

        # If field_ref is 3D, take a 2D slice for plotting
        field_data_to_plot = field_ref[:, :, 0] if len(field_ref.shape) == 3 else field_ref

        # Plot the field
        extent_vals = [0, self.x_dim_nm, 0, self.y_dim_nm]  # x and y dimensions in nm
        im = ax[0].imshow(field_data_to_plot, cmap='viridis', origin='lower', extent=extent_vals)
        ax[0].set_xlabel('Distance (nm)')
        ax[0].set_ylabel('Distance (nm)')
        fig.colorbar(im, ax=ax[0], label=f'{field_name} Value')
        ax[0].set_title(f'Generated {field_name} Field')

        # Plot histogram of field values
        ax[1].hist(field_data_to_plot.flatten(), bins=50, color='blue', edgecolor='black')
        ax[1].set_title(f'Histogram of {field_name} Values')
        ax[1].set_xlabel(f'{field_name} Value')
        ax[1].set_ylabel('Frequency')

        # Compute 2D Power Spectral Density
        f_transform = np.fft.fft2(field_data_to_plot - field_data_to_plot.mean())
        psd2D = np.abs(f_transform)**2
        psd2D_shifted = np.fft.fftshift(psd2D)

        # Radial averaging for PSDF
        y, x = np.indices(psd2D_shifted.shape)
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(int)
        radial_profile = np.bincount(r.ravel(), psd2D_shifted.ravel()) / np.bincount(r.ravel())
        freq_res = 1 / (self.x_dim_nm * 10)  # Convert nm to Å for frequency calculation
        radius_values = np.arange(radial_profile.size) * freq_res

        ax[2].plot(radius_values, radial_profile)
        ax[2].set_yscale('log')
        ax[2].set_xlabel('Spatial Frequency (Å$^{-1}$)')
        ax[2].set_ylabel('Power Spectrum')
        ax[2].set_title('Radial PSDF')

        plt.tight_layout()
        return fig, ax
    
    def new_fibril(self):
        """ Initializes a new fibril object.

        Returns:
            Fibril: new fibril object 
        """

        # Set starting center of new fibril
        center = self.get_random_point()

        # Set long axis fibril direction
        direction = self.get_random_direction(center)

        # Set radius from specified distribution
        radius = np.random.normal(self.radius_avg, self.radius_std)

        # Initialize length to be minimum fibril length
        length = self.min_fibril_length

        return Fibril(center, direction, radius, length)

    def grow_fibril(self, fibril: Fibril, grow_direction: int):
        """ Grow the fibril in the grow_direction until intersection (with another fibril or the bounding box)
        or the maximum fibril length is reached.

        Args:
            fibril (Fibril): Fibril object that undergoes growth
            grow_direction (int): +1, grow left. -1, grow right.

        Returns:
            Fibril: Fibril object after growth
        """
        
        # Check that fibril is not intersecting and is the length is less than max length
        while (self.is_mesh_valid(fibril.intersection_mesh) and fibril.length <= self.max_fibril_length):
            fibril.length += self.pitch_nm
            fibril.center += self.pitch_nm * fibril.direction * grow_direction / 2
            fibril.make_intersection_mesh()
        
        # Returns false if fibril could not grow beyond initial (minimum) fibril length
        if (fibril.length != self.min_fibril_length):
            # Fibril growth terminated due to exceeding maximum fibril length
            if (fibril.length > self.max_fibril_length):
                diff = fibril.length - self.max_fibril_length
                fibril.center -= self.pitch_nm * diff * fibril.direction * grow_direction / 2
                fibril.length = self.max_fibril_length
                fibril.make_intersection_mesh()
            # Fibril growth terminated due to intersection with another fibril or bounding box
            else:
                fibril.length -= self.pitch_nm
                fibril.center -= self.pitch_nm * fibril.direction * grow_direction / 2
                fibril.make_intersection_mesh()

        return fibril
            
    def fill_model(self, timeout=10, plot_histogram=False, visualization_interval=10, gif_filename=None):
        """ Fill morphology with fibrils and optionally create a gif.
        """
        if gif_filename is not None:
            # Create a temporary directory to store frames
            with tempfile.TemporaryDirectory() as frames_dir:
                print(f"Temporary directory created at {frames_dir}")
    
                if self.rand_orientation >= 2:
                    print('Generating psi field...')
                    self.generate_psi_field()
                
                if (self.theta_distribution_csv is not None) and (self.rand_orientation >= 3):
                    # Load data from CSV
                    df = pd.read_csv(self.theta_distribution_csv)
                    
                    # Filter out rows with NaN and limit distribution to fit from 2pi/3 to pi/2
                    df = df.dropna()
                    df = df[(df['theta'] >= 60) & (df['theta'] <= 85)]
                    
                    # Get theta values and percentages
                    chi_values = df['theta'].values
                    percentages = df['percentage'].values
                    
                    # Fit the Gaussian function to the histogram data
                    params_new, _ = curve_fit(self.gaussian, chi_values / 180 * np.pi, percentages, p0=[np.pi/5, max(percentages)])
                    
                    # Extract fitted parameters
                    theta_sigma_fit, theta_scale_fit = params_new
                    
                    # Generate fitted curve
                    fit_x = np.linspace(0, np.pi, 101)
                    fit_y = self.gaussian(fit_x, theta_sigma_fit, theta_scale_fit)
                    
                     # Create figure and axis objects
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Plot the original and fitted data on the axes
                    ax.plot(chi_values / 180 * np.pi, percentages, '-', color='black', linewidth=2, label='Original Data')
                    ax.plot(fit_x, fit_y, '--', color='orange', linewidth=2, label=f'Fitted Gaussian ($\sigma$ = {theta_sigma_fit:.2f})')
                    
                    # Add labels and title
                    ax.set_xlabel('Theta (rad)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Filtered Theta Distribution and Fitted Gaussian')
                    
                    # Show legend
                    ax.legend()
                    
                    plt.show()
                    
                    self.theta_sigma_fit = theta_sigma_fit
                    print('Generating theta field from fit sigma...')
                    self.generate_theta_field(normalization_type='psd', new_mean=np.pi/2, new_std=np.sqrt(self.theta_sigma_fit))
                    
                    # Return figure and axis
                    # return fig, ax
                elif self.rand_orientation >= 3:
                    print('Generating theta field...')
                    self.generate_theta_field(normalization_type='psd')
                else:
                    print('random theta sampling...')
        
                def create_fibril():
                    # Initialize a new fibril 
                    fibril = self.new_fibril()
        
                    # Check if fibril is valid. If not, keep generating new fibril until it is valid.
                    # Fibril is invalid if it intersects with the bounding box or any existing fibrils
                    while not self.is_initial_mesh_valid(fibril.intersection_mesh):
                        fibril = self.new_fibril()
        
                    return fibril
                
                # Wrap the range object with tqdm for a loading bar
                for i in tqdm(range(self.max_num_fibrils), desc='Filling model'):
                    fibril_creation_thread = threading.Thread(target=create_fibril)
                    fibril_creation_thread.start()
                    fibril_creation_thread.join(timeout)
        
                    if fibril_creation_thread.is_alive():
                        fibril_creation_thread.join() # Ensure the thread finishes
                        print(f"Timed out while creating fibril {i}. Ending process.")
                        break
        
                    fibril = create_fibril()
                    # Save a copy of the fibril in its initial state
                    init_fibril = copy.deepcopy(fibril)
        
                    # Grow fibrils in each direction until the fibril intersects with any other fibrils or
                    # the bounding box or until the fibril reaches its maximum length
                    fibril = self.grow_fibril(fibril, +1) # Grow fibril to the left
                    fibril = self.grow_fibril(fibril, -1) # Grow fibril to the right
        
                    fibril.make_fibril_mesh()
        
                    #print(f'-- Fibril {i} --')
        
                    self.fibrils.append(fibril)
    
                    # Update visualization and save frame conditionally
                    if i % visualization_interval == 0:
                        clear_output(wait=True)
                        scene = self.get_scene(show_bounding_box=True)
                        display(scene.show())
                        
                        # Calculate volume fraction
                        volume_fraction = self.calculate_volume_fraction()
        
                        # Save frame with fibril count and volume fraction
                        frame_filename = os.path.join(frames_dir, f'frame_{i:04d}.png')
                        self.get_frame(frame_filename, fibril_count=i + 1, volume_fraction=volume_fraction)
            
                    if plot_histogram:
                        self.plot_fibril_histogram()
        
                # Create GIF after the model is filled
                self.make_gif(frames_dir, '.', gif_filename)

        else:
            if self.rand_orientation >= 2:
                print('Generating psi field...')
                self.generate_psi_field()
            
            if (self.theta_distribution_csv is not None) and (self.rand_orientation >= 3):
                # Load data from CSV
                df = pd.read_csv(self.theta_distribution_csv)
                
                # Filter out rows with NaN and limit distribution to fit from 2pi/3 to pi/2
                df = df.dropna()
                df = df[(df['theta'] >= 60) & (df['theta'] <= 85)]
                
                # Get theta values and percentages
                chi_values = df['theta'].values
                percentages = df['percentage'].values
                
                # Fit the Gaussian function to the histogram data
                params_new, _ = curve_fit(self.gaussian, chi_values / 180 * np.pi, percentages, p0=[np.pi/5, max(percentages)])
                
                # Extract fitted parameters
                theta_sigma_fit, theta_scale_fit = params_new
                
                # Generate fitted curve
                fit_x = np.linspace(0, np.pi, 101)
                fit_y = self.gaussian(fit_x, theta_sigma_fit, theta_scale_fit)
                
                 # Create figure and axis objects
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plot the original and fitted data on the axes
                ax.plot(chi_values / 180 * np.pi, percentages, '-', color='black', linewidth=2, label='Original Data')
                ax.plot(fit_x, fit_y, '--', color='orange', linewidth=2, label=f'Fitted Gaussian ($\sigma$ = {theta_sigma_fit:.2f})')
                
                # Add labels and title
                ax.set_xlabel('Theta (rad)')
                ax.set_ylabel('Frequency')
                ax.set_title('Filtered Theta Distribution and Fitted Gaussian')
                
                # Show legend
                ax.legend()
                
                plt.show()
                
                self.theta_sigma_fit = theta_sigma_fit
                print('Generating theta field from fit sigma...')
                self.generate_theta_field(normalization_type='psd', new_mean=np.pi/2, new_std=np.sqrt(self.theta_sigma_fit))
                
                # Return figure and axis
                # return fig, ax
            elif self.rand_orientation >= 3:
                print('Generating theta field...')
                self.generate_theta_field(normalization_type='psd')
            else:
                print('random theta sampling...')
    
            def create_fibril():
                # Initialize a new fibril 
                fibril = self.new_fibril()
    
                # Check if fibril is valid. If not, keep generating new fibril until it is valid.
                # Fibril is invalid if it intersects with the bounding box or any existing fibrils
                while not self.is_initial_mesh_valid(fibril.intersection_mesh):
                    fibril = self.new_fibril()
    
                return fibril
    
            # Wrap the range object with tqdm for a loading bar
            for i in tqdm(range(self.max_num_fibrils), desc='Filling model'):
                fibril_creation_thread = threading.Thread(target=create_fibril)
                fibril_creation_thread.start()
                fibril_creation_thread.join(timeout)
    
                if fibril_creation_thread.is_alive():
                    fibril_creation_thread.join() # Ensure the thread finishes
                    print(f"Timed out while creating fibril {i}. Ending process.")
                    break
    
                fibril = create_fibril()
                # Save a copy of the fibril in its initial state
                init_fibril = copy.deepcopy(fibril)
    
                # Grow fibrils in each direction until the fibril intersects with any other fibrils or
                # the bounding box or until the fibril reaches its maximum length
                fibril = self.grow_fibril(fibril, +1) # Grow fibril to the left
                fibril = self.grow_fibril(fibril, -1) # Grow fibril to the right
    
                fibril.make_fibril_mesh()
    
                #print(f'-- Fibril {i} --')
    
                self.fibrils.append(fibril)
                
                # Update visualization every 'visualization_interval' fibrils
                if i % visualization_interval == 0:
                    clear_output(wait=True)
                    scene = self.get_scene(show_bounding_box=True)
                    display(scene.show())
        
                if plot_histogram:
                    self.plot_fibril_histogram()

    def get_frame(self, filename, fibril_count, volume_fraction):
        """
        Render the current morphology scene as an image, add text with a white background box below the image, and save it to a file.
        """
        scene = self.get_scene(show_voxelized=False, show_bounding_box=True, custom_color=True)
        data = scene.save_image(resolution=(720, 720))
        image = Image.open(io.BytesIO(data))
    
        # Define a larger font size
        font_size = 50
    
        # Try to use a specific font (e.g., Arial), otherwise fall back to default font
        try:
            font_path = r"C:\Users\Phong\Box\Research\Mixed Conduction Project\RSOXS Projects\Fonts\Avenir Regular.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print("Default font loaded as specified font was not found.")
            font = ImageFont.load_default()
    
        # Draw text on the image
        draw = ImageDraw.Draw(image)
        text_fibrils = f"Number of Fibrils: {fibril_count}"
        volume_percent = volume_fraction * 100
        text_volume_percent = f"Fibril Volume Percent: {volume_percent:.2f}%"
    
        # Position for the text (centered below the image)
        textwidth_fibrils, textheight_fibrils = draw.textsize(text_fibrils, font)
        textwidth_volfrac, textheight_volfrac = draw.textsize(text_volume_percent, font)
        x_center = image.width / 2
        y_position = image.height + 10  # Adding a small margin below the image
    
        # Create a new image with extra space for text
        new_image_height = image.height + textheight_fibrils + textheight_volfrac + 30
        new_image = Image.new("RGB", (image.width, new_image_height), "white")
        new_image.paste(image, (0, 0))
    
        draw = ImageDraw.Draw(new_image)
    
        # Draw the text centered
        draw.text((x_center - textwidth_fibrils / 2, y_position), text_fibrils, font=font, fill=(0, 0, 0))
        draw.text((x_center - textwidth_volfrac / 2, y_position + textheight_fibrils + 10), text_volume_percent, font=font, fill=(0, 0, 0))
    
        new_image.save(str(filename))


    def calculate_volume_fraction(self):
        # Calculate the total volume occupied by the fibrils
        total_fibril_volume = sum(fibril.volume for fibril in self.fibrils if fibril.volume is not None)

        # Calculate the volume of the simulation space (ensure units are consistent)
        simulation_space_volume = self.x_dim_nm * self.y_dim_nm * self.z_dim_nm

        # Calculate the volume fraction
        volume_fraction = total_fibril_volume / simulation_space_volume if simulation_space_volume > 0 else 0

        return volume_fraction
    
    def make_gif(self, input_dir, output_dir, gif_name):
        """
        Create a gif from a series of images in a directory and make the final frame appear longer.
    
        Args:
            input_dir (str): Directory containing the input images.
            output_dir (str): Directory to save the output gif.
            gif_name (str): Name of the gif file to create.
        """
        output_path = Path(output_dir) / gif_name
        images = sorted(Path(input_dir).glob("frame_*.png"))
    
        if not images:
            print("No images found for creating GIF.")
            return
    
        # Create frames from images
        frames = [Image.open(image) for image in images]
    
        # Extend the duration of the final frame by adding it multiple times
        final_frame_extension = 10  # Number of times to repeat the final frame
        frames.extend([frames[-1]] * final_frame_extension)
    
        # Save as GIF
        frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=250, loop=1)
    
        print(f"GIF saved to {output_path}")
        """
        Create a gif from a series of images in a directory.

        Args:
            input_dir (str): Directory containing the input images.
            output_dir (str): Directory to save the output gif.
            gif_name (str): Name of the gif file to create.
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List all image files in the input directory
        images = [img for img in sorted(os.listdir(input_dir)) if img.endswith(".png")]
        frames = [Image.open(os.path.join(input_dir, img)) for img in images]

        # Make gif
        gif_path = os.path.join(output_dir, gif_name)
        frames[0].save(gif_path, format='GIF',
                        append_images=frames[1:],
                        save_all=True,
                        duration=500, loop=0)

        print(f"GIF saved to {gif_path}")
    
    def get_scene(self, show_voxelized=False, show_bounding_box=False, custom_color=True):
        scene_mesh_list = []
        for fibril in self.fibrils:
            mesh = fibril.voxel_mesh if show_voxelized else fibril.fibril_mesh
    
            # Set custom color if requested
            if custom_color:
                mesh.visual.face_colors = [139, 86, 176, 255]
        
            scene_mesh_list.append(mesh)
        
        if show_bounding_box:
            scene_mesh_list.append(self.bounding_path)
    
        # Create the scene
        scene = trimesh.Scene(scene_mesh_list)
    
        # Set transparent background color
        scene.bgcolor = [255, 255, 255, 0]  # RGBA where A (alpha) is 0 for transparency
    
        return scene
    
    def voxelize_model(self):
        for fibril in tqdm(self.fibrils, desc="Voxelizing Fibrils"):
            fibril.make_voxelized_fibril_mesh()

    def set_fibril_orientations(self):
        for i in range(len(self.fibrils)):
            self.fibrils[i].set_random_orientation()
    
    def plot_fibril_histogram(self):
        # Extracting lengths, radii, orientations theta, and orientations psi from fibrils
        lengths = [fibril.length * self.pitch_nm for fibril in self.fibrils]
        radii = [fibril.radius * self.pitch_nm for fibril in self.fibrils]
        orientation_thetas = [fibril.orientation_theta for fibril in self.fibrils]
        orientation_psis = [fibril.orientation_psi for fibril in self.fibrils]
    
        # Creating a new figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        
        # Creating a histogram for lengths
        axs[0, 0].hist(lengths, bins=20, edgecolor='black')
        axs[0, 0].set_title('Histogram of Fibril Lengths')
        axs[0, 0].set_xlabel('Length')
        axs[0, 0].set_ylabel('Frequency')
    
        # Creating a histogram for radii
        axs[0, 1].hist(radii, bins=20, edgecolor='black')
        axs[0, 1].set_title('Histogram of Fibril Radii')
        axs[0, 1].set_xlabel('Radius')
        axs[0, 1].set_ylabel('Frequency')
    
        # Creating a plot for orientation theta
        axs[1, 0].hist(orientation_thetas, bins=20, edgecolor='black')
        axs[1, 0].set_title('Histogram of Orientation Theta')
        axs[1, 0].set_xlabel('Theta (rad)')
        axs[1, 0].set_ylabel('Frequency')
    
        # Creating a plot for orientation psi
        axs[1, 1].hist(orientation_psis, bins=20, edgecolor='black')
        axs[1, 1].set_title('Histogram of Orientation Psi')
        axs[1, 1].set_xlabel('Psi (rad)')
        axs[1, 1].set_ylabel('Frequency')
    
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        return fig, axs