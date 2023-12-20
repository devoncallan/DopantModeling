import warnings
from enum import Enum
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.optimize import curve_fit
from FyeldGenerator import generate_field
from tqdm import tqdm

from src.Morphology.Fibril.Fibril import Fibril
from src.Morphology.util.FieldGeneration import generate_field_with_PSD, NormalizationType
from src.Morphology.util.PoissonDiskSampling import generate_points
from src.Morphology.MorphologyData import MorphologyData


############################################
### Named parameters for FibrilGenerator ###
############################################

class Materials:
    VACUUM_ID  = 0
    CRYSTAL_ID = 1
    AMORPH_ID  = 2
    DOPANT_ID  = 3

class FibrilOrientation(Enum):
    RANDOM_FLAT = 0
    RANDOM_VERY_FLAT = 1
    GRF_SAMPLE_FLAT = 2
    GRF_SAMPLE_ALL = 3

class FieldType(Enum):
    PSI = 0
    THETA = 1
    
class FibrilDistribution(Enum):
    RSA = 0 # Random Sequential Addition (RSA)
    PDS = 1 # Poisson disc sampling (PDS)

#####################################################
### Groups of parameters for FibrilGenerator ###
#####################################################

@dataclass
class FibrilSizeParams:
    radius_nm_avg: float
    radius_nm_std: float
    min_fibril_length_nm: float
    max_fibril_length_nm: float
    
@dataclass
class FibrilGrowthParams:
    max_num_fibrils: int = 100
    fibril_distribution: FibrilDistribution = FibrilDistribution.RSA
    c2c_dist_nm: float = None
    symmetrical_growth: bool = False
    periodic_bc: bool = False
    
@dataclass
class FibrilOrientationParams:
    fibril_orientation: FibrilOrientation
    theta_distribution_csv: str = None
    k: float = 1
    std: float = 1.
    theta_sigma: float = 30.

#########################################################
### FibrilGenerator: Create initial fibril morphology ###
#########################################################

class FibrilGenerator:
    """Generates a MorphologyData object of distributed semi-crystalline fibrils
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
        self.pitch_nm = pitch_nm # Dimension of voxel in nm
        
        # Dimensions of morphology in voxels
        self.x_dim = int(round(x_dim_nm / pitch_nm))
        self.y_dim = int(round(y_dim_nm / pitch_nm))
        self.z_dim = int(round(z_dim_nm / pitch_nm))
        self.dims  = np.array([self.x_dim, self.y_dim, self.z_dim])

        self.box_volume = self.x_dim * self.y_dim * self.z_dim
        self.box = np.zeros((self.x_dim, self.y_dim, self.z_dim), dtype=np.bool_)
        
        self.fibrils = []   
        
        self.parameters_set = False
        self.model_filled = False
        
    ############################
    ### Set model parameters ###
    ############################
    
    def set_model_parameters(self,
        fibril_size_params: FibrilSizeParams,
        fibril_growth_params: FibrilGrowthParams,
        fibril_orientation_params: FibrilOrientationParams):
        
        # Set the parameters
        self.fibril_size_params = fibril_size_params
        self.fibril_growth_params = fibril_growth_params
        self.fibril_orientation_params= fibril_orientation_params
        
        self.parameters_set = True
      
    def generate_psi_field(self, normalization_type=NormalizationType.CDF, new_mean=None, new_std=None):
        # self.psi_ref = generate_field_with_PSD(self.k, self.std, self.dims, 2 * np.pi, normalization_type=normalization_type, new_mean=new_mean, new_std=new_std)
        self.psi_ref = generate_field_with_PSD(
            k=self.fibril_orientation_params.k,
            std=self.fibril_orientation_params.std,
            dims=self.dims,
            max_value=2*np.pi,
            normalization_type=normalization_type,
            new_mean=new_mean,
            new_std=new_std
        )
            
    def generate_theta_field(self, normalization_type=NormalizationType.PSD, new_mean=np.pi/2, new_std=np.pi/5):
        # self.theta_ref = generate_field_with_PSD(self.k, self.std, self.dims, np.pi, normalization_type=normalization_type, new_mean=new_mean, new_std=new_std) 
        self.theta_ref = generate_field_with_PSD(
            k=self.fibril_orientation_params.k,
            std=self.fibril_orientation_params.std,
            dims=self.dims,
            max_value=2*np.pi,
            normalization_type=normalization_type,
            new_mean=new_mean,
            new_std=new_std
        )
    
    #############################################################
    ### Helper functions for creating the fibril morphologies ###
    #############################################################
    
    def get_random_point(self) -> np.ndarray:
        """Returns a random point within the morphology volume
        """
        return self.dims * np.random.rand(3)
    
    def get_random_direction(self, point: np.ndarray) -> np.ndarray:
        
        fibril_orientation = self.fibril_orientation_params.fibril_orientation
        
        if fibril_orientation == FibrilOrientation.RANDOM_FLAT:
            theta = np.deg2rad(np.random.normal(90, 30))  # Centered orthogonal to z-axis
            psi   = np.deg2rad(np.random.uniform(0, 360)) # Any direction in xy-plane
            
        elif fibril_orientation == FibrilOrientation.RANDOM_VERY_FLAT:
            theta = np.deg2rad(np.random.normal(90, 1.0))  # Centered very orthogonal to z-axis
            psi   = np.deg2rad(np.random.uniform(0, 360))  # Any direction in xy-plane
            
        elif fibril_orientation == FibrilOrientation.GRF_SAMPLE_FLAT:
            theta = np.deg2rad(np.random.normal(90, 30))   # Centered orthogonal to z-axis
            psi   = self.psi_ref[tuple(point.astype(int))] # Lookup GRF sampled angle
            
        elif fibril_orientation == FibrilOrientation.GRF_SAMPLE_ALL:
            theta = self.theta_ref[tuple(point.astype(int))] # Lookup GRF sampled angle
            psi   = self.psi_ref[tuple(point.astype(int))]   # Lookup GRF sampled angle
           
        # Convert Euler angles to unit vector
        direction = np.asarray([np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta)])
        return direction / np.linalg.norm(direction)
                
    def create_new_fibril(self, center:np.ndarray=None) -> Fibril:
        """Creates a fibril object defined with a center point, direction, and radius.

        Args:
            center (np.ndarray, optional): Center point to the fibril. Defaults to None -> center point is set to be random..

        Returns:
            Fibril: Returns a new fibril object
        """

        # Set starting center to a random point if not defined
        if center is None:
            center = self.get_random_point()

        # Set long axis fibril direction
        direction = self.get_random_direction(center)

        # Set radius from normal distribution
        fibril_radius_avg =  self.fibril_size_params.radius_nm_avg / self.pitch_nm
        fibril_radius_std = self.fibril_size_params.radius_nm_std / self.pitch_nm
        radius = np.random.normal(fibril_radius_avg, fibril_radius_std)

        return Fibril(center, direction, radius)
    
    def generate_fields(self):
        
        # Define Gaussian function to fit to data
        def gaussian(x, sigma, scale):
            return scale * (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - np.pi/2) / sigma)**2)

        fibril_orientation = self.fibril_orientation_params.fibril_orientation
        
        if fibril_orientation == FibrilOrientation.GRF_SAMPLE_ALL or fibril_orientation == FibrilOrientation.GRF_SAMPLE_FLAT:
            self.generate_psi_field()

        if fibril_orientation == FibrilOrientation.GRF_SAMPLE_ALL:

            theta_distribution_csv = self.fibril_orientation_params.theta_distribution_csv
            if theta_distribution_csv is not None:
                # Load data from CSV
                df = pd.read_csv(theta_distribution_csv)
                
                # Filter out rows with NaN and limit distribution to fit from 2pi/3 to pi/2
                df = df.dropna()
                df = df[(df['theta'] >= 60) & (df['theta'] <= 85)]
                
                # Get theta values and percentages
                chi_values = df['theta'].values
                percentages = df['percentage'].values
                
                # Fit the Gaussian function to the histogram data
                params_new, _ = curve_fit(gaussian, chi_values / 180 * np.pi, percentages, p0=[np.pi/5, max(percentages)])
                
                # Extract fitted parameters
                theta_sigma_fit, theta_scale_fit = params_new
                
                # Generate fitted curve
                fit_x = np.linspace(0, np.pi, 101)
                fit_y = gaussian(fit_x, theta_sigma_fit, theta_scale_fit)
                
                # Create figure and axis objects
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plot the original and fitted data on the axes
                ax.plot(chi_values / 180 * np.pi, percentages, '-', color='black', linewidth=2, label='Original Data')
                ax.plot(fit_x, fit_y, '--', color='orange', linewidth=2, label=f'Fitted Gaussian ($\sigma$ = {theta_sigma_fit:.2f})')
                
                # Add labels and title
                ax.set_xlabel('Theta (degrees)')
                ax.set_ylabel('Frequency')
                ax.set_title('Filtered Theta Distribution and Fitted Gaussian')
                
                # Show legend
                ax.legend()
                
                plt.show()
                
                self.theta_sigma_fit = theta_sigma_fit
                print('Generating theta field from fit sigma...')
                self.generate_theta_field(
                    normalization_type=NormalizationType.PSD, 
                    new_mean=np.pi/2, new_std=np.sqrt(self.theta_sigma_fit))
            else:
                print('Generating theta field...')
                self.generate_theta_field(normalization_type='psd')

    def fill_model_RSA(self):

        # Generate fields if needed
        self.generate_fields()

        total_attempts = 0
        successful_attempts = 0
        
        fibril_length_range = [
            self.fibril_size_params.min_fibril_length_nm / self.pitch_nm,
            self.fibril_size_params.max_fibril_length_nm / self.pitch_nm]

        while len(self.fibrils) <= self.fibril_growth_params.max_num_fibrils:
            fibril = self.create_new_fibril()
            

            growth_successful = fibril.grow_fibril_along_direction(
                self.box,
                length_range=fibril_length_range,
                periodic_bc=self.fibril_growth_params.periodic_bc, 
                symmetrical_growth=self.fibril_growth_params.symmetrical_growth
            )

            total_attempts += 1
            if growth_successful:
                successful_attempts += 1
                self.fibrils.append(fibril)
                self.box[fibril.fibril_indices] = True
        print(f'{successful_attempts} / {total_attempts}')
                
    def fill_model_PDS(self):

        # Generate fields if needed
        self.generate_fields()
        
        c2c_dist = self.fibril_growth_params.c2c_dist_nm / self.pitch_nm        
        fibril_length_range = [
            self.fibril_size_params.min_fibril_length_nm / self.pitch_nm,
            self.fibril_size_params.max_fibril_length_nm / self.pitch_nm]
        self.pds_centers = generate_points(self.dims, c2c_dist)
        self.init_centers = []

        total_attempts = 0
        successful_attempts = 0
        
        for center in self.pds_centers:
            fibril = self.create_new_fibril(center)
            
            growth_successful = fibril.grow_fibril_along_direction(
                self.box,
                length_range=fibril_length_range,
                periodic_bc=self.fibril_growth_params.periodic_bc, 
                symmetrical_growth=self.fibril_growth_params.symmetrical_growth
            )

            total_attempts += 1
            if growth_successful:
                successful_attempts += 1
                self.fibrils.append(fibril)
                self.init_centers.append(center)
                self.box[fibril.fibril_indices] = True
        print(f'{successful_attempts} / {total_attempts}')

    def fill_model(self):
        
        fibril_distribution = self.fibril_growth_params.fibril_distribution
        if fibril_distribution == FibrilDistribution.RSA:
            self.fill_model_RSA()
        elif fibril_distribution == FibrilDistribution.PDS:
            self.fill_model_PDS()
            
    #########################################
    ### Create the morphology data object ###
    #########################################
    
    def create_morphology_data(self) -> MorphologyData:
        
        data = MorphologyData(self.dims, 4, self.pitch_nm)
        
        # Initialize the matrices with fibrils
        for fibril in tqdm(self.fibrils, desc='Creating morphology data object.'):
            fibril_indices = fibril.fibril_indices
            
            # Convert XYZ to ZYX convention
            fibril_indices = np.transpose(np.flip(fibril_indices, axis=0))
            
            for index in fibril_indices:
                
                point_inside_box = index[0] < self.z_dim and index[1] < self.y_dim and index[2] < self.x_dim
                if not point_inside_box:
                    print(index)
                    continue
                
                # Set fibrils to pure crystalline region
                data.mat_Vfrac[Materials.CRYSTAL_ID][tuple(index)] = 1
                data.mat_S[Materials.CRYSTAL_ID][tuple(index)]     = 1
                data.mat_theta[Materials.CRYSTAL_ID][tuple(index)] = fibril.orientation_theta
                data.mat_psi[Materials.CRYSTAL_ID][tuple(index)]   = fibril.orientation_psi
        
        return data
            
    ##########################
    ### Plotting functions ###
    ##########################
    
    def plot_field(self, field_type=FieldType.PSI):
        if field_type == FieldType.PSI:
            field_ref = self.psi_ref
            field_name = 'Psi'
        elif field_type == FieldType.THETA:
            field_ref = self.theta_ref
            field_name = 'Theta'
        else:
            warnings.warn(f"Invalid field_type '{field_type}'. Use either 'theta' or 'psi'.")
            return

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Added one more axis for the PSDF

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
    
##################################
### Generate fibril morphology ###
##################################

def generate_morphology(p) -> FibrilGenerator:

    # Initialize fibril generator
    fibgen = FibrilGenerator(p.x_dim_nm, p.y_dim_nm, p.z_dim_nm, p.pitch_nm)

    # Define fibril generator parameters
    fibril_size_params = FibrilSizeParams(
        radius_nm_avg=p.radius_nm_avg, 
        radius_nm_std=p.radius_nm_std,
        min_fibril_length_nm=p.min_fibril_length_nm, 
        max_fibril_length_nm=p.max_fibril_length_nm
    )

    fibril_growth_params = FibrilGrowthParams(
        max_num_fibrils=p.max_num_fibrils, 
        fibril_distribution=p.fibril_distribution,
        c2c_dist_nm=p.c2c_dist_nm, 
        symmetrical_growth=p.symmetrical_growth, 
        periodic_bc=p.periodic_bc
    )

    fibril_orientation_params = FibrilOrientationParams(
        fibril_orientation=FibrilOrientation.GRF_SAMPLE_FLAT,
        k=p.k,
        std=p.std
        # k=1./(fibgen.x_dim*np.pi*f), 
        # std=1./(fibgen.x_dim*np.pi*f)
    )

    # Set fibril generator parameters
    fibgen.set_model_parameters(
        fibril_size_params=fibril_size_params, 
        fibril_growth_params=fibril_growth_params,
        fibril_orientation_params=fibril_orientation_params
    )

    # Fill model with fibrils
    fibgen.fill_model()

    return fibgen