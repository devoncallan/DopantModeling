# Importing required libraries
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from FyeldGenerator import generate_field

class FieldGenerator:
    def __init__(self, dims, rand_orientation):
        self.dims = dims
        self.rand_orientation = rand_orientation
        self.psi_ref = None
        
    def generate_psi_field(self):
        # create PSD function for field generator
        def PSD_gauss(avg,std):
            def Pk(k):
                return (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((k-avg)/std)**2)
            return Pk
        
        # create Gaussian distribution function for stats
        def distrib(shape):
            a = np.random.normal(loc=0, scale=90, size=shape)
            return a
        
        # create array of angles at all points
        psi_shape = tuple(self.dims)
        psi_kav = 1/300.
        psi_kstd = 1/500.
        
        # Generate the field using the Gaussian PSD and random distribution
        self.psi_ref = generate_field(distrib, PSD_gauss(psi_kav, psi_kstd), psi_shape)
        
        # Store a copy for comparison before applying CDF
        self.psi_ref_before_cdf = np.copy(self.psi_ref)
        
        # Normalize the generated field using the Gaussian CDF
        self.psi_ref = 2 * np.pi * scipy.stats.norm.cdf(self.psi_ref, scale=self.psi_ref.std())
    
    def plot_psi_field(self):
        plt.figure(figsize=(20, 5))
        
        # If psi_ref is 3D, take a 2D slice for plotting
        psi_data_to_plot_before = self.psi_ref_before_cdf[0, :, :] if len(self.psi_ref_before_cdf.shape) == 3 else self.psi_ref_before_cdf
        psi_data_to_plot_after = self.psi_ref[0, :, :] if len(self.psi_ref.shape) == 3 else self.psi_ref
        
        # Plot the psi field before applying CDF
        plt.subplot(1, 4, 1)
        plt.imshow(psi_data_to_plot_before, cmap='viridis', origin='lower')
        plt.colorbar(label='Psi Value')
        plt.title('Generated Psi Field (Before CDF)')
        
        # Plot histogram of psi values before applying CDF
        plt.subplot(1, 4, 2)
        plt.hist(psi_data_to_plot_before.flatten(), bins=50, color='blue', edgecolor='black')
        plt.title('Histogram of Psi Values (Before CDF)')
        plt.xlabel('Psi Value')
        plt.ylabel('Frequency')
        
        # Plot the psi field after applying CDF
        plt.subplot(1, 4, 3)
        plt.imshow(psi_data_to_plot_after, cmap='viridis', origin='lower')
        plt.colorbar(label='Psi Value')
        plt.title('Generated Psi Field (After CDF)')
        
        # Plot histogram of psi values after applying CDF
        plt.subplot(1, 4, 4)
        plt.hist(psi_data_to_plot_after.flatten(), bins=50, color='blue', edgecolor='black')
        plt.title('Histogram of Psi Values (After CDF)')
        plt.xlabel('Psi Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

# Example usage
dims = (100, 100, 100)
rand_orientation = 2
field_gen = FieldGenerator(dims, rand_orientation)
field_gen.generate_psi_field()
field_gen.plot_psi_field()
