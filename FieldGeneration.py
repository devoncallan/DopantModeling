import numpy as np
import scipy.stats
from FyeldGenerator import generate_field

def generate_field_with_PSD(k, std, dims, max_value, normalization_type='cdf', new_mean=None, new_std=None):
    """
    Generates a field using a Gaussian Power Spectral Density (PSD) function.
    
    Parameters:
    - k (float): Mean of the Gaussian PSD function.
    - std (float): Standard deviation of the Gaussian PSD function.
    - dims (tuple): Dimensions of the field to generate.
    - max_value (float): Maximum value of the generated field.
    - normalization_type (str, optional): Type of normalization to apply ('cdf' or 'psd'). Default is 'cdf'.
    - new_mean (float, optional): New mean to apply when normalization_type is 'psd'.
    - new_std (float, optional): New standard deviation to apply when normalization_type is 'psd'.
    
    Returns:
    - numpy.ndarray: Generated field.
    
    Notes:
    - PSD refers to Power Spectral Density. It is a function used to describe the distribution of power into frequency components.
    - The generated field is normalized using either a Cumulative Distribution Function (CDF) or directly through its PSD.
    - Passing through the CDF remaps the field from a Gaussian distribution to a distribution of equal probabilities for each outcome.
    """
    
    def PSD_gauss(avg, std):
        def Pk(k):
            return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((k - avg) / std) ** 2)
        return Pk

    def distrib(shape):
        return np.random.normal(loc=0, scale=np.pi/5, size=shape)

    field_shape = tuple(dims)
    generated_field = generate_field(distrib, PSD_gauss(k, std), field_shape)
    
    if normalization_type == 'cdf':
        return max_value * scipy.stats.norm.cdf(generated_field, scale=generated_field.std()) - max_value/2
    elif normalization_type == 'psd':
        if new_mean is None or new_std is None:
            raise ValueError("For normalization_type='psd', new_mean and new_std must be provided.")
        
        old_std = generated_field.std()
        normalized_field = generated_field * (new_std / old_std) + new_mean
        return normalized_field
    else:
        raise ValueError("Invalid normalization_type. Use either 'cdf' or 'psd'")
