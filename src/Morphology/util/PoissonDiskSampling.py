import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def remove_points_near_edge(points: np.ndarray, cutoff:float) -> np.ndarray:
    """Keep points in the range [cutoff, 1-cutoff]

    Args:
        points (np.ndarray): Nx3 array of points whose values range from 0 to 1
        cutoff (float): distance from edge (must be < 0.5)

    Returns:
        np.ndarray: Nx3 array of valid points
    """
    
    # Find the closest distance to an edge for each dimension for each point
    points_edge_dist = np.min(np.stack((points, 1-points), axis=-1), axis=-1) # dim = N x d

    # Remove all points < cutoff distance away from edge
    valid_indices = np.all(points_edge_dist >= cutoff, axis=-1)
    valid_points = points[valid_indices]
    
    return valid_points

def reshape_points(points: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Remove 

    Args:
        points (np.ndarray): _description_
        dims (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    valid_points = np.all(points <= dims, axis=1)
    points = points[valid_points]

    return points

def rescale_and_reshape_points(points, dims):

    print(points.shape)
    print(dims.shape)

    assert(points.shape[1] == dims.shape[0])


    scaled_points = points * np.max(dims)

    scaled_points = scaled_points[np.all(scaled_points <= dims, axis=1)]

    return scaled_points

def plot_points(points, diameter):

    fig, ax = plt.subplots(figsize=(20,20), dpi=300)
    ax.scatter(points[:,0], points[:,1])
    circles = [plt.Circle((point[0], point[1]), radius=diameter/2, fill=False) for point in points]
    collection = PatchCollection(circles, match_original=True)
    ax.add_collection(collection)
    ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$')
    return fig, ax

def generate_points(dims, c2c_dist):
    
    print('Generating fibril centers from Poisson disk sampling...')
    
    rng = np.random.default_rng()
    
    unit_dims = dims / np.max(dims)
    c2c_unit_dist = c2c_dist / np.max(dims)
    
    # radius = minimum distance between points
    engine = qmc.PoissonDisk(d=len(dims), radius=c2c_unit_dist, seed=rng)
    points = engine.fill_space()
    
    points = reshape_points(points, unit_dims)
    points = remove_points_near_edge(points, c2c_unit_dist/2)
    points *= np.max(dims)
    
    return points
    
    
    
    
    