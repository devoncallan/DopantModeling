import numpy as np

def cartesian_to_euler(orientation):
    orientation /= np.linalg.norm(orientation)
    theta = np.arccos(orientation[2])
    psi = np.arctan2(orientation[1], orientation[0])
    return theta, psi

def euler_to_cartesian(self, theta, psi):
    z = np.sin(theta) * np.cos(psi)
    y = np.sin(theta) * np.sin(psi)
    x = np.cos(theta)
    return np.array([z, y, x])