from enum import Enum
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import opensimplex as simplex
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from src.Morphology.Fibril.FibrilGenerator import FibrilGenerator, Materials
from src.Morphology.MorphologyData import MorphologyData
from src.Morphology.util.FieldGeneration import generate_field_with_PSD

# dopant_method:
    # all dopant methods:
        # 0: no dopant 
    # random dopant:
        # 1: uniform random replacing p3ht 
        # 2: Dopant only in amorph matrix
        # 3: Dopant only in fibrils (mainly for f4tcnq, tfsi likely won't do this)
        # 4: Uniformly doped to the dopant frac, all subtracted from P3HT
        # 5: Dope preferentially towards fibrils or matrix
            # crystal_dope_frac: fraction of total dopant in crystals
    # uniform dopant
        # 1: uniform random replacing p3ht 
        # 2: Dopant only in amorph matrix
        # 3: Dopant only in fibrils (mainly for f4tcnq, tfsi likely won't do this)
    # preferential_dopant
        # != 0: add dopant
            # crystal_dope_frac: fraction of total dopant in crystals
     
################################################
### Named parameters for FibrilPostProcessor ###
################################################
    
class DopantLocation(Enum):
    EVERYWHERE   = 0
    AMORPH_ONLY  = 1
    CRYSTAL_ONLY = 2
    PREFERENTIAL = 3
    
# class DopantDistribution(Enum):
#     NO_DOPANT    = 0
#     RANDOM       = 1
#     UNIFORM      = 2
    
# class DopantDistribution(Enum):
#     NO_DOPANT    = 0
#     UNIFORM      = 1
#     PREFERENTIAL = 2
    
# Overall dopant volume fraction
# Fraction of dopant in crystalline region vs. amorphous


class DopantOrientation(Enum):
    NONE = 0
    ISOTROPIC = 1
    PARALLEL = 2
    PERPENDICULAR = 3
    
####################################################
### Groups of parameters for FibrilPostProcessor ###
####################################################

@dataclass
class MatrixParams:
    amorph_matrix_Vfrac: float = 0.90
    amorph_orientation: bool = True
    
@dataclass
class DopantParams:
    dopant_vol_frac: float = 0.
    crystal_dopant_frac: float = 0.
    uniform_doping: bool = False
    dopant_orientation: DopantOrientation = DopantOrientation.NONE
    
@dataclass
class SurfaceRoughnessParams:
    height_feature: float = 0.
    max_valley_nm: float = 0.
    
@dataclass
class CoreShellParams:
    gaussian_std: float = 3.
    fibril_shell_cutoff: float = 0.2
    
@dataclass
class MaterialParams:
    mw: dict
    density: dict
    num_materials: int = 4

#############################
### FibrilPostProcessor:  ###
#############################
    
class FibrilPostProcessor:
    
    def __init__(self,
        material_params:MaterialParams,
        matrix_params:MatrixParams,
        dopant_params:DopantParams=None,
        core_shell_params:CoreShellParams=None,
        surface_roughness_params:SurfaceRoughnessParams=None):
        
        self.material_params = material_params
        self.matrix_params = matrix_params
        self.dopant_params = dopant_params
        self.core_shell_params = core_shell_params
        self.surface_roughness_params = surface_roughness_params
        
    def process(self, data:MorphologyData) -> MorphologyData:

        if self.core_shell_params is not None:
            data = self._add_fibril_shell(data)

        data = self._set_amorphous_matrix(data) # Also handles surface roughness
        data = self._set_amorphous_orientation(data)

        if self.dopant_params is not None:
            data = self._add_uniform_dopant(data)
            data = self._set_dopant_orientation(data)

        return data
            
    def _add_fibril_shell(self, data:MorphologyData) -> MorphologyData:
        
        # Isolate crystalline region
        fibril_core_Vfrac = data.mat_Vfrac[Materials.CRYSTAL_ID]
        fibril_core_mask  = np.where(fibril_core_Vfrac == 1)

        # Create and isolate shell region
        fibril_shell_Vfrac = gaussian_filter(fibril_core_Vfrac, self.core_shell_params.gaussian_std)
        fibril_shell_Vfrac[fibril_core_mask] = 0
        fibril_shell_mask = np.where(fibril_shell_Vfrac >= self.core_shell_params.fibril_shell_cutoff)

        # Set the fibril shells to 100% amorphous
        fibril_shell_Vfrac = np.zeros_like(fibril_shell_Vfrac)
        fibril_shell_Vfrac[fibril_shell_mask] = 1.0
        data.mat_Vfrac[Materials.AMORPH_ID] = fibril_shell_Vfrac
        return data
    
    def _set_amorphous_matrix(self, data: MorphologyData) -> MorphologyData:
        
        if self.surface_roughness_params is not None:
            print('Adding surface roughness...')
            feature_size = int(data.x_dim / self.surface_roughness_params.height_feature)
            max_valley = int(self.surface_roughness_params.max_valley_nm / data.pitch_nm)
            
            # Calculate surface roughness depth map for entire grid
            x_range = np.arange(data.x_dim)
            y_range = np.arange(data.y_dim)
            depth_map = (max_valley/2 * (simplex.noise2array(x_range / feature_size, y_range / feature_size) + 1)).astype(int)
            max_z_dim = data.z_dim - depth_map
            
            z_range = np.arange(data.z_dim)
            z_mask = z_range[:,None,None] < max_z_dim
            current_Vfrac = data.mat_Vfrac[Materials.AMORPH_ID] + data.mat_Vfrac[Materials.CRYSTAL_ID]
            amorph_matrix_mask = z_mask & (current_Vfrac < 1)
        else:
            amorph_matrix_mask = np.where(data.mat_Vfrac[Materials.CRYSTAL_ID] != 1)
        
        print('Setting amorphous matrix...')
        data.mat_Vfrac[Materials.AMORPH_ID][amorph_matrix_mask] += self.matrix_params.amorph_matrix_Vfrac
        data.mat_Vfrac[Materials.AMORPH_ID] = np.clip(data.mat_Vfrac[Materials.AMORPH_ID], 0, 1)
        data.mat_Vfrac[Materials.VACUUM_ID] = 1 - data.mat_Vfrac[Materials.CRYSTAL_ID] - data.mat_Vfrac[Materials.AMORPH_ID] - data.mat_Vfrac[Materials.DOPANT_ID]
        
        return data

    def _add_uniform_dopant(self, data:MorphologyData) -> MorphologyData:
        x_D = self.dopant_params.dopant_vol_frac        # Final dopant volume fraction
        f_DC = self.dopant_params.crystal_dopant_frac   # Fraction of dopant in crystalline region
        uniform_doping = self.dopant_params.uniform_doping  # Check for uniform doping
    
        if x_D == 0:
            return data
        elif x_D >= 1:
            raise Exception('Target dopant volume fraction must be < 1.')
            
        print('Adding dopant...')
    
        # Calculate overall volume fractions
        crystal_vol_frac, amorph_vol_frac, _ = analyze_vol_fractions(data.mat_Vfrac)
    
        # Override f_DC for uniform doping
        if uniform_doping:
            f_DC = crystal_vol_frac
    
        # Calculate replacement ratios
        R_crystal = (crystal_vol_frac - x_D * f_DC) / crystal_vol_frac if crystal_vol_frac != 0 else 0
        R_amorph = (amorph_vol_frac - x_D * (1 - f_DC)) / amorph_vol_frac if amorph_vol_frac != 0 else 0
    
        # Replace the specified fraction of each material with dopant in each voxel
        data.mat_Vfrac[Materials.DOPANT_ID] += data.mat_Vfrac[Materials.CRYSTAL_ID] * (1 - R_crystal)
        data.mat_Vfrac[Materials.DOPANT_ID] += data.mat_Vfrac[Materials.AMORPH_ID] * (1 - R_amorph)
        data.mat_Vfrac[Materials.CRYSTAL_ID] *= R_crystal
        data.mat_Vfrac[Materials.AMORPH_ID] *= R_amorph
    
        # Ensure the sum of all volume fractions equals 1
        data.mat_Vfrac[Materials.VACUUM_ID] = 1 - np.sum(data.mat_Vfrac, axis=0)
    
        return data

    def _set_amorphous_orientation(self, data:MorphologyData) -> MorphologyData:

        if self.matrix_params is not None and self.matrix_params.amorph_orientation == False:
            return data

        amorph = data.mat_Vfrac[Materials.AMORPH_ID]
        amorph_mask = amorph != 0
        
        # Generate randomly sampled theta and psi (samples unit sphere)
        theta = np.arccos(1 - 2 * np.random.uniform(0, 1, amorph.shape))
        psi = np.random.uniform(-np.pi, np.pi, amorph.shape)

        data.mat_S[Materials.AMORPH_ID][amorph_mask] = 1
        data.mat_theta[Materials.AMORPH_ID][amorph_mask] = theta[amorph_mask]
        data.mat_psi[Materials.AMORPH_ID][amorph_mask] = psi[amorph_mask]

        return data
    
    def _set_dopant_orientation(self, data:MorphologyData) -> MorphologyData:

        if self.dopant_params.dopant_orientation == DopantOrientation.NONE:
            return data

        # Set to zero, parallel in crystalline, perpendicular in crystalline, and/or in amorphous, random
        crystal = data.mat_Vfrac[Materials.CRYSTAL_ID]
        crystal_mask = crystal != 0
        
        amorph = data.mat_Vfrac[Materials.AMORPH_ID]
        amorph_mask = amorph != 0
        
        dopant_in_crystal = np.sum(data.mat_Vfrac[Materials.DOPANT_ID][crystal_mask]) > 0.
        dopant_in_amorph  = np.sum(data.mat_Vfrac[Materials.DOPANT_ID][amorph_mask]) > 0.
        
        if dopant_in_crystal:
            data.mat_S[Materials.DOPANT_ID][crystal_mask] = 1
            data.mat_theta[Materials.DOPANT_ID][crystal_mask] = data.mat_theta[Materials.CRYSTAL_ID][crystal_mask]
            data.mat_psi[Materials.DOPANT_ID][crystal_mask] = data.mat_psi[Materials.CRYSTAL_ID][crystal_mask]
        if dopant_in_amorph:
            data.mat_S[Materials.DOPANT_ID][amorph_mask] = 1
            data.mat_theta[Materials.DOPANT_ID][amorph_mask] = data.mat_theta[Materials.AMORPH_ID][amorph_mask]
            data.mat_psi[Materials.DOPANT_ID][amorph_mask] = data.mat_psi[Materials.AMORPH_ID][amorph_mask]
        
        return data
    
    def save_parameters(self, data:MorphologyData, fibgen:FibrilGenerator, filename:str=''):

        if filename == '':
            filename = 'parameters.txt'
        else:
            filename = '_'.join(['parameters', filename]) + '.txt'
        
        with open(filename, 'w') as f:
            f.write(filename+'\n')

            # Box dimensions
            f.write('\nBox Dimensions:\n')
            f.write(f'x: {fibgen.x_dim_nm} nm ({fibgen.x_dim} voxels)\n')
            f.write(f'y: {fibgen.y_dim_nm} nm ({fibgen.y_dim} voxels)\n')
            f.write(f'z: {fibgen.z_dim_nm} nm ({fibgen.z_dim} voxels)\n')
            f.write(f'Pitch: {fibgen.pitch_nm} nm\n')

            f.write('\nFibril Generator Parameters:\n')
            f.write(f'\tSize:\t{fibgen.fibril_size_params}\n')
            f.write(f'\tGrowth:\t{fibgen.fibril_growth_params}\n')
            f.write(f'\tOrientation:\t{fibgen.fibril_orientation_params}\n')

            f.write('\nPost Processor Parameters:\n')
            f.write(f'\tMaterials:\t{self.material_params}\n')
            f.write(f'\tCoreShell:\t{self.core_shell_params}\n')
            f.write(f'\tMatrix:\t{self.matrix_params}\n')
            f.write(f'\tSurface Roughness:\t{self.surface_roughness_params}\n')
            f.write(f'\tDopant:\t{self.dopant_params}\n')


            crystal_mol_frac, amorph_mol_frac, dopant_mol_frac = analyze_mol_fractions(data.mat_Vfrac, self.material_params.density, self.material_params.mw)
            crystal_vol_frac, amorph_vol_frac, dopant_vol_frac = analyze_vol_fractions(data.mat_Vfrac)
            f.write('\nCalculated Mole Fractions:\n')
            f.write(f'\tCrystalline Mole Fraction: {crystal_mol_frac}\n')
            f.write(f'\tAmorphous Mole Fraction: {amorph_mol_frac}\n')
            f.write(f'\tDopant Mole Fraction: {dopant_mol_frac}\n')

            f.write('\nCalculated Volume Fractions:\n')
            f.write(f'\tCrystalline Volume Fraction: {crystal_vol_frac}\n')
            f.write(f'\tAmorphous Volume Fraction: {amorph_vol_frac}\n')
            f.write(f'\tDopant Volume Fraction: {dopant_vol_frac}\n')

            per_crystal = analyze_percent_crystallinity(data.mat_Vfrac)
            f.write('\nPercent Crystallinity:\n')
            f.write(f'\tCrystallinity (%): {per_crystal}')

### Post Process Morphology ###

def process_morphology(fibgen: FibrilGenerator, p) -> MorphologyData:
    material_params = MaterialParams(
        num_materials=p.num_materials, 
        mw=p.mw, 
        density=p.density
    )

    core_shell_params = CoreShellParams(
        gaussian_std=p.gaussian_std, 
        fibril_shell_cutoff=p.fibril_shell_cutoff)

    matrix_params = MatrixParams(
        amorph_matrix_Vfrac=p.amorph_matrix_Vfrac, 
        amorph_orientation=p.amorph_orientation)

    surface_roughness_params = None
    if p.use_surface_roughness_params:
        surface_roughness_params = SurfaceRoughnessParams(
            height_feature=p.height_feature,
            max_valley_nm=p.max_valley_nm)
    
    dopant_params = None
    if p.use_dopant_params:
        dopant_params = DopantParams(
            dopant_vol_frac=p.dopant_vol_frac, 
            crystal_dopant_frac=p.crystal_dopant_frac,
            dopant_orientation=DopantOrientation.ISOTROPIC)

    post_processor = FibrilPostProcessor(
        material_params=material_params,
        matrix_params=matrix_params,
        surface_roughness_params=surface_roughness_params,
        core_shell_params=core_shell_params,
        dopant_params=dopant_params)
    
    data = fibgen.create_morphology_data()
    data = post_processor.process(data)
    post_processor.save_parameters(data, fibgen)

    return data


#######################################################
### Functions for calculating volume/mole fractions ###
#######################################################

def analyze_percent_crystallinity(mat_Vfrac):

    # Sum the volume of each component
    crystal_vol = np.sum(mat_Vfrac[Materials.CRYSTAL_ID] != 0)
    amorph_vol  = np.sum(mat_Vfrac[Materials.AMORPH_ID] != 0)

    total_vol = crystal_vol + amorph_vol
    percent_crystallinity = crystal_vol / total_vol * 100
    
    return percent_crystallinity
    
def analyze_vol_fractions(mat_Vfrac):

    # Sum the volume of each component
    crystal_vol = np.sum(mat_Vfrac[Materials.CRYSTAL_ID])
    amorph_vol  = np.sum(mat_Vfrac[Materials.AMORPH_ID])
    dopant_vol  = np.sum(mat_Vfrac[Materials.DOPANT_ID])
    vacuum_vol  = np.sum(mat_Vfrac[Materials.VACUUM_ID])

    # Calculate the total occupied volume
    total_vol = crystal_vol + amorph_vol + dopant_vol + 0*vacuum_vol

    # Calculate volume fractions
    crystal_vol_frac = crystal_vol / total_vol
    amorph_vol_frac  = amorph_vol  / total_vol
    dopant_vol_frac  = dopant_vol  / total_vol

    return crystal_vol_frac, amorph_vol_frac, dopant_vol_frac

def analyze_mol_fractions(mat_Vfrac, density, MW):

    # Sum the volume of each component
    crystal_vol = np.sum(mat_Vfrac[Materials.CRYSTAL_ID])
    amorph_vol  = np.sum(mat_Vfrac[Materials.AMORPH_ID])
    dopant_vol  = np.sum(mat_Vfrac[Materials.DOPANT_ID])

    # Calculate molar amounts
    crystal_mol = (crystal_vol * density[Materials.CRYSTAL_ID]) / MW[Materials.CRYSTAL_ID]
    amorph_mol  = (amorph_vol  * density[Materials.AMORPH_ID])  / MW[Materials.AMORPH_ID]
    dopant_mol  = (dopant_vol  * density[Materials.DOPANT_ID])  / MW[Materials.DOPANT_ID]

    # Calculate the total molar amount
    total_mol = crystal_mol + amorph_mol + dopant_mol

    # Calculate volume fractions
    crystal_mol_frac = crystal_mol / total_mol
    amorph_mol_frac  = amorph_mol  / total_mol
    dopant_mol_frac  = dopant_mol  / total_mol

    return crystal_mol_frac, amorph_mol_frac, dopant_mol_frac