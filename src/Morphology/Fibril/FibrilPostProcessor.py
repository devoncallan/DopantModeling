from enum import Enum
from dataclasses import dataclass
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import opensimplex as simplex
import re
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from src.Morphology.Fibril.FibrilGenerator import FibrilGenerator, Materials
from src.Morphology.MorphologyData import MorphologyData
from src.Morphology.util.FieldGeneration import generate_field_with_PSD
     
################################################
### Named parameters for FibrilPostProcessor ###
################################################
    
class DopantLocation(Enum):
    EVERYWHERE   = 0
    AMORPH_ONLY  = 1
    CRYSTAL_ONLY = 2
    PREFERENTIAL = 3
    
class DopantOrientation(Enum):
    NONE = 0
    ISOTROPIC = 1
    PARALLEL = 2
    PERPENDICULAR = 3
    ISOTROPIC_S0 = 4
    
####################################################
### Groups of parameters for FibrilPostProcessor ###
####################################################

@dataclass
class MatrixParams:
    amorph_matrix_Vfrac: float = 0.90
    amorph_orientation: bool = True
    
@dataclass
class DopantParams:
    dopant_vol_frac: float = 0.0
    crystal_dopant_frac: float = 0.0
    uniform_doping: bool = False
    dopant_orientation: DopantOrientation = DopantOrientation.NONE
    
@dataclass
class SurfaceRoughnessParams:
    height_feature: float = 0.0
    max_valley_nm: float = 0.0
    
@dataclass
class CoreShellParams:
    gaussian_std: float = 3.0
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
        dopant_params:DopantParams,
        core_shell_params:CoreShellParams,
        surface_roughness_params:SurfaceRoughnessParams):
        
        self.material_params = material_params
        self.matrix_params = matrix_params
        self.dopant_params = dopant_params
        self.core_shell_params = core_shell_params
        self.surface_roughness_params = surface_roughness_params
        
    def process(self, data:MorphologyData, just_add_dopants: bool = False) -> MorphologyData:
        
        if just_add_dopants:
            if self.dopant_params is not None:
                data = self._add_uniform_dopant(data)
                data = self._set_dopant_orientation(data)
        else:
            if self.core_shell_params is not None:
                data = self._add_fibril_shell(data)
    
            data = self._set_amorphous_matrix(data) # Also handles surface roughness
            
            if self.dopant_params is not None:
                data = self._add_uniform_dopant(data)
                data = self._set_dopant_orientation(data)
                
            data = self._adjust_amorphous_density(data)
            data = self._set_amorphous_orientation(data)
            
            data.mat_Vfrac[Materials.VACUUM_ID] = 1.0 - np.sum(data.mat_Vfrac, axis=0)
            
        return data
            
    def _add_fibril_shell(self, data:MorphologyData) -> MorphologyData:
        
        # Isolate crystalline region
        fibril_core_Vfrac = data.mat_Vfrac[Materials.CRYSTAL_ID]
        fibril_core_mask  = np.where(fibril_core_Vfrac == 1.0)

        # Create and isolate shell region
        fibril_shell_Vfrac = gaussian_filter(fibril_core_Vfrac, self.core_shell_params.gaussian_std)
        fibril_shell_Vfrac[fibril_core_mask] = 0.0
        fibril_shell_mask = np.where(fibril_shell_Vfrac >= self.core_shell_params.fibril_shell_cutoff)

        # Set the fibril shells to 100% amorphous
        data.mat_Vfrac[Materials.AMORPH_ID][fibril_shell_mask] = 1
        
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
            amorph_matrix_mask = z_mask & (current_Vfrac < 1.0)
        else:
            amorph_matrix_mask = np.where(data.mat_Vfrac[Materials.CRYSTAL_ID] < 1.0)
        
        print('Setting amorphous matrix...')
        data.mat_Vfrac[Materials.AMORPH_ID][amorph_matrix_mask] = (1.0 - data.mat_Vfrac[Materials.CRYSTAL_ID][amorph_matrix_mask])
        
        return data

    def _add_uniform_dopant(self, data:MorphologyData) -> MorphologyData:
        x_D_target = self.dopant_params.dopant_vol_frac        # Final dopant volume fraction
        f_DC = self.dopant_params.crystal_dopant_frac          # Fraction of dopant in crystalline region
        uniform_doping = self.dopant_params.uniform_doping     # Check for uniform doping
    
        if x_D_target == 0.0:
            return data
        elif x_D_target >= 1.0:
            raise Exception('Target dopant volume fraction must be < 1.')
        
        # Calculate overall volume fractions
        x_C, x_A, x_D, x_V = analyze_vol_fractions(data.mat_Vfrac)
            
        # Override f_DC for uniform doping
        if uniform_doping:
            f_DC = x_C / (x_C + x_A + x_V)
    
        # Function to find the root
        def find_x_DT(x_DT):
            # Create a copy of data.mat_Vfrac for simulation
            sim_Vfrac = np.copy(data.mat_Vfrac)
    
            # Calculate replacement ratios
            R_crystal = (x_C - x_DT * f_DC) / x_C if x_C != 0.0 else 0.0
            R_amorph = (x_A - x_DT * (1.0 - f_DC)) / x_A if x_A != 0.0 else 0.0
    
            # Simulated doping
            sim_Vfrac[Materials.DOPANT_ID] += sim_Vfrac[Materials.CRYSTAL_ID] * (1.0 - R_crystal)
            sim_Vfrac[Materials.DOPANT_ID] += (data.mat_Vfrac[Materials.AMORPH_ID]) * (1.0 - R_amorph)
            sim_Vfrac[Materials.CRYSTAL_ID] *= R_crystal
            sim_Vfrac[Materials.AMORPH_ID] *= R_amorph
    
            # Recalculate volume fractions after simulated doping
            sim_x_C, sim_x_A, sim_x_D, sim_x_V = analyze_vol_fractions(sim_Vfrac)
            sim_x_D = sim_x_D / (sim_x_C + sim_x_A + sim_x_D + sim_x_V)
    
            return sim_x_D - x_D_target
    
        # Initial guess for x_DT
        x_DT_guess = x_D_target
    
        # Solve for x_DT
        x_DT = fsolve(find_x_DT, x_DT_guess)[0]
    
        # Calculate replacement ratios using the solved x_DT
        R_crystal = (x_C - x_DT * f_DC) / x_C if x_C != 0.0 else 0.0
        R_amorph = (x_A - x_DT * (1.0 - f_DC)) / x_A if x_A != 0.0 else 0.0
    
        print('Adding dopant...')
    
        # Replace the specified fraction of each material with dopant in each voxel
        data.mat_Vfrac[Materials.DOPANT_ID] += data.mat_Vfrac[Materials.CRYSTAL_ID] * (1.0 - R_crystal) 
        data.mat_Vfrac[Materials.DOPANT_ID] += (data.mat_Vfrac[Materials.AMORPH_ID]) * (1.0 - R_amorph)
        data.mat_Vfrac[Materials.CRYSTAL_ID] *= R_crystal
        data.mat_Vfrac[Materials.AMORPH_ID] *= R_amorph
    
        return data

    def _adjust_amorphous_density(self, data: MorphologyData) -> MorphologyData:
        print('Adjusting amorphous matrix for density...')
    
        # Adjust the amorphous matrix volume fraction
        data.mat_Vfrac[Materials.AMORPH_ID] *= self.matrix_params.amorph_matrix_Vfrac
    
        return data

    def _set_amorphous_orientation(self, data:MorphologyData) -> MorphologyData:

        if self.matrix_params is not None and self.matrix_params.amorph_orientation == False:
            return data

        amorph = data.mat_Vfrac[Materials.AMORPH_ID]
        amorph_mask = amorph != 0.0
        
        # Generate randomly sampled theta and psi (samples unit sphere)
        theta = np.arccos(1.0 - 2.0 * np.random.uniform(0.0, 1.0, amorph.shape))
        psi = np.random.uniform(-np.pi, np.pi, amorph.shape)

        data.mat_S[Materials.AMORPH_ID][amorph_mask] = 1.0
        data.mat_theta[Materials.AMORPH_ID][amorph_mask] = theta[amorph_mask]
        data.mat_psi[Materials.AMORPH_ID][amorph_mask] = psi[amorph_mask]

        return data
    
    def _set_dopant_orientation(self, data: MorphologyData) -> MorphologyData:
        if self.dopant_params.dopant_orientation == DopantOrientation.NONE:
            return data
    
        # Creating masks using np.where
        dopant_mask = np.where(data.mat_Vfrac[Materials.DOPANT_ID] > 0)
        crystal_mask = np.where(data.mat_Vfrac[Materials.CRYSTAL_ID] > 0)
        amorph_mask = np.where(data.mat_Vfrac[Materials.AMORPH_ID] > 0)
    
        # Determine overlapping regions for dopant with crystal and amorphous phases
        overlapping_with_crystal = (data.mat_Vfrac[Materials.DOPANT_ID] > 0) & (data.mat_Vfrac[Materials.CRYSTAL_ID] > 0)
        overlapping_with_amorph = (data.mat_Vfrac[Materials.DOPANT_ID] > 0) & (data.mat_Vfrac[Materials.AMORPH_ID] > 0)
    
        # Set default orientation scalar
        data.mat_S[Materials.DOPANT_ID][dopant_mask] = 1.0
    
        if self.dopant_params.dopant_orientation == DopantOrientation.ISOTROPIC:
            # Generating isotropic orientation angles
            theta = np.arccos(1.0 - 2.0 * np.random.rand(*data.mat_Vfrac[Materials.DOPANT_ID].shape))
            psi = np.random.uniform(-np.pi, np.pi, data.mat_Vfrac[Materials.DOPANT_ID].shape)
            data.mat_theta[Materials.DOPANT_ID] = theta
            data.mat_psi[Materials.DOPANT_ID] = psi
    
        elif self.dopant_params.dopant_orientation == DopantOrientation.PARALLEL:
            # Parallel orientation adjustment
            data.mat_theta[Materials.DOPANT_ID][overlapping_with_crystal] = data.mat_theta[Materials.CRYSTAL_ID][overlapping_with_crystal]
            data.mat_psi[Materials.DOPANT_ID][overlapping_with_crystal] = data.mat_psi[Materials.CRYSTAL_ID][overlapping_with_crystal]
            data.mat_theta[Materials.DOPANT_ID][overlapping_with_amorph] = data.mat_theta[Materials.AMORPH_ID][overlapping_with_amorph]
            data.mat_psi[Materials.DOPANT_ID][overlapping_with_amorph] = data.mat_psi[Materials.AMORPH_ID][overlapping_with_amorph]
    
        elif self.dopant_params.dopant_orientation == DopantOrientation.PERPENDICULAR:
            # Perpendicular orientation adjustment
            
            if np.any(overlapping_with_crystal):
                theta_crystal = data.mat_theta[Materials.CRYSTAL_ID][overlapping_with_crystal]
                psi_crystal = data.mat_psi[Materials.CRYSTAL_ID][overlapping_with_crystal]
                r_cryst = R.from_euler('zyz', np.stack((np.zeros_like(theta_crystal), theta_crystal, psi_crystal), axis=-1))
                r_cryst_dopant = r_cryst * R.from_euler('X', np.pi / 2)
                
                # there are infinite orthogonal directions in the orthogonal plane, so pick randomly
                dist_angle = np.random.uniform(-np.pi, np.pi, data.mat_theta[Materials.DOPANT_ID][overlapping_with_crystal].shape) 
                r_cryst_dopant = r_cryst_dopant * R.from_euler("Y", dist_angle)

                dopant_euler_crystal = r_cryst_dopant.as_euler('zyz')
                data.mat_theta[Materials.DOPANT_ID][overlapping_with_crystal] = dopant_euler_crystal[:, 1]
                data.mat_psi[Materials.DOPANT_ID][overlapping_with_crystal] = dopant_euler_crystal[:, 2]
    
            if np.any(overlapping_with_amorph):
                theta_amorph = data.mat_theta[Materials.AMORPH_ID][overlapping_with_amorph]
                psi_amorph = data.mat_psi[Materials.AMORPH_ID][overlapping_with_amorph]
                r_amorph = R.from_euler('zyz', np.stack((np.zeros_like(theta_amorph), theta_amorph, psi_amorph), axis=-1))
                r_amorph_dopant = r_amorph * R.from_euler('X', np.pi / 2)
                
                # there are infinite orthogonal directions in the orthogonal plane, so pick randomly
                dist_angle = np.random.uniform(-np.pi, np.pi, data.mat_theta[Materials.AMORPH_ID][overlapping_with_amorph].shape) 
                r_amorph_dopant = r_amorph_dopant * R.from_euler("Y", dist_angle)
                
                dopant_euler_amorph = r_amorph_dopant.as_euler('zyz')
                data.mat_theta[Materials.DOPANT_ID][overlapping_with_amorph] = dopant_euler_amorph[:, 1]
                data.mat_psi[Materials.DOPANT_ID][overlapping_with_amorph] = dopant_euler_amorph[:, 2]
            
        elif self.dopant_params.dopant_orientation == DopantOrientation.ISOTROPIC_S0:
            # same as DopantOrientation.ISOTROPIC, but set S = 0.0
            data.mat_S[Materials.DOPANT_ID][dopant_mask] = 0.0

            # Generating isotropic orientation angles
            theta = np.arccos(1.0 - 2.0 * np.random.rand(*data.mat_Vfrac[Materials.DOPANT_ID].shape))
            psi = np.random.uniform(-np.pi, np.pi, data.mat_Vfrac[Materials.DOPANT_ID].shape)
            data.mat_theta[Materials.DOPANT_ID] = theta
            data.mat_psi[Materials.DOPANT_ID] = psi
        return data
    
    def save_parameters(self, data:MorphologyData, fibgen:FibrilGenerator, p, filename:str=''):
        if filename == '':
            filename = 'parameters.txt'
        else:
            filename = '_'.join(['parameters', filename]) + '.txt'
        
        with open(filename, 'w') as f:
            f.write(f'File: {filename}\n')

            # Box dimensions
            f.write('\nBox Dimensions:\n')
            f.write(f'x: {fibgen.x_dim_nm} nm ({fibgen.x_dim} voxels)\n')
            f.write(f'y: {fibgen.y_dim_nm} nm ({fibgen.y_dim} voxels)\n')
            f.write(f'z: {fibgen.z_dim_nm} nm ({fibgen.z_dim} voxels)\n')
            f.write(f'Pitch: {fibgen.pitch_nm} nm\n')

            # Fibril Generator Parameters
            f.write('\nFibril Generator Parameters:\n')
            f.write(f'\tSize:\t{fibgen.fibril_size_params}\n')
            f.write(f'\tGrowth:\t{fibgen.fibril_growth_params}\n')
            f.write(f'\tOrientation:\t{fibgen.fibril_orientation_params}\n')

            # Post Processor Parameters
            f.write('\nPost Processor Parameters:\n')
            f.write(f'\tMaterials:\t{self.material_params}\n')
            f.write(f'\tCoreShell:\t{self.core_shell_params}\n')
            f.write(f'\tMatrix:\t{self.matrix_params}\n')
            f.write(f'\tSurface Roughness:\t{self.surface_roughness_params}\n')
            f.write(f'\tDopant:\t{self.dopant_params}\n')

            # Calculated Mole and Volume Fractions
            crystal_mol_frac, amorph_mol_frac, dopant_mol_frac = analyze_mol_fractions(data.mat_Vfrac, self.material_params.density, self.material_params.mw)
            crystal_vol_frac, amorph_vol_frac, dopant_vol_frac, vacuum_vol_frac = analyze_vol_fractions(data.mat_Vfrac)
            f.write('\nCalculated Mole Fractions:\n')
            f.write(f'\tCrystalline Mole Fraction: {crystal_mol_frac}\n')
            f.write(f'\tAmorphous Mole Fraction: {amorph_mol_frac}\n')
            f.write(f'\tDopant Mole Fraction: {dopant_mol_frac}\n')

            f.write('\nCalculated Volume Fractions:\n')
            f.write(f'\tCrystalline Volume Fraction: {crystal_vol_frac}\n')
            f.write(f'\tAmorphous Volume Fraction: {amorph_vol_frac}\n')
            f.write(f'\tDopant Volume Fraction: {dopant_vol_frac}\n')

            # # Percent Crystallinity
            # per_crystal = analyze_percent_crystallinity(data.mat_Vfrac)
            # f.write('\nPercent Crystallinity:\n')
            # f.write(f'\tCrystallinity (%): {per_crystal}\n')

            # Energies and Materials
            f.write('\nSimulation Energies:\n')
            f.write(f'\t{p.edge_to_find}: {p.energies}\n')

            f.write('\nMaterials Dictionary:\n')
            for material, path in p.material_dict.items():
                f.write(f'\t{material}: {path}\n')

### Post Process Morphology ###

def process_morphology(fibgen: FibrilGenerator, p, existing_data: MorphologyData = None, just_add_dopants: bool = False) -> MorphologyData:
    material_params = MaterialParams(
        num_materials=p.num_materials, 
        mw=p.mw, 
        density=p.density
    )

    core_shell_params = CoreShellParams(
        gaussian_std=p.gaussian_std, 
        fibril_shell_cutoff=p.fibril_shell_cutoff
    )

    matrix_params = MatrixParams(
        amorph_matrix_Vfrac=p.amorph_matrix_Vfrac,
        amorph_orientation=p.amorph_orientation
    )

    surface_roughness_params = SurfaceRoughnessParams(
        height_feature=p.height_feature,
        max_valley_nm=p.max_valley_nm
        )
    
    if p.height_feature == 0 and p.max_valley_nm == 0:
        surface_roughness_params = None

    dopant_params = DopantParams(
        dopant_vol_frac=p.dopant_vol_frac, 
        crystal_dopant_frac=p.crystal_dopant_frac,
        uniform_doping=p.uniform_doping,
        dopant_orientation=p.dopant_orientation
        )
    
    if p.dopant == None:
        dopant_params = None

    post_processor = FibrilPostProcessor(
        material_params=material_params,
        matrix_params=matrix_params,
        surface_roughness_params=surface_roughness_params,
        core_shell_params=core_shell_params,
        dopant_params=dopant_params
    )
    
    if existing_data is not None and just_add_dopants:
        data = post_processor.process(existing_data, just_add_dopants=True)
    else:
        data = fibgen.create_morphology_data()
        data = post_processor.process(data)

    return data, post_processor

#######################################################
### Functions for calculating volume/mole fractions ###
#######################################################

def analyze_percent_crystallinity(mat_Vfrac):

    # Sum the volume of each component
    crystal_vol = np.sum(mat_Vfrac[Materials.CRYSTAL_ID] != 0.0)
    amorph_vol  = np.sum(mat_Vfrac[Materials.AMORPH_ID] != 0.0)

    total_vol = crystal_vol + amorph_vol
    percent_crystallinity = crystal_vol / total_vol * 100.0
    
    return percent_crystallinity
    
def analyze_vol_fractions(mat_Vfrac):

    # Sum the volume of each component
    crystal_vol = np.sum(mat_Vfrac[Materials.CRYSTAL_ID])
    amorph_vol  = np.sum(mat_Vfrac[Materials.AMORPH_ID])
    dopant_vol  = np.sum(mat_Vfrac[Materials.DOPANT_ID])
    vacuum_vol  = np.sum(mat_Vfrac[Materials.VACUUM_ID])

    # Calculate the total occupied volume
    total_vol = crystal_vol + amorph_vol + dopant_vol + vacuum_vol

    # Calculate volume fractions
    crystal_vol_frac = crystal_vol / total_vol
    amorph_vol_frac  = amorph_vol  / total_vol
    dopant_vol_frac  = dopant_vol  / total_vol
    vacuum_vol_frac = vacuum_vol / total_vol

    return crystal_vol_frac, amorph_vol_frac, dopant_vol_frac, vacuum_vol_frac

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

def read_crystalline_mol_frac_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'Crystalline Mole Fraction:' in line:
                return float(re.findall("\d+\.\d+", line)[0])
    return None