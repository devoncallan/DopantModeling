import subprocess
from src.Common.files import move, make_output_dir, delete_path

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

from src.Morphology.MorphologyData import MorphologyData 

from NRSS.writer import write_materials, write_hdf5, write_config
from NRSS.checkH5 import checkH5

CONFIG_FILE = 'config.txt'
LOG_FILE = 'CyRSoXS.log'
PARAM_FILE = 'parameters.txt'
HDF5_DIR = 'HDF5/'


def cleanup(p, delete_morph_file=False):

    delete_path(CONFIG_FILE)
    delete_path(LOG_FILE)
    delete_path(PARAM_FILE)
    delete_path(HDF5_DIR)
    for i in range(p.num_materials):
        delete_path(f'Material{i+1}.txt')
    
    if delete_morph_file:
        delete_path(p.DEFAULT_MORPH_FILE)

def create_hdf5(data:MorphologyData, p):
    write_hdf5(data.get_data()[0:p.num_materials], float(p.pitch_nm), p.DEFAULT_MORPH_FILE)

def create_inputs(p):
    write_materials(p.energies, p.material_dict, p.energy_dict, p.num_materials)
    write_config(list(p.energies), [0.0, 1.0, 360.0], CaseType=0, MorphologyType=0)

def run(p, save_dir:str=''):
    subprocess.run(['CyRSoXS', p.DEFAULT_MORPH_FILE])

    if save_dir == '':
        return
    
    move(src=CONFIG_FILE, dest_dir=save_dir)
    move(src=LOG_FILE, dest_dir=save_dir)
    move(src=PARAM_FILE, dest_dir=save_dir)
    move(src=HDF5_DIR, dest_dir=save_dir)
    for i in range(p.num_materials):
        move(src=f'Material{i+1}.txt', dest_dir=save_dir)

def load(base_path, pitch_nm=2):

    load = cyrsoxsLoader()
    raw_data = load.loadDirectory(base_path, PhysSize=pitch_nm)

    integ = WPIntegrator()
    integ_data = integ.integrateImageStack(raw_data)

    return raw_data, integ_data

def get_Iq2_ISI():
    return

def get_para_perp_AR(integ_data, q_range):

    q_min, q_max = q_range

    para = integ_data.rsoxs.slice_chi(90, chi_width = 45).sel(q = slice(q_min, q_max)) + \
        integ_data.rsoxs.slice_chi(-90, chi_width = 45).sel(q = slice(q_min, q_max))
    perp = integ_data.rsoxs.slice_chi(0, chi_width = 45).sel(q = slice(q_min, q_max)) + \
        integ_data.rsoxs.slice_chi(180, chi_width = 45).sel(q = slice(q_min, q_max))
    AR = (para - perp)/(para + perp)

    return para, perp, AR