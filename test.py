import h5py
import numpy as np
from pathlib import Path
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import Validators as vd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

home_path = str(Path().resolve()) + '\\measuring results\\'

file_path = inquirer.filepath(
    message='Choose scan file to load:\n(CTRL+Z to cancel)\n',
    default=home_path,
    mandatory=False,
    validate=PathValidator(is_file=True, message='Input is not a file')
).execute()
if file_path == None:
    print(f'{bcolors.WARNING}Data loading canceled!{bcolors.ENDC}')

with h5py.File(file_path,'r') as file:
    group = file['PA data']
    raw_ds = group['raw data']

    raw_data = np.zeros((
        len(raw_ds.keys()),
        raw_ds.attrs['max dataset len'] + 1))
    
    for i, key in zip(range(len(raw_ds.keys())), raw_ds.keys()):
        raw_data[i,0] = raw_ds[key].attrs['parameter value']
        raw_data[i,1:raw_ds[key].len() + 1] = raw_ds[key][:]

print(f'shape of loaded data {raw_data.shape}')
print(raw_data[0,0])
print(raw_data[1,0])
print(raw_data[1,1])
    
