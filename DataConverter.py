import h5py
import numpy as np
import os.path
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

def load_data(old_data):
    """Loads data from a file"""

    home_path = str(Path().resolve()) + '\\measuring results\\'
    file_path = inquirer.filepath(
        message='Choose spectral file to load:\n(CTRL+Z to cancel)\n',
        default=home_path,
        mandatory=False,
        validate=PathValidator(is_file=True, message='Input is not a file')
    ).execute()
    if file_path == None:
        print(f'{bcolors.WARNING}Data loading canceled!{bcolors.ENDC}')
        return old_data

    if file_path.split('.')[-1] != 'npy':
        print(f'{bcolors.WARNING} Wrong data format! *.npy is required{bcolors.ENDC}')
        return old_data
    
    data = np.load(file_path)
    print(f'Data with shape {bcolors.OKGREEN}{data.shape}{bcolors.ENDC} loaded!')
    return data, file_path

if __name__ == '__main__':

    data, file_path = load_data([])
    
    h5_file_path = file_path.split('.npy')[0] + '.hdf5'
    with h5py.File(h5_file_path,'w') as file:
        general = file.create_group('general')
        general.attrs['parameter name'] = 'wavelength'
        general.attrs['parameter units'] = 'nm'
        general.attrs['data points'] = data.shape[0]
        raw_data = file.create_group('raw data')
        raw_data.attrs['x var name'] = 'time'
        raw_data.attrs['x var units'] = 's'
        raw_data.attrs['y var name'] = 'PA signal'
        raw_data.attrs['y var units'] = 'V'
        raw_data.attrs['max dataset len'] = 0
        filt_data = file.create_group('filtered data')
        filt_data.attrs['x var name'] = 'time'
        filt_data.attrs['x var units'] = 's'
        filt_data.attrs['y var name'] = 'filtered PA signal'
        filt_data.attrs['y var units'] = 'V'
        filt_data.attrs['max dataset len'] = 0
        freq_data = file.create_group('freq data')
        freq_data.attrs['x var name'] = 'frequency'
        freq_data.attrs['x var units'] = 'Hz'
        freq_data.attrs['y var name'] = 'FFT of PA signal'
        freq_data.attrs['y var units'] = 'V'
        freq_data.attrs['max dataset len'] = 0
        

        for i in range(data.shape[0]):
            if i<10:
                ds_name = 'point00' + str(i)
            elif i<100:
                ds_name = 'point0' + str(i)
            else:
                ds_name = 'point' + str(i)
            ds_raw = raw_data.create_dataset(ds_name, data=data[i,0,6:])
            if ds_raw.len() > raw_data.attrs['max dataset len']:
                raw_data.attrs['max dataset len'] = ds_raw.len()
                filt_data.attrs['max dataset len'] = ds_raw.len()
            
            if i < (data.shape[0]-1):
                wl = data[0,0,0] + i*data[0,0,2]
            else:
                wl = data[0,0,1]
            ds_raw.attrs['parameter value'] = wl
            ds_raw.attrs['x var step'] = data[i,0,3]
            ds_raw.attrs['x var start'] = 0
            ds_raw.attrs['x var stop'] = 0 + (ds_raw.len()-1)*data[i,0,3] 
            ds_raw.attrs['max amp'] = data[i,0,4]
            ds_raw.attrs['PM energy'] = data[i,0,5]
            ds_raw.attrs['sample energy'] = data[i,1,5]

            
            ds_filt = filt_data.create_dataset(ds_name, data=data[i,1,6:])
            ds_filt.attrs['parameter value'] = wl
            ds_filt.attrs['x var step'] = data[i,0,3]
            ds_filt.attrs['x var start'] = 0
            ds_filt.attrs['x var stop'] = 0 + (ds_raw.len()-1)*data[i,0,3]
            ds_filt.attrs['max amp'] = data[i,1,4]
            ds_filt.attrs['PM energy'] = data[i,0,5]
            ds_filt.attrs['sample energy'] = data[i,1,5]

            freq_range = int((data[0,2,1]-data[0,2,0])/data[0,2,2])+1
            ds_freq = freq_data.create_dataset(ds_name,data=data[i,2,3:freq_range+3])
            if ds_freq.len() > freq_data.attrs['max dataset len']:
                freq_data.attrs['max dataset len'] = ds_freq.len()
            ds_freq.attrs['parameter value'] = wl
            ds_freq.attrs['x var step'] = data[0,2,2]
            ds_freq.attrs['x var start'] = data[0,2,0]
            ds_freq.attrs['x var stop'] = data[0,2,1]
            ds_freq.attrs['PM energy'] = data[i,0,5]
            ds_freq.attrs['sample energy'] = data[i,1,5]