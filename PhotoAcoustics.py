from __future__ import annotations
from traceback import print_tb

import warnings
import os.path
from pathlib import Path
import time
import math
from itertools import combinations
from typing import Any, Iterable, TypedDict
from datetime import datetime

from pylablib.devices import Thorlabs
from scipy.fftpack import rfft, irfft, fftfreq
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning # type: ignore
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import matplotlib.gridspec as gridspec
import keyboard
import h5py

import validators as vd
import oscilloscope

config = {
    'pre_time':2,#[us] used for zoom data. Ref is max of filt PA signal
    'post_time':13#[us] same for post time
}

osc_params = {
    'pre_time': 100, # [us] start time of data storage before trigger
    'frame_duration': 250, # [us] whole duration of the stored frame
    'pm_response_time': 2500, # [us] response time of the power meter
    'pm_pre_time': 300,
    'laser_calib_uj': 2500000,
    'trigger_channel': 'CHAN1',
    'pa_channel': 'CHAN2',
}
       
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

class MeasuredData:
    """Class for data storage and manipulations.
    Data has 4 dicsts:
    {attrs} stores general information about the data
    other 3 dicts are {raw_data}, {filt_data} and {freq_data}
    each contains {attrs} dics with specific information about  
    the data group and dicts named {point1}, {point2}, etc, 
    which correspond to the measured datasets.
    Each dataset has numpy array with actual data and 
    additional information about the dataset."""

    def __init__(self) -> None:

        #attributes of data
        self.attrs = {
            'parameter name': 'Unknown',
            'parameter units': 'Unknown',
            'data points': 0,
            'created': self._get_cur_time(),
            'updated': self._get_cur_time(),
            'path': '',
            'filename': '',
            'zoom pre time': 0.000002,
            'zoom post time': 0.000013,
            'zoom units': 's'
        }

        #raw ds for each measured points + dict with attributes
        self.raw_data = {
            'attrs': {
                'max dataset len': 0,
                'x var name': 'Unknown',
                'x var units': 'Unknown',
                'y var name': 'Unknown',
                'y var units': 'Unknown'
            }
        }
        #filtered ds for each measured points + dict with attributes
        self.filt_data = {
            'attrs': {
                'max dataset len': 0,
                'x var name': 'Unknown',
                'x var units': 'Unknown',
                'y var name': 'Unknown',
                'y var units': 'Unknown'
            }
        } 
        #FFT of each measured point + dict with attributes
        self.freq_data = {
            'attrs': {
                'max dataset len': 0,
                'x var name': 'Frequency',
                'x var units': 'Hz',
                'y var name': 'FFT amplitude',
                'y var units': 'Unknown'
            }
        } 

    def set_metadata(self, data_group: str, metadata: dict) -> None:
        """set attributes for the data_group"""

        if data_group == 'raw_data':
            self.raw_data['attrs'].update(metadata)
            self.attrs['updated'] = self._get_cur_time()
        elif data_group == 'filt_data':
            self.filt_data['attrs'].update(metadata)
            self.attrs['updated'] = self._get_cur_time()
        elif data_group == 'freq_data':
            self.freq_data['attrs'].update(metadata)
            self.attrs['updated'] = self._get_cur_time()
        elif data_group == 'general':
            self.attrs.update(metadata)
            self.attrs['updated'] = self._get_cur_time()
        else:
            print((f'{bcolors.WARNING}'
                  + 'Unknown data_group for metadata!'
                  + f'{bcolors.ENDC}'))

    def add_measurement(self, data: np.ndarray, attributes: dict) -> None:
        """Adds a datapoint to raw_data,
        also adds empty datapoint to other data groups.
        attributes are set to both raw and filt datapoints"""

        ds_name = self.build_ds_name(self.attrs['data points'])
        if not ds_name:
            print((f'{bcolors.WARNING}'
                  + 'Max data points reached! Data cannot be added!'
                  + f'{bcolors.ENDC}'))
            return
        ds = {}
        ds.update({'data': data,
                   'parameter value': 0,
                   'x var step': 0,
                   'x var start': 0,
                   'x var stop': 0,
                   'PM energy': 0,
                   'sample energy': 0,
                   'max amp': 0})

        if len(data) > self.raw_data['attrs']['max dataset len']:
            self.raw_data['attrs']['max dataset len'] = len(data)
        ds.update(attributes)
        self.raw_data.update({ds_name:ds})

        self.filt_data.update({ds_name:{}})
        self.filt_data[ds_name].update(attributes)
        self.freq_data.update({ds_name:{}})
        self.attrs['data points'] += 1
        self.attrs['updated'] = self._get_cur_time()

    def _get_cur_time (self) -> str:
        """returns timestamp of current time"""
        
        cur_time = time.time()
        date_time = datetime.fromtimestamp(cur_time)
        date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

        return date_time

    def save(self, filename: str='') -> None:
        """Saves data to file"""

        if filename != None:
            self.attrs['filename'] = filename.split('\\')[-1]
            self.attrs['path'] = filename.split(self.attrs['filename'])[0]

        elif self.attrs['filename'] and self.attrs['path']:
            filename = self.attrs['path'] + self.attrs['filename']
        
        else:
            print(f'{bcolors.WARNING}'
                  + 'Filename is not set. Data cannot be saved!'
                  + f'{bcolors.ENDC}')
            return
        
        self._flush(filename)

    def save_tmp(self) -> None:
        """saves current data to TmpData.hdf5"""

        Path('measuring results/').mkdir(parents=True, exist_ok=True)
        filename = 'measuring results/TmpData.hdf5'
        self._flush(filename)

    def _flush(self, filename: str) -> None:
        """Actually writes to disk"""

        with h5py.File(filename,'w') as file:
            general = file.create_group('general')
            general.attrs.update(self.attrs)

            raw_data = file.create_group('raw data')
            raw_data.attrs.update(self.raw_data['attrs'])
            for key, value in self.raw_data.items():
                if key !='attrs':
                    ds_raw = raw_data.create_dataset(key,data=value['data'])
                    for attr_name, attr_value in value.items():
                        if attr_name != 'data':
                            ds_raw.attrs.update({attr_name:attr_value})
        
            filt_data = file.create_group('filtered data')
            filt_data.attrs.update(self.filt_data['attrs'])
            for key, value in self.filt_data.items():
                if key !='attrs':
                    ds_filt = filt_data.create_dataset(key,data=value['data'])
                    for attr_name, attr_value in value.items():
                        if attr_name != 'data':
                            ds_filt.attrs.update({attr_name:attr_value})
            
            freq_data = file.create_group('freq data')
            freq_data.attrs.update(self.freq_data['attrs'])
            for key, value in self.freq_data.items():
                if key !='attrs':
                    ds_freq = freq_data.create_dataset(key,data=value['data'])
                    for attr_name, attr_value in value.items():
                        if attr_name != 'data':
                            ds_freq.attrs.update({attr_name:attr_value})

    def load(self, filename: str) -> None:
        """Loads data from file"""

        self.attrs['filename'] = filename.split('\\')[-1]
        self.attrs['path'] = filename.split(self.attrs['filename'])[0]

        with h5py.File(filename,'r') as file:
            
            #load general metadata
            general = file['general']
            self.attrs.update(general.attrs)
            
            #raw_data
            raw_data = file['raw data']
            #metadata of raw_data
            self.raw_data['attrs'].update(raw_data.attrs)
            #cycle for loading measured points (datasets)
            for ds_name in raw_data.keys(): # type: ignore
                self.raw_data.update({ds_name:{}})
                #load actual data
                self.raw_data[ds_name].update(
                    {'data': raw_data[ds_name][:]}) #type: ignore
                #load metadata of the dataset
                self.raw_data[ds_name].update(
                    raw_data[ds_name].attrs) #type: ignore

            #filt_data
            filt_data = file['filtered data']
            #metadata of filt_data
            self.filt_data['attrs'].update(filt_data.attrs)
            #cycle for loading measured points (datasets)
            for ds_name in filt_data.keys(): # type: ignore
                self.filt_data.update({ds_name:{}})
                #load actual data
                self.filt_data[ds_name].update(
                    {'data': filt_data[ds_name][:]}) # type: ignore
                #load metadata of the dataset
                self.filt_data[ds_name].update(
                    filt_data[ds_name].attrs) # type: ignore

            #freq_data
            freq_data = file['freq data']
            #metadata of freq_data
            self.freq_data['attrs'].update(freq_data.attrs)
            #cycle for loading measured points (datasets)
            for ds_name in freq_data.keys(): # type: ignore
                self.freq_data.update({ds_name:{}})
                #load actual data
                self.freq_data[ds_name].update(
                    {'data': freq_data[ds_name][:]}) # type: ignore
                #load metadata of the dataset
                self.freq_data[ds_name].update(
                    freq_data[ds_name].attrs) # type: ignore

        #set amount of data points for compatibility with old data
        if self.attrs.get('data points', 0) < (len(self.raw_data) - 1):
            self.attrs.update({'data points': len(self.raw_data) - 1})

    def build_ds_name(self, n: int) -> str:
        """Builds and returns name of dataset"""
        
        if n <10:
            n_str = '00' + str(n)
        elif n<100:
            n_str = '0' + str(n)
        elif n<1000:
            n_str = str(n)
        else:
            return ''
        return 'point' + n_str
    
    def get_ds_index(self, ds_name: str) -> int:
        """Returns index from dataset name"""

        n_str = ds_name.split('point')[-1]
        n = int(n_str)
        return n

    def plot(self) -> None:
        """Plots current data"""

        if not self.attrs['data points']:
            print(f'{bcolors.WARNING}'
                  + 'No data to display'
                  + f'{bcolors.ENDC}')

        #plot initialization
        warnings.filterwarnings('ignore',
                                category=MatplotlibDeprecationWarning)

        self._fig = plt.figure(tight_layout=True)
        filename = self.attrs['path'] + self.attrs['filename']
        self._fig.suptitle(filename)
        gs = gridspec.GridSpec(2,3)
        self._ax_sp = self._fig.add_subplot(gs[0,0])
        self._ax_raw = self._fig.add_subplot(gs[0,1])
        self._ax_freq = self._fig.add_subplot(gs[1,0])
        self._ax_filt = self._fig.add_subplot(gs[1,1])
        self._ax_raw_zoom = self._fig.add_subplot(gs[0,2])
        self._ax_filt_zoom = self._fig.add_subplot(gs[1,2])

        self._param_ind = 0 #index of active data on plot

        #plot max_signal_amp(parameter)
        self._param_values = self.get_dependance('raw_data','parameter value')
        self._raw_amps = self.get_dependance('raw_data', 'max amp')
        self._filt_amps = self.get_dependance('filt_data', 'max amp')
        
        self._ax_sp.plot(
            self._param_values,
            self._raw_amps,
            label='raw data')
        self._ax_sp.plot(
            self._param_values,
            self._filt_amps,
            label='filt data'
        )
        self._ax_sp.legend(loc='upper right')
        self._ax_sp.set_ylim(bottom=0)
        x_label = (self.attrs['parameter name']
                   + ', ['
                   + self.attrs['parameter units']
                   + ']')
        self._ax_sp.set_xlabel(x_label)
        y_label = self.raw_data['attrs']['y var name']
        self._ax_sp.set_ylabel(y_label)

        self._plot_selected_raw, = self._ax_sp.plot(
            self._param_values[self._param_ind], # type: ignore
            self._raw_amps[self._param_ind],  # type: ignore
            'o', alpha=0.4, ms=12, color='yellow')
        
        self._plot_selected_filt, = self._ax_sp.plot(
            self._param_values[self._param_ind],  # type: ignore
            self._filt_amps[self._param_ind],  # type: ignore
            'o', alpha=0.4, ms=12, color='yellow')

        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._plot_update()
        plt.show()

    def _on_key_press(self, event) -> None:
        """Callback function for changing active data on plot"""

        if event.key == 'left':
            if self._param_ind == 0:
                pass
            else:
                self._param_ind -= 1
                self._plot_update()
        elif event.key == 'right':
            if self._param_ind == (self.attrs['data points'] - 1):
                pass
            else:
                self._param_ind += 1
                self._plot_update()

    def _plot_update(self) -> None:
        """Updates plotted data"""

        ds_name = self.build_ds_name(self._param_ind)
        start = self.raw_data[ds_name]['x var start']
        step = self.raw_data[ds_name]['x var step']
        stop = self.raw_data[ds_name]['x var stop']
        num = len(self.raw_data[ds_name]['data'])
        time_points = np.linspace(start,stop,num)

        #check if datapoint is empty
        if not step:
            return None

        #update max_signal_amp(parameter) plot
        title = (self.attrs['parameter name']
                 + ': '
                 + str(int(self._param_values[self._param_ind]))
                 + self.attrs['parameter units'])
        self._ax_sp.set_title(title)
        self._plot_selected_raw.set_data(
            self._param_values[self._param_ind], # type: ignore
            self._raw_amps[self._param_ind]) # type: ignore
        self._plot_selected_filt.set_data(
            self._param_values[self._param_ind], # type: ignore
            self._filt_amps[self._param_ind]) # type: ignore
        
        #update filt data
        self._ax_filt.clear()
        self._ax_filt.plot(time_points,
                                 self.filt_data[ds_name]['data'])
        x_label = (self.filt_data['attrs']['x var name']
                   + ', ['
                   + self.filt_data['attrs']['x var units']
                   + ']')
        self._ax_filt.set_xlabel(x_label)
        self._ax_filt_zoom.set_xlabel(x_label)

        y_label = (self.filt_data['attrs']['y var name']
                   + ', ['
                   + self.filt_data['attrs']['y var units']
                   + ']')
        self._ax_filt.set_ylabel(y_label)
        self._ax_filt_zoom.set_ylabel(y_label)

        title = self.filt_data['attrs']['y var name']
        self._ax_filt.set_title(title)

        #marker for max value
        filt_max = np.amax(self.filt_data[ds_name]['data'])
        filt_max_ind = np.argwhere(
            self.filt_data[ds_name]['data']==filt_max)[0][0]
        filt_max_t = time_points[filt_max_ind]
        self._ax_filt.plot(filt_max_t,
                           filt_max,
                           'o',
                           alpha=0.4,
                           ms=12,
                           color='yellow')
        
        #marker for min value
        filt_min = np.amin(self.filt_data[ds_name]['data'])
        filt_min_t = time_points[
            np.argwhere(self.filt_data[ds_name]['data']==filt_min)[0]]
        self._ax_filt.plot(filt_min_t,
                           filt_min,
                           'o',
                           alpha=0.4,
                           ms=12,
                           color='yellow')
        
        #marker for zoomed area
        if self.filt_data['attrs']['x var units'] == self.attrs['zoom units']:
            pre_points = int(self.attrs['zoom pre time']/step)
            post_points = int(self.attrs['zoom post time']/step)
            start_zoom_ind = filt_max_ind-pre_points
            if start_zoom_ind < 0:
                start_zoom_ind = 0
            stop_zoom_ind = filt_max_ind + post_points
            if stop_zoom_ind > (len(time_points) - 1):
                stop_zoom_ind = len(time_points) - 1
            self._ax_filt.fill_betweenx([filt_min,filt_max],
                                    time_points[start_zoom_ind],
                                    time_points[stop_zoom_ind],
                                    alpha=0.3,
                                    color='g')
        else:
            print(f'{bcolors.WARNING}'
                  + 'Zoom units do not match x var units!'
                  + f'{bcolors.ENDC}')
            
        #update raw data
        self._ax_raw.clear()
        self._ax_raw.plot(time_points,
                         self.raw_data[ds_name]['data'])
        x_label = (self.raw_data['attrs']['x var name']
                   + ', ['
                   + self.raw_data['attrs']['x var units']
                   + ']')
        self._ax_raw.set_xlabel(x_label)
        self._ax_raw_zoom.set_xlabel(x_label)

        y_label = (self.raw_data['attrs']['y var name']
                   + ', ['
                   + self.raw_data['attrs']['y var units']
                   + ']')
        self._ax_raw.set_ylabel(y_label)
        self._ax_raw_zoom.set_ylabel(y_label)

        title = self.raw_data['attrs']['y var name']
        self._ax_raw.set_title(title)

        #marker for max value
        raw_max = np.amax(self.raw_data[ds_name]['data'])
        raw_max_ind = np.argwhere(
            self.raw_data[ds_name]['data']==raw_max)[0][0]
        raw_max_t = time_points[raw_max_ind]
        self._ax_raw.plot(raw_max_t,
                          raw_max,
                          'o',
                          alpha=0.4,
                          ms=12,
                          color='yellow')

        #marker for min value
        raw_min = np.amin(self.raw_data[ds_name]['data'])
        raw_min_t = time_points[
            np.argwhere(self.raw_data[ds_name]['data']==raw_min)[0]]
        self._ax_raw.plot(raw_min_t,
                          raw_min,
                          'o',
                          alpha=0.4,
                          ms=12,
                          color='yellow')
        
        #marker for zoomed area
        self._ax_raw.fill_betweenx([raw_min,raw_max],
                                   time_points[start_zoom_ind],
                                   time_points[stop_zoom_ind],
                                   alpha=0.3,
                                   color='g')
        
        #update raw zoom data
        self._ax_raw_zoom.clear()
        self._ax_raw_zoom.plot(
            time_points[start_zoom_ind:stop_zoom_ind+1],
            self.raw_data[ds_name]['data'][start_zoom_ind:stop_zoom_ind+1])
        self._ax_raw_zoom.set_title('Zoom of raw PA data')

        #update raw zoom data
        self._ax_filt_zoom.clear()
        self._ax_filt_zoom.plot(
            time_points[start_zoom_ind:stop_zoom_ind+1],
            self.filt_data[ds_name]['data'][start_zoom_ind:stop_zoom_ind+1])
        self._ax_filt_zoom.set_title('Zoom of filtered PA data')

        #update freq data
        start_freq = self.freq_data[ds_name]['x var start']
        step_freq = self.freq_data[ds_name]['x var step']
        num_freq = len(self.freq_data[ds_name]['data'])
        stop_freq = (num_freq - 1)*step_freq + start_freq
        freq_points = np.linspace(start_freq,stop_freq,num_freq)

        self._ax_freq.clear()
        self._ax_freq.plot(freq_points,
                          self.freq_data[ds_name]['data'])
        x_label = (self.freq_data['attrs']['x var name']
                   + ', ['
                   + self.freq_data['attrs']['x var units']
                   + ']')
        self._ax_freq.set_xlabel(x_label)

        y_label = (self.freq_data['attrs']['y var name']
                   + ', ['
                   + self.freq_data['attrs']['y var units']
                   + ']')
        self._ax_freq.set_ylabel(y_label)
        
        title = self.freq_data['attrs']['y var name']
        self._ax_freq.set_title(title)
        
        #general update
        self._fig.align_labels()
        self._fig.canvas.draw()

    def get_dependance(self, data_group: str, value: str) -> Iterable:
        """Returns an array with value 
        from each dataset in the data_group.
        data_group: 'raw_data'|'filt_data|'freq_data'
        """

        dep = [] #array for return values
        if not self.attrs['data points']:
            print(f'{bcolors.WARNING}'
                  + 'Attempt to read dependence from empty data'
                  + f'{bcolors.ENDC}')
            return []
        
        if data_group == 'raw_data':
            ds_name = self.build_ds_name(0)
            if self.raw_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}'
                      + 'Attempt to read dependence of unknown VALUE from RAW_data'
                      + f'{bcolors.ENDC}')
                return []
            for ds_name, ds in self.raw_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])

        
        elif data_group == 'filt_data':
            ds_name = self.build_ds_name(0)
            if self.filt_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}'
                      + 'Attempt to read dependence of unknown VALUE from FILT_data'
                      + f'{bcolors.ENDC}')
                return []
            for ds_name, ds in self.filt_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])
        
        elif data_group == 'freq_data':
            ds_name = self.build_ds_name(0)
            if self.freq_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}'
                      + 'Attempt to read dependence of unknown VALUE from FREQ_data'
                      + f'{bcolors.ENDC}')
                return []
            for ds_name, ds in self.freq_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])
        
        else:
            print(f'{bcolors.WARNING}'
                  + 'Attempt to read dependence from unknow GROUP'
                  + f'{bcolors.ENDC}')
            return []
        return dep

    def bp_filter(self,
                  low: int = 1000000,
                  high: int = 10000000) -> None:
        """Perform bandpass filtration on data.
        low is high pass cutoff frequency in Hz
        high is low pass cutoff frequency in Hz"""

        for ds_name, ds in self.raw_data.items():
            if ds_name != 'attrs':
                if self.raw_data['attrs']['x var units'] != 's':
                    print(f'{bcolors.WARNING}'
                          + 'FFT conversion require x var step in seconds'
                          + f'{bcolors.ENDC}')
                    return
                dt = ds['x var step']
                W = fftfreq(len(ds['data']), dt) # array with freqs
                f_signal = rfft(ds['data']) # signal in f-space

                filtered_f_signal = f_signal.copy()
                filtered_f_signal[:,(W<low)] = 0 # high pass filtering

                if high > 1/(2.5*dt): # Nyquist frequency check
                    filtered_f_signal[:,(W>1/(2.5*dt))] = 0 
                else:
                    filtered_f_signal[:,(W>high)] = 0

                #pass frequencies
                filtered_freq = W[(W>low)*(W<high)]

                self.freq_data.update({ds_name:{}})
                #start freq, end freq, step freq
                self.freq_data[ds_name].update(
                    {'x var start':filtered_freq.min()})
                self.freq_data[ds_name].update(
                    {'x var stop':filtered_freq.max()})
                self.freq_data[ds_name].update(
                    {'x var step':filtered_freq[1]-filtered_freq[0]})

                #Fourier amplitudes
                self.freq_data[ds_name].update(
                    {'data':f_signal[:,(W>low)*(W<high)]})
                freq_points = len(self.freq_data[ds_name]['data'])
                #update 'max dataset len' metadata
                if self.freq_data['attrs']['max dataset len'] < freq_points:
                    self.freq_data['attrs'].update(
                        {'max dataset len':freq_points})

                #filtered PA data
                #filt dataset does not exist
                #then create the datapoints
                if not self.filt_data.get(ds_name):
                    self.filt_data[ds_name].update(
                        {'data':irfft(filtered_f_signal)})
                    for key, value in self.raw_data[ds_name].items():
                        if key != 'data':
                            self.filt_data[ds_name].update({key:value})
                        max_amp = (np.amax(self.filt_data[ds_name]['data'])
                                 - np.amin(self.filt_data[ds_name]['data']))
                        self.filt_data[ds_name].update({'max amp': max_amp})
                #if the ds already exists, update its values
                else:
                    self.filt_data[ds_name].update(
                        {'data':irfft(filtered_f_signal)})
                    max_amp = (np.amax(self.filt_data[ds_name]['data'])
                             - np.amin(self.filt_data[ds_name]['data']))
                    self.filt_data[ds_name].update({'max amp': max_amp})

        #update metadata
        self.freq_data['attrs']['y var units'] = self.raw_data['attrs']['y var units']
        self.attrs['updated'] = self._get_cur_time()

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware"""

    stage_x: Any
    stage_y: Any
    osc: oscilloscope.Oscilloscope

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware"""
    
    power_meter: oscilloscope.PowerMeter

def init_hardware(hardware: Hardware) -> None:
    """Initialize all hardware"""

    if not hardware['stage_x'] and not hardware['stage_y']:
        init_stages(hardware)   
    else:
        print(f'{bcolors.WARNING}Stages already initiated!{bcolors.ENDC}')

    if hardware['osc'].not_found:
        hardware['osc'].initialize()
    else:
        print(f'{bcolors.WARNING}Oscilloscope already initiated!{bcolors.ENDC}')

    if hardware['stage_x'] and hardware['stage_y'] and not hardware['osc'].not_found:
        hardware['power_meter'] = oscilloscope.PowerMeter(hardware['osc'])
        print(f'{bcolors.OKGREEN}Initialization complete!{bcolors.ENDC}')

def init_stages(hardware: Hardware) -> None:
    """Initiate stages."""

    print('Initializing stages...')
    stages = Thorlabs.list_kinesis_devices() # type: ignore

    if len(stages) < 2:
        print(f'{bcolors.WARNING}Less than 2 stages detected! Try again!{bcolors.ENDC}')

    else:
        stage1_ID = stages.pop()[0]
        #motor units [m]
        stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') # type: ignore
        print(f'{bcolors.OKBLUE}Stage X{bcolors.ENDC} initiated. Stage X ID = {stage1_ID}')
        hardware['stage_x'] = stage1

        stage2_ID = stages.pop()[0]
        #motor units [m]
        stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') # type: ignore
        print(f'{bcolors.OKBLUE}Stage Y{bcolors.ENDC} initiated. Stage X ID = {stage2_ID}')
        hardware['stage_y'] = stage2

def move_to(X: float, Y: float, hardware: Hardware) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    
    hardware['stage_x'].move_to(X/1000)
    hardware['stage_y'].move_to(Y/1000)

def wait_stages_stop(hardware: Hardware) -> None:
    """Waits untill all specified stages stop"""

    if hardware['stage_x']:
        hardware['stage_x'].wait_for_stop()
    
    if hardware['stage_y']:
        hardware['stage_y'].wait_for_stop()

def scan(hardware):
    """Scan an area, which starts at 
    at (x_start, y_start) and has a size (x_size, y_size) in mm.
    Checks upper scan boundary.
    Returns 2D array with normalized signal amplitudes and
    3D array with the whole normalized PA data for each scan point.
    Updates global state."""

    stage_X = hardware['stage x']
    stage_Y = hardware['stage y']
    osc = hardware['osc']

    if state['stages init'] and state['osc init']:
        x_start = inquirer.text(
            message='Enter X starting position [mm] ' + vd.cancel_option,
            default='1.0',
            mandatory=False,
            validate=vd.ScanRangeValidator()
        ).execute()
        if x_start == None:
            print(f'{bcolors.WARNING} Intput terminated!{bcolors.ENDC}')
            return hardware
        else:
            x_start = float(x_start)
        
        y_start = inquirer.text(
            message='Enter Y starting position [mm] ' + vd.cancel_option,
            default='1.0',
            mandatory=False,
            validate=vd.ScanRangeValidator()
        ).execute()
        if y_start == None:
            print(f'{bcolors.WARNING} Intput terminated!{bcolors.ENDC}')
            return hardware
        else:
            y_start = float(y_start)

        x_size = inquirer.text(
            message='Enter X scan size [mm] ' + vd.cancel_option,
            default= str(x_start + 3.0),
            mandatory=False,
            validate=vd.ScanRangeValidator()
        ).execute()
        if x_size == None:
            print(f'{bcolors.WARNING} Intput terminated!{bcolors.ENDC}')
            return hardware
        else:
            x_size = float(x_size)

        y_size = inquirer.text(
            message='Enter Y scan size [mm] ' + vd.cancel_option,
            default= str(y_start + 3.0),
            mandatory=False,
            validate=vd.ScanRangeValidator()
        ).execute()
        if y_size == None:
            print(f'{bcolors.WARNING} Intput terminated!{bcolors.ENDC}')
            return hardware
        else:
            y_size = float(y_size)

        x_points = inquirer.text(
            message='Enter number of X scan points ' + vd.cancel_option,
            default= '5',
            mandatory=False,
            validate=vd.ScanPointsValidator()
        ).execute()
        if x_points == None:
            print(f'{bcolors.WARNING} Intput terminated!{bcolors.ENDC}')
            return hardware
        else:
            x_points = int(x_points)

        y_points = inquirer.text(
            message='Enter number of Y scan points' + vd.cancel_option,
            default= '5',
            mandatory=False,
            validate=vd.ScanPointsValidator()
        ).execute()
        if y_points == None:
            print(f'{bcolors.WARNING} Intput terminated!{bcolors.ENDC}')
            return hardware
        else: 
            y_points = int(y_points)

        scan_frame = np.zeros((x_points,y_points)) #scan image of normalized amplitudes
        scan_frame_full = np.zeros((x_points,y_points,4,osc.pa_frame_size)) #0-raw data, 1-filt data, 2-freq, 3-FFT

        print('Scan starting...')
        move_to(x_start, y_start, hardware) # move to starting point
        wait_stages_stop(hardware)

        fig, ax = plt.subplots(1,1)
        im = ax.imshow(scan_frame)
        fig.show()

        for i in range(x_points):
            for j in range(y_points):
                x = x_start + i*(x_size/x_points)
                y = y_start + j*(y_size/y_points)

                move_to(x,y,hardware)
                wait_stages_stop(hardware)

                osc.measure()
                if not osc.bad_read:
                    if osc.laser_amp >1:
                        scan_frame[i,j] = osc.signal_amp/osc.laser_amp
                        scan_frame_full[i,j,0,:] = osc.current_pa_data/osc.laser_amp
                        print(f'normalizaed amp at ({i}, {j}) is {scan_frame[i,j]:.3f}\n')
                    else:
                        scan_frame[i,j] = 0
                        scan_frame_full[i,j,0,:] = 0
                        print(f'{bcolors.WARNING} Bad data at point ({i},{j}){bcolors.ENDC}\n')
                else:
                    scan_frame[i,j] = 0
                    scan_frame_full[i,j,0,:] = 0
                    print(f'{bcolors.WARNING} Bad data at point ({i},{j}){bcolors.ENDC}\n')
                    
                im.set_data(scan_frame.transpose())
                im.set_clim(vmax=np.amax(scan_frame))
                fig.canvas.draw()
                plt.pause(0.1)

        print(f'{bcolors.OKGREEN}...Scan complete!{bcolors.ENDC}')

        max_amp_index = np.unravel_index(scan_frame.argmax(), scan_frame.shape) # find position with max PA amp
        if x_points > 1 and y_points > 1:
            opt_x = x_start + max_amp_index[0]*x_size/(x_points-1)
            opt_y = y_start + max_amp_index[1]*y_size/(y_points-1)
            print(f'best pos indexes {max_amp_index}')
            print(f'best X pos = {opt_x:.2f}')
            print(f'best Y pos = {opt_y:.2f}')

            confirm_move = inquirer.confirm(message='Move to optimal position?').execute()
            if confirm_move:
                print(f'Start moving to the optimal position...')
                move_to(opt_x, opt_y, hardware)
                wait_stages_stop(hardware)
                print(f'{bcolors.OKGREEN}PA detector came to the optimal position!{bcolors.ENDC}')
            
            print(f'{bcolors.UNDERLINE} Do not forget to adjust datactor position along laser beam (manually)!{bcolors.ENDC}')

    else:
        if not state['stages init']:
            print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')
        if not state['osc init']:
            print(f'{bcolors.WARNING} Oscilloscope is not initialized!{bcolors.ENDC}')
        return 0, 0, 0 

    dt = 1/osc.sample_rate
    state['scan data'] = True
    return scan_frame, scan_frame_full, dt

def save_data(data: MeasuredData) -> None:
    """"Save data"""

    if not data.attrs['filename']:
        filename = inquirer.text(
            message='Enter Sample name' + vd.cancel_option,
            default='Unknown',
            mandatory=False
        ).execute()
        if filename == None:
            print(f'{bcolors.WARNING}Save terminated!{bcolors.ENDC}')
            return
        full_name = 'measuring results/' + filename + '.hdf5'
    else:
        filename = data.attrs['filename']
        full_name = data.attrs['path'] + filename

    if os.path.exists(full_name):
        override = inquirer.confirm(
            message='Do you want to override file ' + filename + '?'
        ).execute()
        
        if not override:
            i = 1
            full_name_tmp = full_name.split('.hdf5')[0] + str(i) + '.hdf5'
            while os.path.exists(full_name_tmp):
                i +=1
                full_name_tmp = full_name.split('.hdf5')[0] + str(i) + '.hdf5'
            full_name = full_name_tmp

    data.save(full_name)
    print(f'File updated: {bcolors.OKGREEN}{filename}{bcolors.ENDC}')

def load_data(old_data: MeasuredData) -> MeasuredData:
    """Return loaded data in the related format"""

    home_path = str(Path().resolve()) + '\\measuring results\\'
    file_path = inquirer.filepath(
        message='Choose spectral file to load:' + vd.cancel_option,
        default=home_path,
        mandatory=False,
        validate=PathValidator(is_file=True, message='Input is not a file')
    ).execute()
    if file_path == None:
        print(f'{bcolors.WARNING}Data loading canceled!{bcolors.ENDC}')
        return old_data
    
    if file_path.split('.')[-1] != 'hdf5':
        print(f'{bcolors.WARNING} Wrong data format! *.hdf5 is required{bcolors.ENDC}')
        return old_data
    new_data = MeasuredData()
    new_data.load(file_path)
    print(f'... data with {len(new_data.raw_data)-1} PA measurements loaded!')
    return new_data

def bp_filter(data: MeasuredData) -> None:
    """Perform bandpass filtration on data
    low is high pass cutoff frequency in Hz
    high is low pass cutoff frequency in Hz
    dt is time step in seconds"""

    low_cutof = inquirer.text(
        message='Enter low cutoff frequency [Hz]' + vd.cancel_option,
        default='1000000',
        mandatory=False,
        validate=vd.FreqValidator()
    ).execute()
    if low_cutof is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return
    low_cutof = int(low_cutof)

    high_cutof = inquirer.text(
        message='Enter high cutoff frequency [Hz]' + vd.cancel_option,
        default='10000000',
        mandatory=False,
        validate=vd.FreqValidator()
    ).execute()
    if high_cutof is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return
    high_cutof = int(high_cutof)

    data.bp_filter(low_cutof,high_cutof)

def print_status(hardware: Hardware) -> None:
    """Prints current status and position of stages and oscilloscope"""
    
    if hardware['stage_x'] and hardware['stage_y']:
        stage_X = hardware['stage_x']
        stage_Y = hardware['stage_y']
        print(f'{bcolors.OKBLUE}Stages are initiated!{bcolors.ENDC}')
        print(f'{bcolors.OKBLUE}X stage{bcolors.ENDC} '
              + f'homing status: {stage_X.is_homed()}, '
              + f'status: {stage_X.get_status()}, '
              + f'position: {stage_X.get_position()*1000:.2f} mm.')
        print(f'{bcolors.OKBLUE}Y stage{bcolors.ENDC} '
              + f'homing status: {stage_Y.is_homed()}, '
              + f'status: {stage_Y.get_status()}, '
              + f'position: {stage_Y.get_position()*1000:.2f} mm.')
    else:
        print(f'{bcolors.WARNING}Stages are not initialized!{bcolors.ENDC}')

    if not hardware['osc'].not_found:
        print(f'{bcolors.OKBLUE}Oscilloscope is initiated!{bcolors.ENDC}')
    else:
        print(f'{bcolors.WARNING} Oscilloscope is not initialized!{bcolors.ENDC}')

    if hardware['stage_x'] and hardware['stage_y'] and not hardware['osc'].not_found:
        print(f'{bcolors.OKGREEN} All hardware is initiated!{bcolors.ENDC}')

def home(hardware: Hardware) -> None:
    """Homes stages"""

    if hardware['stage_x'] and hardware['stage_y']:
        hardware['stage_x'].home(sync=False,force=True)
        hardware['stage_y'].home(sync=False,force=True)
        print('Homing started...')
        wait_stages_stop(hardware)
        print(f'{bcolors.OKGREEN}...Homing complete!{bcolors.ENDC}')
    else:
        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

def spectra(hardware: Hardware,
            old_data: MeasuredData,
            store_raw: bool=False,
            chan_pm: str='CHAN1',
            chan_pa: str='CHAN2') -> MeasuredData:
    """Measures spectral data.
    store_raw flag will result in saving datapoints in uint8 format
    chan_pa ='CHAN1'|'CHAN2', is channel to which PA sendor is connected"""

    osc = hardware['osc']

    #set read channel for power meter
    hardware['power_meter'].set_channel(chan_pm) #type:ignore
    pm = hardware['power_meter'] #type:ignore

    #return old data if osc is not init
    if osc.not_found:
        print(f'{bcolors.WARNING}'
              + 'Oscilloscope is not initializaed!'
              + f'{bcolors.ENDC}')
        return old_data
    
    #CLI to get measuring options
    power_control = inquirer.select(
        message='Choose method for laser energy control:',
        choices=[
            'Glan prism',
            'Filters'
        ],
        mandatory=False
    ).execute()
    if power_control is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return old_data

    start_wl = inquirer.text(
        message='Set start wavelength, [nm]' + vd.cancel_option,
        default='950',
        mandatory=False,
        validate=vd.WavelengthValidator()
    ).execute()
    if start_wl is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return old_data
    start_wl = int(start_wl)

    end_wl = inquirer.text(
        message='Set end wavelength, [nm]' + vd.cancel_option,
        default='690',
        mandatory=False,
        validate=vd.WavelengthValidator()
    ).execute()
    if end_wl is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return old_data
    end_wl = int(end_wl)

    step = inquirer.text(
        message='Set step, [nm]' + vd.cancel_option,
        default='10',
        mandatory=False,
        validate=vd.StepWlValidator()
    ).execute()
    if step is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return old_data
    step = int(step)

    target_energy = inquirer.text(
        message='Set target energy in [mJ]' + vd.cancel_option,
        default='0.5',
        mandatory=False,
        validate=vd.EnergyValidator()
    ).execute()
    if target_energy is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return old_data
    target_energy = float(target_energy)*1000

    # set default value.
    max_combinations = 0

    if power_control == 'Filters':
        max_combinations = inquirer.text(
            message='Set maximum amount of filters' + vd.cancel_option,
            default='2',
            mandatory=False,
            validate=vd.FilterNumberValidator()
        ).execute()
        if max_combinations is None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return old_data
        max_combinations = int(max_combinations)

    averaging = inquirer.text(
        message='Set averaging' + vd.cancel_option,
        default='5',
        mandatory=False,
        validate=vd.AveragingValidator()
    ).execute()
    if averaging is None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return old_data   
    averaging = int(averaging)  

    #make steps negative if going from long WLs to short
    if start_wl > end_wl:
        step = -step
        
    print(f'{bcolors.UNDERLINE}'
          + 'Start measuring spectra!'
          + f'{bcolors.ENDC}')

    #calculate amount of data points
    d_wl = end_wl-start_wl
    if d_wl%step:
        spectral_points = int(d_wl/step) + 2
    else:
        spectral_points = int(d_wl/step) + 1

    #create data class and set basic metadata
    data = MeasuredData()
    data.set_metadata(data_group='general', metadata={
        'parameter name': 'wavelength',
        'parameter units': 'nm'
    })
    data.set_metadata(data_group='raw_data', metadata={
        'x var name': 'time',
        'x var units': 's',
        'y var name': 'PA signal',
        'y var units': 'V/uJ'
    })
    data.set_metadata(data_group='filt_data', metadata={
        'x var name': 'time',
        'x var units': 's',
        'y var name': 'PA signal',
        'y var units': 'V/uJ'
    })

    #main measurement cycle
    for i in range(spectral_points):
        
        #calculate current wavelength
        if abs(step*i) < abs(d_wl):
            current_wl = start_wl + step*i
        else:
            current_wl = end_wl

        # temp vars for averaging
        # should be reset in each sycle
        tmp_signal = 0
        tmp_laser = 0
        counter = 0

        print(f'\n{bcolors.HEADER}'
              + f'Start measuring point {(i+1)}'
              + f'{bcolors.ENDC}')
        print('Current wavelength is'
              + f'{bcolors.OKBLUE}{current_wl}{bcolors.ENDC}.'
              + 'Please set it!')
        
        #adjust laser energy with color glasses
        if power_control == 'Filters':
            print(f'{bcolors.UNDERLINE}'
                  + 'Please remove all filters!'
                  + f'{bcolors.ENDC}')
            #measure mean energy at glass reflection
            energy = track_power(hardware, 50)
            print(f'Power meter energy = {energy:.0f} [uJ]')

            #find valid filters combinations for current parameters
            filters,_,_ = glass_calculator(
                current_wl,
                energy,
                target_energy,
                max_combinations,
                no_print=True)
            if not len(filters):
                print(f'{bcolors.WARNING}'
                      + 'WARNING! No valid filter combination for '
                      + f'{current_wl} [nm]!{bcolors.ENDC}')
                cont_ans = inquirer.confirm(
                    message='Do you want to continue?').execute()
                if not cont_ans:
                    print(f'{bcolors.WARNING}'
                          + 'Spectral measurements terminated!'
                          + f'{bcolors.ENDC}')
                    return old_data

            #call glass_calculator again to print valid filter combinations
            _, target_pm_value,_ = glass_calculator(
                current_wl,
                energy,
                target_energy,
                max_combinations)
            
            print(f'Target power meter energy is {target_pm_value}!')
            print('Please set it using'
                  + f'{bcolors.UNDERLINE}laser software{bcolors.ENDC}')
        
        elif power_control == 'Glan prism':
            #for energy control by Glan prism target power meter
            #energy have to be calculated only once
            if i == 0:
                target_pm_value = glan_calc_reverse(target_energy*1000)
            print(f'Target power meter energy is {target_pm_value}!') #type: ignore
            print(f'Please set it using {bcolors.UNDERLINE}Glan prism{bcolors.ENDC}!')
            _ = track_power(hardware, 50)
        else:
            print(f'{bcolors.WARNING}'
                  + 'Unknown power control method! Measurements terminated!'
                  + f'{bcolors.ENDC}')
            return old_data

        #start measurements and averaging
        while counter < averaging:
            print('Signal at current WL should be measured '
                  + f'{averaging-counter} more times.')
            measure_ans = inquirer.rawlist(
                message='Chose an action:',
                choices=['Tune power','Measure','Stop measurements']
            ).execute()

            #adjust energy
            if measure_ans == 'Tune power':
                track_power(hardware, 40)

            #actual measurement
            elif measure_ans == 'Measure':
                
                #measure data on both channels
                osc.measure()
                dt = 1/osc.sample_rate

                #set signal data depending on channel assignment
                if chan_pa == 'CHAN1':
                    pa_signal = osc.ch1_data
                    pa_amp = osc.ch1_amp
                    pm_signal = osc.ch2_data
                elif chan_pa == 'CHAN2':
                    pa_signal = osc.ch2_data
                    pa_amp = osc.ch2_amp
                    pm_signal = osc.ch1_data
                else:
                    print(f'{bcolors.WARNING}'
                          + 'Uknown channel assignment in spectra'
                          + f'{bcolors.ENDC}')
                    return old_data
                
                #calculate laser energy at power meter
                cur_pm_laser = pm.energy_from_data(
                        pm_signal,
                        1/osc.sample_rate)
                print(f'Start ind = {pm.start_ind}')
                print(f'Stop ind = {pm.stop_ind}')
                #show measured data to visually check if data is OK
                fig = plt.figure(tight_layout=True)
                gs = gridspec.GridSpec(1,2)
                ax_pm = fig.add_subplot(gs[0,0])
                ax_pm.plot(pm_signal)
                #ax_pm.plot(signal.decimate(pm_signal, 100))
                #add markers for data start and stop
                ax_pm.plot(
                    pm.start_ind,
                    pm_signal[pm.start_ind],
                    'o',
                    alpha=0.4,
                    ms=12,
                    color='green')
                ax_pm.plot(
                    pm.stop_ind,
                    pm_signal[pm.stop_ind],
                    'o',
                    alpha=0.4,
                    ms=12,
                    color='red')
                ax_pa = fig.add_subplot(gs[0,1])
                ax_pa.plot(pa_signal)
                plt.show()

                #confirm that the data is OK
                good_data = inquirer.confirm(
                    message='Data looks good?').execute()
                if good_data:
                    #note that tmp_signal is only amplitude of signal for averaging
                    tmp_signal += pa_amp
                    tmp_laser += cur_pm_laser
                    counter += 1
                    if counter == averaging:
                        if power_control == 'Filters':
                            _, ___, sample_energy_aver = glass_calculator(
                                current_wl,
                                tmp_laser/averaging, #average energy
                                target_energy,
                                max_combinations,
                                no_print=True)

                            _, ___, sample_energy_cur = glass_calculator(
                                current_wl,
                                cur_pm_laser, #current energy
                                target_energy,
                                max_combinations,
                                no_print=True)

                            #add datapoint with metadata
                            max_amp = tmp_signal/(
                                averaging*sample_energy_aver)
                            data.add_measurement(
                                pa_signal/sample_energy_cur, #last mes
                                {'parameter value': current_wl,
                                'x var step': dt,
                                'x var start': 0,
                                'x var stop': dt*(len(pa_signal)-1),
                                'PM energy': tmp_laser/averaging,
                                'sample energy': sample_energy_aver,
                                'max amp': max_amp})
                            
                            print(f'{bcolors.OKBLUE}'
                                  + 'Average laser at sample = '
                                  + f'{sample_energy_aver:.0f} [uJ]')
                            print(f'Average PA signal = {max_amp}'
                                  + f'{bcolors.ENDC}')
                            
                        elif power_control == 'Glan prism':
                            #add datapoint with metadata
                            sample_energy_aver = glan_calc(
                                tmp_laser/averaging)
                            max_amp = tmp_signal/(
                                averaging*sample_energy_aver)
                            data.add_measurement(
                                pa_signal/glan_calc(cur_pm_laser), #last mes
                                {'parameter value': current_wl,
                                'x var step': dt,
                                'x var start': 0,
                                'x var stop': dt*(len(pa_signal)-1),
                                'PM energy': tmp_laser/averaging,
                                'sample energy': sample_energy_aver,
                                'max amp': max_amp})
                            
                            print(f'{bcolors.OKBLUE}'
                                  + 'Average laser at sample = '
                                  + f'{sample_energy_aver:.0f} [uJ]')
                            print(f'Average PA signal = {max_amp}'
                                  + f'{bcolors.ENDC}')
                            
                        else:
                            print(f'{bcolors.WARNING}'
                                  + 'Unknown power control method in writing '
                                  + f'laser energy{bcolors.ENDC}')
                        data.save_tmp()
                        
            elif measure_ans == 'Stop measurements':
                confirm = inquirer.confirm(
                    message='Are you sure?'
                ).execute()
                if confirm:
                    print(f'{bcolors.WARNING}'
                          + 'Spectral measurements terminated!'
                          + f'{bcolors.ENDC}')
                    data.bp_filter()
                    return data
            else:
                print(f'{bcolors.WARNING}'
                      + 'Unknown command in Spectral measure menu!'
                      + f'{bcolors.ENDC}')
    
    print(f'{bcolors.OKGREEN}'
          + 'Spectral scanning complete!'
          + f'{bcolors.ENDC}')
    data.bp_filter()
    return data
          
def track_power(hardware: Hardware, tune_width: int) -> float:
    """Build energy graph.
    Return mean energy for last aver=10 measurements"""

    ### config parameters
    #Averaging for mean and std calculations
    aver = 10
    # ignore read if it is smaller than threshold*mean
    threshold = 0.01
    # time delay between measurements in s
    measure_delya = 0.05
    ###

    pm = hardware['power_meter'] #type: ignore
    
    #tune_width cannot be smaller than averaging
    if tune_width < aver:
        print(f'{bcolors.WARNING}'
              + 'Wrong tune_width value!'
              + f'{bcolors.ENDC}')
        return 0
    
    print(f'{bcolors.OKGREEN}'
          + 'Hold q button to stop power measurements'
          + f'{bcolors.ENDC}')
    
    # init arrays for data storage
    data = np.zeros(tune_width)
    tmp_data = np.zeros(tune_width)

    #init plot
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1,2)
    #axis for signal from osc
    ax_pm = fig.add_subplot(gs[0,0])
    #axis for energy graph
    ax_pa = fig.add_subplot(gs[0,1])
    
    #mean energy
    mean = 0

    #measuring loop
    i = 0
    while True:
        laser_amp = pm.get_energy_scr()
        if not laser_amp:
            continue
        if i == 0:
            title = (f'Energy={laser_amp:.1f} [uJ], '
                     + f'Mean (last {aver}) = {laser_amp:.1f} [uJ], '
                     + f'Std (last {aver}) = {data[:i+1].std():.1f} [uJ]')
            data[i] = laser_amp
            mean = laser_amp

        elif i < tune_width:
            if i < aver:
                if laser_amp < threshold*data[:i].mean():
                    continue
                mean = data[:i].mean()
                title = (f'Energy={laser_amp:.1f} [uJ], '
                         + f'Mean (last {aver}) = {mean:.1f} [uJ], '
                         + f'Std (last {aver}) = {data[:i].std():.1f} [uJ]')
            else:
                if laser_amp < threshold*data[i-aver:i].mean():
                    continue
                mean = data[i-aver:i].mean()
                title = (f'Energy={laser_amp:.1f} [uJ], '
                         + f'Mean (last {aver}) = {mean:.1f} [uJ], '
                         + f'Std (last {aver}) = '
                         + f'{data[i-aver:i].std():.1f} [uJ]')
            data[i] = laser_amp
        
        else:
            tmp_data[:-1] = data[1:].copy()
            tmp_data[tune_width-1] = laser_amp
            mean = tmp_data[tune_width-aver:-1].mean()
            title = (f'Energy={laser_amp:.1f} [uJ], '
                     + f'Mean (last {aver}) = {mean:.1f} [uJ], '
                     + f'Std (last {aver}) = '
                     + f'{tmp_data[tune_width-aver:-1].std():.1f} [uJ]')
            if tmp_data[tune_width-1] < threshold*tmp_data[tune_width-aver:-1].mean():
                continue
            data = tmp_data.copy()
        
        #increase index
        i += 1
        #plotting data
        ax_pm.clear()
        ax_pa.clear()
        ax_pm.plot(pm.data)
        #add markers for data start and stop
        ax_pm.plot(
            pm.start_ind,
            pm.data[pm.start_ind],
            'o',
            alpha=0.4,
            ms=12,
            color='green')
        ax_pm.plot(
            pm.stop_ind,
            pm.data[pm.stop_ind],
            'o',
            alpha=0.4,
            ms=12,
            color='red')
        ax_pa.plot(data)
        ax_pa.set_ylabel('Laser energy, [uJ]')
        ax_pa.set_title(title)
        ax_pa.set_ylim(bottom=0)
        fig.canvas.draw()
        plt.pause(0.01)
            
        if keyboard.is_pressed('q'):
            break
        time.sleep(measure_delya)

    return mean

def set_new_position(hardware: Hardware) -> None:
    """Queries new position and move PA detector to this position"""

    if not hardware['stage_x'] or not hardware['stage_y']:
        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

    x_dest = inquirer.text(
        message='Enter X destination [mm] ' + vd.cancel_option,
        default='0.0',
        validate=vd.ScanRangeValidator(),
        mandatory=False
    ).execute()
    if x_dest is None:
        print(f'{bcolors.WARNING} Input terminated! {bcolors.ENDC}')
        return
    x_dest = float(x_dest)
    
    y_dest = inquirer.text(
        message='Enter Y destination [mm] ' + vd.cancel_option,
        default='0.0',
        validate=vd.ScanRangeValidator(),
        mandatory=False
    ).execute()
    if y_dest is None:
        print(f'{bcolors.WARNING} Input terminated! {bcolors.ENDC}')
        return
    y_dest = float(y_dest)

    print(f'Moving to ({x_dest},{y_dest})...')
    move_to(x_dest, y_dest, hardware)
    wait_stages_stop(hardware)
    pos_x = hardware['stage_x'].get_position(scale=True)*1000
    pos_y = hardware['stage_y'].get_position(scale=True)*1000
    print(f'{bcolors.OKGREEN}'
          + f'...Mooving complete!{bcolors.ENDC}'
          + f'Current position ({pos_x:.2f},{pos_y:.2f})')

def remove_zeros(data: np.ndarray) -> np.ndarray:
    """Replaces zeros in filters data by linear fit from nearest values"""

    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            if data[i+1,j+2] == 0:
                if i == 0:
                    if data[i+2,j+2] == 0 or data[i+3,j+2] == 0:
                        print('missing value for the smallest WL cannot be calculated!')
                        return data
                    else:
                        data[i+1,j+2] = 2*data[i+2,j+2] - data[i+3,j+2]
                elif i == data.shape[0]-2:
                    if data[i,j+2] == 0 or data[i-1,j+2] == 0:
                        print('missing value for the smallest WL cannot be calculated!')
                        return data
                    else:
                        data[i+1,j+2] = 2*data[i,j+2] - data[i-1,j+2]
                else:
                    if data[i,j+2] == 0 or data[i+2,j+2] == 0:
                        print('adjacent zeros in filter data are not supported!')
                        return data
                    else:
                        data[i+1,j+2] = (data[i,j+2] + data[i+2,j+2])/2
    return data

def calc_od(data: np.ndarray) -> np.ndarray:
    """calculates OD using thickness of filters"""

    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            data[i+1,j+2] = data[i+1,j+2]*data[0,j+2]
    return data

def glass_calculator(wavelength: int,
                     current_energy_pm: float,
                     target_energy: float,
                     max_combinations: int,
                     no_print: bool=False
                     ) -> tuple[dict,float,float]:
    """Return a dict with filter combinations, 
    having the closest transmissions,
    which are higher than required but not more than 2.5 folds higher.
    Also returns energy at glass reflection, which will correspond
    to the required energy at sample and current energy at sample.
    Accept only wavelengthes, which are present in 'ColorGlass.txt'
    Units:
    wavelength [nm];
    current_energy_pm [uJ];
    target_energy [uJ]"""

    #dict for storing found valid combinations
    result = {}
    #file with filter's properties
    filename = 'ColorGlass.txt'

    try:
        data = np.loadtxt(filename,skiprows=1)
        header = open(filename).readline()
    except FileNotFoundError:
        print(f'{bcolors.WARNING}'
              + 'File with color glass data not found!'
              + f'{bcolors.ENDC}')
        return ({},0,0)
    
    except ValueError as er:
        print(f'Error message: {str(er)}')
        print(f'{bcolors.WARNING}'
              + 'Error while loading color glass data!'
              + f'{bcolors.ENDC}')
        return ({},0,0)
    
    data = remove_zeros(data)
    data = calc_od(data)
    filter_titles = header.split('\n')[0].split('\t')[2:]

    #find index of wavelength
    try:
        wl_index = np.where(data[1:,0] == wavelength)[0][0] + 1
    except IndexError:
        print(f'{bcolors.WARNING}'
              + 'Target WL is missing in color glass data table!'
              + f'{bcolors.ENDC}')
        return ({},0,0)

    #{filter_name: OD at wavelength} for all filters
    filter_dict = {}
    for key, value in zip(filter_titles,data[wl_index,2:]):
        filter_dict.update({key:value})

    #build a dict with all possible combinations of filters
    #and convert OD to transmission for the combinations
    filter_combinations = {}
    for i in range(max_combinations):
        for comb in combinations(filter_dict.items(),i+1):
            key = ''
            value = 0
            for k,v in comb:
                key +=k
                value+=v
            filter_combinations.update({key:math.pow(10,-value)})    

    # calculated laser energy at sample
    laser_energy = current_energy_pm/data[wl_index,1]*100

    if laser_energy == 0:
        print(f'{bcolors.WARNING} Laser radiation is not detected!{bcolors.ENDC}')
        return ({},0,0)
    
    #required total transmission of filters
    target_transm = target_energy/laser_energy
    
    if not no_print:
        print(f'Target energy = {target_energy} [uJ]')
        print(f'Current laser output = {laser_energy:.0f} [uJ]')
        print(f'Target transmission = {target_transm*100:.1f} %')
        print(f'{bcolors.HEADER} Valid filter combinations:{bcolors.ENDC}')
    
    #sort filter combinations by transmission
    filter_combinations = dict(sorted(
        filter_combinations.copy().items(),
        key=lambda item: item[1]))

    #build dict with filter combinations,
    #which have transmissions higher that required,
    #but not more than 2.5 folds higher
    i=0
    for key, value in filter_combinations.items():
        if (value-target_transm) > 0 and value/target_transm < 2.5:
            result.update({key: value})
            #print up to 5 best combinations with color code
            if not no_print:
                if (value/target_transm < 1.25) and i<5:
                    print(f'{bcolors.OKGREEN} {key}, transmission = {value*100:.1f}%{bcolors.ENDC} (target= {target_transm*100:.1f}%)')
                elif (value/target_transm < 1.5) and i<5:
                    print(f'{bcolors.OKCYAN} {key}, transmission = {value*100:.1f}%{bcolors.ENDC} (target= {target_transm*100:.1f}%)')
                elif value/target_transm < 2 and i<5:
                    print(f'{bcolors.OKBLUE} {key}, transmission = {value*100:.1f}%{bcolors.ENDC} (target= {target_transm*100:.1f}%)')
                elif value/target_transm < 2.5 and i<5:
                    print(f'{bcolors.WARNING} {key}, transmission = {value*100:.1f}%{bcolors.ENDC} (target= {target_transm*100:.1f}%)')
            i+=1
    
    if not no_print:
        print('\n')

    # energy at glass reflection, which correspond to 
    #the required energy at sample
    target_pm_value = target_energy*data[wl_index,1]/100

    return result, target_pm_value, laser_energy

def calc_filters_for_energy(hardware: Hardware) -> None:
    """Provides required filter combination for an energy"""

    #max filters for calculation
    max_combinations = 3

    wl = inquirer.text(
        message='Set wavelength, [nm]' + vd.cancel_option,
        default='750',
        mandatory=False,
        validate=vd.WavelengthValidator()
    ).execute()
    if wl == None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return
    else:
        wl = int(wl)

    target_energy = inquirer.text(
        message='Set target energy in [mJ]' + vd.cancel_option,
        default='0.5',
        mandatory=False,
        validate=vd.EnergyValidator()
    ).execute()
    if target_energy == None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return
    else:
        target_energy = float(target_energy)*1000

    print(f'{bcolors.UNDERLINE}Please remove all filters!{bcolors.ENDC}')
    energy = track_power(hardware, 50)
    print(f'Power meter energy = {energy:.0f} [uJ]')
    filters, _, _ = glass_calculator(wl,
                                  energy,
                                  target_energy,
                                  max_combinations,
                                  no_print=True)
    if len(filters):
        print(f'{bcolors.WARNING}'
              + 'WARNING! Valid filter combination were not found!'
              + f'{bcolors.ENDC}')

    _, target_pm_value, _ = glass_calculator(
        wl,
        energy,
        target_energy,
        max_combinations)
    print(f'Target power meter energy is {target_pm_value}!')
    print(f'Please set it using {bcolors.UNDERLINE}'
          + f'laser software{bcolors.ENDC}')

def glan_calc(energy: float) -> float:
    """Calculates energy at sample for a given energy"""

    filename = 'GlanCalibr.txt' # file with Glan calibrations
    fit_order = 1 #order of the polynom for fitting data

    try:
        calibr_data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f'{bcolors.WARNING} File with color glass data not found!{bcolors.ENDC}')
        return 0
    except ValueError as er:
        print(f'Error message: {str(er)}')
        print(f'{bcolors.WARNING} Error while loading color glass data!{bcolors.ENDC}')
        return 0

    #get coefficients which fit calibration data with fit_order polynom
    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    #init polynom with the coefficients
    fit = np.poly1d(coef)

    #return the value of polynom at energy
    return fit(energy)

def glan_calc_reverse(target_energy: float) -> float:
    """Calculates energy at power meter placed 
    at glass reflection to obtain target_energy"""

    filename = 'GlanCalibr.txt' # file with Glan calibrations
    fit_order = 1 #order of the polynom for fitting data

    try:
        calibr_data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f'{bcolors.WARNING}'
              + 'File with color glass data not found!'
              + f'{bcolors.ENDC}')
        return 0
    except ValueError as er:
        print(f'Error message: {str(er)}')
        print(f'{bcolors.WARNING}'
              + 'Error while loading color glass data!'
              + f'{bcolors.ENDC}')
        return 0

    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    if fit_order == 1:
        # target_energy = coef[0]*energy + coef[1]
        energy = (target_energy - coef[1])/coef[0]
    else:
        print(f'{bcolors.WARNING}'
              + 'Reverse Glan calculation for nonlinear fit is not realized!'
              + f'{bcolors.ENDC}')
        return 0
    
    return energy

def glan_check(hardware: Hardware) -> None:
    """Used to check glan performance"""

    # [uJ] maximum value, which does not damage PM
    damage_threshold = 800 

    print(f'{bcolors.HEADER}'
          + 'Start procedure to check Glan performance'
          + f'{bcolors.ENDC}')
    print(f'Do not try energies at sample large than'
          + f'{bcolors.UNDERLINE} 800 uJ {bcolors.ENDC}!')
    
    while True:
        print(f'\nSet some energy at glass reflection')
        energy = track_power(hardware, 50)
        target_energy = glan_calc(energy)
        if target_energy > damage_threshold:
            print(f'{bcolors.WARNING}'
                  + 'Energy at sample will damage the PM, set smaller energy!'
                  + f'{bcolors.ENDC}')
            continue
        print(f'Energy at sample should be ~{target_energy} uJ. Check it!')
        track_power(hardware,50)

        option = inquirer.rawlist(
            message='Choose an action:',
            choices=[
                'Measure again',
                'Back'
            ]
        ).execute()

        if option == 'Measure again':
            continue
        elif option == 'Back':
            break
        else:
            print(f'{bcolors.WARNING}'
                  + 'Unknown command in Glan chack menu!'
                  + f'{bcolors.ENDC}')

def export_to_txt(data: MeasuredData) -> None:
    """CLI method for export data to txt"""

    export_type = inquirer.rawlist(
        message='Choose data to export:',
        choices=[
            'Raw data',
            'Filtered data',
            'Freq data',
            'Spectral',
            'back'
        ]
    ).execute()
    
    if export_type == 'back':
        return
    
    elif export_type in ('Raw data', 'Filtered data', 'Freq data'):
        #build full filename
        filename = data.attrs['path'] + data.attrs['filename']
        if not filename:
            filename = 'measuring results/txt data/Spectral-Unknown'
        else:
            filename = filename.split('.hdf5')[0]
        if export_type == 'Raw data':
            filename += '-raw.txt'
        elif export_type == 'Filtered data':
            filename += '-filt.txt'
        elif export_type == 'Freq data':
            filename += '-freq.txt'

        #if file already exists, ask to override it
        if os.path.exists(filename):
            override = inquirer.confirm(
                message=f'Do you want to override file {filename}?'
            ).execute()
            if override:
                try:
                    os.remove(filename)
                except OSError:
                    pass
            else:
                filename_tmp = filename.split('.txt')[0]
                i = 1
                while os.path.exists(filename_tmp + str(i) + '.txt'):
                    i += 1
                filename = filename_tmp + str(i) + '.txt'
        
        #build file header; h1,h2,h3 lines of header
        h1 = ''
        h2 = ''
        h3 = ''
        ds_data = np.empty(1)

        #build arra for txt data
        if export_type in ('Raw data', 'Filtered data'):

            ds_name = ''
            if export_type == 'Raw data':
                ds_name = 'raw_data'
            elif export_type == 'Filtered data':
                ds_name = 'filt_data'
            # 1D array with values of parameter
            param_vals = np.array(
                data.get_dependance(ds_name,'parameter value'))
            #1D array with values of laser at sample
            laser_vals = np.array(
                data.get_dependance(ds_name, 'sample energy'))
            #2D array with Y values of datasets for each datapoint
            y_data = np.array(data.get_dependance(ds_name,'data')).T
            # filt data will be used to find signal position
            y_data_filt = np.array(data.get_dependance('filt_data','data')).T

            #build 2D array with dataset in format (X,Y) for all data points
            start_vals = np.array(
                data.get_dependance(ds_name, 'x var start'))
            step_vals = np.array(
                data.get_dependance(ds_name, 'x var step'))
            stop_vals = np.array(
                data.get_dependance(ds_name, 'x var stop'))
            
            #init array with NaNs, which will not be saved to the file
            ds_data = np.empty((y_data.shape[0] + 2, 2*y_data.shape[1]))
            ds_data[:] = np.nan

            #loop for filling data
            for start, stop, step, col in zip(start_vals,
                                            stop_vals,
                                            step_vals,
                                            range(len(param_vals))):
                #find indexes of signal within pre_time:post_time limits
                filt = y_data_filt[:,col]
                filt_max = np.amax(filt)
                #check if dataset is empty
                if stop:
                    #convert pre and post time to points
                    pre_points = int(data.attrs['zoom pre time']/step)
                    post_points = int(data.attrs['zoom post time']/step)
                    
                    #index of maximum of filt data
                    max_ind = 0
                    #check if there is some non zero data in filt
                    if filt_max:
                        max_ind = np.argwhere(filt == filt_max)[0][0]
                    #check not go outside of dataset boundaries
                    if pre_points > max_ind:
                        pre_points = max_ind
                    if (post_points + max_ind) > len(filt):
                        post_points = len(filt) - max_ind
                    
                    zoom = pre_points + post_points
                    start_ind = max_ind - pre_points
                    stop_ind = max_ind + post_points

                    #add (X,Y) data of the dataset and fill headers lines
                    x_vals = np.arange(start, stop, step)
                    ds_data[2:zoom+2,2*col] = x_vals[start_ind:stop_ind].T
                    h1 += data.raw_data['attrs']['x var name'] + ';'
                    h2 += data.raw_data['attrs']['x var units'] + ';'
                    ds_data[2:zoom+2,2*col+1] = y_data[start_ind:stop_ind,col]
                    h1 += data.raw_data['attrs']['y var name'] + ';'
                    h2 += data.raw_data['attrs']['y var units'] + ';'

                #add parameter and laser values
                ds_data[0,2*col:2*col+2] = param_vals[col]
                ds_data[1,2*col:2*col+2] = laser_vals[col]

            #build last line of header
            param = data.attrs['parameter name']
            param_units = data.attrs['parameter units']
            h3 = (f'First line is {param} in [{param_units}].'
                + 'Second line is laser energy in [uJ]')
    
        if export_type == 'Freq data':

            ds_name = 'freq_data'
            # 1D array with values of parameter
            param_vals = np.array(
                data.get_dependance(ds_name,'parameter value'))
            #2D array with Y values of datasets for each datapoint
            y_data = np.array(data.get_dependance(ds_name,'data')).T

            #build 2D array with dataset in format (X,Y) for all data points
            start_vals = np.array(
                data.get_dependance(ds_name, 'x var start'))
            step_vals = np.array(
                data.get_dependance(ds_name, 'x var step'))
            stop_vals = np.array(
                data.get_dependance(ds_name, 'x var stop'))
            
            #init array with NaNs, which will not be saved to the file
            ds_data = np.empty((y_data.shape[0] + 1, 2*y_data.shape[1]))
            ds_data[:] = np.nan

            #loop for filling data
            for start, stop, step, col in zip(start_vals,
                                            stop_vals,
                                            step_vals,
                                            range(len(param_vals))):

                #check if dataset is empty
                if stop:
                    #add (X,Y) data of the dataset and fill headers lines
                    x_vals = np.arange(start, stop, step)
                    ds_data[1:,2*col] = x_vals.T
                    h1 += data.freq_data['attrs']['x var name'] + ';'
                    h2 += data.freq_data['attrs']['x var units'] + ';'
                    ds_data[1:,2*col+1] = y_data[:,col]
                    h1 += data.freq_data['attrs']['y var name'] + ';'
                    h2 += data.freq_data['attrs']['y var units'] + ';'

                #add parameter values
                ds_data[0,2*col:2*col+2] = param_vals[col]

            #build last line of header
            param = data.attrs['parameter name']
            param_units = data.attrs['parameter units']
            h3 = (f'First line is {param} in [{param_units}].')
        
        header = h1 + '\n' + h2 + '\n' + h3
        #save the data to the file
        np.savetxt(
            filename,
            ds_data,
            header=header,
            delimiter=';')
        print(f'Data exported to'
              + f'{bcolors.OKGREEN}{filename}{bcolors.ENDC}')

    elif export_type == 'Spectral':
        
        #build full filename
        filename = data.attrs['path'] + data.attrs['filename']
        if not filename:
            filename = 'measuring results/txt data/Spectral-Unknown'
        else:
            filename = filename.split('.hdf5')[0]
        filename += '-spectral.txt'

        #if file already exists, ask to override it
        if os.path.exists(filename):
            override = inquirer.confirm(
                message=f'Do you want to override file {filename}?'
            ).execute()
            if override:
                try:
                    os.remove(filename)
                except OSError:
                    pass
            else:
                filename_tmp = filename.split('.txt')[0]
                i = 1
                while os.path.exists(filename_tmp + str(i) + '.txt'):
                    i += 1
                filename = filename_tmp + str(i) + '.txt'
        
        #build data array and header
        param_vals = np.array(data.get_dependance(
            'raw_data',
            'parameter value'))
        h1 = data.attrs['parameter name'] + ';'
        h2 = data.attrs['parameter units'] + ';'

        laser_vals = np.array(data.get_dependance(
            'raw_data',
            'sample energy'))
        h1 += 'Laser;'
        h2 += 'uJ;'

        raw_amps = np.array(data.get_dependance(
            'raw_data',
            'max amp'))
        h1 += data.raw_data['attrs']['y var name'] + ';'
        h2 += data.raw_data['attrs']['y var units'] + ';'
        
        filt_amps = np.array(data.get_dependance(
            'filt_data',
            'max amp'))
        h1 += data.filt_data['attrs']['y var name'] + ';'
        h2 += data.filt_data['attrs']['y var units'] + ';'

        ds_data = np.column_stack((
            param_vals,
            laser_vals,
            raw_amps,
            filt_amps))
        header = h1 + '\n' + h2

        #save the data to the file
        np.savetxt(
            filename,
            ds_data,
            header=header,
            delimiter=';')
        print('Data exported to'
              + f'{bcolors.OKGREEN}{filename}{bcolors.ENDC}')

    else:
        print(f'{bcolors.WARNING} Unknown command in data export menu {bcolors.ENDC}')

if __name__ == "__main__":
    
    #dict for keeping references to hardware
    hardware: Hardware = {
        'stage_x': 0,
        'stage_y': 0,
        'osc': oscilloscope.Oscilloscope()
    }

    # init class for data storage
    data = MeasuredData()

    while True: #main execution loop
        menu_ans = inquirer.rawlist(
            message='Choose an action',
            choices=[
                'Init and status',
                'Power meter',
                'Energy',
                'Move to',
                'Stage scanning',
                'Spectral scanning',
                'Exit'
            ],
            height=9
        ).execute()

        if menu_ans == 'Init and status':
            while True:
                stat_ans = inquirer.rawlist(
                message='Choose an action',
                choices=[
                    'Init hardware',
                    'Get status',
                    'Home stages',
                    'Back'
                ]
            ).execute()

                if stat_ans == 'Init hardware':
                    init_hardware(hardware)

                elif stat_ans == 'Home stages':
                    home(hardware)

                elif stat_ans == 'Get status':
                    print_status(hardware)

                elif stat_ans == 'Back':
                    break

        elif menu_ans == 'Power meter':
            _ = track_power(hardware, 100)

        elif menu_ans == 'Energy':
            while True:
                energy_menu = inquirer.rawlist(
                    message='Choose an option',
                    choices = [
                        'Glan check',
                        'Filter caclulation',
                        'Back'
                    ]
                ).execute()
                if energy_menu == 'Glan check':
                    glan_check(hardware)
                elif energy_menu == 'Filter caclulation':
                    calc_filters_for_energy(hardware)
                elif energy_menu == 'Back':
                    break
                else:
                    print(f'{bcolors.WARNING}'
                          + 'Unknown command in energy menu!'
                          + f'{bcolors.ENDC}')

        elif menu_ans == 'Move to':
            set_new_position(hardware)

        elif menu_ans == 'Stage scanning':
            while True:
                data_ans = inquirer.rawlist(
                    message='Choose scan action',
                    choices=[
                        'Scan',
                        'View data', 
                        'FFT filtration',
                        'Load data',
                        'Save data',
                        'Back to main menu'
                    ]
                ).execute()
                
                if data_ans == 'Scan':
                    scan_image, scan_data, dt = scan(hardware)

                elif data_ans == 'View data':
                    print(f'{bcolors.WARNING}'
                          + 'Scan data view is not implemented!'
                          + f'{bcolors.ENDC}')

                elif data_ans == 'FFT filtration':
                    print(f'{bcolors.WARNING}'
                          + 'FFT of scan data is not implemented!'
                          + f'{bcolors.ENDC}')

                elif data_ans == 'Save data':
                    print(f'{bcolors.WARNING}'
                          + 'Save scan data is not implemented!'
                          + f'{bcolors.ENDC}')

                elif data_ans == 'Load data':
                    print(f'{bcolors.WARNING}'
                          + 'Load scan data is not implemented!'
                          + f'{bcolors.ENDC}')

                elif data_ans == 'Back to main menu':
                        break
                
                else:
                    print(f'{bcolors.WARNING}'
                          + 'Unknown option is stage scanning menu!'
                          + f'{bcolors.ENDC}')

        elif menu_ans == 'Spectral scanning':
            while True:
                data_ans = inquirer.rawlist(
                    message='Choose spectral data action',
                    choices=[
                        'Measure spectrum',
                        'View data', 
                        'FFT filtration',
                        'Load data',
                        'Save data',
                        'Export to txt',
                        'Back to main menu'
                    ]
                ).execute()
                
                if data_ans == 'Measure spectrum':
                    data = spectra(hardware, data)  

                elif data_ans == 'View data':
                    data.plot()

                elif data_ans == 'FFT filtration':
                    bp_filter(data)

                elif data_ans == 'Save data':
                    save_data(data)

                elif data_ans == 'Export to txt':
                    export_to_txt(data)

                elif data_ans == 'Load data':
                    data = load_data(data)

                elif data_ans == 'Back to main menu':
                        break         

                else:
                    print(f'{bcolors.WARNING}Unknown command in Spectral scanning menu!{bcolors.ENDC}')

        elif menu_ans == 'Exit':
            exit_ans = inquirer.confirm(
                message='Do you really want to exit?'
                ).execute()
            if exit_ans:
                if hardware['stage_x']:
                    hardware['stage_x'].close()
                if hardware['stage_y']:
                    hardware['stage_y'].close()
                exit()

        else:
            print(f'{bcolors.WARNING}Unknown action in the main menu!{bcolors.ENDC}')