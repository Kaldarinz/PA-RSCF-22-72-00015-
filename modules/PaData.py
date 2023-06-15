"""
Operations with PA data

Data structure:
    
    Measured data:
    |--'attrs'
    |  |--'parameter name': str - independent parameter, changed between measured PA signals
    |  |--'parameter units': str - parameters units
    |  |--'data points': int - amount of stored PA signals
    |  |--'created': timestamp - date and time of data measurement
    |  |--'updated': timestamp - date and time of last data update
    |  |--'path': str - relative path to the datafile.
    |  |--'filename': str - name of the file where data is stored
    |  |--'zoom pre time': float - start time from the center of the data frame for zoom in data view
    |  |--'zoom post time': float - end time from the center of the data frame for zoom in data view
    |  |--'zoom units': str - units for pre and post zoom time
    |
    |--'raw_data'
    |  |--'attrs'
    |  |  |--'max dataset len': int - amount of points in PA signal
    |  |  |--'x var name': str - name of the X variable in PA signal
    |  |  |--'x var units': str - units of the X variable
    |  |  |--'y var name': str - name of the Y variable in PA signal
    |  |  |--'y var units': str - name of the X variable
    |  |
    |  |--point1
    |  |  |--'data': ndarray - measured PA signal
    |  |  |--'parameter value': int - value of independent parameter
    |  |  |--'x var step': float
    |  |  |--'x var start': float
    |  |  |--'x var stop': float
    |  |  |--'PM energy': float - laser energy in [uJ] measured by power meter in glass reflection
    |  |  |--'sample energy': float - laser energy at sample in [uJ]
    |  |  |--'max amp': float - (y_max - y_min)
    |  |
    |  |--point2
    |  |  |--'data': ndarray - measured PA signal
    |  |  ...
    |  ...
    |
    |--'filt_data'
    |  |--'attrs'
    |  |  |--'max dataset len': int - amount of points in PA signal
    |  |  |--'x var name': str - name of the X variable in PA signal
    |  |  |--'x var units': str - units of the X variable
    |  |  |--'y var name': str - name of the Y variable in PA signal
    |  |  |--'y var units': str - name of the X variable
    |  |
    |  |--point1
    |  |  |--'data': ndarray - filtered PA signal
    |  |  |--'parameter value': int - value of independent parameter
    |  |  |--'x var step': float
    |  |  |--'x var start': float
    |  |  |--'x var stop': float
    |  |  |--'PM energy': float - laser energy in [uJ] measured by power meter in glass reflection
    |  |  |--'sample energy': float - laser energy at sample in [uJ]
    |  |  |--'max amp': float - (y_max - y_min)
    |  |
    |  |--point2
    |  |  |--'data': ndarray - measured PA signal
    |  |  ...
    |  ...
    |
    |--'freq_data'
    |  |--'attrs'
    |  |  |--'max dataset len': int - amount of frequency data points
    |  |  |--'x var name': str - name of the X variable
    |  |  |--'x var units': str - units of the X variable
    |  |  |--'y var name': str - name of the Y variable
    |  |  |--'y var units': str - name of the X variable
    |  |
    |  |--point1
    |  |  |--'data': ndarray - frequncies present in filt_data
    |  |  |--'parameter value': int - value of independent parameter
    |  |  |--'x var step': float
    |  |  |--'x var start': float
    |  |  |--'x var stop': float
    |  |  |--'PM energy': float - laser energy in [uJ] measured by power meter in glass reflection
    |  |  |--'sample energy': float - laser energy at sample in [uJ]
    |  |  |--'max amp': float - (y_max - y_min)
    |  |
    |  |--point2
    |  |  |--'data': ndarray - measured PA signal
    |  |  ...
    |  ...
"""
import warnings
from typing import Iterable
import numpy as np
from datetime import datetime
import h5py
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import MatplotlibDeprecationWarning # type: ignore
from scipy.fftpack import rfft, irfft, fftfreq

from modules.bcolors import bcolors

class PaData:
    """Class for PA data storage and manipulations"""

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
            print(f'{bcolors.WARNING}\
                  Unknown data_group for metadata!\
                  {bcolors.ENDC}')

    def add_measurement(self, data: np.ndarray, attributes: dict) -> None:
        """Adds a datapoint to raw_data,
        also adds empty datapoint to other data groups.
        attributes are set to both raw and filt datapoints"""

        ds_name = self.build_ds_name(self.attrs['data points'])
        if not ds_name:
            print(f'{bcolors.WARNING}\
                  Max data points reached! Data cannot be added!\
                  {bcolors.ENDC}')
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
            print(f'{bcolors.WARNING}\
                  Filename is not set. Data cannot be saved!\
                  {bcolors.ENDC}')
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
            print(f'{bcolors.WARNING}\
                  No data to display\
                  {bcolors.ENDC}')

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
        x_label = self.attrs['parameter name']\
                  + ', ['\
                  + self.attrs['parameter units']\
                  + ']'
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
        title = self.attrs['parameter name']\
                + ': '\
                + str(int(self._param_values[self._param_ind]))\
                + self.attrs['parameter units']
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
        x_label = self.filt_data['attrs']['x var name'] +\
            ', [' + self.filt_data['attrs']['x var units']+']'
        self._ax_filt.set_xlabel(x_label)
        self._ax_filt_zoom.set_xlabel(x_label)

        y_label = self.filt_data['attrs']['y var name'] +\
            ', [' + self.filt_data['attrs']['y var units']+']'
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
            print(f'{bcolors.WARNING}\
                  Zoom units do not match x var units!\
                  {bcolors.ENDC}')
            
        #update raw data
        self._ax_raw.clear()
        self._ax_raw.plot(time_points,
                         self.raw_data[ds_name]['data'])
        x_label = self.raw_data['attrs']['x var name'] +\
            ', [' + self.raw_data['attrs']['x var units']+']'
        self._ax_raw.set_xlabel(x_label)
        self._ax_raw_zoom.set_xlabel(x_label)

        y_label = self.raw_data['attrs']['y var name'] +\
            ', [' + self.raw_data['attrs']['y var units']+']'
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
        x_label = self.freq_data['attrs']['x var name'] +\
            ', [' + self.freq_data['attrs']['x var units']+']'
        self._ax_freq.set_xlabel(x_label)

        y_label = self.freq_data['attrs']['y var name'] +\
            ', [' + self.freq_data['attrs']['y var units']+']'
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
            print(f'{bcolors.WARNING}\
                  Attempt to read dependence from empty data\
                  {bcolors.ENDC}')
            return []
        
        if data_group == 'raw_data':
            ds_name = self.build_ds_name(0)
            if self.raw_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from RAW_data\
                    {bcolors.ENDC}')
                return []
            for ds_name, ds in self.raw_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])

        
        elif data_group == 'filt_data':
            ds_name = self.build_ds_name(0)
            if self.filt_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from FILT_data\
                    {bcolors.ENDC}')
                return []
            for ds_name, ds in self.filt_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])
        
        elif data_group == 'freq_data':
            ds_name = self.build_ds_name(0)
            if self.freq_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from FREQ_data\
                    {bcolors.ENDC}')
                return []
            for ds_name, ds in self.freq_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])
        
        else:
            print(f'{bcolors.WARNING}\
                  Attempt to read dependence from unknow GROUP\
                  {bcolors.ENDC}')
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
                    print(f'{bcolors.WARNING}\
                          FFT conversion require x var step in seconds\
                          {bcolors.ENDC}')
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
