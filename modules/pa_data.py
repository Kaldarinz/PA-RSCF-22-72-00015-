"""
Operations with PA data.
Requires Python > 3.10

Workflow with PaData:
1. init class instance.
2. Set 'measurement_dims' and 'parameter_name' in root metadata.
3. Add data points.

Data structure of PaData class:
    
    PaData:
    |--'attrs'
    |  |--'version': float - version data structure
    |  |--'measurement_dims': int - dimensionality of the stored measurement
    |  |--'parameter_name': str - independent parameter, changed between measured PA signals
    |  |--'parameter_units': str - parameters units
    |  |--'data_points': int - amount of stored PA measurements
    |  |--'created': timestamp - date and time of data measurement
    |  |--'updated': timestamp - date and time of last data update
    |  |--'filename': os.PathLike - full path to the data file
    |  |--'zoom_pre time': float - start time from the center of the PA data frame for zoom in data view
    |  |--'zoom_post time': float - end time from the center of the PA data frame for zoom in data view
    |  |--'zoom_units': str - units for pre and post zoom time
    |
    |--'raw_data'
    |  |--'attrs'
    |  |  |--'max_len': int - maximum amount of points in PA signal
    |  |  |--'x_var_name': str - name of the X variable in PA signal
    |  |  |--'x_var_units': str - units of the X variable
    |  |  |--'y_var_name': str - name of the Y variable in PA signal
    |  |  |--'y_var_units': str - name of the Y variable
    |  |
    |  |--point1
    |  |  |--'data': ndarray[uint8] - measured PA signal
    |  |  |--'param_val': ndarray - value of independent parameter
    |  |  |--'a': float
    |  |  |--'b': float - y = a*x+b, where 'x' - values from 'data', 'y' - values in 'y var units' scale
    |  |  |--'x_var_step': float
    |  |  |--'x_var_start': float
    |  |  |--'x_var_stop': float
    |  |  |--'pm_en': float - laser energy measured by power meter in glass reflection
    |  |  |--'sample_energy': float - laser energy at sample in [uJ]
    |  |  |--'max_amp': float - (y_max - y_min)
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
    |  |  |--'y var units': str - name of the Y variable
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
    |  |  |--'max dataset len': int - amount of frequency data_points
    |  |  |--'x var name': str - name of the X variable
    |  |  |--'x var units': str - units of the X variable
    |  |  |--'y var name': str - name of the Y variable
    |  |  |--'y var units': str - name of the Y variable
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
from typing import Iterable, Any, TypedDict, List
from datetime import datetime
import time
import os, os.path
import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import MatplotlibDeprecationWarning # type: ignore
from scipy.fftpack import rfft, irfft, fftfreq
import pint
import h5py
import numpy as np
import numpy.typing as npt

from .pa_logic import Data_point
from . import ureg

logger = logging.getLogger(__name__)

class BaseMetadata(TypedDict):
    """Typed dict for general metadata."""

    version: float
    measurement_dims: int
    parameter_name: List[str]
    data_points: int
    created: str
    updated: str
    filename: str
    zoom_pre_time: pint.Quantity
    zoom_post_time: pint.Quantity

class RawMetadata(TypedDict):
    """Typed dict for raw_data metadata."""

    max_len: int
    x_var_name: str
    y_var_name: str

class FiltMetadata(TypedDict):
    """Typed dict for filt_data metadata."""

    x_var_name: str
    y_var_name: str

class RawData(TypedDict):
    """Typed dict for a data point."""

    data: List[pint.Quantity]
    data_raw: npt.NDArray[np.uint8]
    param_val: List[pint.Quantity]
    x_var_step: pint.Quantity
    x_var_start: pint.Quantity
    x_var_stop: pint.Quantity
    pm_en: pint.Quantity
    sample_en: pint.Quantity
    max_amp: pint.Quantity

class FreqData(TypedDict):
    """Typed dict for a frequency data."""

    data: List[pint.Quantity]
    x_var_step: pint.Quantity
    x_var_start: pint.Quantity
    x_var_stop: pint.Quantity
    max_amp: pint.Quantity

class PaData:
    """Class for PA data storage and manipulations"""

    def __init__(self) -> None:

        #general metadata
        self.attrs: BaseMetadata = {
            'version': 1.0,
            'measurement_dims': -1,
            'parameter_name': [],
            'data_points': 0,
            'created': self._get_cur_time(),
            'updated': self._get_cur_time(),
            'filename': '',
            'zoom_pre_time': 2*ureg.us,
            'zoom_post_time': 13*ureg.us
        }
        raw_attrs: RawMetadata = {
            'max_len': 0,
            'x_var_name': 'time',
            'y_var_name': 'PhotoAcoustic signal'
        }
        self.raw_data = {}
        self.raw_data.update({'attrs': raw_attrs})
        
        filt_attrs: FiltMetadata = {
            'x_var_name': 'time',
            'y_var_name': 'Filtered photoAcoustic signal'
        }

        self.filt_data = {}
        self.filt_data.update({'attrs': filt_attrs})
        
        freq_attrs: RawMetadata = {
            'max_len': 0,
            'x_var_name': 'Frequency',
            'y_var_name': 'FFT amplitude'
        }
        self.freq_data = {}
        self.freq_data.update({'attrs': freq_attrs})
        
        logger.debug('PaData instance created')

    def set_metadata(self, data_group: str, metadata: dict) -> None:
        """Set attributes for the <data_group>.
        
        <data_group> is 'general'|'raw_data'|'filt_data'|'freq_data'.
        """

        logger.debug(f'Updating metadata for {data_group} in '
                     + f'{self.attrs["filename"]}')
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
            logger.warning('Unknown data_group for metadata!')

    def add_measurement(
            self, 
            data: Data_point,
            param_val: List[pint.Quantity]
        ) -> None:
        """Add a single data point.
        
        Add a datapoint to raw_data, filt_data and freq_data.
        """

        ds_name = self._build_ds_name(self.attrs['data_points'])
        if not ds_name:
            logger.error('Max data_points reached! Data cannot be added!')
            return
        logger.debug(f'Adding {ds_name}...')
        ds: RawData = {
            'data': data['pa_signal'],
            'data_raw': data['pa_signal_raw'],
            'param_val': param_val,
            'x_var_step': data['dt'],
            'x_var_start': data['start_time'],
            'x_var_stop': data['stop_time'],
            'pm_en': data['pm_energy'],
            'sample_en': data['sample_energy'],
            'max_amp': data['max_amp']
        }

        cur_data_len = len(ds['data'])
        old_data_len = self.raw_data['attrs']['max_len']
        if cur_data_len > old_data_len:
            self.raw_data['attrs']['max_len'] = cur_data_len
            logger.debug(f'max_len updated from {old_data_len} '
                         + f'to {cur_data_len}')
        self.raw_data.update({ds_name: ds})
        self.bp_filter(ds_name=ds_name)
        
        self.attrs['data_points'] += 1
        self.attrs['updated'] = self._get_cur_time()

    def _get_cur_time (self) -> str:
        """returns timestamp of current time"""
        
        cur_time = time.time()
        date_time = datetime.fromtimestamp(cur_time)
        date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

        return date_time

    def save(self, filename: os.PathLike='') -> None:
        """Saves data to file"""

        if filename:
            path, filename = os.path.split(filename)
            self.attrs['file_path'] = path
            self.attrs['filename'] = filename

        elif self.attrs['filename'] and self.attrs['file_path']:
            path = self.attrs['file_path']
            name = self.attrs['filename']
            filename = os.path.join(path, name)
        
        else:
            logger.warning('Filename is not set. Data cannot be saved!')
            return
        
        self._flush(filename)

    def save_tmp(self) -> None:
        """saves current data to TmpData.hdf5"""

        Path('measuring results/').mkdir(parents=True, exist_ok=True)
        filename = 'measuring results/TmpData.hdf5'
        self._flush(filename)

    def _flush(self, filename: os.PathLike) -> None:
        """Write data to disk."""

        logger.debug(f'Start friting data to {filename}')
        with h5py.File(filename, 'w') as file:
            file.attrs.update(self.attrs)
            raw_data = file.create_group('raw data')
            raw_data.attrs.update(self.raw_data['attrs'])
            for key, value in self.raw_data.items():
                if key !='attrs':
                    ds_raw = raw_data.create_dataset(key, data=value['data'])
                    for attr_name, attr_value in value.items():
                        if attr_name != 'data':
                            ds_raw.attrs.update({attr_name:attr_value})
        
            filt_data = file.create_group('filtered data')
            filt_data.attrs.update(self.filt_data['attrs'])
            for key, value in self.filt_data.items():
                if key !='attrs':
                    ds_filt = filt_data.create_dataset(key, data=value['data'])
                    for attr_name, attr_value in value.items():
                        if attr_name != 'data':
                            ds_filt.attrs.update({attr_name:attr_value})
            
            freq_data = file.create_group('freq data')
            freq_data.attrs.update(self.freq_data['attrs'])
            for key, value in self.freq_data.items():
                if key !='attrs':
                    ds_freq = freq_data.create_dataset(key, data=value['data'])
                    for attr_name, attr_value in value.items():
                        if attr_name != 'data':
                            ds_freq.attrs.update({attr_name:attr_value})

    def load(self, filename: str) -> None:
        """Loads data from file"""

        logger.debug('load procedure is starting...')
        file_path, file_name = os.path.split(filename)
        self.attrs['file_path'] = file_path
        logger.debug(f'"file_path" set to {file_path}')
        self.attrs['filename'] = file_name
        logger.debug(f'"filename" set to {file_name}')

        with h5py.File(filename,'r') as file:
            
            #load general metadata
            general = file['general']
            self.attrs.update(general.attrs) # type: ignore
            logger.debug(f'General metadata with {len(general.attrs)}'
                         + ' records loaded.')
            
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

        #set amount of data_points for compatibility with old data
        if self.attrs.get('data_points', 0) < (len(self.raw_data) - 1):
            self.attrs.update({'data_points': len(self.raw_data) - 1})

    def _build_ds_name(self, n: int) -> str:
        """Build and return name for dataset."""
        
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

        if not self.attrs['data_points']:
            print(f'{bcolors.WARNING}\
                  No data to display\
                  {bcolors.ENDC}')

        #plot initialization
        warnings.filterwarnings('ignore',
                                category=MatplotlibDeprecationWarning)

        self._fig = plt.figure(tight_layout=True)
        filename = os.path.join(self.attrs['file_path'],
                                self.attrs['filename'])
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
        x_label = self.attrs['parameter_name']\
                  + ', ['\
                  + self.attrs['parameter_units']\
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
            if self._param_ind == (self.attrs['data_points'] - 1):
                pass
            else:
                self._param_ind += 1
                self._plot_update()

    def _plot_update(self) -> None:
        """Updates plotted data"""

        ds_name = self._build_ds_name(self._param_ind)
        start = self.raw_data[ds_name]['x var start']
        step = self.raw_data[ds_name]['x var step']
        stop = self.raw_data[ds_name]['x var stop']
        num = len(self.raw_data[ds_name]['data'])
        time_points = np.linspace(start,stop,num)

        #check if datapoint is empty
        if not step:
            return None

        #update max_signal_amp(parameter) plot
        title = self.attrs['parameter_name']\
                + ': '\
                + str(int(self._param_values[self._param_ind]))\
                + self.attrs['parameter_units']
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
        if self.filt_data['attrs']['x var units'] == self.attrs['zoom_units']:
            pre_points = int(self.attrs['zoom_pre_time']/step)
            post_points = int(self.attrs['zoom_post_time']/step)
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
                  zoom_units do not match x var units!\
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
        if not self.attrs['data_points']:
            print(f'{bcolors.WARNING}\
                  Attempt to read dependence from empty data\
                  {bcolors.ENDC}')
            return []
        
        if data_group == 'raw_data':
            ds_name = self._build_ds_name(0)
            if self.raw_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from RAW_data\
                    {bcolors.ENDC}')
                return []
            for ds_name, ds in self.raw_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])

        
        elif data_group == 'filt_data':
            ds_name = self._build_ds_name(0)
            if self.filt_data[ds_name].get(value) is None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from FILT_data\
                    {bcolors.ENDC}')
                return []
            for ds_name, ds in self.filt_data.items():
                if ds_name != 'attrs':
                    dep.append(ds[value])
        
        elif data_group == 'freq_data':
            ds_name = self._build_ds_name(0)
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
                  low: pint.Quantity=1*ureg.MHz,
                  high: pint.Quantity=10*ureg.MHz,
                  ds_name: str='') -> None:
        """Perform bandpass filtration on data.
        
        if <ds_name> is empty, then perform filtration for all data,
        otherwise perform filtration only for specified <ds_name>.
        Create necessary datasets in self.filt_data and self.freq_data.
        """

        if ds_name:
            ds = self.raw_data[ds_name]
            self._bp_filter_single(low, high, ds_name, ds)
        else:
            for ds_name, ds in self.raw_data.items():
                if ds_name != 'attrs':
                    self._bp_filter_single(low, high, ds_name, ds)

    def _bp_filter_single(self,
                    low: pint.Quantity,
                    high: pint.Quantity,
                    ds_name: str,
                    ds: RawData) -> None:
        """Internal bandpass filtration method.
        
        Actually do the filtration."""

        logger.debug(f'Starting FFT for {ds_name} '
                     + f'with bp filter ({low}:{high})')
        dt = ds['x_var_step'].to('s').m
        logger.debug(f'{dt=}')
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
        filtered_data = f_signal[:,(W>low)*(W<high)]
        freq_ds: FreqData = {
            'data': filtered_data*ureg.Hz,
            'x_var_step': (filtered_freq[1]-filtered_freq[0])*ureg.Hz,
            'x_var_start': filtered_freq.min()*ureg.Hz,
            'x_var_stop': filtered_freq.max()*ureg.Hz,
            'max_amp': (filtered_data.max() - filtered_data.min())*ureg.Hz
        }
        self.freq_data.update({ds_name: freq_ds})
        freq_points = len(self.freq_data[ds_name]['data'])

        if self.freq_data['attrs']['max_len'] < freq_points:
            self.freq_data['attrs']['max_len'] = freq_points

        self.filt_data.update(
            {ds_name: self.raw_data[ds_name]})
        final_filt_data = irfft(filtered_f_signal)
        
        filt_max_amp = final_filt_data.max()-final_filt_data.min()
        self.filt_data[ds_name].update(
            {'data':final_filt_data})
        self.filt_data[ds_name].update({'max_amp': filt_max_amp})

        self.attrs['updated'] = self._get_cur_time()
