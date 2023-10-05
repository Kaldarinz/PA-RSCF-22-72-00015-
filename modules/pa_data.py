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
    |  |--'data_points': int - amount of stored PA measurements
    |  |--'created': str - date and time of data measurement
    |  |--'updated': str - date and time of last data update
    |  |--'filename': str - full path to the data file
    |  |--'zoom_pre_time': Quantity - start time from the center of the PA data frame for zoom in data view
    |  |--'zoom_post_time': Quantity - end time from the center of the PA data frame for zoom in data view
    |
    |--'raw_data'
    |  |--'attrs'
    |  |  |--'max_len': int - maximum amount of points in PA signal
    |  |  |--'x_var_name': str - name of the X variable in PA signal
    |  |  |--'y_var_name': str - name of the Y variable in PA signal
    |  |
    |  |--point001
    |  |  |--'data': List[Quantity] - measured PA signal
    |  |  |--'data_raw': ndarray[int8]
    |  |  |--'param_val': List[Quantity] - value of independent parameter
    |  |  |--'x_var_step': Quantity
    |  |  |--'x_var_start': Quantity
    |  |  |--'x_var_stop': Quantity
    |  |  |--'pm_en': Quantity - laser energy measured by power meter in glass reflection
    |  |  |--'sample_en': Quantity - laser energy at sample in [uJ]
    |  |  |--'max_amp': Quantity - (y_max - y_min)
    |  |
    |  |--point002
    |  |  |--'data': List[Quantity] - measured PA signal
    |  |  ...
    |  ...
    |
    |--'filt_data'
    |  |--'attrs'
    |  |  |--'x var name': str - name of the X variable in PA signal
    |  |  |--'y var name': str - name of the Y variable in PA signal
    |  |
    |  |--point001
    |  |  |--'data': List[Quantity] - measured PA signal
    |  |  |--'data_raw': ndarray[int8]
    |  |  |--'param_val': List[Quantity] - value of independent parameter
    |  |  |--'x_var_step': Quantity
    |  |  |--'x_var_start': Quantity
    |  |  |--'x_var_stop': Quantity
    |  |  |--'pm_en': Quantity - laser energy measured by power meter in glass reflection
    |  |  |--'sample_energy': Quantity - laser energy at sample in [uJ]
    |  |  |--'max_amp': Quantity - (y_max - y_min)
    |  |
    |  |--point002
    |  |  |--'data': List[Quantity] - measured PA signal
    |  |  ...
    |  ...
    |
    |--'freq_data'
       |--'attrs'
       |  |--'max_len': int - maximum amount of points in PA signal
       |  |--'x_var_name': str - name of the X variable in PA signal
       |  |--'y_var_name': str - name of the Y variable in PA signal
       |
       |--point001
       |  |--'data': List[Quantity] - frequncies present in filt_data
       |  |--'x var step': float
       |  |--'x var start': float
       |  |--'x var stop': float
       |  |--'max amp': float - (y_max - y_min)
       |
       |--point002
       |  |--'data': List[Quantity] - frequncies present in filt_data
       |  ...
       ...
"""
import warnings
from typing import Iterable, Any, Tuple, TypedDict, List
from datetime import datetime
import time
import os, os.path
import logging
from itertools import zip_longest

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import MatplotlibDeprecationWarning # type: ignore
from scipy.fftpack import rfft, irfft, fftfreq
import pint
from pint.facets.plain.quantity import PlainQuantity
import h5py
import numpy as np
import numpy.typing as npt

from . import Q_
from . import data_classes as dc
from modules.exceptions import (
    PlotError
)

logger = logging.getLogger(__name__)

class PaData:
    """Class for PA data storage and manipulations"""

    VERSION = 1.1

    _ax_sp: plt.Axes
    "Axes for plotting dependence of max_amp on parameter."
    _ax_raw: plt.Axes
    "Axes for plotting raw signal."
    _ax_freq: plt.Axes
    "Axes for plotting signal in frequnecy domain."
    _ax_filt: plt.Axes
    "Axes for plotting filtered data."
    _ax_raw_zoom: plt.Axes
    "Axes for plotting a region containing actual RAW signal."
    _ax_filt_zoom: plt.Axes
    "Axes for plotting a region containing actual FILTERED signal."

    def __init__(self, dims: int=-1, params: List[str]=[]) -> None:
        """Class init.
        
        <dims> dimensionality of the stored measurement.
        <params> contain names of the dimensions."""

        #general metadata
        self.attrs: dc.BaseMetadata = {
            'version': self.VERSION,
            'measurement_dims': dims,
            'parameter_name': params,
            'data_points': 0,
            'created': self._get_cur_time(),
            'updated': self._get_cur_time(),
            'filename': '',
            'zoom_pre_time': Q_(2, 'us'),
            'zoom_post_time': Q_(13, 'us')
        }
        raw_attrs: dc.RawMetadata = {
            'max_len': 0,
            'x_var_name': 'Time',
            'y_var_name': 'PhotoAcoustic signal'
        }
        self.raw_data = {}
        self.raw_data.update({'attrs': raw_attrs})
        
        filt_attrs: dc.FiltMetadata = {
            'x_var_name': 'Time',
            'y_var_name': 'Filtered photoAcoustic signal'
        }

        self.filt_data = {}
        self.filt_data.update({'attrs': filt_attrs})
        
        freq_attrs: dc.RawMetadata = {
            'max_len': 0,
            'x_var_name': 'Frequency',
            'y_var_name': 'FFT amplitude'
        }
        self.freq_data = {}
        self.freq_data.update({'attrs': freq_attrs})
        
        logger.debug('PaData instance created')

    def add_measurement(
            self, 
            data: dc.DataPoint,
            param_val: List[PlainQuantity] = []
        ) -> None:
        """Add a single data point.
        
        Add a datapoint to raw_data, filt_data and freq_data.
        """

        logger.debug('Starting datapoint addition to file...')
        if data.pa_signal is None or not len(data.pa_signal):
            logger.debug('...Terminating datapoint addition. PA signal is missing.')
            return None
        if data.pa_signal_raw is None or not len(data.pa_signal_raw):
            logger.debug('...Terminating datapoint addition. PA signal_raw is missing.')
            return None
        ds_name = self._build_ds_name(self.attrs['data_points']+1)
        if not ds_name:
            logger.error('Max data_points reached! Data cannot be added!')
            return None
        logger.debug(f'{ds_name=}')
        # ensure that units of parameter and data for new point
        # is the same as units of already present points
        if self.attrs['data_points']:
            params = self.raw_data['point001']['param_val']
            #should be changed for 0D case, when there is no parameter
            logger.debug(f'Param values changed from {param_val}...')
            param_val = [x.to(y.u) for x,y in zip(param_val,params)]
            logger.debug(f'... to {param_val}')
            data_u = self.raw_data['point001']['data'].u
            if data_u != data.pa_signal.u:
                logger.debug(f'Changing units of data from {data.pa_signal.u} '
                             +f'to {data_u}')
                data.pa_signal = data.pa_signal.to(data_u)
        ds: dc.RawData = {
            'data': data.pa_signal, #type: ignore
            'data_raw': data.pa_signal_raw,
            'param_val': param_val,
            'x_var_step': data.dt.to('us'),
            'x_var_start': data.start_time.to('us'),
            'x_var_stop': data.stop_time.to('us'),
            'pm_en': data.pm_energy,
            'sample_en': data.sample_energy,
            'max_amp': data.max_amp
        }
        cur_data_len = len(ds['data'])
        old_data_len = self.raw_data['attrs']['max_len']
        if cur_data_len > old_data_len:
            self.raw_data['attrs']['max_len'] = cur_data_len
            logger.debug(f'max_len updated from {old_data_len} '
                         + f'to {cur_data_len}')
        self.raw_data.update({ds_name: ds})
        a, b = self._calc_data_fit(ds_name)
        self.raw_data[ds_name]['a'] = a
        self.raw_data[ds_name]['b'] = b
        self.bp_filter(ds_name=ds_name)
        
        self.attrs['data_points'] += 1
        self.attrs['updated'] = self._get_cur_time()
        logger.debug('...Finishing data point addition to file.')

    def _get_cur_time (self) -> str:
        """Return timestamp of current time."""
        
        cur_time = time.time()
        date_time = datetime.fromtimestamp(cur_time)
        date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

        return date_time

    def save(self, filename: str='') -> None:
        """Saves data to file."""

        if filename:
            self.attrs['filename'] = filename
        elif self.attrs['filename']:
            filename = self.attrs['filename']
        else:
            logger.warning('Filename is not set. Data cannot be saved!')
            return
        self._flush(filename)

    def save_tmp(self) -> None:
        """Save current data to TmpData.hdf5."""

        logger.debug('Starting tmp save...')
        base = os.getcwd()
        filename = os.path.join(base, 'measuring results', 'TmpData.hdf5')
        self._flush(filename)
        logger.debug('...Finishing tmp save.')

    def export_txt(
            self,
            group_name: str,
            filename: str,
            full: bool=False) -> None:
        """Export all data points to txt format.
        
        <group_name> is 'raw_data'|'filt_data'|'freq_data'.
        Export waveforms from <group_name> to <filname> file in 
        'measuring results' folder.
        The whole waveform is exported if <fool> flag is set,
        otherwise only zoomed data will be exported.
        """

        logger.debug(f'Start export procedure from {group_name}.')

        group = getattr(self, group_name, None)
        if group is None:
            logger.warning(f'{group_name} is missing in PaData class.')
            return
        if len(self.attrs['parameter_name']) > 1:
            logger.warning('Export to txt is supported only for 0D and 1D data.')
            return
        data = []
        header = ('# 1st line: name of data. 2nd line: units of data.'
                  + '3rd line: paramter name. 4th line: parameter units.'
                  + '5th line: excitation energy units. 6th line: '
                  + 'parameter value. 7th line: excitation energy value.'
                  + 'following lines: data.\n')
        for ds_name, datapoint in group.items():
            if ds_name == 'attrs':
                continue
            logger.debug(f'Building list with x data for {ds_name}')
            col_x = []
            col_x.append(group['attrs']['x_var_name'])
            col_x.append(datapoint['x_var_step'].u)
            param_name = self.attrs['parameter_name'][0]
            col_x.append(param_name)
            param_units = datapoint['param_val'][0].u
            param_val = datapoint['param_val'][0].m
            col_x.append(param_units)
            energy_units = datapoint['sample_en'].u
            energy_val = datapoint['sample_en'].m
            col_x.append(energy_units)
            col_x.append(param_val)
            col_x.append(energy_val)
            
            col_y = []
            col_y.append(group['attrs']['y_var_name'])
            col_y.append(datapoint['data'][0].u)
            col_y.append(param_name)
            col_y.append(param_units)
            col_y.append(energy_units)
            col_y.append(param_val)
            col_y.append(energy_val)

            if group_name in ('raw_data', 'filt_data') and not full:
                max_y = np.amax(datapoint['data'])
                max_ind = np.flatnonzero(datapoint['data']==max_y)[0]
                x_step = datapoint['x_var_step']
                pre_time = self.attrs['zoom_pre_time']
                post_time = self.attrs['zoom_post_time']
                pre_points = int(pre_time.to(x_step.u).m/x_step.m)
                post_points = int(post_time.to(x_step.u).m/x_step.m)
                start_zoom_ind = max_ind-pre_points
                if start_zoom_ind < 0:
                    start_zoom_ind = 0
                t_points = int((datapoint['x_var_stop']
                                - datapoint['x_var_start'])/x_step)
                stop_zoom_ind = max_ind + post_points
                if stop_zoom_ind > t_points:
                    stop_zoom_ind = t_points
                for i in range(start_zoom_ind, stop_zoom_ind):
                    col_x.append((datapoint['x_var_start']+i*x_step).m)
                    col_y.append(datapoint['data'][i].m)
            else:
                x = datapoint['x_var_start'].m
                col_x.append(x)
                while x < (datapoint['x_var_stop'].m - datapoint['x_var_step'].m):
                    x += datapoint['x_var_step'].m
                    col_x.append(x)
                col_y.extend((val.m for val in datapoint['data']))

            data.append(col_x)
            data.append(col_y)
        
        with open(filename, 'w') as file:
            file.write(header)
            for row in zip_longest(*data, fillvalue=''):
                file.write(';'.join(map(str,row)) + '\n')
    
    def _flush(self, filename: str) -> None:
        """Write data to disk."""

        logger.debug(f'Start friting data to {filename}')
        with h5py.File(filename, 'w') as file:
            file.attrs.update(self._root_attrs())
            raw_data = file.create_group('raw_data')
            raw_data.attrs.update(self._raw_attrs())
            for ds_name, ds in self.raw_data.items():
                if ds_name !='attrs':
                    ds_raw = raw_data.create_dataset(
                        ds_name,
                        data = ds['data_raw'])
                    ds_raw.attrs.update(self._ds_attrs(ds_name))

    def _root_attrs(self) -> dict:
        """"Build dict with root attributes."""

        param_vals = self.raw_data['point001']['param_val']
        attrs = {
            'version': self.attrs['version'],
            'measurement_dims': self.attrs['measurement_dims'],
            'parameter_name': self.attrs['parameter_name'],
            'parameter_u': [str(param.u) for param in param_vals],
            'data_points': self.attrs['data_points'],
            'created': self.attrs['created'],
            'updated': self.attrs['updated'],
            'filename': self.attrs['filename'],
            'zoom_pre_time': self.attrs['zoom_pre_time'].m,
            'zoom_post_time': self.attrs['zoom_post_time'].m,
            'zoom_u': str(self.attrs['zoom_pre_time'].u)
        }
        return attrs

    def _raw_attrs(self) -> dict:
        """"Build dict with raw_data attributes."""

        attrs = {
            'max_len': self.raw_data['attrs']['max_len'],
            'x_var_name': self.raw_data['attrs']['x_var_name'],
            'x_var_u': str(self.raw_data['point001']['x_var_step'].u),
            'y_var_name': self.raw_data['attrs']['y_var_name'],
            'y_var_u': str(self.raw_data['point001']['data'][0].u)
        }
        return attrs

    def _ds_attrs(self, ds_name: str) -> dict:
        """Build dict with attributes for <ds_name>"""
        
        ds_attrs = self.raw_data[ds_name]
        attrs = {
            'param_val': [param.m for param in ds_attrs['param_val']],
            'a': ds_attrs['a'],
            'b': ds_attrs['b'],
            'x_var_step': ds_attrs['x_var_step'].m,
            'x_var_start': ds_attrs['x_var_start'].m,
            'x_var_stop': ds_attrs['x_var_stop'].m,
            'pm_en': ds_attrs['pm_en'],
            'pm_en_u': str(ds_attrs['pm_en'].u),
            'sample_en': ds_attrs['sample_en'].m,
            'sample_en_u': str(ds_attrs['sample_en'].u),
            'max_amp': ds_attrs['max_amp'].m,
            'max_amp_u': str(ds_attrs['max_amp'].u)
        }
        return attrs

    def _calc_data_fit(self,
                       ds_name: str,
                       points: int=20
        ) -> Tuple[float,float]:
        """Calculate coefs to convert data_raw to data.
        
        Return (a,b), where <data> = a*<data_raw> + b.
        <points> is amount of points used for fitting.
        """

        DIFF_VALS = 5

        data = self.raw_data[ds_name]['data']
        data_raw = self.raw_data[ds_name]['data_raw']
        max_ind = np.flatnonzero(data==data.max())[0]
        x = data_raw[max_ind:max_ind+points]
        if len(np.unique(x))< DIFF_VALS:
            for i in range(len(self.raw_data[ds_name]['data_raw'])):
                stop_ind = max_ind+points+i
                x = data_raw[max_ind:stop_ind].copy()
                if len(np.unique(x)) == DIFF_VALS:
                    break
        logger.debug(f'{i} additional points added to find '
                     + f'{DIFF_VALS} unique values.')
        y = [quantity.m for quantity, _ in zip(data[max_ind:], x)]
        coef = np.polyfit(x, y, 1)
        return coef

    def load(self, filename: str) -> None:
        """Loads data from file"""

        logger.debug('load procedure is starting...')
        self.attrs['filename'] = filename
        logger.debug(f'"filename" set to {filename}')

        with h5py.File(filename, 'r') as file:
            
            #load general metadata
            general = file.attrs
            if 'version' not in general:
                logger.warning('File has old structure and cannot be loaded.')
                return
            time_unit = general['zoom_u']
            self.attrs.update(
                {
                'version': general['version'],
                'measurement_dims': general['measurement_dims'],
                'parameter_name': general['parameter_name'],
                'data_points': general['data_points'],
                'created': general['created'],
                'updated': general['updated'],
                'zoom_pre_time': Q_(general['zoom_pre_time'], time_unit), # type: ignore
                'zoom_post_time': Q_(general['zoom_post_time'], time_unit) # type: ignore
                }
            )
            #in old version of 0D data save parameter name was missed
            #but it was wavelength in all cases. Fix it on load old data.
            if not len(self.attrs['parameter_name']):
                self.attrs['parameter_name'] = ['Wavelength']
            logger.debug(f'General metadata with {len(general)}'
                         + ' records loaded.')
        
            raw_data = file['raw_data']
            #metadata of raw_data
            self.raw_data['attrs'].update(
                {
                    'max_len': raw_data.attrs['max_len'],
                    'x_var_name': raw_data.attrs['x_var_name'],
                    'y_var_name': raw_data.attrs['y_var_name']
                }
            )
            logger.debug(f'raw_data metadata with {len(raw_data.attrs)}'
                         + ' records loaded.')

            for ds_name in raw_data: # type: ignore
                self._load_ds(
                    ds_name,
                    raw_data[ds_name], # type: ignore
                    raw_data.attrs['y_var_u'], # type: ignore
                    raw_data.attrs['x_var_u'], # type: ignore
                    general['parameter_u'] # type: ignore
                )
            self.bp_filter()
            logger.debug('...Data loaded!')

    def _load_ds(self,
                 ds_name: str,
                 ds: h5py.Dataset,
                 y_var_unit: str,
                 x_var_unit: str,
                 param_units: List[str]
                 ) -> None:
        """Load <ds_name> dataset from <ds>."""

        p = np.poly1d([ds.attrs['a'], ds.attrs['b']]) # type: ignore
        data_raw = ds[...]
        data = Q_(p(data_raw), y_var_unit)
        p_vals = ds.attrs['param_val']
        param_val = [Q_(x, y) for x, y in zip(p_vals,param_units)] # type: ignore
        x_var_step = Q_(ds.attrs['x_var_step'], x_var_unit)
        x_var_start = Q_(ds.attrs['x_var_start'], x_var_unit)
        x_var_stop = Q_(ds.attrs['x_var_stop'], x_var_unit)
        pm_en = Q_(ds.attrs['pm_en'], ds.attrs['pm_en_u']) # type: ignore
        sample_en = Q_(ds.attrs['sample_en'], ds.attrs['sample_en_u']) # type: ignore
        max_amp = Q_(ds.attrs['max_amp'], ds.attrs['max_amp_u']) # type: ignore

        self.raw_data.update(
                    {
                        ds_name:{
                            'data': data,
                            'data_raw': data_raw,
                            'param_val': param_val,
                            'x_var_step': x_var_step,
                            'x_var_start': x_var_start,
                            'x_var_stop': x_var_stop,
                            'pm_en': pm_en,
                            'sample_en': sample_en,
                            'max_amp': max_amp
                        }
                    }
                )

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
    
    def _get_ds_index(self, ds_name: str) -> int:
        """Return index from dataset name."""

        n_str = ds_name.split('point')[-1]
        n = int(n_str)
        return n

    def plot(self) -> None:
        """Plot data."""

        if not self.attrs['data_points']:
            logger.warning('No data to plot!')
            return

        #plot initialization
        warnings.filterwarnings('ignore',
                                category=MatplotlibDeprecationWarning)
        self._fig = plt.figure(tight_layout=True)
        filename = os.path.basename(self.attrs['filename'])
        self._fig.suptitle(filename)
        gs = gridspec.GridSpec(2,3)
        self._ax_sp = self._fig.add_subplot(gs[0,0])
        self._ax_raw = self._fig.add_subplot(gs[0,1])
        self._ax_freq = self._fig.add_subplot(gs[1,0])
        self._ax_filt = self._fig.add_subplot(gs[1,1])
        self._ax_raw_zoom = self._fig.add_subplot(gs[0,2])
        self._ax_filt_zoom = self._fig.add_subplot(gs[1,2])

        self._marker_style = {
            'marker': 'o',
            'alpha': 0.4,
            'ms': 12,
            'color': 'yellow'
        }
        self._fill_style = {
            'alpha': 0.3,
            'color': 'g'
        }

        dims = self.attrs['measurement_dims']
        try:
            if dims == 0:
                self._plot_0d()
            elif dims == 1:
                self._plot_1d()
            elif dims == 2:
                self._plot_2d()
            elif dims == 3:
                self._plot_3d()
            else:
                logger.warning(f'Unsupported dimensionality: ({dims})')
        except PlotError as err:
            logger.warning(f'Plot attempt failed. {err.value}')

    def _plot_0d(self) -> None:
        """Plot 0D data."""

        logger.debug('Starting 0D plotting...')
        for ds_name, ds in self.filt_data.items():
            if ds_name != 'attrs':
                break
        #update filt plot withh filt_zoom
        self._plot_update_signal(
            self.filt_data[ds_name],
            self.filt_data['attrs'],
            'Filtered data',
            self._ax_filt,
            self._ax_filt_zoom,
        )
        #update raw plot with raw_zoom
        self._plot_update_signal( 
            self.raw_data[ds_name],
            self.raw_data['attrs'],
            'Raw data',
            self._ax_raw,
            self._ax_raw_zoom,
        )
        #update freq plot
        self._plot_update_signal(
            self.freq_data[ds_name],
            self.freq_data['attrs'],
            'FFT data',
            self._ax_freq
        )
        plt.show()

    def _plot_1d(self) -> None:
        """Plot 1D data."""

        logger.debug('Starting 1D plotting...')
        self._param_ind = 0 #index of active data on plot
        self._plot_param()
        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._plot_update()
        plt.show()

    def _plot_2d(self) -> None:
        """Plot 2D data."""

        logger.warning('Plotting 2D data is not implemented!')

    def _plot_3d(self) -> None:
        """Plot 3D data."""

        logger.warning('Plotting 3D data is not implemented!')

    def _plot_param(
            self,
            value: str = 'max_amp'
        ) -> None:
        """Plot parameter data.
        
        <value> define which quantity will be plotted.
        """

        param_values = self.get_dependance('raw_data','param_val[0]')
        if param_values is None:
            err_msg = 'Cannot get param_val for all datapoints.'
            logger.debug(err_msg)
            raise PlotError(err_msg)
        else:
            self._param_values = param_values

        raw_vals = self.get_dependance('raw_data', value)
        if raw_vals is None:
            err_msg = f'Cant get {value} of raw_data.'
            logger.debug(err_msg)
            raise PlotError(err_msg)
        else:
            self._raw_vals = raw_vals

        filt_vals = self.get_dependance('filt_data', value)
        if filt_vals is None:
            err_msg = f'Cant get {value} of raw_data.'
            logger.debug(err_msg)
            raise PlotError(err_msg)
        else:
            units = self._raw_vals.u
            self._filt_vals = filt_vals.to(units)
        
        self._ax_sp.plot(
            self._param_values.m,
            self._raw_vals.m,
            label=f'{value} of raw data')
        self._ax_sp.plot(
            self._param_values.m,
            self._filt_vals.m,
            label=f'{value} of filtered data'
        )
        self._ax_sp.legend(loc='upper right')
        self._ax_sp.set_ylim(bottom=0)
        x_label = (self.attrs['parameter_name'][0] 
                   + ', ['
                   + f'{self._param_values.u}'
                   + ']')
        self._ax_sp.set_xlabel(x_label)
        y_label = (self.raw_data['attrs']['y_var_name']
                   + ', ['
                   + f'{self._raw_vals.u}'
                   + ']')
        self._ax_sp.set_ylabel(y_label)

        self._plot_init_cur_param()

    def _plot_init_cur_param(self) -> None:
        """Plot initial param value.
        
        This method should be called after _plot_param.
        """

        if (not len(self._param_values) 
            or not len(self._raw_vals)
            or not len(self._filt_vals)):
            err_msg = 'No data for displaying current param value.'
            logger.debug(err_msg)
            raise PlotError(err_msg)

        self._plot_selected_raw, = self._ax_sp.plot(
            self._param_values[self._param_ind].m, #type:ignore
            self._raw_vals[self._param_ind].m, #type:ignore
            **self._marker_style)
        
        self._plot_selected_filt, = self._ax_sp.plot(
            self._param_values[self._param_ind].m, #type:ignore
            self._filt_vals[self._param_ind].m, #type:ignore
            **self._marker_style)

    def _on_key_press(self, event) -> None:
        """Callback function for changing active data on plot."""

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
        """Update plotted data."""

        ds_name = self._build_ds_name(self._param_ind+1)
        #check if datapoint is empty
        if self.filt_data[ds_name].get('data') is None:
            return None
        #update parameter plot
        self._plot_update_cur_param()
        #update filt plot withh filt_zoom
        self._plot_update_signal(
            self.filt_data[ds_name],
            self.filt_data['attrs'],
            'Filtered data',
            self._ax_filt,
            self._ax_filt_zoom,
        )
        #update raw plot with raw_zoom
        self._plot_update_signal( 
            self.raw_data[ds_name],
            self.raw_data['attrs'],
            'Raw data',
            self._ax_raw,
            self._ax_raw_zoom,
        )
        #update freq plot
        self._plot_update_signal(
            self.freq_data[ds_name],
            self.freq_data['attrs'],
            'FFT data',
            self._ax_freq
        )
        
        #general update
        self._fig.align_labels()
        self._fig.canvas.draw()

    def _plot_update_cur_param(self) -> None:
        """Update current position in parameter subplot."""

        title = (self.attrs['parameter_name'][0] + ': '
                  + str(self._param_values[self._param_ind])) #type:ignore
        self._ax_sp.set_title(title)
        self._plot_selected_raw.set_data(
            self._param_values[self._param_ind].m, #type:ignore
            self._raw_vals[self._param_ind].m) #type:ignore
        self._plot_selected_filt.set_data(
            self._param_values[self._param_ind].m, #type:ignore
            self._filt_vals[self._param_ind].m) #type:ignore

    def _plot_update_signal(
            self,
            ds: dict,
            attrs: dict,
            title: str,
            ax: plt.Axes,
            zoom_ax: plt.Axes|None=None,
        ) -> None:
        """Update plot on <ax> and <zoom_ax> with signal from <ds>.
        
        <ds> is datasets for plotting.
        <attrs> is dict with attributes of group containing <ds>.
        If <zoom_ax> is not set, then zoom version of plot is
        not updated AND markers for max and min values are not plotted.
        """

        logger.debug(f'Start updating subplot {title}')
        
        start = ds['x_var_start']
        stop = ds['x_var_stop']
        step = ds['x_var_step']
        num = len(ds['data'])
        time_data = Q_(np.linspace(start.m,stop.m,num), start.u)

        x_label = (attrs['x_var_name']
                   + ', ['
                   + str(start.u)
                   + ']')
        y_label = (attrs['y_var_name']
                   + ', ['
                   + str(ds['data'].u)
                   + ']')
        ax.clear()
        ax.plot(time_data.m, ds['data'].m)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if zoom_ax is not None:
            #marker for max value
            max_val = np.amax(ds['data'])
            max_ind = np.flatnonzero(ds['data']==max_val)[0]
            max_t = time_data[max_ind] #type:ignore
            ax.plot(max_t.m, max_val.m, **self._marker_style)
            
            #marker for min value
            min_val = np.amin(ds['data'])
            min_ind = np.flatnonzero(ds['data']==min_val)[0]
            min_t = time_data[min_ind] #type:ignore
            ax.plot(min_t.m, min_val.m, **self._marker_style)
        
            #marker for zoomed area
            pre_time = self.attrs['zoom_pre_time']
            post_time = self.attrs['zoom_post_time']
            pre_points = int(pre_time.to(step.u).m/step.m)
            post_points = int(post_time.to(step.u).m/step.m)
            start_zoom_ind = max_ind-pre_points
            if start_zoom_ind < 0:
                start_zoom_ind = 0
            stop_zoom_ind = min_ind + post_points
            if stop_zoom_ind > (len(time_data.m) - 1):
                stop_zoom_ind = len(time_data.m) - 1
            ax.fill_betweenx(
                [min_val.m, max_val.m],
                time_data[start_zoom_ind].m, #type:ignore
                time_data[stop_zoom_ind].m, #type:ignore
                **self._fill_style
            )

            #plot zoomed
            zoom_ax.clear()
            # uncomment to change into absolute x_axis for zoomed adrea
            # z_time_off = time_data[start_zoom_ind].m #type:ignore
            # zoom_ax.plot(
            #     time_data[start_zoom_ind:stop_zoom_ind+1].m-z_time_off, #type:ignore
            #     ds['data'][start_zoom_ind:stop_zoom_ind+1].m
            # )
            zoom_ax.plot(
                time_data[start_zoom_ind:stop_zoom_ind+1].m,
                ds['data'][start_zoom_ind:stop_zoom_ind+1].m
            )
            zoom_ax.set_xlabel(x_label)
            zoom_ax.set_ylabel(y_label)
            zoom_ax.set_title('Zoom of ' + title)

    def get_dependance(self, 
                       data_group: str, 
                       value: str) -> PlainQuantity|None:
        """Return array with value from each dataset in the data_group.
        
        data_group: 'raw_data'|'filt_data|'freq_data'.
        """

        logger.debug(f'Start building array of {value} from {data_group}.')
        dep = [] #array for return values
        if not self.attrs['data_points']:
            logger.error(f'PaData instance contains no data points.')
            return None
        
        if value.startswith('param_val'):
            ind = int(value.split('[')[-1].split(']')[0])
            value = 'param_val'
            check_st = ('self.'
                        + data_group
                        + '[self._build_ds_name(1)].get(value)'
                        + f'[{ind}]'
            )
        else:
            ind = None
            check_st = ('self.'
                        + data_group
                        + '[self._build_ds_name(1)].get(value)'
            )
        if eval(check_st) is None:
            logger.error(f'Attempt to read unknown attribute: {value} '
                         + f'from {data_group}.')
            return None
        
        if ind is not None:
            if data_group == 'raw_data':
                for ds_name, ds in self.raw_data.items():
                    if ds_name != 'attrs':
                        dep.append(ds[value][ind])
            elif data_group == 'filt_data':
                for ds_name, ds in self.filt_data.items():
                    if ds_name != 'attrs':
                        dep.append(ds[value][ind])
            elif data_group == 'freq_data':
                for ds_name, ds in self.freq_data.items():
                    if ds_name != 'attrs':
                        dep.append(ds[value][ind])
        else:
            if data_group == 'raw_data':
                for ds_name, ds in self.raw_data.items():
                    if ds_name != 'attrs':
                        dep.append(ds[value])
            elif data_group == 'filt_data':
                for ds_name, ds in self.filt_data.items():
                    if ds_name != 'attrs':
                        dep.append(ds[value])
            elif data_group == 'freq_data':
                for ds_name, ds in self.freq_data.items():
                    if ds_name != 'attrs':
                        dep.append(ds[value])
        dep = pint.Quantity.from_list(dep)
        return dep

    def bp_filter(self,
                  low: PlainQuantity=Q_(1, 'MHz'),
                  high: PlainQuantity=Q_(10, 'MHz'),
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
                    low: PlainQuantity,
                    high: PlainQuantity,
                    ds_name: str,
                    ds: dc.RawData) -> None:
        """Internal bandpass filtration method.
        
        Actually do the filtration."""

        logger.debug(f'Starting FFT for {ds_name} '
                     + f'with bp filter ({low}:{high})...')
        dt = ds['x_var_step'].to('s').m
        low = low.to('Hz').m
        high = high.to('Hz').m
        logger.debug(f'{dt=}')
        W = fftfreq(len(ds['data'].m), dt) # array with freqs
        f_signal = rfft(ds['data'].m) # signal in f-space

        filtered_f_signal = f_signal.copy()
        filtered_f_signal[(W<low)] = 0 # high pass filtering

        if high > 1/(2.5*dt): # Nyquist frequency check
            filtered_f_signal[(W>1/(2.5*dt))] = 0 
        else:
            filtered_f_signal[(W>high)] = 0

        #pass frequencies
        filtered_freq = W[(W>low)*(W<high)]
        filtered_data = f_signal[(W>low)*(W<high)]
        freq_ds: dc.FreqData = {
            'data': Q_(filtered_data, ds['data'].u), #type: ignore
            'x_var_step': Q_((filtered_freq[1]-filtered_freq[0]), 'Hz'),
            'x_var_start': Q_(filtered_freq.min(), 'Hz'),
            'x_var_stop': Q_(filtered_freq.max(), 'Hz'),
            'max_amp': filtered_data.ptp()
        }
        logger.debug(f'freq step: {freq_ds["x_var_step"]}')
        logger.debug(f'FFT amplitude: {freq_ds["max_amp"]}')
        self.freq_data.update({ds_name: freq_ds})
        freq_points = len(self.freq_data[ds_name]['data'])

        if self.freq_data['attrs']['max_len'] < freq_points:
            self.freq_data['attrs']['max_len'] = freq_points

        self.filt_data.update(
            {ds_name: self.raw_data[ds_name].copy()})
        final_filt_data = Q_(irfft(filtered_f_signal), ds['data'].u)
        
        filt_max_amp = final_filt_data.ptp() #type: ignore
        self.filt_data[ds_name].update(
            {'data':final_filt_data})
        self.filt_data[ds_name].update({'max_amp': filt_max_amp})

        self.attrs['updated'] = self._get_cur_time()
