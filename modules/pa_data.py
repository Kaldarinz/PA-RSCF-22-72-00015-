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
    |  |  |--'data_raw': ndarray[uint8]
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
    |  |  |--'data_raw': ndarray[uint8]
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

    VERSION = 1.0

    def __init__(self, dims: int=-1, params: List[str]=[]) -> None:
        """Class init.
        
        <dims> dimensionality of the stored measurement.
        <params> contain names of the dimensions."""

        #general metadata
        self.attrs: BaseMetadata = {
            'version': self.VERSION,
            'measurement_dims': dims,
            'parameter_name': params,
            'data_points': 0,
            'created': self._get_cur_time(),
            'updated': self._get_cur_time(),
            'filename': '',
            'zoom_pre_time': 2*ureg.us,
            'zoom_post_time': 13*ureg.us
        }
        raw_attrs: RawMetadata = {
            'max_len': 0,
            'x_var_name': 'Time',
            'y_var_name': 'PhotoAcoustic signal'
        }
        self.raw_data = {}
        self.raw_data.update({'attrs': raw_attrs})
        
        filt_attrs: FiltMetadata = {
            'x_var_name': 'Time',
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

    def add_measurement(
            self, 
            data: Data_point,
            param_val: List[pint.Quantity]
        ) -> None:
        """Add a single data point.
        
        Add a datapoint to raw_data, filt_data and freq_data.
        """

        ds_name = self._build_ds_name(self.attrs['data_points']+1)
        if not ds_name:
            logger.error('Max data_points reached! Data cannot be added!')
            return
        logger.debug(f'Adding {ds_name}...')
        if self.attrs['data_points']:
            params = self.raw_data['point001']['param_val']
            logger.debug(f'Param values changed from {param_val}...')
            param_val = [x.to(y.u) for x,y in zip(param_val,params)] # type: ignore
            logger.debug(f'... to {param_val}')
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

        base = os.getcwd()
        filename = os.path.join(base, 'measuring results', 'TmpData.hdf5')
        self._flush(filename)

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
            'parameter_u': [param.u for param in param_vals],
            'data_points': self.attrs['data_points'],
            'created': self.attrs['created'],
            'updated': self.attrs['updated'],
            'filename': self.attrs['filename'],
            'zoom_pre_time': self.attrs['zoom_pre_time'].m,
            'zoom_post_time': self.attrs['zoom_post_time'].m,
            'zoom_u': self.attrs['zoom_pre_time'].u
        }
        return attrs

    def _raw_attrs(self) -> dict:
        """"Build dict with raw_data attributes."""

        attrs = {
            'max_len': self.raw_data['attrs']['max_len'],
            'x_var_name': self.raw_data['attrs']['x_var_name'],
            'x_var_u': self.raw_data['point001']['x_var_step'].u,
            'y_var_name': self.raw_data['attrs']['y_var_name'],
            'y_var_u': self.raw_data['attrs']['data'][0].u
        }
        return attrs

    def _ds_attrs(self, ds_name: str) -> dict:
        """Build dict with attributes for <ds_name>"""
        
        ds_attrs = self.raw_data[ds_name]
        a, b = self._calc_data_fit(ds_name)
        attrs = {
            'param_val': [param.m for param in ds_attrs['param_val']],
            'a': a,
            'b': b,
            'x_var_step': ds_attrs['x_var_step'].m,
            'x_var_start': ds_attrs['x_var_start'].m,
            'x_var_stop': ds_attrs['x_var_stop'].m,
            'pm_en': ds_attrs['pm_en'],
            'pm_en_u': ds_attrs['pm_en'].u,
            'sample_en': ds_attrs['sample_en'].m,
            'sample_en_u': ds_attrs['sample_en'].u,
            'max_amp': ds_attrs['max_amp'].m,
            'max_amp_u': ds_attrs['max_amp'].u
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

        x = self.raw_data[ds_name]['data_raw'][:points]
        tmp = self.raw_data[ds_name]['data'][:points]
        y = [quantity.m for quantity in tmp]
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
            time_unit = general['zoom_u']
            self.attrs.update(
                {
                'version': general['version'],
                'measurement_dims': general['measurement_dims'],
                'parameter_name': general['parameter_name'],
                'data_points': general['data_points'],
                'created': general['created'],
                'updated': general['updated'],
                'filename': general['filename'],
                'zoom_pre_time': general['zoom_pre_time']*ureg(time_unit), # type: ignore
                'zoom_post_time': general['zoom_post_time']*ureg(time_unit) # type: ignore
                }
            )
            logger.debug(f'General metadata with {len(general)}'
                         + ' records loaded.')
        
            raw_data = file['raw data']
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

        p = np.poly1d(ds.attrs['a'], ds.attrs['b']) # type: ignore
        data_raw = ds[...]
        data = p(data_raw)*ureg(y_var_unit)
        p_vals = ds.attrs['param_val']
        param_val = [x*ureg(y) for x, y in zip(p_vals,param_units)] # type: ignore
        x_var_step = ds.attrs['x_var_step']*ureg(x_var_unit)
        x_var_start = ds.attrs['x_var_start']*ureg(x_var_unit)
        x_var_stop = ds.attrs['x_var_stop']*ureg(x_var_unit)
        pm_en = ds.attrs['pm_en']*ureg(ds.attrs['pm_en_u']) # type: ignore
        sample_en = ds.attrs['sample_en']* ureg(ds.attrs['sample_en_u']) # type: ignore
        max_amp = ds.attrs['max_amp']*ureg(ds.attrs['max_amp_u']) # type: ignore

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

    def _plot_0d(self) -> None:
        """Plot 0D data."""

        logger.warning('Plotting 0D data is not implemented!')

    def _plot_1d(self) -> None:
        """Plot 1D data."""

        self._param_ind = 0 #index of active data on plot

        #plot max_signal_amp(parameter)
        self._param_values = self.get_dependance('raw_data','param_val')
        self._raw_amps = self.get_dependance('raw_data', 'max_amp')
        self._filt_amps = self.get_dependance('filt_data', 'max_amp')
        
        self._ax_sp.plot(
            self._param_values,
            self._raw_amps,
            label='Max amplitude of raw data')
        self._ax_sp.plot(
            self._param_values,
            self._filt_amps,
            label='Max amplitude of filtered data'
        )
        self._ax_sp.legend(loc='upper right')
        self._ax_sp.set_ylim(bottom=0)
        x_label = (self.attrs['parameter_name'][0] 
                   + ', ' + self._ax_sp.get_xlabel())
        self._ax_sp.set_xlabel(x_label)
        y_label = (self.raw_data['attrs']['y_var_name']
                   + ', ' + self._ax_sp.get_ylabel())
        self._ax_sp.set_ylabel(y_label)

        self._plot_selected_raw, = self._ax_sp.plot(
            self._param_values[self._param_ind],
            self._raw_amps[self._param_ind],
            **self._marker_style)
        
        self._plot_selected_filt, = self._ax_sp.plot(
            self._param_values[self._param_ind],
            self._filt_amps[self._param_ind],
            **self._marker_style)

        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._plot_update()
        plt.show()

    def _plot_2d(self) -> None:
        """Plot 2D data."""

        logger.warning('Plotting 2D data is not implemented!')

    def _plot_3d(self) -> None:
        """Plot 3D data."""

        logger.warning('Plotting 3D data is not implemented!')

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

        ds_name = self._build_ds_name(self._param_ind)
        start = self.raw_data[ds_name]['x_var_start'].to('us').m
        stop = self.raw_data[ds_name]['x_var_stop'].to('us').m
        num = len(self.raw_data[ds_name]['data'])
        self._time_points = np.linspace(start,stop,num)*ureg.us

        #check if datapoint is empty
        if not step:
            return None
        self._update_pos_ax_sp()
        #update filt data
        self._plot_update_signal(
            self._ax_filt,
            self._ax_filt_zoom,
            self.filt_data[ds_name],
            self.filt_data['attrs'],
            'Filtered data'
        )
        #update raw data
        self._plot_update_signal(
            self._ax_raw,
            self._ax_raw_zoom, 
            self.raw_data[ds_name],
            self.raw_data['attrs'],
            'Raw data'
        )
        #update freq data
        self._plot_update_signal(
            self._ax_freq,
            self._ax_raw_zoom, 
            self.raw_data[ds_name],
            self.raw_data['attrs'],
            'Raw data'
        )
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

    def _update_pos_ax_sp(self) -> None:
        """Update current position in parameter subplot."""

        title = (self.attrs['parameter_name'][0] + ': '
                  + str(self._param_values[self._param_ind]))
        self._ax_sp.set_title(title)
        self._plot_selected_raw.set_data(
            self._param_values[self._param_ind],
            self._raw_amps[self._param_ind])
        self._plot_selected_filt.set_data(
            self._param_values[self._param_ind],
            self._filt_amps[self._param_ind])

    def _plot_update_signal(
            self,
            ax: plt.Axes,
            zoom_ax: plt.Axes|None=None,
            ds: dict,
            attrs: dict,
            title: str) -> None:
        """Update plot on <ax> and <zoom_ax> with signal from <ds>.
        
        <ds> is datasets for plotting.
        <attrs> is dict with attributes of group containing <ds>.
        """

        logger.debug(f'Start updating subplot {title}')

        ax.clear()
        ax.plot(self._time_points, ds['data'])
        x_label = attrs['x_var_name'] + ', ' + ax.get_xlabel()
        ax.set_xlabel(x_label)

        y_label = attrs['y_var_name'] + ', ' + ax.get_ylabel()
        ax.set_ylabel(y_label)
        ax.set_title(title)

        #marker for max value
        max_val = np.amax(ds['data'])
        max_ind = np.flatnonzero(ds['data']==max_val)[0]
        max_t = self._time_points[max_ind]
        ax.plot(max_t, max_val, **self._marker_style)
        
        #marker for min value
        min_val = np.amin(ds['data'])
        min_ind = np.flatnonzero(ds['data']==min_val)[0]
        min_t = self._time_points[min_ind]
        ax.plot(min_t, min_val, **self._marker_style)
        
        #marker for zoomed area
        step = ds['x_var_step']
        pre_time = self.attrs['zoom_pre_time']
        post_time = self.attrs['zoom_post_time']
        pre_points = int(pre_time.to(step.u).m/step.m)
        post_points = int(post_time.to(step.u).m/step.m)
        start_zoom_ind = max_ind-pre_points
        if start_zoom_ind < 0:
            start_zoom_ind = 0
        stop_zoom_ind = max_ind + post_points
        if stop_zoom_ind > (len(self._time_points) - 1):
            stop_zoom_ind = len(self._time_points) - 1
        ax.fill_betweenx(
            [min_val, max_val],
            self._time_points[start_zoom_ind],
            self._time_points[stop_zoom_ind],
            **self._fill_style
        )

        zoom_ax.clear()
        zoom_ax.plot(
            self._time_points[start_zoom_ind:stop_zoom_ind+1],
            ds['data'][start_zoom_ind:stop_zoom_ind+1]
        )
        zoom_ax.set_xlabel(x_label)
        zoom_ax.set_ylabel(y_label)
        zoom_ax.set_title('Zoom of ' + title)

    def get_dependance(self, data_group: str, value: str) -> List:
        """Return array with value from each dataset in the data_group.
        
        data_group: 'raw_data'|'filt_data|'freq_data'.
        """

        logger.debug(f'Start building array of {value} from {data_group}.')
        dep = [] #array for return values
        if not self.attrs['data_points']:
            logger.error(f'PaData instance contains no data points.')
            return dep
        check_st = 'self.'+data_group+'[self._build_ds_name(1)].get(value)'
        if eval(check_st) is None:
            logger.error(f'Attempt to read unknown attribute: {value} '
                         + f'from {data_group}.')
            return dep
        
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
