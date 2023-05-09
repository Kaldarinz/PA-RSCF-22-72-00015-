from pylablib.devices import Thorlabs
from scipy.fftpack import rfft, irfft, fftfreq
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
import warnings
import os.path
from pathlib import Path
import Oscilloscope
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import matplotlib.gridspec as gridspec
import keyboard
import time
from datetime import datetime
import math
from itertools import combinations
import Validators as vd
import h5py

# data formats
# scan_data[X scan points, Y scan points, 4, signal data points]
# in scan_data.shape[2]: 0 - raw data, 1 - FFT filtered data, 2 - frquencies, 3 - spectral data

# spec_data[WL index, data type, data points]
# data type: 0 - raw data, 1 - filtered data, 2 - FFT data
# additional data:
# spec_data[0,0,0] - start WL 
# spec_data[0,0,1] - end WL
# spec_data[0,0,2] - step WL
# spec_data[:,0,3] - dt
# spec_data[:,0,4] - max amp raw
# spec_data[:,1,4] - max amp filtered
# spec_data[:,0,5] - laser energy (integral) at PM (reflection from glass)
# spec_data[:,1,5] - laser energy (integral) at sample
# spec_data[0,2,0] - start freq
# spec_data[0,2,1] - end freq
# spec_data[0,2,2] - step freq

config = {
    'pre_time':2, # [us] pre time for zoom in data. Reference is max of filtered PA signal
    'post_time':13 # [us] post time for zoom in data. Reference is max of filtered PA signal
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

state = {
    'stages init': False,
    'osc init': False,
    'scan data': False,
    'spectral data': False,
    'filtered spec data': False,
    'filtered scan data': False,
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
    each contains {attrs} dics with specific information about the data group
    and dicts named {point1}, {point2}, etc, which correspond to the measured datasets
    each dataset has numpy array with actual data and additional information about the dataset."""

    def __init__(self) -> None:

        #raw data series for each measured points + dict with attributes
        self.raw_data = {
            'attrs': {
                'max dataset len': 0,
                'x var name': 'Unknown',
                'x var units': 'Unknown',
                'y var name': 'Unknown',
                'y var units': 'Unknown'
            }
        }
        #filtered data series for each measured points + dict with attributes
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
                'x var name': 'Unknown',
                'x var units': 'Unknown',
                'y var name': 'Unknown',
                'y var units': 'Unknown'
            }
        } 

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

    def set_metadata(self, data_group, **metadata):
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

    def add_measurement(self, data, **attributes):
        """Adds a datapoint to raw_data"""

        ds_name = self.build_ds_name(self.attrs['data points'])
        if not ds_name:
            print(f'{bcolors.WARNING}\
                  Max data points reached! Data cannot be added!\
                  {bcolors.ENDC}')
            return
        ds = {'data': data}
        if len(data) > self.raw_data['attrs']['max dataset len']:
            self.raw_data['attrs']['max dataset len'] = len(data)
        ds.update(attributes)
        self.raw_data.update({ds_name:ds})
        self.attrs['data points'] += 1
        self.attrs['updated'] = self._get_cur_time()

    def _get_cur_time (self):
        """returns timestamp of current time"""
        
        cur_time = time.time()
        date_time = datetime.fromtimestamp(cur_time)
        date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

        return date_time

    def save(self, filename=None):
        """Saves data to file"""

        if filename != None:
            self.attrs['filename'] = filename.split('\\')[-1]
            self.attrs['path'] = filename.split(self.attrs['filename'])[0] 

        elif self.filename and self.path:
            filename = self.attrs['path'] + self.attrs['filename']
        
        else:
            print(f'{bcolors.WARNING}\
                  Filename is not set. Data cannot be saved!\
                  {bcolors.ENDC}')
            return

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

    def load(self, filename):
        """Loads data from file"""

        self.attrs['filename'] = filename.split('\\')[-1]
        self.attrs['path'] = filename.split(self.attrs['filename'])[0] 

        with h5py.File(filename,'r') as file:
            general = file['general']
            self.attrs.update(general.attrs)
            
            raw_data = file['raw data']
            self.raw_data['attrs'].update(raw_data.attrs)
            for key in raw_data.keys():
                self.raw_data.update({key:{}})
                self.raw_data[key].update({'data': raw_data[key][:]})
                self.raw_data[key].update(raw_data[key].attrs)

            filt_data = file['filtered data']
            self.filt_data['attrs'].update(filt_data.attrs)
            for key in filt_data.keys():
                self.filt_data.update({key:{}})
                self.filt_data[key].update({'data': filt_data[key][:]})
                self.filt_data[key].update(filt_data[key].attrs)

            freq_data = file['freq data']
            self.freq_data['attrs'].update(freq_data.attrs)
            for key in freq_data.keys():
                self.freq_data.update({key:{}})
                self.freq_data[key].update({'data': freq_data[key][:]})
                self.freq_data[key].update(freq_data[key].attrs)

        #for compatibility with old data
        if self.attrs.get('data points', default=0) < (len(self.raw_data) - 1):
            self.attrs.update({'data points': len(self.raw_data) - 1})

    def build_ds_name(self, n):
        """Builds and returns name of dataset"""
        
        if n <10:
            n = '00' + str(n)
        elif n<100:
            n = '0' + str(n)
        elif n<1000:
            n = str(n)
        else:
            return None
        return 'point' + n
    
    def get_ds_index(self, ds_name):
        """Returns index from dataset name"""

        n_str = ds_name.split('point')[-1]
        n = int(n_str)
        return n

    def plot(self):
        """Plots current data"""

        #plot initialization
        warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

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
        self._param_values = np.zeros(self.attrs['data points'])
        self._raw_amps = np.zeros(self.attrs['data points'])
        self._filt_amps = np.zeros(self.attrs['data points'])
        i = 0
        for ds_name, ds in self.raw_data.items():
            if ds_name != 'attrs':
                self._param_values[i] = ds['parameter value']
                self._raw_amps[i] = ds['max amp']
                i += 1
        i = 0
        for ds_name, ds in self.filt_data.items():
            if ds_name != 'attrs':
                self._filt_amps[i] = ds['max amp']
                i += 1
        
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
        x_label = self.attrs['parameter name']+', ['+self.attrs['parameter units']+']'
        self._ax_sp.set_xlabel(x_label)
        y_label = self.raw_data['attrs']['y var name']
        self._ax_sp.set_ylabel(y_label)

        self._plot_selected_raw, = self._ax_sp.plot(
            self._param_values[self._param_ind],
            self._raw_amps[self._param_ind], 
            'o', alpha=0.4, ms=12, color='yellow')
        
        self._plot_selected_filt, = self._ax_sp.plot(
            self._param_values[self._param_ind], 
            self._filt_amps[self._param_ind], 
            'o', alpha=0.4, ms=12, color='yellow')

        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._plot_update()
        plt.show()

    def _on_key_press(self, event):
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

    def _plot_update(self):
        """Updates plotted data"""

        ds_name = self.build_ds_name(self._param_ind)
        start = self.raw_data[ds_name]['x var start']
        step = self.raw_data[ds_name]['x var step']
        num = len(self.raw_data[ds_name]['data'])
        stop = (num - 1)*step + start
        time_points = np.linspace(start,stop,num)

        #update max_signal_amp(parameter) plot
        title = self.attrs['parameter name'] + ': ' + \
            str(int(self._param_values[self._param_ind])) + \
            self.attrs['parameter units']
        self._ax_sp.set_title(title)
        self._plot_selected_raw.set_data(
            self._param_values[self._param_ind],
            self._raw_amps[self._param_ind])
        self._plot_selected_filt.set_data(
            self._param_values[self._param_ind],
            self._filt_amps[self._param_ind])
        
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
        filt_max_ind = np.argwhere(self.filt_data[ds_name]['data']==filt_max)[0][0]
        filt_max_t = time_points[filt_max_ind]
        self._ax_filt.plot(filt_max_t, filt_max, 'o', alpha=0.4, ms=12, color='yellow')
        #marker for min value
        filt_min = np.amin(self.filt_data[ds_name]['data'])
        filt_min_t = time_points[np.argwhere(self.filt_data[ds_name]['data']==filt_min)[0]]
        self._ax_filt.plot(filt_min_t, filt_min, 'o', alpha=0.4, ms=12, color='yellow')
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
        raw_max_ind = np.argwhere(self.raw_data[ds_name]['data']==raw_max)[0][0]
        raw_max_t = time_points[raw_max_ind]
        self._ax_raw.plot(raw_max_t, raw_max, 'o', alpha=0.4, ms=12, color='yellow')
        #marker for min value
        raw_min = np.amin(self.raw_data[ds_name]['data'])
        raw_min_t = time_points[np.argwhere(self.raw_data[ds_name]['data']==raw_min)[0]]
        self._ax_raw.plot(raw_min_t, raw_min, 'o', alpha=0.4, ms=12, color='yellow')
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

    def get_dependance(self, data_group, value):
        """Returns an array with value from each dataset in the data_group"""

        if not self.attrs['data points']:
            print(f'{bcolors.WARNING}\
                  Attempt to read dependence from empty data\
                  {bcolors.ENDC}')
            return []
        
        if data_group == 'raw_data':
            ds_name = self.build_ds_name(0)
            if self.raw_data[ds_name].get(value) == None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from RAW_data\
                    {bcolors.ENDC}')
                return []
        
        elif data_group == 'filt_data':
            ds_name = self.build_ds_name(0)
            if self.filt_data[ds_name].get(value) == None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from FILT_data\
                    {bcolors.ENDC}')
                return []
        
        elif data_group == 'freq_data':
            ds_name = self.build_ds_name(0)
            if self.freq_data[ds_name].get(value) == None:
                print(f'{bcolors.WARNING}\
                    Attempt to read dependence of unknown VALUE from FREQ_data\
                    {bcolors.ENDC}')
                return []
        
        else:
            print(f'{bcolors.WARNING}\
                  Attempt to read dependence from unknow GROUP\
                  {bcolors.ENDC}')
            return []


def set_spec_preamble(data, start, stop, step, d_type='time'):
    """Sets preamble of spectral data"""

    if d_type == 'time':
        data[:,0:2,0] = start
        data[:,0:2,1] = stop
        data[:,0:2,2] = step
    elif d_type == 'freq':
        data[:,2,0] = start
        data[:,2,1] = stop
        data[:,2,2] = step

    return data

def init_hardware(hardware, osc_params):
    """Initialize all hardware and return them.
    Updates global state."""

    if not state['stages init']:
        init_stages(hardware) #global state for stages is updated in init_stages()      
    else:
        print(f'{bcolors.WARNING}Stages already initiated!{bcolors.ENDC}')

    if not state['osc init']:
        osc = Oscilloscope.Oscilloscope(osc_params)
        hardware.update({'osc':osc})
        if not osc.not_found:
            state['osc init'] = True
    else:
        print(f'{bcolors.WARNING}Oscilloscope already initiated!{bcolors.ENDC}')

    if state['stages init'] and state['osc init']:
        print(f'{bcolors.OKGREEN}Initialization complete!{bcolors.ENDC}')

def init_stages(hardware):
    """Initiate stages. Return their ID
    and updates global state['stages init']"""

    print('Initializing stages...')
    stages = Thorlabs.list_kinesis_devices()

    if len(stages) < 2:
        print(f'{bcolors.WARNING}Less than 2 stages detected! Try again!{bcolors.ENDC}')

    else:
        stage1_ID = stages.pop()[0]
        stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') #motor units [m]
        print(f'{bcolors.OKBLUE}Stage X{bcolors.ENDC} initiated. Stage X ID = {stage1_ID}')
        hardware.update({'stage x': stage1})

        stage2_ID = stages.pop()[0]
        stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') #motor units [m]
        print(f'{bcolors.OKBLUE}Stage Y{bcolors.ENDC} initiated. Stage X ID = {stage2_ID}')
        hardware.update({'stage y': stage2})

        state['stages init'] = True

def move_to(X, Y, hardware) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    
    hardware['stage x'].move_to(X/1000)
    hardware['stage y'].move_to(Y/1000)

def wait_stages_stop(hardware):
    """Waits untill all specified stages stop"""

    hardware['stage x'].wait_for_stop()
    hardware['stage y'].wait_for_stop()

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
            message='Enter X starting position [mm] \n(CTRL+Z to cancel)\n',
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
            message='Enter Y starting position [mm] \n(CTRL+Z to cancel)\n',
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
            message='Enter X scan size [mm] \n (CTRL+Z to cancel)\n',
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
            message='Enter Y scan size [mm] \n (CTRL+Z to cancel)\n',
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
            message='Enter number of X scan points \n(CTRL+Z to cancel)\n',
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
            message='Enter number of Y scan points\n(CTRL+Z to cancel)\n',
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

def save_data(data):
    """"Save data"""

    if not data.attrs['filename']:
        filename = inquirer.text(
            message='Enter Sample name\n(CTRL+Z to cancel)\n',
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
    return

def save_tmp_data(data):
    """"Saves temp data"""

    Path('measuring results/').mkdir(parents=True, exist_ok=True)

    filename = 'measuring results/TmpData'

    try:
        os.remove(filename)
    except OSError:
        pass
    np.save(filename, data)

def load_data(data):
    """Return loaded data in the related format"""

    home_path = str(Path().resolve()) + '\\measuring results\\'
    file_path = inquirer.filepath(
        message='Choose spectral file to load:\n(CTRL+Z to cancel)\n',
        default=home_path,
        mandatory=False,
        validate=PathValidator(is_file=True, message='Input is not a file')
    ).execute()
    if file_path == None:
        print(f'{bcolors.WARNING}Data loading canceled!{bcolors.ENDC}')
        return data
    
    if file_path.split('.')[-1] != 'hdf5':
        print(f'{bcolors.WARNING} Wrong data format! *.hdf5 is required{bcolors.ENDC}')
        return data
    new_data = MeasuredData()
    new_data.load(file_path)
    data = new_data
    state['spectral data'] = True
    print(f'... data with {len(data.raw_data)-1} PA measurements loaded!')
    return data

def bp_filter(data, data_type='spectral'):
    """Perform bandpass filtration on data
    low is high pass cutoff frequency in Hz
    high is low pass cutoff frequency in Hz
    dt is time step in seconds"""

    low_cutof = inquirer.text(
        message='Enter low cutoff frequency [Hz]\n(CTRL+Z to cancel)\n',
        default='1000000',
        mandatory=False,
        validate=vd.FreqValidator()
    ).execute()
    if low_cutof == None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return data
    else:
        low_cutof = int(low_cutof)

    high_cutof = inquirer.text(
        message='Enter high cutoff frequency [Hz]\n(CTRL+Z to cancel)\n',
        default='10000000',
        mandatory=False,
        validate=vd.FreqValidator()
    ).execute()
    if high_cutof == None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return data
    else:
        high_cutof = int(high_cutof)

    if data_type == 'spectral':
        temp_data = data[:,0,6:].copy()
        dt = data[0,0,3]
        W = fftfreq(temp_data.shape[1], dt) # array with frequencies
        f_signal = rfft(temp_data[:,:]) # signal in f-space

        filtered_f_signal = f_signal.copy()
        filtered_f_signal[:,(W<low_cutof)] = 0   # high pass filtering

        if high_cutof > 1/(2.5*dt): # Nyquist frequency check
            filtered_f_signal[:,(W>1/(2.5*dt))] = 0 
        else:
            filtered_f_signal[:,(W>high_cutof)] = 0

        #pass frequencies
        filtered_freq = W[(W>low_cutof)*(W<high_cutof)]

        #start freq, end freq, step freq
        data[:,2,0] = filtered_freq.min()
        data[:,2,1] = filtered_freq.max()
        data[:,2,2] = filtered_freq[1]-filtered_freq[0]

        #Fourier amplitudes
        data[:,2,3:len(filtered_freq)+3] = f_signal[:,(W>low_cutof)*(W<high_cutof)]
        
        #filtered PA data
        data[:,1,6:] = irfft(filtered_f_signal)

        #norm filtered laser amp
        data[:,1,4] = np.amax(data[:,1,6:], axis=1)-np.amin(data[:,1,6:], axis=1)

        state['filtered spec data'] = True
        print(f'{bcolors.OKGREEN} FFT filtration of spectral data complete!{bcolors.ENDC}')
    
    return data

def print_status(hardware):
    """Prints current status and position of stages and oscilloscope"""
    
    if state['stages init']:
        stage_X = hardware['stage x']
        stage_Y = hardware['stage y']
        print(f'{bcolors.OKBLUE} Stages are initiated!{bcolors.ENDC}')
        print(f'{bcolors.OKBLUE}X stage{bcolors.ENDC} \
              homing status: {stage_X.is_homed()}, \
              status: {stage_X.get_status()}, \
              position: {stage_X.get_position()*1000:.2f} mm.')
        print(f'{bcolors.OKBLUE}Y stage{bcolors.ENDC} \
              homing status: {stage_Y.is_homed()}, \
              status: {stage_Y.get_status()}, \
              position: {stage_Y.get_position()*1000:.2f} mm.')
    else:
        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

    if state['osc init']:
        print(f'{bcolors.OKBLUE}Oscilloscope is initiated!{bcolors.ENDC}')
    else:
        print(f'{bcolors.WARNING} Oscilloscope is not initialized!{bcolors.ENDC}')

    if state['stages init'] and state['osc init']:
        print(f'{bcolors.OKGREEN} All hardware is initiated!{bcolors.ENDC}')

def home(hardware):
    """Homes stages"""

    if state['stages init']:
        hardware['stage x'].home(sync=False,force=True)
        hardware['stage y'].home(sync=False,force=True)
        print('Homing started...')
        wait_stages_stop(hardware)
        print(f'{bcolors.OKGREEN}...Homing complete!{bcolors.ENDC}')
    else:
        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

def spectra(hardware):
    """Measures spectral data.
    Updates global state"""

    osc = hardware['osc']

    if state['osc init']:
        power_control = inquirer.select(
            message='Choose method for laser energy control:',
            choices=[
                'Glan prism',
                'Filters'
            ],
            mandatory=False
        ).execute()
        if power_control == None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return 0

        start_wl = inquirer.text(
            message='Set start wavelength, [nm]\n(CTRL+Z to cancel)\n',
            default='950',
            mandatory=False,
            validate=vd.WavelengthValidator()
        ).execute()
        if start_wl == None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return 0
        else:
            start_wl = int(start_wl)

        end_wl = inquirer.text(
            message='Set end wavelength, [nm]\n(CTRL+Z to cancel)\n',
            default='690',
            mandatory=False,
            validate=vd.WavelengthValidator()
        ).execute()
        if end_wl == None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return 0
        else:
            end_wl = int(end_wl)

        step = inquirer.text(
            message='Set step, [nm]\n(CTRL+Z to cancel)\n',
            default='10',
            mandatory=False,
            validate=vd.StepWlValidator()
        ).execute()
        if step == None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return 0
        else:
            step = int(step)

        target_energy = inquirer.text(
            message='Set target energy in [mJ]\n(CTRL+Z to cancel)\n',
            default='0.5',
            mandatory=False,
            validate=vd.EnergyValidator()
        ).execute()
        if target_energy == None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return 0
        else:
            target_energy = float(target_energy)

        if power_control == 'Filters':
            max_combinations = inquirer.text(
                message='Set maximum amount of filters\n(CTRL+Z to cancel)\n',
                default='2',
                mandatory=False,
                validate=vd.FilterNumberValidator()
            ).execute()
            if max_combinations == None:
                print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
                return 0
            else:
                max_combinations = int(max_combinations)

        averaging = inquirer.text(
            message='Set averaging\n(CTRL+Z to cancel)\n',
            default='5',
            mandatory=False,
            validate=vd.AveragingValidator()
        ).execute()
        if averaging == None:
            print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
            return 0     
        else:
            averaging = int(averaging)  

        if start_wl > end_wl:
            step = -step
            
        print('Start measuring spectra!')
    
        d_wl = end_wl-start_wl

        if d_wl%step:
            spectral_points = int(d_wl/step) + 2
        else:
            spectral_points = int(d_wl/step) + 1

        frame_size = osc.time_to_points(osc.frame_duration)
        spec_data = np.zeros((spectral_points,3,frame_size+6))
        spec_data = set_spec_preamble(spec_data,start_wl,end_wl,step)

        for i in range(spectral_points):
            if abs(step*i) < abs(d_wl):
                current_wl = start_wl + step*i
            else:
                current_wl = end_wl

            tmp_signal = 0
            tmp_laser = 0
            counter = 0
            print(f'\n{bcolors.HEADER}Start measuring point {(i+1)}{bcolors.ENDC}')
            print(f'Current wavelength is {bcolors.OKBLUE}{current_wl}{bcolors.ENDC}. Please set it!')
            if power_control == 'Filters':
                print(f'{bcolors.UNDERLINE}Please remove all filters!{bcolors.ENDC}')
                energy = track_power(hardware, 50)
                print(f'Power meter energy = {energy:.0f} [uJ]')
                filters, n,_,_ = glass_calculator(current_wl,energy,target_energy,max_combinations,no_print=True)
                if n==0:
                    print(f'{bcolors.WARNING} WARNING! No valid filter combination for {current_wl} [nm]!{bcolors.ENDC}')
                    cont_ans = inquirer.confirm(message='Do you want to continue?').execute()
                    if not cont_ans:
                        print(f'{bcolors.WARNING} Spectral measurements terminated!{bcolors.ENDC}')
                        state['spectral data'] = True
                        return spec_data

                _,__, target_pm_value,_ = glass_calculator(current_wl,energy,target_energy, max_combinations)
                print(f'Target power meter energy is {target_pm_value}!')
                print(f'Please set it using {bcolors.UNDERLINE}laser software{bcolors.ENDC}')
            elif power_control == 'Glan prism':
                if i == 0:
                    target_pm_value = glan_calc_reverse(target_energy*1000)
                print(f'Target power meter energy is {target_pm_value}!')
                print(f'Please set it using {bcolors.UNDERLINE}Glan prism{bcolors.ENDC}!')
                _ = track_power(hardware, 50)
            else:
                print(f'{bcolors.WARNING}Unknown power control method! Measurements terminated!')
                return 0

            while counter < averaging:
                print(f'Signal at current WL should be measured {averaging-counter} more times.')
                measure_ans = inquirer.rawlist(
                    message='Chose an action:',
                    choices=['Tune power','Measure','Stop measurements']
                ).execute()

                if measure_ans == 'Tune power':
                    track_power(hardware, 40)

                elif measure_ans == 'Measure':
                    osc.measure()
                    dt = 1/osc.sample_rate

                    fig = plt.figure(tight_layout=True)
                    gs = gridspec.GridSpec(1,2)
                    ax_pm = fig.add_subplot(gs[0,0])
                    ax_pm.plot(signal.decimate(osc.current_pm_data, 100))
                    ax_pa = fig.add_subplot(gs[0,1])
                    ax_pa.plot(osc.current_pa_data)
                    plt.show()

                    good_data = inquirer.confirm(message='Data looks good?').execute()
                    if good_data:
                        tmp_signal += osc.signal_amp
                        tmp_laser += osc.laser_amp
                        counter += 1
                        if counter == averaging:
                            spec_data[i,0:2,3] = dt
                            spec_data[i,0,5] = tmp_laser/averaging #PM laser energy
                            if power_control == 'Filters':
                                _,__, ___, spec_data[i,1,5] = glass_calculator(
                                    current_wl,
                                    spec_data[i,0,5], #cuz average energy is saved
                                    target_energy,
                                    max_combinations,
                                    no_print=True)
                                spec_data[i,0,4] = tmp_signal/(averaging*spec_data[i,1,5]) # PA signal norm to sample energy
                                _,__, ___, sample_energy = glass_calculator(
                                    current_wl,
                                    osc.laser_amp, #cuz signal is normed to this particular energy
                                    target_energy,
                                    max_combinations,
                                    no_print=True)
                                spec_data[i,0,6:] = osc.current_pa_data/sample_energy
                            elif power_control == 'Glan prism':
                                spec_data[i,1,5] = glan_calc(spec_data[i,0,5]) #energy at sample in average
                                spec_data[i,0,4] = tmp_signal/(averaging*spec_data[i,1,5]) # PA signal norm to sample energy
                                spec_data[i,0,6:] = osc.current_pa_data/glan_calc(osc.laser_amp) #norm to this particular energy
                            else:
                                print(f'{bcolors.WARNING}\
                                      Unknown power control method in writing laser energy\
                                      {bcolors.ENDC}')
                            save_tmp_data(spec_data)
                            print(f'{bcolors.OKBLUE} Average laser = {spec_data[i,0,5]} [uJ]')
                            print(f'Average PA signal = {spec_data[i,0,4]}{bcolors.ENDC}')
                elif measure_ans == 'Stop measurements':
                    confirm = inquirer.confirm(
                        message='Are you sure?'
                    ).execute()
                    if confirm:
                        print(f'{bcolors.WARNING} Spectral measurements terminated!{bcolors.ENDC}')
                        state['spectral data'] = True
                        return spec_data
                else:
                    print(f'{bcolors.WARNING}Unknown command in Spectral measure menu!{bcolors.ENDC}')
        
        print(f'{bcolors.OKGREEN}Spectral scanning complete!{bcolors.ENDC}')
        state['spectral data'] = True
        return spec_data

    else:
        print(f'{bcolors.WARNING} Oscilloscope is not initializaed!{bcolors.ENDC}')
        return np.zeros((10,3,10))
    
def track_power(hardware, tune_width):
    """Build power graph"""

    if state['osc init']:
        osc = hardware['osc']
        if tune_width <11:
            print(f'{bcolors.WARNING} Wrong tune_width value!{bcolors.ENDC}')
            return
        print(f'{bcolors.OKGREEN} Hold q button to stop power measurements{bcolors.ENDC}')
        threshold = 0.1
        data = np.zeros(tune_width)
        tmp_data = np.zeros(tune_width)
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1,2)
        ax_pm = fig.add_subplot(gs[0,0])
        ax_pa = fig.add_subplot(gs[0,1])
        i = 0 #iterator
        mean = 0
        bad_read_flag = False

        while True:
            if i == 0:
                osc.read_screen(osc.pm_channel)
                bad_read_flag = osc.bad_read
                if osc.screen_laser_amp < 1:
                    bad_read_flag = True
                title = f'Energy={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {osc.screen_laser_amp:.1f} [uJ], Std (last 10) = {data[:i+1].std():.1f} [uJ]'
                if not bad_read_flag:
                    data[i] = osc.screen_laser_amp

            elif i <tune_width:
                osc.read_screen(osc.pm_channel)
                bad_read_flag = osc.bad_read

                if i <11:
                    if osc.screen_laser_amp < threshold*data[:i].mean():
                        bad_read_flag = True
                    mean = data[:i].mean()
                    title = f'Energy={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {mean:.1f} [uJ], Std (last 10) = {data[:i].std():.1f} [uJ]'
                else:
                    if osc.screen_laser_amp < threshold*data[i-11:i].mean():
                        bad_read_flag = True
                    mean = data[i-11:i].mean()
                    title = f'Energy={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {mean:.1f} [uJ], Std (last 10) = {data[i-11:i].std():.1f} [uJ]'
                if not bad_read_flag:
                    data[i] = osc.screen_laser_amp
            
            else:
                tmp_data[:-1] = data[1:].copy()
                osc.read_screen(osc.pm_channel)
                bad_read_flag = osc.bad_read
                tmp_data[tune_width-1] = osc.screen_laser_amp
                mean = tmp_data[tune_width-11:-1].mean()
                title = f'Energy={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {mean:.1f} [uJ], Std (last 10) = {tmp_data[tune_width-11:-1].std():.1f} [uJ]'
                if tmp_data[tune_width-1] < threshold*tmp_data[tune_width-11:-1].mean():
                    bad_read_flag = True
                else:
                    data = tmp_data.copy()
            ax_pm.clear()
            ax_pa.clear()
            ax_pm.plot(osc.screen_data)
            ax_pa.plot(data)
            ax_pa.set_ylabel('Laser energy, [uJ]')
            ax_pa.set_title(title)
            ax_pa.set_ylim(bottom=0)
            fig.canvas.draw()
            plt.pause(0.1)
            if bad_read_flag:
                bad_read_flag = False
            else:
                i += 1
            if keyboard.is_pressed('q'):
                break
            time.sleep(0.1)

        return mean
    else:
        print(f'{bcolors.WARNING}Oscilloscope in not initialized!{bcolors.ENDC}')
        return 0

def set_new_position(hardware):
    """Queries new position and move PA detector to this position"""

    if state['stages init']:
        x_dest = inquirer.text(
            message='Enter X destination [mm] \n(CTRL + Z to cancel)\n',
            default='0.0',
            validate=vd.ScanRangeValidator(),
            mandatory=False
        ).execute()
        if x_dest == None:
            print(f'{bcolors.WARNING} Input terminated! {bcolors.ENDC}')
            return
        else:
            x_dest = float(x_dest)
        
        y_dest = inquirer.text(
            message='Enter Y destination [mm] \n(CTRL + Z to cancel)\n',
            default='0.0',
            validate=vd.ScanRangeValidator(),
            mandatory=False
        ).execute()
        if y_dest == None:
            print(f'{bcolors.WARNING} Input terminated! {bcolors.ENDC}')
            return
        else:
            y_dest = float(y_dest)

        print(f'Moving to ({x_dest},{y_dest})...')
        move_to(x_dest, y_dest, hardware)
        wait_stages_stop(hardware)
        pos_x = hardware['stage x'].get_position(scale=True)*1000
        pos_y = hardware['stage y'].get_position(scale=True)*1000
        print(f'{bcolors.OKGREEN}...Mooving complete!{bcolors.ENDC} Current position ({pos_x:.2f},{pos_y:.2f})')
    else:
        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

def remove_zeros(data):
    """change zeros in filters data by linear fit from nearest values"""

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

def calc_od(data):
    """calculates OD using thickness of filters"""
    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            data[i+1,j+2] = data[i+1,j+2]*data[0,j+2]
    return data

def glass_calculator(wavelength, current_energy_pm, target_energy, max_combinations, no_print=False):
    """Return filter combinations whith close transmissions
    which are higher than required"""

    result = {}
    filename = 'ColorGlass.txt'

    try:
        data = np.loadtxt(filename,skiprows=1)
        header = open(filename).readline()
    except FileNotFoundError:
        print(f'{bcolors.WARNING} File with color glass data not found!{bcolors.ENDC}')
        return {}
    except ValueError as er:
        print(f'Error message: {str(er)}')
        print(f'{bcolors.WARNING} Error while loading color glass data!{bcolors.ENDC}')
        return {}
    
    data = remove_zeros(data)
    data = calc_od(data)
    filter_titles = header.split('\n')[0].split('\t')[2:]

    try:
        wl_index = np.where(data[1:,0] == wavelength)[0][0] + 1
    except IndexError:
        print(f'{bcolors.WARNING} Target WL is missing in color glass data table!{bcolors.ENDC}')
        return {}

    filter_dict = {}
    for key, value in zip(filter_titles,data[wl_index,2:]):
        filter_dict.update({key:value})

    filter_combinations = {}
    for i in range(max_combinations):
        for comb in combinations(filter_dict.items(),i+1):
            key = ''
            value = 0
            for k,v in comb:
                key +=k
                value+=v
            filter_combinations.update({key:math.pow(10,-value)})    

    target_energy = target_energy*1000
    laser_energy = current_energy_pm/data[wl_index,1]*100

    if laser_energy == 0:
        print(f'{bcolors.WARNING} Laser radiation is not detected!{bcolors.ENDC}')
        return 0,0,0,0
    
    target_transm = target_energy/laser_energy
    if not no_print:
        print(f'Target energy = {target_energy} [uJ]')
        print(f'Current laser output = {laser_energy:.0f} [uJ]')
        print(f'Target transmission = {target_transm*100:.1f} %')
        print(f'{bcolors.HEADER} Valid filter combinations:{bcolors.ENDC}')
    filter_combinations = dict(sorted(filter_combinations.copy().items(), key=lambda item: item[1]))

    i=0
    for key, value in filter_combinations.items():
        if (value-target_transm) > 0 and value/target_transm < 2.5:
            result.update({key: value})
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

    target_pm_value = target_energy*data[wl_index,1]/100
    return result, i, target_pm_value, laser_energy

def calc_filters_for_energy(hardware):
    """Provides required filter combination for an energy"""

    max_combinations = 3 #max filters

    wl = inquirer.text(
        message='Set wavelength, [nm]\n(CTRL+Z to cancel)\n',
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
        message='Set target energy in [mJ]\n(CTRL+Z to cancel)\n',
        default='0.5',
        mandatory=False,
        validate=vd.EnergyValidator()
    ).execute()
    if target_energy == None:
        print(f'{bcolors.WARNING}Intup terminated!{bcolors.WARNING}')
        return
    else:
        target_energy = float(target_energy)

    print(f'{bcolors.UNDERLINE}Please remove all filters!{bcolors.ENDC}')
    energy = track_power(hardware, 50)
    print(f'Power meter energy = {energy:.0f} [uJ]')
    filters, n, _, _ = glass_calculator(wl,energy,target_energy,max_combinations,no_print=True)
    if n==0:
        print(f'{bcolors.WARNING} WARNING! No valid filter combination!{bcolors.ENDC}')

    _,__, target_pm_value, _ = glass_calculator(wl,energy,target_energy, max_combinations)
    print(f'Target power meter energy is {target_pm_value}!')
    print(f'Please set it using {bcolors.UNDERLINE}laser software{bcolors.ENDC}')

def glan_calc(energy):
    """Calculates energy at sample for a given energy"""

    filename = 'GlanCalibr.txt' # file with Glan calibrations
    fit_order = 1 #order of the polynom for fitting data

    try:
        calibr_data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f'{bcolors.WARNING} File with color glass data not found!{bcolors.ENDC}')
        return [0]
    except ValueError as er:
        print(f'Error message: {str(er)}')
        print(f'{bcolors.WARNING} Error while loading color glass data!{bcolors.ENDC}')
        return [0]

    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    fit = np.poly1d(coef)

    return fit(energy)

def glan_calc_reverse(target_energy):
    """Calculates energy at power meter placed at glass reflection
    to obtain target_energy"""

    filename = 'GlanCalibr.txt' # file with Glan calibrations
    fit_order = 1 #order of the polynom for fitting data

    try:
        calibr_data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f'{bcolors.WARNING} File with color glass data not found!{bcolors.ENDC}')
        return [0]
    except ValueError as er:
        print(f'Error message: {str(er)}')
        print(f'{bcolors.WARNING} Error while loading color glass data!{bcolors.ENDC}')
        return [0]

    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    if fit_order == 1:
        # target_energy = coef[0]*energy + coef[1]
        energy = (target_energy - coef[1])/coef[0]
    else:
        print(f'{bcolors.WARNING} Reverse Glan calculation for nonlinear fit is not realized!{bcolors.ENDC}')
        return None    
    
    return energy

def glan_check(hardware):
    """Used to check glan performance"""

    damage_threshold = 800 # [uJ] maximum value, which does not damage PM

    print(f'{bcolors.HEADER}Start procedure to check Glan performance{bcolors.ENDC}')
    print(f'Do not try energies at sample large than {bcolors.UNDERLINE} 800 uJ {bcolors.ENDC}!')
    
    while True:
        print(f'\nSet some energy at glass reflection')
        energy = track_power(hardware, 50)
        target_energy = glan_calc(energy)
        if target_energy > damage_threshold:
            print(f'{bcolors.WARNING} Energy at sample will damage the PM, set smaller energy!{bcolors.ENDC}')
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
            print(f'{bcolors.WARNING}Unknown command in Glan chack menu!{bcolors.ENDC}')

def export_to_txt(data, sample, data_type='spectral'):
    """CLI method for export data to txt"""

    if data_type == 'spectral':
        if state['spectral data'] and state['filtered spec data']:
            
            #backward compatibility with old data format
            power_control = ''
            if data[0,1,5] == 0:
                power_control = inquirer.rawlist(
                    message='Choose method of power control',
                    choices=[
                        'Filters',
                        'Glan prism'
                    ]
                ).execute()
            export_type = inquirer.rawlist(
                message='Choose data to export:',
                choices=[
                    'Raw data',
                    'Filtered data',
                    'Freq data',
                    'Spectral',
                    'All',
                    'back'
                ]
            ).execute()
            
            if export_type == 'back':
                return
            elif export_type == 'Raw data':
                if len(power_control):
                    save_spectr_raw_txt(data,sample,power_control)
                else:
                    save_spectr_raw_txt(data,sample)
            elif export_type == 'Filtered data':
                if len(power_control):
                    save_spectr_filt_txt(data,sample,power_control)
                else:
                    save_spectr_filt_txt(data,sample)
            elif export_type == 'Freq data':
                if len(power_control):
                    save_spectr_freq_txt(data,sample, power_control)
                else:
                    save_spectr_freq_txt(data,sample)
            elif export_type == 'Spectral':
                if len(power_control):
                    save_spectr_txt(data,sample,power_control)
                else:
                    save_spectr_txt(data,sample)
            elif export_type == 'All':
                if len(power_control):
                    save_spectr_raw_txt(data,sample,power_control)
                    save_spectr_filt_txt(data,sample, power_control)
                    save_spectr_freq_txt(data,sample, power_control)
                    save_spectr_txt(data,sample, power_control)
                else:
                    save_spectr_raw_txt(data,sample)
                    save_spectr_filt_txt(data,sample)
                    save_spectr_freq_txt(data,sample)
                    save_spectr_txt(data,sample)
            else:
                print(f'{bcolors.WARNING} Unknown command in data export menu {bcolors.ENDC}')

        else:
            if not state['spectral data']:
                print(f'{bcolors.WARNING}Spectral data is missing!{bcolors.ENDC}')
            elif not state['filtered spec data']:
                print(f'{bcolors.WARNING}Spectral data is not filtered!{bcolors.ENDC}')
            else:
                print(f'{bcolors.WARNING}Unknown state of spectral data!{bcolors.ENDC}')
            return

def save_spectr_filt_txt(data,sample, power_control = ''):
    """Saves filtered data to txt
    corrects for sample energy"""

    if not len(sample):
        filename = 'measuring results/txt data/Spectral-Unknown-filt.txt'
    else:
        filename = sample.split('.npy')[0]
        filename += '-filt.txt'
    if os.path.exists(filename):
        filename_tmp = filename.split('/')[-1]
        override = inquirer.confirm(
            message='Do you want to override file ' + filename_tmp + '?'
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

    start_freq = data[0,2,0]/1000000
    end_freq = data[0,2,1]/1000000

    dt = data[0,0,3]
    start_wl = data[0,0,0]
    end_wl = data[0,0,1]
    step_wl = data[0,0,2]
    duration = (config['pre_time'] + config['post_time'])/1000000
    spectr_points = int(duration/dt)+1
    pre_points = int(config['pre_time']/1000000/dt)
    post_points = spectr_points - pre_points
    data_txt = np.zeros((spectr_points+2,data.shape[0]+1))

    header = 'Time;'
    for _ in range(data_txt.shape[1]-1):
        header +='filt signal;'
    header += '\nus;'
    for _ in range(data_txt.shape[1]-1):
        header +='V;'
    header += '\n'
    header += f'Filtered data with band pass filter ({start_freq:.1f}:{end_freq:.1f}) MHz\n'
    header += 'First line is WL, Second line is laser energy in [uJ]'
    
    #build aray for txt data
    for i in range(data.shape[0]):
        if i < (data.shape[0]-1):
            data_txt[0,i+1] = start_wl + i*step_wl
        else: #handels case when the last step is smaller then others
            data_txt[0,i+1] = end_wl

        pm_energy = data[i,0,5]
        max_amp_ind = np.argmax(data[i,1,6:])
        #copy zoom data and check not go outside boundaries
        if (max_amp_ind-pre_points) > 0 and (6+max_amp_ind+post_points)< data.shape[2]+1:
            tmp = data[i,1,6+max_amp_ind-pre_points:6+max_amp_ind+post_points].copy()
        elif (max_amp_ind-pre_points) < 0:
            tmp = data[i,1,6:6+max_amp_ind+post_points].copy()
        elif (6+max_amp_ind+post_points)> data.shape[2]+1:
            tmp = data[i,1,6+max_amp_ind-pre_points:-1].copy()
        #if tmp is too small, add zeros to the end
        if len(tmp) < len(data_txt[2:,i+1]):
            filled_arr = np.zeros(len(data_txt[2:,i+1]))
            filled_arr[:len(tmp)] = tmp
            data_txt[2:,i+1] = filled_arr
        else:
            data_txt[2:,i+1] = tmp
        
        if power_control == 'Filters':
            _,__,___,sample_energy = glass_calculator(
               data_txt[0,i+1],
               pm_energy,
               pm_energy*20,
               2,
               no_print=True
            )
            data_txt[1,i+1] = sample_energy #laser energy at sample
            data_txt[2:,i+1] = data_txt[2:,i+1]*pm_energy/sample_energy
        elif power_control == 'Glan prism':
            sample_energy = glan_calc(pm_energy)
            data_txt[1,i+1] = sample_energy #laser energy at sample
            data_txt[2:,i+1] = data_txt[2:,i+1]*pm_energy/sample_energy
        else:
            data_txt[1,i+1] = data[i,1,5] #laser energy

    for i in range(spectr_points):
        data_txt[i+2,0] = i*dt*1000000
    
    np.savetxt(filename,data_txt,header=header,fmt='%1.3e', delimiter=';')
    print(f'Data exported to {bcolors.OKGREEN}{filename}{bcolors.ENDC}')

def save_spectr_raw_txt(data,sample, power_control = ''):
    """Saves raw data to txt"""

    if not len(sample):
        filename = 'measuring results/txt data/Spectral-Unknown-raw.txt'
    else:
        filename = sample.split('.npy')[0]
        filename += '-raw.txt'
    if os.path.exists(filename):
        filename_tmp = filename.split('/')[-1]
        override = inquirer.confirm(
            message='Do you want to override file ' + filename_tmp + '?'
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

    dt = data[0,0,3]
    start_wl = data[0,0,0]
    end_wl = data[0,0,1]
    step_wl = data[0,0,2]
    duration = (config['pre_time'] + config['post_time'])/1000000
    spectr_points = int(duration/dt)+1
    pre_points = int(config['pre_time']/1000000/dt)
    post_points = spectr_points - pre_points
    data_txt = np.zeros((spectr_points+2,data.shape[0]+1))

    header = 'Time;'
    for _ in range(data_txt.shape[1]-1):
        header +='raw signal;'
    header += '\nus;'
    for _ in range(data_txt.shape[1]-1):
        header +='V;'
    header += '\n'
    header += f'Raw data\n'
    header += 'First line is WL, Second line is laser energy in [uJ]'
    
    #build aray for txt data
    for i in range(data.shape[0]):
        if i < (data.shape[0]-1):
            data_txt[0,i+1] = start_wl + i*step_wl
        else: #handels case when the last step is smaller then others
            data_txt[0,i+1] = end_wl

        
        pm_energy = data[i,0,5]
        max_amp_ind = np.argmax(data[i,1,6:])
        #copy zoom data and check not go outside boundaries
        if (max_amp_ind-pre_points) > 0 and (6+max_amp_ind+post_points)< data.shape[2]+1:
            tmp = data[i,0,6+max_amp_ind-pre_points:6+max_amp_ind+post_points].copy()
        elif (max_amp_ind-pre_points) < 0:
            tmp = data[i,0,6:6+max_amp_ind+post_points].copy()
        elif (6+max_amp_ind+post_points)> data.shape[2]+1:
            tmp = data[i,0,6+max_amp_ind-pre_points:-1].copy()
        
        data_txt[2:len(tmp),i+1] = tmp
        
        if power_control == 'Filters':
            _,__,___,sample_energy = glass_calculator(
               data_txt[0,i+1],
               pm_energy,
               pm_energy*20,
               2,
               no_print=True
            )
            data_txt[1,i+1] = sample_energy #laser energy at sample
            data_txt[2:,i+1] = data_txt[2:,i+1]*pm_energy/sample_energy
        elif power_control == 'Glan prism':
            sample_energy = glan_calc(pm_energy)
            data_txt[1,i+1] = sample_energy #laser energy at sample
            data_txt[2:,i+1] = data_txt[2:,i+1]*pm_energy/sample_energy
        else:
            data_txt[1,i+1] = data[i,1,5] #laser energy

    for i in range(spectr_points):
        data_txt[i+2,0] = i*dt*1000000
    
    np.savetxt(filename,data_txt,header=header,fmt='%1.3e', delimiter=';')
    print(f'Data exported to {bcolors.OKGREEN}{filename}{bcolors.ENDC}')

def save_spectr_freq_txt(data,sample, power_control = ''):
    """Saves freq data to txt"""

    if not len(sample):
        filename = 'measuring results/txt data/Spectral-Unknown-freq.txt'
    else:
        filename = sample.split('.npy')[0]
        filename += '-freq.txt'
    if os.path.exists(filename):
        filename_tmp = filename.split('/')[-1]
        override = inquirer.confirm(
            message='Do you want to override file ' + filename_tmp + '?'
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

    start_freq = data[0,2,0]/1000000
    end_freq = data[0,2,1]/1000000

    start_wl = data[0,0,0]
    end_wl = data[0,0,1]
    step_wl = data[0,0,2]
    start_freq = data[0,2,0]
    end_freq = data[0,2,1]
    step_freq = data[0,2,2]
    spectr_points = int((end_freq-start_freq)/step_freq)+1
    data_txt = np.zeros((spectr_points+2,data.shape[0]+1))

    header = 'Frequency;'
    for _ in range(data_txt.shape[1]-1):
        header +='FFT amplitude;'
    header += '\nMHz;'
    for _ in range(data_txt.shape[1]-1):
        header +='V;'
    header += '\n'
    header += f'Filtered data with band pass filter ({start_freq:.1f}:{end_freq:.1f}) MHz\n'
    header += 'First line is WL, Second line is laser energy in [uJ]'
    
    #build aray for txt data
    for i in range(data.shape[0]):
        if i < (data.shape[0]-1):
            data_txt[0,i+1] = start_wl + i*step_wl
        else: #handels case when the last step is smaller then others
            data_txt[0,i+1] = end_wl
        if data[i,1,5]:
            data_txt[1,i+1] = data[i,1,5]#laser energy at sample
        else:
            data_txt[1,i+1] = data[i,0,5] #laser energy at PM
        data_txt[2:,i+1] = data[i,2,3:3+spectr_points].copy()

    for i in range(spectr_points):
        data_txt[i+2,0] = (start_freq + i*step_freq)/1000000
    
    np.savetxt(filename,data_txt,header=header,fmt='%1.3e', delimiter=';')
    print(f'Data exported to {bcolors.OKGREEN}{filename}{bcolors.ENDC}')

def save_spectr_txt(data,sample, power_control = ''):
    """Saves spectral data to txt
    corrects for sample energy"""

    #path and filename handling
    if not len(sample):
        filename = 'measuring results/txt data/Spectral-Unknown-spectral.txt'
    else:
        filename = sample.split('.npy')[0]
        filename += '-spectral.txt'
    if os.path.exists(filename):
        filename_tmp = filename.split('/')[-1]
        override = inquirer.confirm(
            message='Do you want to override file ' + filename_tmp + '?'
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

    #formation of data
    start_freq = data[0,2,0]/1000000
    end_freq = data[0,2,1]/1000000

    header = ''
    header +='Wavelength   laser   raw filt\n'
    header +='nm   uJ  V   V\n'
    header +=f'Filtered data in range ({start_freq:.1f}:{end_freq:.1f}) MHz\n'
    header +='First col is wavelength in [nm]\n'
    header +='Second col is laser energy in [uJ]\n'
    header +='Third col is normalized (to laser energy) raw PA amp in [V]\n'
    header +='Forth col is normalized (to laser energy) filt PA amp in [V]'

    start_wl = data[0,0,0]
    end_wl = data[0,0,1]
    step_wl = data[0,0,2]
    data_txt = np.zeros((data.shape[0],4))
    
    for i in range(data.shape[0]):
        if i < (data.shape[0]-1):
            data_txt[i,0] = start_wl + i*step_wl
        else: #handels case when the last step is smaller then others
            data_txt[i,0] = end_wl
        pm_energy = data[i,0,5]
        if power_control == 'Filters':
            _,__,___,sample_energy = glass_calculator(
                data_txt[i,0],
                pm_energy,
                pm_energy*20,
                2,
                no_print=True
            )
            if not sample_energy:
                data_txt[i,1] = sample_energy #laser energy
                data_txt[i,2] = data[i,0,4]*pm_energy/sample_energy #raw norm PA amp
                data_txt[i,3] = data[i,1,4]*pm_energy/sample_energy #filt norm PA amp
            else:
                data_txt[i,1] = 0 #laser energy
                data_txt[i,2] = 0 #raw norm PA amp
                data_txt[i,3] = 0 #filt norm PA amp
        elif power_control == 'Glan prism':
            sample_energy = glan_calc(pm_energy)
            data_txt[i,1] = sample_energy #laser energy
            data_txt[i,2] = data[i,0,4]*pm_energy/sample_energy #raw norm PA amp
            data_txt[i,3] = data[i,1,4]*pm_energy/sample_energy #filt norm PA amp
        else:
            data_txt[i,1] = data[i,1,5] #laser energy
            data_txt[i,2] = data[i,0,4] #raw norm PA amp
            data_txt[i,3] = data[i,1,4] #filt norm PA amp
    
    np.savetxt(filename,data_txt,header=header,fmt='%1.3e')
    print(f'Data exported to {bcolors.OKGREEN}{filename}{bcolors.ENDC}')

if __name__ == "__main__":
    
    hardware = {
        'stage x': 0,
        'stage y': 0,
        'osc': 0
    }

    sample = '' # sample name

    spec_data = MeasuredData()
    scan_data = [] # array for scan_data
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
                ],
                height=9
            ).execute()

                if stat_ans == 'Init hardware':
                    init_hardware(hardware, osc_params)

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
                    print(f'{bcolors.WARNING}Unknown command in energy menu!{bcolors.ENDC}')

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
                    if state['scan data']:
                        scan_vizualization(scan_data, dt)
                    else:
                        print(f'{bcolors.WARNING} Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'FFT filtration':
                    if state['scan data']:
                        print(f'{bcolors.WARNING} FFT of scan data is not implemented!{bcolors.ENDC}')
                    else:
                        print(f'{bcolors.WARNING} Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'Save data':
                    if state['scan data']:
                        sample = inquirer.text(
                            message='Enter Sample name\n(CTRL+Z to cancel)\n',
                            default='Unknown',
                            mandatory=False
                        ).execute()
                        if sample == None:
                            print(f'{bcolors.WARNING}Save terminated!{bcolors.ENDC}')
                            break
                        save_scan_data(sample, scan_data, dt)
                    else:
                        print(f'{bcolors.WARNING}Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'Load data':
                    tmp_scan_data, tmp_dt = load_data('Scan', scan_data)
                    if len(tmp_scan_data) > 1:
                        scan_data = tmp_scan_data
                        dt = tmp_dt

                elif data_ans == 'Back to main menu':
                        break
                
                else:
                    print(f'{bcolors.WARNING} Unknown option is stage scanning menu!{bcolors.ENDC}')

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
                    spec_data = spectra(hardware)  

                elif data_ans == 'View data':
                    if state['spectral data']:
                        spec_data.plot()
                    else:
                        print(f'{bcolors.WARNING} Spectral data missing!{bcolors.ENDC}')

                elif data_ans == 'FFT filtration':
                   if state['spectral data']:
                       bp_filter(spec_data)
                   else:
                       print(f'{bcolors.WARNING}Spectral data is missing!{bcolors.ENDC}')

                elif data_ans == 'Save data':
                    if state['spectral data']:
                        save_data(spec_data)
                    else:
                        print(f'{bcolors.WARNING}Spectral data is missing!{bcolors.ENDC}')

                elif data_ans == 'Export to txt':
                    export_to_txt(spec_data,sample)

                elif data_ans == 'Load data':
                    spec_data = load_data(spec_data)

                elif data_ans == 'Back to main menu':
                        break         

                else:
                    print(f'{bcolors.WARNING}Unknown command in Spectral scanning menu!{bcolors.ENDC}')

        elif menu_ans == 'Exit':
            exit_ans = inquirer.confirm(
                message='Do you really want to exit?'
                ).execute()
            if exit_ans:
                if state['stages init']:
                    hardware['stage x'].close()
                    hardware['stage y'].close()
                exit()

        else:
            print(f'{bcolors.WARNING}Unknown action in the main menu!{bcolors.ENDC}')