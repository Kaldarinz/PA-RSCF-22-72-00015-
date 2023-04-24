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
import math
from itertools import combinations
import Validators as vd

# data formats
# scan_data[X scan points, Y scan points, 4, signal data points]
# in scan_data.shape[2]: 0 - raw data, 1 - FFT filtered data, 2 - frquencies, 3 - spectral data

#spec_data[WL index, data type, data points]
#data type: 0 - raw data, 1 - filtered data, 2 - FFT data
#for real data: 
# data points[:,0:2,0] - start WL 
# data points[:,0:2,1] - end WL
# data points[:,0:2,2] - step WL
# data points[:,0:2,3] - dt
# data points[:,0:2,4] - max amp
# data points[:,0:2,5] - laser energy (integral)
# data points[:,2,0] - start freq
# data points[:,2,1] - end freq
# data points[:,2,2] - step freq

osc_params = {
    'pre_time': 30, # [us] start time of data storage before trigger
    'frame_duration': 150, # [us] whole duration of the stored frame
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

class IndexTracker:
    '''Class for scan image vizualization'''

    def __init__(self, fig, ax_freq, ax_raw, ax_filt, data, dt):
        
        self.ax_freq = ax_freq
        self.ax_raw = ax_raw
        self.ax_filt = ax_filt
        self.fig = fig
        self.dt = dt*1000000
        self.freq_data = data[0,0,2,:]/1000
        self.raw_data = data[:,:,0,:]
        self.filt_data = data[:,:,1,:]
        self.fft_data = data[:,:,3,:]
        self.x_max = data.shape[0]
        self.y_max = data.shape[1]
        self.x_ind = 0
        self.y_ind = 0

        self.time_data = np.linspace(0,self.dt*(self.raw_data.shape[2]-1),self.raw_data.shape[2])
        self.update()

    def on_key_press(self, event):
        if event.key == 'left':
            if self.x_ind == 0:
                pass
            else:
                self.x_ind -= 1
                self.update()
        elif event.key == 'right':
            if self.x_ind == (self.x_max - 1):
                pass
            else:
                self.x_ind += 1
                self.update()
        elif event.key == 'down':
            if self.y_ind == 0:
                pass
            else: 
                self.y_ind -= 1
                self.update()
        elif event.key == 'up':
            if self.y_ind == (self.y_max - 1):
                pass
            else:
                self.y_ind += 1
                self.update()

    def update(self):
        self.ax_freq.clear()
        self.ax_raw.clear()
        self.ax_filt.clear()
        self.ax_raw.plot(self.time_data, self.raw_data[self.x_ind,self.y_ind,:])
        self.ax_filt.plot(self.time_data, self.filt_data[self.x_ind,self.y_ind,:])
        self.ax_freq.plot(self.freq_data, self.fft_data[self.x_ind,self.y_ind,:])
        title = 'X index = ' + str(self.x_ind) + '/' + str(self.x_max-1) + '. Y index = ' + str(self.y_ind) + '/' + str(self.y_max-1)
        self.ax_freq.set_title(title)
        
        self.ax_raw.set_ylabel('PA detector signal, [V]')
        self.ax_raw.set_xlabel('Time, [us]')
        self.ax_filt.set_ylabel('PA detector signal, [V]')
        self.ax_filt.set_xlabel('Time, [us]')
        self.ax_freq.set_ylabel('FFT signal amp')
        self.ax_freq.set_xlabel('Frequency, [kHz]')
        self.fig.align_labels()
        self.fig.canvas.draw()

class SpectralIndexTracker:
    '''Class for spectral data vizualization'''

    def __init__(self, fig, ax_sp, ax_raw, ax_freq, ax_filt, data):
        
        warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

        self.data = data
        self.fig = fig
        self.ax_sp = ax_sp
        self.ax_raw = ax_raw
        self.ax_freq = ax_freq
        self.ax_filt = ax_filt
        
        self.dt = data[0,0,3]
        self.dw = data[0,2,2]
        
        self.time_data = np.linspace(0,self.dt*(data.shape[2]-7),data.shape[2]-6)*1000000
        self.raw_data = data[:,0,6:]
        self.filt_data = data[:,1,6:]
        
        if self.dw:
            freq_points = int((data[0,2,1] - data[0,2,0])/self.dw) + 1
        else:
            freq_points = 2
        self.freq_data = np.linspace(data[0,2,0],data[0,2,1],freq_points)/1000
        self.fft_data = data[:,2,3:(freq_points+3)]

        self.x_max = data.shape[0]
        self.x_ind = 0
        self.step_wl = data[0,0,2]
        self.start_wl = data[0,0,0]
        self.stop_wl = data[0,0,1]
        wl_points = int((self.stop_wl - self.start_wl)/self.step_wl) + 1
        self.wl_data = np.linspace(self.start_wl,self.stop_wl,wl_points)
        self.spec_data = data[:,0,4]
        self.ax_sp.plot(self.wl_data,self.spec_data)
        
        self.ax_sp.set_ylabel('Normalizaed PA amp')
        self.ax_sp.set_xlabel('Wavelength, [nm]')
        self.selected, = self.ax_sp.plot(self.wl_data[self.x_ind], data[self.x_ind,0,4], 
                                         'o', alpha=0.4, ms=12, color='yellow')
        self.update()

    def on_key_press(self, event):
        if event.key == 'left':
            if self.x_ind == 0:
                pass
            else:
                self.x_ind -= 1
                self.update()
        elif event.key == 'right':
            if self.x_ind == (self.x_max - 1):
                pass
            else:
                self.x_ind += 1
                self.update()

    def update(self):
        self.ax_raw.clear()
        self.ax_freq.clear()
        self.ax_filt.clear()
        self.ax_raw.plot(self.time_data, self.raw_data[self.x_ind,:])
        self.ax_freq.plot(self.freq_data, self.fft_data[self.x_ind,:])
        self.ax_filt.plot(self.time_data, self.filt_data[self.x_ind,:])
        title = 'Wavelength:' + str(int(self.x_ind*self.step_wl+self.start_wl)) + 'nm'
        self.ax_freq.set_title(title)

        self.selected.set_data(self.wl_data[self.x_ind], self.data[self.x_ind,0,4])
        
        self.ax_raw.set_ylabel('PA detector signal, [V]')
        self.ax_raw.set_xlabel('Time, [us]')
        self.ax_filt.set_ylabel('PA detector signal, [V]')
        self.ax_filt.set_xlabel('Time, [us]')
        self.ax_freq.set_ylabel('FFT signal amp')
        self.ax_freq.set_xlabel('Frequency, [kHz]')
        self.fig.align_labels()
        self.fig.canvas.draw()

def scan_vizualization(data, dt):
    """Vizualization of scan data."""

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax_freq = fig.add_subplot(gs[1, :])
    ax_raw = fig.add_subplot(gs[0,0])
    ax_filt = fig.add_subplot(gs[0,1])
    tracker = IndexTracker(fig, ax_freq, ax_raw, ax_filt, data, dt)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
    plt.show()

def spectral_vizualization(data):
    """Vizualization of spectral data."""

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2,2)
    ax_sp = fig.add_subplot(gs[0,0])
    ax_raw = fig.add_subplot(gs[0,1])
    ax_freq = fig.add_subplot(gs[1,0])
    ax_filt = fig.add_subplot(gs[1,1])
    tracker = SpectralIndexTracker(fig,ax_sp,ax_raw,ax_freq,ax_filt,data)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
    plt.show()

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
            message='Enter X starting position [mm]',
            default='1.0',
            validate=vd.ScanRangeValidator(),
            filter=lambda result: float(result)
        ).execute()

        y_start = inquirer.text(
            message='Enter Y starting position [mm]',
            default='1.0',
            validate=vd.ScanRangeValidator(),
            filter=lambda result: float(result)
        ).execute()

        x_size = inquirer.text(
            message='Enter X scan size [mm]',
            default= str(x_start + 3.0),
            validate=vd.ScanRangeValidator(),
            filter=lambda result: float(result)
        ).execute()

        y_size = inquirer.text(
            message='Enter Y scan size [mm]',
            default= str(y_start + 3.0),
            validate=vd.ScanRangeValidator(),
            filter=lambda result: float(result)
        ).execute()

        x_points = inquirer.text(
            message='Enter number of X scan points',
            default= '5',
            validate=vd.ScanPointsValidator(),
            filter=lambda result: int(result)
        ).execute()

        y_points = inquirer.text(
            message='Enter number of Y scan points',
            default= '5',
            validate=vd.ScanPointsValidator(),
            filter=lambda result: int(result)
        ).execute()

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

                move_to(x,y,stage_X,stage_Y)
                wait_stages_stop(stage_X,stage_Y)

                osc.measure()
                if not osc.bad_read:
                    scan_frame[i,j] = osc.signal_amp/osc.laser_amp
                    scan_frame_full[i,j,0,:] = osc.current_pa_data/osc.laser_amp
                    print(f'normalizaed amp at ({i}, {j}) is {scan_frame[i,j]:.3f}\n')
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

    else:
        if not state['stages init']:
            print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')
        if not state['osc init']:
            print(f'{bcolors.WARNING} Oscilloscope is not initialized!{bcolors.ENDC}')
        return 0, 0, 0 

    dt = 1/osc.sample_rate
    state['scan data'] = True
    return scan_frame, scan_frame_full, dt

def save_scan_data(sample, data, dt):
    """Saves full data in npy format."""

    Path('measuring results/').mkdir(parents=True, exist_ok=True)
    dt = dt*1000000000

    filename = 'measuring results/Scan-' + sample + '-dt' + str(int(dt)) + 'ns'

    i = 1
    while (os.path.exists(filename + str(i) + '.npy')):
        i += 1
    filename = filename + str(i) + '.npy'
    
    np.save(filename, data)
    print('Scan data saved to ', filename)

def save_spectral_data(sample, data):
    """"Saves spectral data"""

    #format: spec_data[WL index, data type, data points]
    #data type: 0 - raw data, 1 - filtered data, 2 - FFT data
    #for real data data points[0] - start WL, [1] - end WL, [2] - step WL, [3] - dt, [4] - max amp
    #for FFT data data points[0] - start freq, [1] - end freq, [2] - step freq

    Path('measuring results/').mkdir(parents=True, exist_ok=True)

    filename = 'measuring results/Spectral-' + sample

    i = 1
    while (os.path.exists(filename + str(i) + '.npy')):
        i += 1
    filename = filename + str(i) + '.npy'
    
    np.save(filename, data)
    print('Spectral data saved to ', filename)

def load_data(data_type):
    """Return loaded data in the related format"""

    home_path = str(Path().resolve()) + '\\measuring results'
    if data_type == 'Scan':
        file_path = inquirer.filepath(
            message='Choose scan file to load:',
            default=home_path,
            validate=PathValidator(is_file=True, message='Input is not a file')
        ).execute()

        dt = int(file_path.split('dt')[1].split('ns')[0])/1000000000
        data = np.load(file_path)
        state['scan data'] = True
        print(f'...Scan data with shape {data.shape} loaded!')
        return data, dt
    
    elif data_type == 'Spectral':
        file_path = inquirer.filepath(
            message='Choose spectral file to load:',
            default=home_path,
            validate=PathValidator(is_file=True, message='Input is not a file')
        ).execute()

        data = np.load(file_path)
        state['spectral data'] = True
        print(f'...Spectral data with shape {data.shape} loaded!')
        return data

    else:
        print(f'{bcolors.WARNING}Unknown data type in Load data!{bcolors.ENDC}')

def bp_filter(data, low, high, dt):
    """Perform bandpass filtration on data
    low is high pass cutoff frequency in Hz
    high is low pass cutoff frequency in Hz
    dt is time step in seconds"""

    temp_data = data.copy()
    temp_data[:,:,2,:] = 0
    temp_data[:,:,3,:] = 0
    W = fftfreq(temp_data.shape[3], dt) # array with frequencies
    f_signal = rfft(temp_data[:,:,0,:]) # signal in f-space

    filtered_f_signal = f_signal.copy()
    filtered_f_signal[:,:,(W<low)] = 0   # high pass filtering

    if high > 1/(2.5*dt): # Nyquist frequency check
        filtered_f_signal[:,:,(W>1/(2.5*dt))] = 0 
    else:
        filtered_f_signal[:,:,(W>high_cutof)] = 0

    filtered_freq = W[(W>low)*(W<high_cutof)]

    temp_data[:,:,2,:len(filtered_freq)] = filtered_freq
    temp_data[:,:,3,:len(filtered_freq)] = f_signal[:,:,(W>low)*(W<high_cutof)]

    temp_data[:,:,1,:] = irfft(filtered_f_signal)

    return temp_data

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
        start_wl = inquirer.text(
            message='Set start wavelength, [nm]',
            default='950',
            validate=vd.WavelengthValidator(),
            filter=lambda result: int(result)
        ).execute()
        end_wl = inquirer.text(
            message='Set end wavelength, [nm]',
            default='690',
            validate=vd.WavelengthValidator(),
            filter=lambda result: int(result)
        ).execute()
        step = inquirer.text(
            message='Set step, [nm]',
            default='10',
            validate=vd.StepWlValidator(),
            filter=lambda result: int(result)
        ).execute()
        target_energy = inquirer.text(
            message='Set target energy in [mJ]',
            default='0.5',
            validate=vd.EnergyValidator(),
            filter=lambda result: float(result)
        ).execute()
        max_combinations = inquirer.text(
            message='Set maximum amount of filters',
            default='2',
            validate=vd.FilterNumberValidator(),
            filter=lambda result: int(result)
        ).execute()

        if start_wl > end_wl:
            step = -step
            
        print('Start measuring spectra!')
    
        d_wl = end_wl-start_wl
        spectral_points = int(d_wl/step) + 1

        frame_size = osc.time_to_points(osc.frame_duration)
        spec_data = np.zeros((spectral_points,3,frame_size+6))
        spec_data = set_spec_preamble(spec_data,start_wl,end_wl,step)

        for i in range(spectral_points):
            current_wl = start_wl + step*i

            print(f'{bcolors.UNDERLINE}Please remove all filters!{bcolors.ENDC}')
            energy = track_power(hardware, 50)
            print(f'Power meter energy = {energy:.0f} [uJ]')
            filters, n, _ = glass_calculator(current_wl,energy,target_energy,max_combinations,no_print=True)
            if n==0:
                print(f'{bcolors.WARNING} WARNING! No valid filter combination for {current_wl} [nm]!{bcolors.ENDC}')
                cont_ans = inquirer.confirm(message='Do you want to continue?').execute()
                if cont_ans:
                    print(f'{bcolors.WARNING} Spectral measurements terminated!{bcolors.ENDC}')
                    return spec_data

            print(f'\n{bcolors.HEADER}Start measuring point {(i+1)}{bcolors.ENDC}')
            print(f'Current wavelength is {bcolors.OKBLUE}{current_wl}{bcolors.ENDC}. Please set it!')
            _,__, target_pm_value = glass_calculator(current_wl,energy,target_energy, max_combinations)
            print(f'Target power meter energy is {target_pm_value}!')
            print(f'Please set it using {bcolors.UNDERLINE}laser software{bcolors.ENDC}')

            measure_ans = 'Empty'
            while measure_ans != 'Measure':
                measure_ans = inquirer.rawlist(
                    message='Chose an action:',
                    choices=['Tune power','Measure','Stop measurements']
                ).execute()

                if measure_ans == 'Tune power':
                    track_power(40)

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
                        spec_data[i,0:2,3] = dt
                        spec_data[i,0,4] = osc.signal_amp
                        spec_data[i,0,5] = osc.laser_amp
                        spec_data[i,0,6:] = osc.current_pa_data/osc.laser_amp
                    else:
                        measure_ans = 'Bad data' #trigger additional while cycle
                elif measure_ans == 'Stop measurements':
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
                title = f'Power={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {osc.screen_laser_amp:.1f} [uJ], Std (last 10) = {data[:i+1].std():.1f} [uJ]'

            elif i <tune_width:
                osc.read_screen(osc.pm_channel)
                bad_read_flag = osc.bad_read

                if i <11:
                    if osc.screen_laser_amp < threshold*data[:i].mean():
                        bad_read_flag = True
                    mean = data[:i].mean()
                    title = f'Power={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {mean:.1f} [uJ], Std (last 10) = {data[:i].std():.1f} [uJ]'
                else:
                    if osc.screen_laser_amp < threshold*data[i-11:i].mean():
                        bad_read_flag = True
                    mean = data[i-11:i].mean()
                    title = f'Power={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {mean:.1f} [uJ], Std (last 10) = {data[i-11:i].std():.1f} [uJ]'
                if not bad_read_flag:
                    data[i] = osc.screen_laser_amp
            else:
                tmp_data[:-1] = data[1:].copy()
                osc.read_screen(osc.pm_channel)
                bad_read_flag = osc.bad_read
                tmp_data[tune_width-1] = osc.screen_laser_amp
                mean = tmp_data[tune_width-11:-1].mean()
                title = f'Power={osc.screen_laser_amp:.1f} [uJ], Mean (last 10) = {mean:.1f} [uJ], Std (last 10) = {tmp_data[tune_width-11:-1].std():.1f} [uJ]'
                if tmp_data[tune_width-1] < threshold*tmp_data[tune_width-11:-1].mean():
                    bad_read_flag = True
                else:
                    data = tmp_data.copy()
            ax_pm.clear()
            ax_pa.clear()
            ax_pm.plot(osc.screen_data)
            ax_pa.plot(data)
            ax_pa.set_ylabel('Laser power, [uJ]')
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
        return mean

def set_new_position(stage_X, stage_Y):
    """Queries new position and move PA detector to this position"""

    if state['stages init']:
        x_dest = inquirer.text(
            message='Enter X destination [mm]',
            default='0.0',
            validate=vd.ScanRangeValidator(),
            filter=lambda result: float(result)
        ).execute()
        y_dest = inquirer.text(
            message='Enter Y destination [mm]',
            default='0.0',
            validate=vd.ScanPointsValidator(),
            filter=lambda result: float(result)
        ).execute()

        print(f'Moving to ({x_dest},{y_dest})...')
        move_to(x_dest, y_dest, stage_X, stage_Y)
        wait_stages_stop(stage_X,stage_Y)
        pos_x = stage_X.get_position(scale=True)*1000
        pos_y = stage_Y.get_position(scale=True)*1000
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
    return result, i, target_pm_value

if __name__ == "__main__":
    
    hardware = {
        'stage x': 0,
        'stage y': 0,
        'osc': 0
    }
    while True: #main execution loop
        menu_ans = inquirer.rawlist(
            message='Choose an action',
            choices=[
                'Init and status',
                'Power meter',
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
                        low_cutof = inquirer.text(
                                message='Enter low cutoff frequency [Hz]',
                                default='100000',
                                validate=vd.FreqValidator(),
                                filter=lambda result: int(result)
                            ).execute()

                        high_cutof = inquirer.text(
                                message='Enter high cutoff frequency [Hz]',
                                default='10000000',
                                validate=vd.FreqValidator(),
                                filter=lambda result: int(result)
                            ).execute()

                        if state['osc init']:
                            dt = 1/hardware['osc'].sample_rate
                        else:
                            dt = inquirer.text(
                                message='Set dt in [ns]',
                                default='20',
                                validate=vd.DtValidator(),
                                filter=lambda result: int(result)/1000000000
                            ).execute()
                        scan_data = bp_filter(scan_data, low_cutof, high_cutof, dt)
                        state['filtered scan data'] = True
                        print('FFT filtration of scan data complete!')
                    else:
                        print(f'{bcolors.WARNING} Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'Save data':
                    if state['scan data']:
                        sample = inquirer.text(
                            message='Enter Sample name',
                            default='Unknown'
                        ).execute()
                        save_scan_data(sample, scan_data, dt)
                    else:
                        print(f'{bcolors.WARNING}Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'Load data':
                    scan_data, dt = load_data('Scan')

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
                        'Back to main menu'
                    ]
                ).execute()
                
                if data_ans == 'Measure spectrum':
                    spec_data = spectra(hardware)  

                elif data_ans == 'View data':
                    if state['spectral data']:
                        spectral_vizualization(spec_data)
                    else:
                        print(f'{bcolors.WARNING} Spectral data missing!{bcolors.ENDC}')

                elif data_ans == 'FFT filtration':
                    print(f'{bcolors.WARNING} FFT filtration of spectral data in not implemented!{bcolors.ENDC}')

                elif data_ans == 'Save data':
                    if state['spectral data']:
                        sample = inquirer.text(
                            message='Enter Sample name',
                            default='Unknown'
                        ).execute()
                        save_spectral_data(sample, spec_data)
                    else:
                        print(f'{bcolors.WARNING}Spectral data is missing!{bcolors.ENDC}')

                elif data_ans == 'Load data':
                    spec_data = load_data('Spectral')

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

