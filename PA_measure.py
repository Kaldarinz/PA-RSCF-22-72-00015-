from pylablib.devices import Thorlabs
from scipy.fftpack import rfft, irfft, fftfreq
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import Oscilloscope
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import matplotlib.gridspec as gridspec
import keyboard
import time

# data formats
# scan_data[X scan points, Y scan points, 4, signal data points]
# in scan_data.shape[2]: 0 - raw data, 1 - FFT filtered data, 2 - frquencies, 3 - spectral data

#spec_data[WL index, data type, data points]
#data type: 0 - raw data, 1 - filtered data, 2 - FFT data
#for real data data points[0] - start WL, [1] - end WL, [2] - step WL, [3] - dt, [4] - max amp
#for FFT data data points[0] - start freq, [1] - end freq, [2] - step freq

osc_params = {
    'pre_time': 30, # [us] start time of data storage before trigger
    'frame_duration': 150, # [us] whole duration of the stored frame
    'pm_response_time': 2500, # [us] response time of the power meter
    'pm_pre_time': 300,
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
        
        self.fig = fig
        self.ax_sp = ax_sp
        self.ax_raw = ax_raw
        self.ax_freq = ax_freq
        self.ax_filt = ax_filt
        
        self.dt = data[0,0,3]
        self.dw = data[0,2,2]
        
        self.time_data = np.linspace(0,self.dt*(data.shape[2]-6),data.shape[2]-5)*1000
        self.raw_data = data[:,0,5:]
        self.filt_data = data[:,1,5:]

        wl_points = int((data[0,0,1] - data[0,0,0])/data[0,0,2]) + 1
        self.wl_data = np.linspace(data[0,0,0],data[0,0,1],wl_points)
        self.spec_data = data[:,0,4]
        self.ax_sp.plot(self.wl_data,self.spec_data)
        self.ax_sp.set_ylabel('Normalizaed PA amp')
        self.ax_sp.set_xlabel('Wavelength, [nm]')

        freq_points = int((data[0,2,1] - data[0,2,0])/self.dw) + 1
        self.freq_data = np.linspace(data[0,2,0],data[0,2,1],freq_points)/1000
        self.fft_data = data[:,2,3:(freq_points+3)]

        self.x_max = data.shape[0]
        self.x_ind = 0
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
        title = 'X index = ' + str(self.x_ind) + '/' + str(self.x_max-1)
        self.ax_freq.set_title(title)
        
        self.ax_raw.set_ylabel('PA detector signal, [V]')
        self.ax_raw.set_xlabel('Time, [us]')
        self.ax_filt.set_ylabel('PA detector signal, [V]')
        self.ax_filt.set_xlabel('Time, [us]')
        self.ax_freq.set_ylabel('FFT signal amp')
        self.ax_freq.set_xlabel('Frequency, [kHz]')
        self.fig.align_labels()
        self.fig.canvas.draw()

def scan_vizualization(data):
    """Vizualization of scan data."""

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax_freq = fig.add_subplot(gs[1, :])
    ax_raw = fig.add_subplot(gs[0,0])
    ax_filt = fig.add_subplot(gs[0,1])
    tracker = SpectralIndexTracker(fig, ax_freq, ax_raw, ax_filt, data)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
    plt.show()

def spectral_vizualization(data, dt):
    """Vizualization of spectral data."""

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2,2)
    ax_sp = fig.add_subplot(gs[0,0])
    ax_raw = fig.add_subplot(gs[0,1])
    ax_freq = fig.add_subplot(gs[1,0])
    ax_filt = fig.add_subplot(gs[1,1])
    tracker = SpectralIndexTracker(fig,ax_sp,ax_raw,ax_freq,ax_filt,data, dt)

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

def init_stages():
    """Initiate stages"""

    print('Initializing stages...')
    stages = Thorlabs.list_kinesis_devices()

    if len(stages) < 2:
        print('Less than 2 stages detected!')
        print('Program terminated!')
        exit()

    stage1_ID = stages.pop()[0]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') #motor units [m]
    print(f'{bcolors.OKBLUE}Stage X{bcolors.ENDC} initiated. Stage X ID = {stage1_ID}')

    stage2_ID = stages.pop()[0]
    stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') #motor units [m]
    print(f'{bcolors.OKBLUE}Stage Y{bcolors.ENDC} initiated. Stage X ID = {stage2_ID}')

    return stage1, stage2

def move_to(X, Y, stage_X, stage_Y) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    
    stage_X.move_to(X/1000)
    stage_Y.move_to(Y/1000)

def wait_stages_stop(stage1 = None, stage2 = None):
    """Waits untill all specified stages stop"""

    stage1.wait_for_stop()
    stage2.wait_for_stop()

def scan(x_start, y_start, x_size, y_size, x_points, y_points):
    """Scan an area, which starts at bottom left side 
    at (x_start, y_start) and has a size (x_size, y_size) in mm.
    Checks upper scan boundary.
    Returns 2D array with normalized signal amplitudes and
    3D array with the whole normalized PA data for each scan point"""

    if (x_start + x_size ) >25:
        x_size = 25 - x_start
        print(f'{bcolors.WARNING}X scan range exceeds stage limitation!')
        print(f'X scan range reduced to {x_size}! {bcolors.ENDC}')

    if (y_start + y_size)>25:
        y_size = 25 - y_start
        print(f'{bcolors.WARNING}Y scan range exceeds stage limitation!')
        print(f'Y scan range reduced to {y_size}! {bcolors.ENDC}')

    move_to(x_start, y_start, stage_X, stage_Y) # move to starting point
    wait_stages_stop(stage_X, stage_Y)

    scan_frame = np.zeros((x_points,y_points)) #scan image of normalized amplitudes
    scan_frame_full = np.zeros((x_points,y_points,4,osc.pa_frame_size)) #0-raw data, 1-filt data, 2-freq, 3-FFT

    fig, ax = plt.subplots(1,1)
    im = ax.imshow(scan_frame, vmin = 0, vmax = 0.5)
    fig.show()

    for i in range(x_points):
        for j in range(y_points):
            x = x_start + i*(x_size/x_points)
            y = y_start + j*(y_size/y_points)

            move_to(x,y,stage_X,stage_Y)
            wait_stages_stop(stage_X,stage_Y)

            osc.measure()

            scan_frame[i,j] = osc.signal_amp/osc.laser_amp
            scan_frame_full[i,j,0,:] = osc.current_pa_data/osc.laser_amp
            print(f'normalizaed amp at ({i}, {j}) is {scan_frame[i,j]:.3f}\n')
            
            im.set_data(scan_frame)
            fig.canvas.draw()
            plt.pause(0.1)

    return scan_frame, scan_frame_full

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

def print_status(stage_X, stage_Y):
    """Prints current status and position of stages"""

    print(f'{bcolors.OKBLUE}X stage{bcolors.ENDC} homing status: {stage_X.is_homed()}, status: {stage_X.get_status()}, position: {stage_X.get_position()*1000:.2f} mm.')
    print(f'{bcolors.OKBLUE}Y stage{bcolors.ENDC} homing status: {stage_Y.is_homed()}, status: {stage_Y.get_status()}, position: {stage_Y.get_position()*1000:.2f} mm.')

def home(stage_X, stage_Y):
    """Homes stages"""

    stage_X.home(sync=False,force=True)
    stage_Y.home(sync=False,force=True)
    print('Homing started...')
    wait_stages_stop(stage_X,stage_Y)
    print('...Homing complete!')

def spectra(osc, start_wl, end_wl, step):
    """Measures spectral data"""

    print('Start measuring spectra!')
    dt = 0
    d_wl = abs(end_wl-start_wl)
    if d_wl % step:
        print(f'{bcolors.WARNING} Wrong step size!{bcolors.ENDC}')
    else:
        spectral_points = int(d_wl/step) + 1
    
    if start_wl>end_wl:
        step = -step

    if state['osc init']:
        spec_data = np.zeros((spectral_points,3,osc.pa_frame_size+5))
        print(f'Spec_data init with shape {spec_data.shape}')
    else:
        spec_data = np.zeros((spectral_points,3,10))

    spec_data = set_spec_preamble(spec_data,start_wl,end_wl,step)

    for i in range(spectral_points):
        current_wl = start_wl + step*i

        measure_ans = 'Empty'
        while measure_ans != 'Measure':
            print(f'Start measuring point {(i+1)}')
            print(f'First wavelength is {bcolors.OKBLUE}{current_wl}{bcolors.ENDC}. Please set it!')
            measure_ans = inquirer.rawlist(
                message='Chose an action:',
                choices=['Tune power','Measure','Stop measurements']
            ).execute()
            if measure_ans == 'Tune power':
                if state['osc init']:
                    track_power(40)
                else:
                    print(f'{bcolors.WARNING}Oscilloscope in not initialized!{bcolors.ENDC}')
            elif measure_ans == 'Measure':
                if state['osc init']:
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
                        spec_data[i,0,5:] = osc.current_pa_data/osc.laser_amp
                    else:
                        measure_ans = 'Bad data' #trigger additional while cycle
                else:
                    print(f'{bcolors.WARNING}Oscilloscope in not initialized!{bcolors.ENDC}')
            elif measure_ans == 'Stop measurements':
                print(f'{bcolors.WARNING} Spectral measurements terminated!{bcolors.ENDC}')
                return spec_data

    print('Spectral scanning complete!')
    return spec_data

def track_power(tune_width):
    """Build power graph"""

    print(f'{bcolors.OKGREEN} Hold q button to stop power measurements{bcolors.ENDC}')
    data = np.zeros(tune_width)
    tmp_data = np.zeros(tune_width)
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1,2)
    ax_pm = fig.add_subplot(gs[0,0])
    ax_pa = fig.add_subplot(gs[0,1])
    i = 0 #iterator
    bad_read_flag = 0
    while True:
        if i <tune_width:
            osc.read_screen(osc.pm_channel)
            data[i] = osc.screen_laser_amp
            if data[i] < 0.2:
                bad_read_flag = 1
        else:
            tmp_data[:-1] = data[1:].copy()
            osc.read_screen(osc.pm_channel)
            tmp_data[tune_width-1] = osc.screen_laser_amp
            if tmp_data[tune_width-1] < 0.2:
                bad_read_flag = 1
            else:
                data = tmp_data.copy()
        ax_pm.clear()
        ax_pa.clear()
        ax_pm.plot(osc.screen_data)
        ax_pa.plot(data)
        ax_pa.set_ylim(bottom=0) 
        fig.canvas.draw()
        plt.pause(0.1)
        if bad_read_flag == 1:
            bad_read_flag = 0
        else:
            i += 1
        if keyboard.is_pressed('q'):
            break
        time.sleep(0.1)

if __name__ == "__main__":
    
    while True: #main execution loop
        menu_ans = inquirer.rawlist(
            message='Choose an action',
            choices=[
                'Init and status',
                'Power meter',
                'Move to',
                'Find beam position (scan)',
                'Measure spectrum',
                'Scan data manipulation',
                'Spectral data manipulation',
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
                    if not state['stages init']:
                        stage_X, stage_Y = init_stages()
                        state['stages init'] = True
                    else:
                        print('Stages already initiated!')

                    if not state['osc init']:
                        osc = Oscilloscope.Oscilloscope(osc_params) #add prompt for osc params
                        state['osc init'] = True
                    else:
                        print('Oscilloscope already initiated!')

                elif stat_ans == 'Home stages':
                    if state['stages init']:
                        home(stage_X, stage_Y)
                    else:
                        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

                elif stat_ans == 'Get status':
                    if state['stages init']:
                        print_status(stage_X, stage_Y)
                    else:
                        print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

                elif stat_ans == 'Back':
                    break

        elif menu_ans == 'Power meter':
            if state['osc init']:
                _ = track_power(100)
            else:
                print('Osc not init!')

        elif menu_ans == 'Move to':
            x_dest = inquirer.number(
                message='Enter X destination [mm]',
                default=0.0,
                float_allowed=True,
                min_allowed=0.0,
                max_allowed=25.0,
                filter=lambda result: float(result)
            ).execute()
            y_dest = inquirer.number(
                message='Enter Y destination [mm]',
                default=0.0,
                float_allowed=True,
                min_allowed=0.0,
                max_allowed=25.0,
                filter=lambda result: float(result)
            ).execute()

            if state['stages init']:
                print(f'Moving to ({x_dest},{y_dest})...')
                move_to(x_dest, y_dest, stage_X, stage_Y)
                wait_stages_stop(stage_X,stage_Y)
                print(f'...Mooving complete! Current position ({stage_X.get_position(scale=True)*1000:.2f},{stage_Y.get_position(scale=True)*1000:.2f})')
            else:
                print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

        elif menu_ans == 'Find beam position (scan)':

            x_start = inquirer.number(
                message='Enter X starting position [mm]',
                default=1.0,
                float_allowed=True,
                min_allowed=0.0,
                max_allowed=25.0,
                filter=lambda result: float(result)
            ).execute()

            y_start = inquirer.number(
                message='Enter Y starting position [mm]',
                default=1.0,
                float_allowed=True,
                min_allowed=0.0,
                max_allowed=25.0,
                filter=lambda result: float(result)
            ).execute()

            x_size = inquirer.number(
                message='Enter X scan size [mm]',
                default= (x_start + 3.0),
                float_allowed=True,
                min_allowed=x_start,
                max_allowed=25.0-x_start,
                filter=lambda result: float(result)
            ).execute()

            y_size = inquirer.number(
                message='Enter Y scan size [mm]',
                default= (y_start + 3.0),
                float_allowed=True,
                min_allowed=y_start,
                max_allowed=25.0-y_start,
                filter=lambda result: float(result)
            ).execute()

            x_points = inquirer.number(
                message='Enter number of X scan points',
                default= 5,
                min_allowed=2,
                max_allowed=50,
                filter=lambda result: int(result)
            ).execute()

            y_points = inquirer.number(
                message='Enter number of Y scan points',
                default= 5,
                min_allowed=2,
                max_allowed=50,
                filter=lambda result: int(result)
            ).execute()

            if state['stages init']:
                print('Scan starting...')
                scan_image, scan_data = scan(x_start, y_start, x_size, y_size, x_points, y_points)
                state['scan data'] = True

                max_amp_index = np.unravel_index(scan_image.argmax(), scan_image.shape) # find position with max PA amp
                if x_points > 1 and y_points > 1:
                    opt_x = x_start + max_amp_index[0]*x_size/(x_points-1)
                    opt_y = y_start + max_amp_index[1]*y_size/(y_points-1)
                    print(f'best pos indexes {max_amp_index}')
                    print(f'best X pos = {opt_x:.2f}')
                    print(f'best Y pos = {opt_y:.2f}')
                
                plt.show()
                print('...Scan complete!')

                confirm_move = inquirer.confirm(
                    message='Move to optimal position?'    
                ).execute()
                if confirm_move:
                    move_to(opt_x, opt_y, stage_X, stage_Y)
                    wait_stages_stop(stage_X,stage_Y)
            else:
                print(f'{bcolors.WARNING} Stages are not initialized!{bcolors.ENDC}')

        elif menu_ans == 'Measure spectrum':
            start_wl = inquirer.number(
                message='Set start wavelength, [nm]',
                default=690,
                filter=lambda result: int(result)
            ).execute()
            end_wl = inquirer.number(
                message='Set end wavelength, [nm]',
                default=950,
                filter=lambda result: int(result)
            ).execute()
            step = inquirer.number(
                message='Set step, [nm]',
                default=10,
                min_allowed=1,
                filter=lambda result: int(result)
            ).execute()

            spec_data = spectra(osc, start_wl, end_wl, step)
            state['spectral data'] = True

        elif menu_ans == 'Scan data manipulation':

            while True:
                data_ans = inquirer.rawlist(
                    message='Choose scan data action',
                    choices=[
                        'View', 
                        'FFT filtration',
                        'Load',
                        'Save',
                        'Back to main menu'
                    ]
                ).execute()
                
                if data_ans == 'View':
                    if state['scan data']:
                        scan_vizualization(scan_data, dt)
                    else:
                        print(f'{bcolors.WARNING} Scan data missing!{bcolors.ENDC}')

                elif data_ans == 'FFT filtration':
                    if state['scan data']:
                        low_cutof = inquirer.number(
                                message='Enter low cutoff frequency [Hz]',
                                default=100000,
                                min_allowed=1,
                                max_allowed=50000000,
                                filter=lambda result: int(result)
                            ).execute()

                        high_cutof = inquirer.number(
                                message='Enter high cutoff frequency [Hz]',
                                default=10000000,
                                min_allowed=(low_cutof+100000),
                                max_allowed=50000000,
                                filter=lambda result: int(result)
                            ).execute()

                        if state['osc init']:
                            dt = 1/osc.sample_rate
                        else:
                            dt = inquirer.number(
                                message='Set dt in [ns]',
                                default=20,
                                min_allowed=1,
                                max_allowed=1000000,
                                filter=lambda result: int(result)/1000000000
                            ).execute()
                        scan_data = bp_filter(scan_data, low_cutof, high_cutof, dt)
                        state['filtered scan data'] = True
                        print('FFT filtration of scan data complete!')
                    else:
                        print(f'{bcolors.WARNING} Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'Save':
                    sample = inquirer.text(
                        message='Enter Sample name',
                        default='Unknown'
                    ).execute()
                    save_scan_data(sample, scan_data, dt)

                elif data_ans == 'Load':
                    scan_data, dt = load_data('Scan')

                elif data_ans == 'Back to main menu':
                        break

        elif menu_ans == 'Spectral data manipulation':
            
            while True:
                data_ans = inquirer.rawlist(
                    message='Choose spectral data action',
                    choices=[
                        'View', 
                        'FFT filtration',
                        'Load',
                        'Save',
                        'Back to main menu'
                    ]
                ).execute()
                
                if data_ans == 'View':
                    if state['spectral data']:
                        scan_vizualization(scan_data)
                    else:
                        print(f'{bcolors.WARNING} Spectral data missing!{bcolors.ENDC}')

                elif data_ans == 'FFT filtration':
                    if state['scan data']:
                        low_cutof = inquirer.number(
                                message='Enter low cutoff frequency [Hz]',
                                default=100000,
                                min_allowed=1,
                                max_allowed=50000000,
                                filter=lambda result: int(result)
                            ).execute()

                        high_cutof = inquirer.number(
                                message='Enter high cutoff frequency [Hz]',
                                default=10000000,
                                min_allowed=(low_cutof+100000),
                                max_allowed=50000000,
                                filter=lambda result: int(result)
                            ).execute()

                        if state['osc init']:
                            dt = 1/osc.sample_rate
                        else:
                            dt = inquirer.number(
                                message='Set dt in [ns]',
                                default=20,
                                min_allowed=1,
                                max_allowed=1000000,
                                filter=lambda result: int(result)/1000000000
                            ).execute()
                        scan_data = bp_filter(scan_data, low_cutof, high_cutof, dt)
                        state['filtered scan data'] = True
                        print('FFT filtration of scan data complete!')
                    else:
                        print(f'{bcolors.WARNING} Scan data is missing!{bcolors.ENDC}')

                elif data_ans == 'Save':
                    sample = inquirer.text(
                        message='Enter Sample name',
                        default='Unknown'
                    ).execute()
                    save_spectral_data(sample, scan_data)

                elif data_ans == 'Load':
                    scan_data = load_data('Spectral')

                elif data_ans == 'Back to main menu':
                        break

        elif menu_ans == 'Exit':
            exit_ans = inquirer.confirm(
                message='Do you really want to exit?'
                ).execute()
            if exit_ans:
                if state['stages init']:
                    stage_X.close()
                    stage_Y.close()
                exit()


