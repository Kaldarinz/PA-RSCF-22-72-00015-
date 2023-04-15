from pylablib.devices import Thorlabs
from scipy.fftpack import rfft, irfft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import time
import Oscilloscope
#from PyInquirer import prompt
from InquirerPy import inquirer
#from examples import custom_style_2
from prompt_toolkit.validation import Validator, ValidationError

### Configuration

sample_name = 'TiN'

osc_params = {
    'pre_time': 0, # [us] start time of data storage before trigger
    'frame_duration': 150, # [us] whole duration of the stored frame
    'pm_response_time': 500, # [us] response time of the power meter
    'trigger_channel': 'CHAN1',
    'pa_channel': 'CHAN2',
}

low_cutof = 300000 # low cutoff frequency
high_cutof = 5000000 # high cutoff frequency

data_storage = 1 # 1 - Save data, 0 - do not save data
       
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

    def __init__(self, fig, ax, data, dt):
        self.ax = ax
        self.fig = fig
        self.dt = dt
        ax.set_title('Photoacoustic signal')
        ax.set_xlabel('us')
        ax.set_ylabel('V')
        self.data = data
        self.data[:,:,0] = 0
        self.x_max = data.shape[0]
        self.y_max = data.shape[1]
        self.x_ind = 0
        self.y_ind = 0

        self.time_data = np.linspace(0,self.dt*(self.data.shape[2]-1),self.data.shape[2])
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
        self.ax.clear()
        self.ax.plot(self.time_data, self.data[self.x_ind,self.y_ind,:])
        title = 'X index = ' + str(self.x_ind) + '/' + str(self.x_max-1) + '. Y index = ' + str(self.y_ind) + '/' + str(self.y_max-1)
        self.ax.set_title(title)
        self.fig.canvas.draw()

def scan_vizualization(data, dt, temporal = True):
    """Vizualization of scan data.
    temporal=True means real signal vizualization, dt is time step in s.
    temporal=False means signal in frequency domain, dt is freq step in Hz."""

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(fig, ax, data, dt)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
    plt.show()

def init_stages():
    """Initiate stages"""

    stages = Thorlabs.list_kinesis_devices()

    if len(stages) < 2:
        print('Less than 2 stages detected!')
        print('Program terminated!')
        exit()

    stage1_ID = stages.pop()[0]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') #motor units [m]
    print('Stage X initiated. Stage X ID = ', stage1_ID)

    stage2_ID = stages.pop()[0]
    stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') #motor units [m]
    print('Stage Y initiated. Stage Y ID = ', stage2_ID)

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
    scan_frame_full = np.zeros((x_points,y_points,osc.pa_frame_size)) #full data

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
            scan_frame_full[i,j,:] = osc.current_pa_data/osc.laser_amp
            print(f'normalizaed amp at ({i}, {j}) is {scan_frame[i,j]:.3f}\n')
            
            im.set_data(scan_frame)
            fig.canvas.draw()
            plt.pause(0.1)

    return scan_frame, scan_frame_full

def save_image(data):
    """Saves image in txt format"""

    Path('measuring results/').mkdir(parents=True, exist_ok=True)
    
    filename = 'measuring results/' + sample_name + '-2D'

    i = 1
    while (os.path.exists(filename + str(i) + '.txt')):
        i += 1
    filename = filename + str(i) + '.txt'
    
    np.savetxt(filename, data)
    print('Data saved to ', filename)

def save_full_data(data_full, dt):
    """Saves full data in npy format.
    By convention the first value in all PA signals in dt"""

    data_full[:,:,0] = dt
    Path('measuring results/').mkdir(parents=True, exist_ok=True)
    
    filename = 'measuring results/Sample_name-' + sample_name + '-Full'

    i = 1
    while (os.path.exists(filename + str(i) + '.npy')):
        i += 1
    filename = filename + str(i) + '.npy'
    
    np.save(filename, data_full)
    print('Data saved to ', filename)

def bp_filter(data, low, high, dt):
    """Perform bandpass filtration on data
    low is high pass cutoff frequency in Hz
    high is low pass cutoff frequency in Hz
    dt is time step in seconds"""

    W = fftfreq(data.shape[2], dt) # array with frequencies
    f_signal = rfft(data) # signal in f-space

    filtered_f_signal = f_signal.copy()
    filtered_f_signal[:,:,(W<low)] = 0   # high pass filtering

    if high > 1/(2.5*dt): # Nyquist frequency check
        filtered_f_signal[:,:,(W>1/(2.5*dt))] = 0 
    else:
        filtered_f_signal[:,:,(W>high_cutof)] = 0

    return irfft(filtered_f_signal), f_signal, W

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

if __name__ == "__main__":
    
    #osc = Oscilloscope.Oscilloscope(osc_params) # initialize oscilloscope
    #stage_X, stage_Y = init_stages() # initialize stages
    print(f"{bcolors.HEADER}Initialization complete! {bcolors.ENDC}")

    while True: #main execution loop
        menu_ans = inquirer.rawlist(
            message='Choose an action',
            choices=[
                'Get status',
                'Home stages',
                'Move to',
                'Find beam position (scan)',
                'Data manipulation',
                'Exit'
            ]
        ).execute()
        if menu_ans == 'Home stages':
            home(stage_X, stage_Y)

        elif menu_ans == 'Get status':
            print_status(stage_X, stage_Y)

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

            print(f'Moving to ({x_dest},{y_dest})...')
            move_to(x_dest, y_dest, stage_X, stage_Y)
            wait_stages_stop(stage_X,stage_Y)
            print(f'...Mooving complete! Current position ({stage_X.get_position(scale=True)*1000:.2f},{stage_Y.get_position(scale=True)*1000:.2f})')

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

            print('Scan starting...')
            scan_image, raw_data = scan(x_start, y_start, x_size, y_size, x_points, y_points)

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

        elif menu_ans == 'Data manipulation':

            while True:
                data_ans = inquirer.rawlist(
                    message='Choose data action',
                    choices=[
                        'View raw scan data', 
                        'FFT filtration of scan data', 
                        'View scan data in Fourier space', 
                        'View filtered data', 
                        'Save all data', 
                        'Back'
                    ]
                ).execute()
                
                if data_ans == 'View raw scan data':
                    dt = 1/osc.sample_rate
                    scan_vizualization(raw_data, dt)

                elif data_ans == 'FFT filtration of scan data':

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
                            max_allowed=50200000,
                            filter=lambda result: int(result)
                        ).execute()

                        filtered_data, fft_data, freq_data = bp_filter(raw_data, low_cutof, high_cutof, 1/osc.sample_rate)
                        print('FFT filtration complete')
                
                elif data_ans == 'View scan data in Fourier space':
                        scan_vizualization(fft_data, 1)
                
                elif data_ans == 'View filtered data':
                        scan_vizualization(filtered_data, 1)

                elif data_ans == 'Save all data':
                        pass

                elif data_ans == 'Back':
                        break

        elif menu_ans == 'Exit':
            exit_ans = inquirer.confirm(
                message='Do you really want to exit?'
                ).execute()
            if exit_ans:
                stage_X.close()
                stage_Y.close()
                exit()

    #if data_storage == 1:
    #    dt = 1 / osc.sample_rate
    #    save_full_data(raw_data, dt)
    #    save_image(scan_image)

    


