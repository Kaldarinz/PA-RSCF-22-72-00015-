from pylablib.devices import Thorlabs
import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import time
import Oscilloscope

### Configuration

sample_name = 'TiN'

osc_params = {
    'pre_time': 40, # [us] start time of data storage before trigger
    'frame_duration': 150, # [us] whole duration of the stored frame
    'pm_response_time': 500, # [us] response time of the power meter
    'trigger_channel': 'CHAN1',
    'pa_channel': 'CHAN2',
    'averaging': 1
}

data_storage = 1 # 1 - Save data, 0 - do not save data

# Stage parameters
# (0,0) of scan area is bottom left corner, when looking in beam direction
x_start = 1 # [mm]
y_start = 1 # [mm]
x_size = 20 # [mm]
y_size = 20 # [mm]
x_points = 10
y_points = 10

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

    if stage1:
        while stage1.is_moving():
            time.sleep(0.01)

    if stage2:
        while stage2.is_moving():
            time.sleep(0.01)

def scan(signal, x_start, y_start, x_size, y_size):
    """Scan an area, which starts at bottom left side 
    at (x_start, y_start) and has a size (x_size, y_size) in mm """
    pass

def save_full_data(data_full):
    """Saves full data in npy format"""

    Path('measuring results/').mkdir(parents=True, exist_ok=True)
    
    filename = 'measuring results/Sample_name-' + sample_name + '-Full'

    i = 1
    while (os.path.exists(filename + str(i) + '.npy')):
        i += 1
    filename = filename + str(i) + '.npy'
    
    np.save(filename, data_full)
    print('Data saved to ', filename)

if __name__ == "__main__":
    
    osc = Oscilloscope.Oscilloscope(osc_params) # initialize oscilloscope
    stage_X, stage_Y = init_stages() # initialize stages

    move_to(x_start, y_start, stage_X, stage_Y) # move to starting point
    wait_stages_stop(stage_X, stage_Y)

    #fig, axc = plt.subplots(2, sharex = True)
    #fig.tight_layout()
    #axc[0].set_title('Channel 1', fontsize=12)
    #axc[0].set_ylabel('Voltage, V', fontsize=11)
    #axc[1].set_title('Channel 2', fontsize=12)
    #axc[1].set_ylabel('Voltage, V', fontsize=11)
    #axc[1].set_xlabel('Time, s')
    #fig.show()

    scan_frame = np.zeros((x_points,y_points)) #scan image of normalized amplitudes
    scan_frame_full = np.zeros((x_points,y_points,osc.pa_frame_size))

    fig, ax = plt.subplots(1,1)
    im = ax.imshow(scan_frame, vmin = 0, vmax = 0.5)
    cbar = plt.colorbar(im)
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
            print('normalizaed amp at ', i, j, ' Value = ', scan_frame[i,j])
            im.set_data(scan_frame)
            fig.canvas.draw()
            plt.pause(0.1)

            #axc[0].clear()
            #axc[0].plot(osc.x_data, osc.current_pm_data[:osc.pa_frame_size], 'tab:orange', linewidth=0.7)
            #axc[1].clear()
            #axc[1].plot(osc.x_data, osc.current_pa_data, 'tab:blue', linewidth=0.3)
            #fig.canvas.draw()
            #plt.pause(0.1)
    

    plt.show()
    save_full_data(scan_frame_full)

    stage_X.close()
    stage_Y.close()


