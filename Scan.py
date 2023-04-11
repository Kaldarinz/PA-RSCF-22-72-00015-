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
    'pre_time': 0, # [us] start time of data storage before trigger
    'frame_duration': 150, # [us] whole duration of the stored frame
    'pm_response_time': 500, # [us] response time of the power meter
    'trigger_channel': 'CHAN1',
    'pa_channel': 'CHAN2',
}

data_storage = 1 # 1 - Save data, 0 - do not save data

# Stage parameters
# (0,0) of scan area is bottom left corner, when looking in beam direction
x_start = 1 # [mm]
y_start = 1 # [mm]
x_size = 20 # [mm]
y_size = 20 # [mm]
x_points = 5
y_points = 5

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

def scan(x_start, y_start, x_size, y_size, x_points, y_points):
    """Scan an area, which starts at bottom left side 
    at (x_start, y_start) and has a size (x_size, y_size) in mm.
    Returns 2D array with normalized signal amplitudes and
    3D array with the whole normalized PA data for each scan point"""

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
            print('normalizaed amp at ', i, j, ' Value = ', scan_frame[i,j])
            
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

if __name__ == "__main__":
    
    osc = Oscilloscope.Oscilloscope(osc_params) # initialize oscilloscope
    stage_X, stage_Y = init_stages() # initialize stages

    move_to(x_start, y_start, stage_X, stage_Y) # move to starting point
    wait_stages_stop(stage_X, stage_Y)

    scan_image, full_data = scan(x_start, y_start, x_size, y_size, x_points, y_points)

    max_amp_index = np.unravel_index(scan_image.argmax(), scan_image.shape)

    if x_points > 1 and y_points > 1:
        opt_x = x_start + max_amp_index[0]*x_size/(x_points-1)
        opt_y = y_start + max_amp_index[1]*y_size/(y_points-1)
        print('best pos indexes', max_amp_index)
        print('best X pos = ', opt_x)
        print('best Y pos = ', opt_y)

        if (input('Move to the best position? (Y/N)')) == 'Y':
            move_to(opt_x, opt_y, stage_X, stage_Y)


    plt.show()

    if data_storage == 1:
        dt = 1 / osc.sample_rate
        save_full_data(full_data, dt)
        save_image(scan_image)

    stage_X.close()
    stage_Y.close()


