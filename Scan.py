from pylablib.devices import Thorlabs
import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import time

### Configuration

sample_name = 'Water'

pre_time = 40 # start time of data storage before trigger in micro seconds
frame_duration = 150 # whole duration of the stored frame in micro seconds
pm_response_time = 500 # response time of the power meter in micro seconds

trigger_channel = 'CHAN1'
pa_channel = 'CHAN2'
averaging = 1

data_storage = 1 # 1 - Save data, 0 - do not save data

# Stage parameters
# (0,0) of scan area is bottom left corner, when looking in beam direction
x_start = 0 # [mm]
y_start = 0 # [mm]
x_size = 2 # [mm]
y_size = 2 # [mm]


def get_signal_amplitude() -> float:
    """Measure PA amplitude"""
    pass

def move_to(X, Y) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    pass

def scan(signal, x_start, y_start, x_size, y_size):
    """Scan an area, which starts at bottom left side 
    at (x_start, y_start) and has a size (x_size, y_size) in mm """
    pass