from pylablib.devices import Thorlabs
import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import time

### Configuration

sample_name = 'Water'

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
x_start = 0 # [mm]
y_start = 0 # [mm]
x_size = 2 # [mm]
y_size = 2 # [mm]

class Oscilloscope:
    __osc = None
    sample_rate = 0
    pa_frame_size = 0
    pm_frame_size = 0
    pre_points = 0
    laser_amp = 0
    signal_amp = 0

    def __init__(self, osc_params) -> None:
        
        self.osc_params = osc_params
        rm = pv.ResourceManager()
        all_instruments = rm.list_resources()
        instrument_name = list(filter(lambda x: 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' in x,
                                    all_instruments))
        if len(instrument_name) == 0:
            print('Oscilloscope was not found!')
            print('Program terminated!')
            exit()
        else:
            self.__osc = rm.open_resource(instrument_name[0])
            print('Oscilloscope found!')
        
        self.sample_rate = float(self.__osc.query(':ACQ:SRAT?'))
        print('Sample rate = ', self.sample_rate)

        self.pa_frame_size = self.time_to_points(osc_params['frame_duration'])
        self.pm_frame_size = self.time_to_points(osc_params['pm_response_time'])
        self.pre_points = self.time_to_points(osc_params['pre_time'])

        self.init_current_data()

        self.pa_channel = osc_params['pa_channel']
        print('PA channel name = ', self.pa_channel)
        self.pm_channel = osc_params['trigger_channel']
        print('power meter channel name = ', self.pm_channel)

        self.set_preamble()

        print('Oscilloscope initiation complete!')
        
    def init_current_data(self):
        """Fill arrays for current data with zeros"""

        self.current_pa_data = np.zeros(self.pa_frame_size)
        self.current_pm_data = np.zeros(self.pm_frame_size)

    def set_preamble(self):
        """Set or update preamble"""

        preamble_raw = self.__osc.query(':WAV:PRE?').split(',')
        self.preamble = {
            'format': int(preamble_raw[0]), # 0 - BYTE, 1 - WORD, 2 - ASC 
            'type': int(preamble_raw[1]), # 0 - NORMal, 1 - MAXimum, 2 RAW
            'points': int(preamble_raw[2]), # between 1 and 240000000
            'count': int(preamble_raw[3]), # the number of averages in the average sample mode and 1 in other modes
            'xincrement': float(preamble_raw[4]), # the time difference brtween two neighboring points in the X direction
            'xorigin': float(preamble_raw[5]), # the start time of the waveform data in the X direction
            'xreference': float(preamble_raw[6]), # the reference time of the data point in the X direction
            'yincrement': float(preamble_raw[7]), # the waveform increment in the Y direction
            'yorigin': float(preamble_raw[8]), # the vertical offset relative to the "Vertical Reference Position" in the Y direction
            'yreference': float(preamble_raw[9]) #the vertical reference position in the Y direction
        }

    def time_to_points (self, duration) -> int:
        """Convert duration [us] into amount of data points"""
        
        points = int(duration*self.sample_rate/1000000) + 1
        return points
    
    def read_data(self, channel):
        """Reads data from the specified channel.
        Automatically handles read size"""

        self.__osc.write(':WAV:SOUR ' + channel)
        self.__osc.write(':WAV:MODE RAW')
        self.__osc.write('"WAV:FORM BYTE')

        self.set_preamble()
        
        # по факту триггерный сигнал в середине сохранённого диапазона.
        data_start = (int(self.preamble['points']/2) - self.pre_points) # выбираем начальную точку

        if channel == self.pm_channel:
                self.__osc.write(':WAV:STAR ' + str(data_start + 1))
                self.__osc.write(':WAV:STOP ' + str(self.pm_frame_size + data_start))
                self.__osc.write(':WAV:DATA?')
                data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.int8)
                data_chunk = (data_chunk - self.preamble['xreference'] - self.preamble['yorigin']) * self.preamble['yincrement'] /2 #2 странная
                data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
                self.current_pm_data += data_chunk[12:]
            
        elif channel == self.pa_channel:
                self.__osc.write(':WAV:STAR ' + str(data_start + 1))
                self.__osc.write(':WAV:STOP ' + str(self.pa_frame_size + data_start))
                self.__osc.write(':WAV:DATA?')
                data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.int8)
                data_chunk = (data_chunk - self.preamble['xreference'] - self.preamble['yorigin']) * self.preamble['yincrement'] /2 #2 странная
                data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
                self.current_pa_data += data_chunk[12:]
        else:
            print('Wrong channel for read!')
            print('Program terminated!')
            exit()

    def baseline_correction(self, channel):
        """Corrects baseline for the selected channel"""

        if channel == self.pm_channel:
            baseline = np.average(self.current_pm_data[:int(self.pm_frame_size/20)])
            self.current_pm_data -= baseline
            print('Baseline corrected')

        elif channel == self.pa_channel:
            baseline = np.average(self.current_pa_data[:int(self.pa_frame_size/20)])
            self.current_pa_data -= baseline
            print('Baseline corrected')
        else:
            print('Wrong channel for base correction! Channel = ', channel)

    def set_laser_amp(self):
        """Updates laser amplitude"""

        self.laser_amp = self.current_pm_data.max()
        print('Laser amplitude updated! New value = ', self.laser_amp)        

    def set_signal_amp(self):
        """Updates PA amplitude"""

        pa_search_start = self.time_to_points(5 + self.osc_params['pre_time'])
        pa_search_stop = self.time_to_points(80) # >то плохо и надо бы переписать
        self.signal_amp = abs(self.current_pa_data[pa_search_start:pa_search_stop].max()-self.current_pa_data[pa_search_start:pa_search_stop].min())
        print('PhotoAcoustic amplitude updated! New value = ', self.signal_amp)

    def measure(self,) -> float:
        """Measure PA amplitude"""

        self.init_current_data()

        while int(self.__osc.query(':TRIG:POS?'))<0: #ждём пока можно будет снова читать данные
            time.sleep(0.1)

        self.__osc.write(':STOP')
        self.read_data(self.pm_channel)
        self.read_data(self.pa_channel)
        self.__osc.write(':RUN')

        self.baseline_correction(self.pa_channel)
        self.baseline_correction(self.pm_channel)

        self.set_laser_amp()
        self.set_signal_amp()

def init_stages():
    """Initiate stages"""

    stages = Thorlabs.list_kinesis_devices()

    if len(stages) < 2:
        print('Less than 2 stages detected!')
        print('Program terminated!')
        exit()

    stage1_ID = stages.pop()[0]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='Stage') #motor units [m]
    print('Stage X initiated. Stage X ID = ', stage1_ID)

    stage2_ID = stages.pop()[0]
    stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='Stage') #motor units [m]
    print('Stage Y initiated. Stage Y ID = ', stage2_ID)

    return stage1, stage2


def move_to(X, Y, stage_X, stage_Y) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    
    stage_X.move_to(X)
    stage_Y.move_to(Y)

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

if __name__ == "__main__":
    
    osc = Oscilloscope(osc_params) # initialize oscilloscope
    stage_X, stage_Y = init_stages() # initialize stages

    move_to(x_start, y_start, stage_X, stage_Y) # move to starting point
    wait_stages_stop(stage_X, stage_Y)
    amp = osc.measure()

    print(amp)

