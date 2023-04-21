import pyvisa as pv
import numpy as np
import time
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt

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

# oscilloscope class. Intended to be used as a module in other scripts.
class Oscilloscope:
    __osc = None
    sample_rate = 0
    laser_calib_uj = 0
    pa_frame_size = 0
    pm_frame_size = 0
    pm_pre_time = 0
    pre_points = 0
    laser_amp = 0
    screen_laser_amp = 0
    signal_amp = 0
    frame_duration = 0
    pre_time = 0
    pm_response_time = 0
    pm_pre_time = 0
    bad_read = False

    def __init__(self, osc_params) -> None:
        
        print('Initiating oscilloscope...')
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

        self.pa_channel = osc_params['pa_channel']
        print('PA channel name = ', self.pa_channel)
        self.pm_channel = osc_params['trigger_channel']
        print('power meter channel name = ', self.pm_channel)

        self.laser_calib_uj = osc_params['laser_calib_uj']
        self.frame_duration = osc_params['frame_duration']
        self.pre_time = osc_params['pre_time']
        self.pa_frame_size = self.time_to_points(self.frame_duration)
        self.pre_points = self.time_to_points(self.pre_time)
        self.init_current_data(self.pa_channel)

        self.pm_response_time = osc_params['pm_response_time']
        self.pm_pre_time = osc_params['pm_pre_time']
        self.pm_frame_size = self.time_to_points(self.pm_response_time + self.pm_pre_time)
        self.pm_pre_points = self.time_to_points(self.pm_pre_time)
        self.init_current_data(self.pm_channel)
        self.screen_data = np.zeros(1200)

        self.set_preamble()

        print('Oscilloscope initiation complete!')
        
    def query(self, message):
        """Sends a querry to the oscilloscope"""

        return self.__osc.query(message)

    def init_current_data(self, channel):
        """Fill arrays for current data with zeros"""

        if channel == self.pm_channel:
            self.current_pm_data = np.zeros(self.pm_frame_size)
        if channel == self.pa_channel:
            self.current_pa_data = np.zeros(self.pa_frame_size)
        
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
        self.sample_rate = float(self.__osc.query(':ACQ:SRAT?'))

    def time_to_points (self, duration) -> int:
        """Convert duration [us] into amount of data points"""
        
        points = int(duration*self.sample_rate/1000000) + 1
        return points
    
    def hp_filter(self, channel, cutoff_freq):
        """High-pass filter for the given channel"""

        if channel == self.pm_channel:
            dt = 1/self.sample_rate
            print(f'self.current_pm_data.shape = {self.current_pm_data.shape}')
            print(f'dt={dt}')
            W = fftfreq(len(self.current_pm_data), dt) # array with frequencies
            f_signal = rfft(self.current_pm_data) # signal in f-space

            filtered_f_signal = f_signal.copy()
            filtered_f_signal[(W<cutoff_freq)] = 0   # high pass filtering

            self.current_pm_data = irfft(filtered_f_signal)

    def rolling_average(self, channel, kernel_size):
        """Smooth data of the given channel using rolling average method
        with kernel_size"""

        kernel = np.ones(kernel_size)/kernel_size
        if channel == self.pm_channel:
            self.current_pm_data = np.convolve(self.current_pm_data,kernel,mode='valid')
        elif channel == 'screen':
            tmp_array = np.zeros(len(self.screen_data))
            border = int(kernel_size/2)
            tmp_array[border:-(border-1)] = np.convolve(self.screen_data,kernel,mode='valid')
            tmp_array[:border] = tmp_array[border]
            tmp_array[-(border):] = tmp_array[-border]
            self.screen_data = tmp_array.copy()
        else:
            print(f'rolling average for channel {channel} is not written!')

    def read_data(self, channel):
        """Reads data from the specified channel.
        Automatically handles read size"""

        self.__osc.write(':WAV:SOUR ' + channel)
        self.__osc.write(':WAV:MODE RAW')
        self.__osc.write(':WAV:FORM BYTE')

        self.set_preamble()
        
        if channel == self.pm_channel:
            self.pm_frame_size = self.time_to_points(self.pm_response_time + self.pm_pre_time)
            self.pm_pre_points = self.time_to_points(self.pm_pre_time)
            self.init_current_data(self.pm_channel)
            # по факту триггерный сигнал в середине сохранённого диапазона.
            data_start = (int(self.preamble['points']/2) - self.pm_pre_points) # выбираем начальную точку
            if self.pm_frame_size <250001:
                self.__osc.write(':WAV:STAR ' + str(data_start + 1))
                self.__osc.write(':WAV:STOP ' + str(self.pm_frame_size + data_start))
                self.__osc.write(':WAV:DATA?')
                data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8)
                data_chunk = (data_chunk - self.preamble['xreference'] - self.preamble['yorigin']) * self.preamble['yincrement']
                data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
                if len(self.current_pm_data) == len(data_chunk[12:]):
                    self.current_pm_data += data_chunk[12:]
                else:
                    self.bad_read = True
            else:
                data_frames = int(self.pm_frame_size/250000) + 1
                print(f'data_frames = {data_frames}')
                for i in range(data_frames):
                    command = ':WAV:STAR ' + str(data_start + 1 + i*250000)
                    self.__osc.write(command)
                    if (self.pm_frame_size - (i+1)*250000) > 0:
                        self.__osc.write(':WAV:STOP ' + str(data_start + (i+1)*250000))
                    else:
                        command = ':WAV:STOP ' + str(data_start + self.pm_frame_size - 1)
                        self.__osc.write(command)
                    self.__osc.write(':WAV:DATA?')
                    data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8)
                    data_chunk = (data_chunk - self.preamble['xreference'] - self.preamble['yorigin']) * self.preamble['yincrement']
                    data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
                    if (self.pm_frame_size - (i+1)*250000) > 0:
                        self.current_pm_data[i*250000:(i+1)*250000] += data_chunk[12:].copy()
                        data_chunk = 0
                    else:
                        self.current_pm_data[i*250000:-1] += data_chunk[12:].copy()
                        data_chunk = 0

        elif channel == self.pa_channel:
            self.pa_frame_size = self.time_to_points(self.frame_duration)
            self.pre_points = self.time_to_points(self.pre_time)
            self.init_current_data(self.pa_channel)
            data_start = (int(self.preamble['points']/2) - self.pre_points) # выбираем начальную точку
            self.__osc.write(':WAV:STAR ' + str(data_start + 1))
            self.__osc.write(':WAV:STOP ' + str(self.pa_frame_size + data_start))
            self.__osc.write(':WAV:DATA?')
            data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8)
            data_chunk = (data_chunk - self.preamble['xreference'] - self.preamble['yorigin']) * self.preamble['yincrement']
            data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
            self.current_pa_data += data_chunk[12:]
        else:
            print('Wrong channel for read!')
            print('Program terminated!')
            exit()

    def baseline_correction(self, channel):
        """Corrects baseline for the selected channel"""

        if channel == self.pm_channel:
            baseline = np.average(self.current_pm_data[:int(self.pm_frame_size/100)])
            self.current_pm_data -= baseline

        elif channel == self.pa_channel:
            baseline = np.average(self.current_pa_data[:int(self.pa_frame_size/20)])
            self.current_pa_data -= baseline
        
        elif channel == 'screen':
            baseline = np.average(self.screen_data[:int(len(self.screen_data)/20)])
            self.screen_data -= baseline
        else:
            print('Wrong channel for base correction! Channel = ', channel)

    def set_laser_amp(self):
        """Updates laser amplitude in uJ
        If this method is called from outside, 
        please set 'osc.bad_read = False' first"""

        threshold = 0.03 # percentage of max amp, when we set begining of the impulse
        

        self.set_preamble()
        self.baseline_correction(self.pm_channel)
        max_pm = np.amax(self.current_pm_data)

        try:
            start_index = np.where(self.current_pm_data>(max_pm*threshold))[0][0]
        except IndexError:
            self.bad_read = True
            start_index = 0
            print(f'{bcolors.WARNING} Problem in set_laser_amp start_index. Laser amp set to 0 {bcolors.ENDC}')

        try:
            stop_index = np.where(self.current_pm_data[start_index:] < 0)[0][0]
        except IndexError:
            self.bad_read = True
            stop_index = 0
            print(f'{bcolors.WARNING} Problem in set_laser_amp stop_index. Laser amp set to 0 {bcolors.ENDC}')

        if not self.bad_read:
            self.laser_amp = np.sum(self.current_pm_data[start_index:(start_index + stop_index)])/self.sample_rate*self.laser_calib_uj
        else:
            self.laser_amp = 0
        print(f'Laser amp = {self.laser_amp:.5f}')

    def set_signal_amp(self):
        """Updates PA amplitude"""

        self.baseline_correction(self.pa_channel)
        pa_search_start = self.time_to_points(5 + self.osc_params['pre_time'])
        pa_search_stop = self.time_to_points(80) # это плохо и надо бы переписать
        self.signal_amp = abs(self.current_pa_data[pa_search_start:pa_search_stop].max()-self.current_pa_data[pa_search_start:pa_search_stop].min())
        print(f'PA amp = {self.signal_amp:.4f}')

    def read_screen(self, channel):
        """Read data from screen"""

        self.bad_read = False
        threshold = 0.10 # percentage of max amp, when we set begining of the impulse

        self.set_preamble()
        self.__osc.write(':WAV:SOUR ' + channel)
        self.__osc.write(':WAV:MODE NORM')
        self.__osc.write(':WAV:FORM BYTE')
        self.__osc.write(':WAV:STAR 1')
        self.__osc.write(':WAV:STOP 1200')
        self.__osc.write(':WAV:DATA?')
        data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8)
        data_chunk = data_chunk.astype(np.float64)

        if len(self.screen_data) == len(data_chunk[12:]):
            self.screen_data = data_chunk[12:]
            self.rolling_average('screen', 10)
            self.baseline_correction('screen')
        else:
            print(f'len of screen_data = {len(self.screen_data )}')
            print(f'len of data_chunk[12:] = {len(data_chunk[12:])}')
            self.bad_read = True
            print(f'{bcolors.WARNING} Bad read (data points number) of screen data {bcolors.ENDC}')
            self.screen_laser_amp = 0
        max_screen = np.amax(self.screen_data)

        try:
            start_index = np.where(self.screen_data>(max_screen*threshold))[0][0]
            stop_index = np.where(self.screen_data[start_index:] < 0)[0][0]
            dt = self.preamble['xincrement']
            dy = self.preamble['yincrement']
            self.screen_laser_amp = np.sum(self.screen_data[start_index:(start_index + stop_index)])*dt*dy*self.laser_calib_uj
            self.screen_data[start_index] = np.amax(self.screen_data)
            self.screen_data[start_index + stop_index] = np.amax(self.screen_data)
        except IndexError:
            self.screen_laser_amp = 0
            self.bad_read = True
            print(f'{bcolors.WARNING} Bad read (index problem) of screen data {bcolors.ENDC}')



    def measure(self,):
        """Measure PA amplitude"""

        self.bad_read = False
        pm_cutoff_freq = 1000 # pm cutoff freq for filtration
        kernel_size = 1000 # kernel size for rolling average filtration 

        self.set_preamble()

        while int(self.__osc.query(':TRIG:POS?'))<0: #ждём пока можно будет снова читать данные
            time.sleep(0.1)

        self.__osc.write(':STOP')
        self.init_current_data(self.pm_channel)
        self.read_data(self.pm_channel)
        kernel_size = int(len(self.current_pm_data)/1000)
        self.rolling_average(self.pm_channel,kernel_size)
        self.init_current_data(self.pa_channel)
        self.read_data(self.pa_channel)
        self.__osc.write(':RUN')

        self.set_laser_amp()
        self.set_signal_amp()
