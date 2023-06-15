"""
Classes for oscilloscope based devices
"""

import pyvisa as pv
import numpy as np
import time

from modules.bcolors import bcolors

class Oscilloscope:
    
    #defaults
    max_read_points = 250000
    scr_data_p = 1200
    preamble = {}
    sample_rate = 0
    ch1_data = np.zeros(0)
    ch1_amp = 0
    ch1_raw = False
    ch2_data = np.zeros(0)
    ch2_amp = 0
    ch2_raw = False
    ch1_scr_data = np.zeros(scr_data_p)
    ch1_scr_amp = 0
    ch1_scr_raw = False
    ch2_scr_data = np.zeros(scr_data_p)
    ch2_scr_amp = 0
    ch2_scr_raw = False
    
    bad_read = False
    not_found = True

    def __init__(self) -> None:
        """oscilloscope class for Rigol MS1000Z device.
        Intended to be used as a module in other scripts.
        Call 'initialize' before working with Oscilloscope."""
        
    def initialize(self,
                 chan1_pre: int=100, #[us] pre time for channel 1
                 chan1_post: int=2500, #[us] post time for channel 1
                 chan2_pre: int=100, #[us] pre time for channel 2
                 chan2_post: int=150, #[us] post time for channel 2
                 ra_kernel_size: int=20 #smoothing by rolling average kernel
                 ) -> None:
        
        print('Initiating oscilloscope...')
        rm = pv.ResourceManager()
        all_instruments = rm.list_resources()
        instrument_name = list(filter(lambda x: 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' in x,
                                    all_instruments))
        if len(instrument_name) == 0:
            print(f'{bcolors.WARNING}Oscilloscope was not found!{bcolors.ENDC}')
            self.not_found = True
            return
        else:
            self.__osc = rm.open_resource(instrument_name[0])
            print('Oscilloscope found!')
        
        self.set_preamble()
        self.set_sample_rate()

        #smoothing parameters
        self.ra_kernel = ra_kernel_size

        #set time intervals for reading frame from chan1 in [us]
        self.ch1_pre_t = chan1_pre
        self.ch1_post_t = chan1_post
        self.ch1_dur_t = chan1_pre + chan1_post

        #set time intervals for reading frame from chan2 in [us]
        self.ch2_pre_t = chan2_pre
        self.ch2_post_t = chan2_post
        self.ch2_dur_t = chan2_pre + chan2_post

        #update time intervals for both channels in points
        self.ch_points()
        
        self.not_found = False

        print(f'{bcolors.OKBLUE}Oscilloscope{bcolors.ENDC} initiation complete!')

    def query(self, message: str) -> str:
        """Sends a querry to the oscilloscope"""

        return self.__osc.query(message) # type: ignore
        
    def set_preamble(self) -> None:
        """Set or update preamble"""

        preamble_raw = self.__osc.query(':WAV:PRE?').split(',') # type: ignore
        self.preamble.update({
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
        })

    def set_sample_rate(self) -> None:
        """Updates sample rate"""

        self.sample_rate = float(self.__osc.query(':ACQ:SRAT?')) # type: ignore

    def time_to_points (self, duration: float) -> int:
        """Convert duration [us] into amount of data points.
        Updates sample_rate before conversion."""
        
        self.set_sample_rate()
        points = int(duration*self.sample_rate/1000000) + 1
        return points

    def ch_points(self) -> None:
        """Updates len of pre, post and dur points for both channels"""

        self.set_sample_rate()
        self.ch1_pre_p = int(self.ch1_pre_t*self.sample_rate/1000000) + 1
        self.ch1_post_p = int(self.ch1_post_t*self.sample_rate/1000000) + 1
        self.ch1_dur_p = int(self.ch1_dur_t*self.sample_rate/1000000) + 1

        self.ch2_pre_p = int(self.ch2_pre_t*self.sample_rate/1000000) + 1
        self.ch2_post_p = int(self.ch2_post_t*self.sample_rate/1000000) + 1
        self.ch2_dur_p = int(self.ch2_dur_t*self.sample_rate/1000000) + 1

    def rolling_average(self, data: np.ndarray) -> np.ndarray:
        """Smooth data using rolling average method"""

        if len(data)<2*self.ra_kernel:
            print(f'{bcolors.WARNING}\
                  Data size is too small for rolling average smoothing\
                  {bcolors.ENDC}')
            return data
        
        kernel = np.ones(self.ra_kernel)/self.ra_kernel
        tmp_array = np.zeros(len(data))
        border = int(self.ra_kernel/2)
        tmp_array[border:-(border-1)] = np.convolve(data,kernel,mode='valid')
        
        #leave edges unfiltered
        tmp_array[:border] = tmp_array[border]
        tmp_array[-(border):] = tmp_array[-border]
        return tmp_array

    def read_data(self, channel: str, raw: bool) -> np.ndarray:
        """Reads data from the specified channel.
        Automatically handles read size"""

        self.__osc.write(':WAV:SOUR ' + channel) # type: ignore
        self.__osc.write(':WAV:MODE RAW') # type: ignore
        self.__osc.write(':WAV:FORM BYTE') # type: ignore
        
        #update preamble, sample rate and len of channel points
        self.set_preamble()
        self.ch_points()

        if channel == 'CHAN1':
            data = np.zeros(self.ch1_dur_p)
            
            # по факту триггерный сигнал в середине сохранённого диапазона.
            data_start = (int(self.preamble['points']/2) - self.ch1_pre_p) # выбираем начальную точку
            
            #if we can read the whole data in 1 read
            if self.max_read_points > self.ch1_dur_p:
                self.__osc.write(':WAV:STAR ' + str(data_start + 1)) # type: ignore
                self.__osc.write(':WAV:STOP ' + str(self.ch1_dur_p + data_start)) # type: ignore
                self.__osc.write(':WAV:DATA?') # type: ignore
                data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8) # type: ignore
                if not raw:
                    data_chunk = (data_chunk - 
                                  self.preamble['xreference'] - 
                                  self.preamble['yorigin']) * self.preamble['yincrement']
                if self.ch1_dur_p == len(data_chunk[12:]):
                    data = data_chunk[12:]
                else:
                    self.bad_read = True
            #if several read are necessary
            else:
                data_frames = int(self.ch1_dur_p/250000) + 1
                print(f'Reading {bcolors.OKBLUE}{data_frames}{bcolors.ENDC} data frames...')
                for i in range(data_frames):
                    command = ':WAV:STAR ' + str(data_start + 1 + i*250000)
                    self.__osc.write(command) # type: ignore
                    if (self.ch1_dur_p - (i+1)*250000) > 0:
                        self.__osc.write(':WAV:STOP ' + str(data_start + (i+1)*250000)) # type: ignore
                    else:
                        command = ':WAV:STOP ' + str(data_start + self.ch1_dur_p - 1)
                        self.__osc.write(command) # type: ignore
                    self.__osc.write(':WAV:DATA?') # type: ignore
                    data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8) # type: ignore
                    if not raw:
                        data_chunk = (data_chunk -
                                      self.preamble['xreference'] -
                                      self.preamble['yorigin']) * self.preamble['yincrement']
                    if (self.ch1_dur_p - (i+1)*250000) > 0:
                        data[i*250000:(i+1)*250000] = data_chunk[12:].copy()
                    else:
                        data[i*250000:-1] = data_chunk[12:].copy()

        elif channel == 'CHAN2':
            data = np.zeros(self.ch2_dur_p)
            
            # по факту триггерный сигнал в середине сохранённого диапазона.
            data_start = (int(self.preamble['points']/2) - self.ch2_pre_p) # выбираем начальную точку
            
            #if we can read the whole data in 1 read
            if self.max_read_points > self.ch2_dur_p:
                self.__osc.write(':WAV:STAR ' + str(data_start + 1)) # type: ignore
                self.__osc.write(':WAV:STOP ' + str(self.ch2_dur_p + data_start)) # type: ignore
                self.__osc.write(':WAV:DATA?') # type: ignore
                data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8) # type: ignore
                if not raw:
                    data_chunk = (data_chunk - 
                                  self.preamble['xreference'] - 
                                  self.preamble['yorigin']) * self.preamble['yincrement']
                if self.ch2_dur_p == len(data_chunk[12:]):
                    data = data_chunk[12:]
                else:
                    self.bad_read = True
            #if several read are necessary
            else:
                data_frames = int(self.ch2_dur_p/250000) + 1
                print(f'Reading {bcolors.OKBLUE}{data_frames}{bcolors.ENDC} data frames...')
                for i in range(data_frames):
                    command = ':WAV:STAR ' + str(data_start + 1 + i*250000)
                    self.__osc.write(command) # type: ignore
                    if (self.ch2_dur_p - (i+1)*250000) > 0:
                        self.__osc.write(':WAV:STOP ' + str(data_start + (i+1)*250000)) # type: ignore
                    else:
                        command = ':WAV:STOP ' + str(data_start + self.ch2_dur_p - 1)
                        self.__osc.write(command) # type: ignore
                    self.__osc.write(':WAV:DATA?') # type: ignore
                    data_chunk = np.frombuffer(self.__osc.read_raw(), dtype=np.uint8) # type: ignore
                    if not raw:
                        data_chunk = (data_chunk -
                                      self.preamble['xreference'] -
                                      self.preamble['yorigin']) * self.preamble['yincrement']
                    if (self.ch2_dur_p - (i+1)*250000) > 0:
                        data[i*250000:(i+1)*250000] = data_chunk[12:].copy()
                    else:
                        data[i*250000:-1] = data_chunk[12:].copy()
        else:
            print('Wrong channel for read!')
            self.bad_read = True
            data = np.zeros(0)

        return data

    def baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """Corrects baseline for the data.
        Assumes that baseline as at the start of measured signal"""

        #fraction of measured datapoints used for calculation of bl
        bl_length = 0.02

        data = data.astype(np.float64)
        baseline = np.average(data[:int(len(data)*bl_length)])
        data -= baseline

        return data

    def measure_scr(self, 
                    read_ch1: bool=True, #read channel 1
                    read_ch2: bool=True, #read channel 2
                    correct_bl: bool=True, #correct baseline
                    smooth: bool=True, #smooth data
                    raw: bool=False #save data in uint8
                    ) -> None:
        """Measure data from screen"""

        #reset bad read flag
        self.bad_read = False

        self.set_preamble()

        if read_ch1:
            self.__osc.write(':WAV:SOUR ' + 'CHAN1') # type: ignore
            self.__osc.write(':WAV:MODE NORM') # type: ignore
            self.__osc.write(':WAV:FORM BYTE') # type: ignore
            self.__osc.write(':WAV:STAR 1') # type: ignore
            self.__osc.write(':WAV:STOP ' + str(self.scr_data_p)) # type: ignore
            self.__osc.write(':WAV:DATA?') # type: ignore
            data_chunk = np.frombuffer(self.__osc.read_raw(), # type: ignore
                                       dtype=np.uint8) 
            #choose format, in which to store data
            if raw:
                data = data_chunk[12:].copy()
            else:
                dy = self.preamble['yincrement']
                data = data_chunk[12:].astype(np.float64)*dy

            if len(data) == self.scr_data_p:
                if smooth:
                    data = self.rolling_average(data)
                if correct_bl:
                    data = self.baseline_correction(data)
                if raw:
                    self.ch1_scr_data = data.astype(np.uint8)
                else:
                    self.ch1_scr_data = data.copy()
                self.ch1_scr_raw = raw
                self.ch1_scr_amp = abs(np.amax(self.ch1_scr_data)-
                                       np.amin(self.ch1_scr_data))
            else:
                self.bad_read = True
                print(f'{bcolors.WARNING}\
                      Bad read (data points number) of screen data\
                      {bcolors.ENDC}')

        if read_ch2:
            self.__osc.write(':WAV:SOUR ' + 'CHAN2') # type: ignore
            self.__osc.write(':WAV:MODE NORM') # type: ignore
            self.__osc.write(':WAV:FORM BYTE') # type: ignore
            self.__osc.write(':WAV:STAR 1') # type: ignore
            self.__osc.write(':WAV:STOP ' + str(self.scr_data_p)) # type: ignore
            self.__osc.write(':WAV:DATA?') # type: ignore
            data_chunk = np.frombuffer(self.__osc.read_raw(), # type: ignore
                                       dtype=np.uint8) 
            data = data_chunk[12:].astype(np.float64)

            if len(data) == self.scr_data_p:
                if smooth:
                    data = self.rolling_average(data)
                if correct_bl:
                    data = self.baseline_correction(data)
                if raw:
                    self.ch2_scr_data = data.astype(np.uint8)
                self.ch2_scr_raw = raw
                self.ch2_scr_amp = abs(np.amax(self.ch2_scr_data)-
                                       np.amin(self.ch2_scr_data))
            else:
                self.bad_read = True
                print(f'{bcolors.WARNING}\
                      Bad read (data points number) of screen data\
                      {bcolors.ENDC}')

    def measure(self,
                read_ch1: bool=True, #read channel 1
                read_ch2: bool=True, #read channel 2
                correct_bl: bool=True, #correct baseline
                smooth: bool=True, #smooth data
                raw: bool=False #save data in uint8
                ) -> None:
        """Measure data from memory"""

        #reset bad_read flag
        self.bad_read = False

        #wait for ready to read
        while int(self.__osc.query(':TRIG:POS?'))<0: # type: ignore
            time.sleep(0.1)

        self.__osc.write(':STOP') # type: ignore
        if read_ch1:
            data = self.read_data('CHAN1', raw)
            if not self.bad_read:
                if smooth:
                    data = self.rolling_average(data)
                if correct_bl:
                    data = self.baseline_correction(data)
                if raw:
                    self.ch1_data = data.astype(np.uint8)
                else:
                    self.ch1_data = data.copy()
                self.ch1_raw = raw
                self.ch1_amp = abs(np.amax(self.ch1_data)-
                                   np.amin(self.ch1_data))

        if read_ch2:
            data = self.read_data('CHAN2', raw)
            if not self.bad_read:
                if smooth:
                    data = self.rolling_average(data)
                if correct_bl:
                    data = self.baseline_correction(data)
                if raw:
                    self.ch2_data = data.astype(np.uint8)
                else:
                    self.ch2_data = data.copy()
                self.ch2_raw = raw
                self.ch2_amp = abs(np.amax(self.ch2_data)-
                                   np.amin(self.ch2_data))
        
        self.__osc.write(':RUN') # type: ignore

class PowerMeter:
    ###DEFAULTS###

    #scalar coef to convert integral readings into [uJ]
    sclr_sens = 2630000
    ch = 'CHAN1'
    osc = Oscilloscope()
    # percentage of max amp, when we set begining of the impulse
    threshold = 0.05
    data = np.zeros((0))
    #start and stop indexes of the measured signal
    start_ind = 0
    stop_ind = 0

    def __init__(self, 
                 osc: Oscilloscope,
                 chan: str='CHAN1',
                 threshold: float=0.05) -> None:
        """PowerMeter class for working with
        Thorlabs ES111C pyroelectric detector.
        osc is as instance of Oscilloscope class, which is used for reading data.
        chan is a channel to which the detector is connected."""

        self.osc = osc
        self.ch = chan
        self.threshold = threshold

    def get_energy_scr(self) -> float:
        """Measure energy from screen (fast)."""

        if self.osc.not_found:
            print(f'{bcolors.WARNING}\
                  Attempt to measure energy from not init oscilloscope\
                  {bcolors.ENDC}')
            return 0
        
        if self.ch == 'CHAN1':
            self.osc.measure_scr(read_ch2=False,
                                 correct_bl=True)
        elif self.ch == 'CHAN2':
            self.osc.measure_scr(read_ch1=False,
                                 correct_bl=True)

        #return 0, if read data was not successfull
        #Oscilloscope class will promt warning
        if self.osc.bad_read:
            return 0
        
        self.data = self.osc.ch1_scr_data.copy()
        laser_amp = self.energy_from_data(self.data,
                                          self.osc.preamble['xincrement'])

        return laser_amp
    
    def energy_from_data(self, data: np.ndarray, step: float) -> float:
        """Calculate laser energy from data.
        step is time step for the data."""

        #indexes for start and stop of laser impulse
        start_index = 0
        stop_index = 1

        if len(data) < 10:
            print(f'{bcolors.WARNING}\
                  Data for energy calculation is too short, len(data)={len(data)}\
                  {bcolors.ENDC}')
            return 0

        max_amp = np.amax(data)
        try:
            start_index = np.where(data>(max_amp*self.threshold))[0][0]
            self.start_ind = start_index
        except IndexError:
            print(f'{bcolors.WARNING}\
                  Problem in set_laser_amp start_index. Laser amp set to 0!\
                  {bcolors.ENDC}')
            return 0

        try:
            stop_ind_arr = np.where(data[start_index:] < 0)[0] + start_index
            stop_index = 0
            #check interval. If data at index + check_int less than zero
            #then we assume that the laser pulse is really finished
            check_ind = int(len(data)/100)
            for x in stop_ind_arr:
                if (x+check_ind) < len(data) and data[x+check_ind] < 0:
                    stop_index = x
                    break
            if not stop_index:
                print(f'{bcolors.WARNING}'
                      + 'End of laser impulse was not found'
                      + f'{bcolors.ENDC}')
                return 0
            self.stop_ind = stop_index
        except IndexError:
            print(f'{bcolors.WARNING}\
                  Problem in set_laser_amp stop_index. Laser amp set to 0!\
                  {bcolors.ENDC}')
            return 0

        laser_amp = np.sum(
            data[start_index:stop_index])*step*self.sclr_sens
        
        print(f'Laser amp = {laser_amp:.1f} [uJ]')

        return laser_amp
    
    def set_channel(self, chan: str) -> None:
        """Sets read channel"""

        self.ch = chan