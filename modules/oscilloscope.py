"""
Oscilloscope based devices

General usage procedure:

1. Create class instance
2. Call initialize
3. Call measure or measure_scr

data can be accessed from ch_data or ch_scr_data accordingly.
"""
import typing

import pyvisa as pv
import numpy as np
import numpy.typing as npt
import time
import pint
from zmq import CHANNEL

import modules.exceptions as exceptions
from . import ureg

class Oscilloscope:
    """Rigol MSO1000Z/DS1000Z"""  

    #defaults
    MAX_MEMORY_READ = 250000 #max read data points from osc memory
    MAX_SCR_POINTS = 1200 # fixed length of screen data
    HEADER_LEN = 12 # length of header in read data
    SMOOTH_LEN_FACTOR = 10 # determines minimum len of data for smoothing
    BL_LENGTH = 0.02 #fraction of datapoints for calculation of baseline
    OSC_ID = 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' # osc name

    #attributes
    sample_rate: pint.Quantity
    format: int # 0 - BYTE, 1 - WORD, 2 - ASC 
    read_type: int # 0 - NORMal, 1 - MAXimum, 2 RAW
    points: int # between 1 and 240000000
    averages: int # number of averages in average mode, 1 in other modes
    xincrement: pint.Quantity # time diff between points
    xorigin: pint.Quantity # start time of the data
    xreference: pint.Quantity # reference time of data
    yincrement: float # the waveform increment in the Y direction
    yorigin: float # vertical offset relative to the yreference
    yreference: float # vertical reference position in the Y direction

    ch1_pre: pint.Quantity # time bef trig to save chan 1
    ch1_post: pint.Quantity # time after trigger
    ch1_dur: pint.Quantity # duration of data
    ch1_pre_p: int # same in points
    ch1_post_p: int 
    ch1_dur_p: int 
    ch1_data: pint.Quantity
    ch1_data_raw: npt.NDArray[np.uint8]
    ch1_amp: pint.Quantity # amplitude of data in channel 1
    ch1_scr_data: pint.Quantity
    ch1_scr_data_raw: npt.NDArray[np.uint8]
    ch1_scr_amp: pint.Quantity

    ch2_pre: pint.Quantity # time bef trig to save chan 2
    ch2_post: pint.Quantity # time after trigger
    ch2_dur: pint.Quantity # duration of data
    ch1_pre_p: int # same in points
    ch1_post_p: int 
    ch1_dur_p: int 
    ch2_data: pint.Quantity
    ch2_data_raw: npt.NDArray[np.uint8]
    ch2_amp: pint.Quantity
    ch2_scr_data: pint.Quantity
    ch2_scr_data_raw: npt.NDArray[np.uint8]
    ch2_scr_amp: pint.Quantity

    # data smoothing parameters
    ra_kernel: int# kernel size for rolling average smoothing
    
    bad_read = False # flag for indication of error during read
    not_found = True # flag for state of osc
    read_chunks: int# amount of reads required for a chan

    def __init__(self) -> None:
        """oscilloscope class for Rigol MSO1000Z/DS1000Z device.
        Intended to be used as a module in other scripts.
        Call 'initialize' before working with Oscilloscope."""
        
    def initialize(self,
                 chan1_pre: pint.Quantity=ureg('150us'),
                 chan1_post: pint.Quantity=ureg('2500us'),
                 chan2_pre: pint.Quantity=ureg('100us'),
                 chan2_post: pint.Quantity=ureg('150us'),
                 ra_kernel_size: int=20 #smoothing by rolling average
                 ) -> None:
        """Oscilloscope initializator.
        chan_pre and chan_post are time intervals before and after
        trigger for saving data from corresponding channels"""
        
        rm = pv.ResourceManager()
        all_instruments = rm.list_resources()
        instrument_name = list(filter(lambda x: self.OSC_ID in x,
                                    all_instruments))
        if len(instrument_name) == 0:
            raise exceptions.OscilloscopeError('Oscilloscope was not found')
        else:
            self.__osc = rm.open_resource(instrument_name[0])
        
        self.set_preamble()
        self.set_sample_rate()

        #smoothing parameters
        self.ra_kernel = ra_kernel_size

        #set time intervals for reading frame from chan1 in [us]
        self.ch1_pre = chan1_pre
        self.ch1_post = chan1_post
        self.ch1_dur = chan1_pre + chan1_post # type: ignore

        #set time intervals for reading frame from chan2 in [us]
        self.ch2_pre = chan2_pre
        self.ch2_post = chan2_post
        self.ch2_dur = chan2_pre + chan2_post # type: ignore

        #update time intervals for both channels in points
        self.ch_points()
        
        self.not_found = False

    def connection_check(self) -> None:
        """Checks connection to the oscilloscope.
        Does not work."""
        try:
            self.__osc.write(':SYST:GAM?') # type: ignore
            np.frombuffer(self.__osc.read_raw(), dtype=np.uint8) # type: ignore
        except:
            raise exceptions.OscilloscopeError('No connection to the oscilloscope')

    def query(self, message: str) -> str:
        """Sends a querry to the oscilloscope"""

        return self.__osc.query(message) # type: ignore
        
    def set_preamble(self) -> None:
        """Sets osc params"""

        preamble_raw = self.__osc.query(':WAV:PRE?').split(',') # type: ignore
        self.format = int(preamble_raw[0]) 
        self.read_type = int(preamble_raw[1])
        self.points = int(preamble_raw[2])
        self.averages = int(preamble_raw[3])
        self.xincrement = float(preamble_raw[4])*ureg('second')
        self.xorigin = float(preamble_raw[5])*ureg('second')
        self.xreference = float(preamble_raw[6])*ureg('second')
        self.yincrement = float(preamble_raw[7])
        self.yorigin = float(preamble_raw[8])
        self.yreference = float(preamble_raw[9])

    def set_sample_rate(self) -> None:
        """Updates sample rate"""

        self.sample_rate = (float(self.__osc.query(':ACQ:SRAT?')) # type: ignore
                            *ureg('hertz'))

    def time_to_points (self, duration: pint.Quantity) -> int:
        """Convert duration into amount of data points"""
        
        points = int((duration*self.sample_rate).magnitude) + 1
        return points

    def ch_points(self) -> None:
        """Updates len of pre, post and dur points for both channels"""

        self.set_sample_rate()
        self.ch1_pre_p = self.time_to_points(self.ch1_pre)
        self.ch1_post_p = self.time_to_points(self.ch1_post)
        self.ch1_dur_p = self.time_to_points(self.ch1_dur)

        self.ch2_pre_p = self.time_to_points(self.ch2_pre)
        self.ch2_post_p = self.time_to_points(self.ch2_post)
        self.ch2_dur_p = self.time_to_points(self.ch2_dur)

    def rolling_average(self,
                        data: npt.NDArray[np.uint8]
                        ) -> npt.NDArray[np.uint8]:
        """Smooth data using rolling average method"""

        if len(data)<self.SMOOTH_LEN_FACTOR*self.ra_kernel:
            raise exceptions.OscilloscopeError('Data too small for smoothing')
        
        kernel = np.ones(self.ra_kernel)/self.ra_kernel
        tmp_array = np.zeros(len(data))
        border = int(self.ra_kernel/2)

        tmp_array[border:-(border-1)] = np.convolve(data,kernel,mode='valid')
        
        #leave edges unfiltered
        tmp_array[:border] = tmp_array[border]
        tmp_array[-(border):] = tmp_array[-border]
        return tmp_array.astype(np.uint8)

    def _one_chunk_read(self,
                        start: int,
                        dur: int) -> npt.NDArray[np.uint8]:
        """read from memory in single chunk"""

        self.__osc.write(':WAV:STAR ' # type: ignore
                                 + str(start + 1))
        self.__osc.write(':WAV:STOP ' # type: ignore
                            + str(dur + start))
        self.__osc.write(':WAV:DATA?') # type: ignore
        data = np.frombuffer(self.__osc.read_raw(), # type: ignore
                             dtype=np.uint8)[self.HEADER_LEN:]
        return data.astype(np.uint8)

    def _multi_chunk_read(self,
                          start: int,
                          dur: int) -> npt.NDArray[np.uint8]:
        """reads from memory in multiple chunks"""

        data = np.zeros(dur).astype(np.uint8)
        data_frames = int(dur/self.MAX_MEMORY_READ) + 1
        for i in range(data_frames):
            start_i = i*self.MAX_MEMORY_READ
            stop_i = (i+1)*self.MAX_MEMORY_READ

            command = ':WAV:STAR ' + str(start + start_i + 1)
            self.__osc.write(command) # type: ignore
            
            if (dur - stop_i) > 0:
                command = (':WAV:STOP ' + str(start + stop_i))
            else:
                command = (':WAV:STOP ' + str(start + dur - 1))
            self.__osc.write(command) # type: ignore
            
            self.__osc.write(':WAV:DATA?') # type: ignore
            #read data without header
            data_chunk = np.frombuffer(self.__osc.read_raw(), # type: ignore
                                       dtype=np.uint8)[self.HEADER_LEN:]

            if (dur - stop_i) > 0:
                if self._ok_read(self.MAX_MEMORY_READ,data_chunk):
                    data[start_i:stop_i] = data_chunk.copy()
            else:
                if self._ok_read(len(data[start_i:-1]), data_chunk):
                    data[start_i:-1] = data_chunk.copy()

        return data

    def to_volts(self, data: npt.NDArray[np.uint8]) -> pint.Quantity:
        """converts data to volts"""

        return (data.astype(np.float64)
                - self.xreference.magnitude
                - self.yorigin)*self.yincrement*ureg('volt')
    
    def to_volts_scr(self,
                     data: npt.NDArray[np.uint8]) -> pint.Quantity:
        """converts screen data to volts"""

        dy = self.yincrement
        return data.astype(np.float64)*dy*ureg('volt')

    def _ok_read(self, dur: int,
                 data_chunk: npt.NDArray[np.uint8]) -> bool:
        """verify that read data have necessary size"""

        if dur == len(data_chunk):
            return True
        else:
            self.bad_read = True
            return False

    def read_data(self, channel: str) -> None:
        """Reads data from the specified channel.
        Sets data to ch_data_raw attribute"""

        self.__osc.write(':WAV:SOUR ' + channel) # type: ignore
        self.__osc.write(':WAV:MODE RAW') # type: ignore
        self.__osc.write(':WAV:FORM BYTE') # type: ignore
        
        #update preamble, sample rate and len of channel points
        self.set_preamble()
        self.ch_points()

        if channel == 'CHAN1':   
            # trigger is in the mid of memory
            # set starting point
            data_start = (int(self.points/2) - self.ch1_pre_p)
            
            #if one can read the whole data in 1 read
            if self.MAX_MEMORY_READ > self.ch1_dur_p:
                data_chunk = self._one_chunk_read(data_start,
                                                  self.ch1_dur_p)
            else:
                data_chunk = self._multi_chunk_read(data_start,
                                                    self.ch1_dur_p)
 
            if self._ok_read(self.ch1_dur_p, data_chunk):
                self.ch1_data_raw = data_chunk

        elif channel == 'CHAN2':
            # trigger is in the mid of memory
            # set starting point
            data_start = (int(self.points/2) - self.ch2_pre_p)
            
            #if we can read the whole data in 1 read
            if self.MAX_MEMORY_READ > self.ch2_dur_p:
                data_chunk = self._one_chunk_read(data_start,
                                                  self.ch2_dur_p)
            else:
                data_chunk = self._multi_chunk_read(data_start,
                                                    self.ch2_dur_p)
            if self._ok_read(self.ch2_dur_p, data_chunk):
                self.ch2_data_raw = data_chunk
        
        else:
            self.bad_read = True
            raise exceptions.OscilloscopeError('Wrong read channel name')

    def baseline_correction(self,
                            data: npt.NDArray[np.uint8]
                            ) -> npt.NDArray[np.uint8]:
        """Corrects baseline for the data.
        Assumes that baseline as at the start of measured signal"""

        baseline = np.average(data[:int(len(data)*self.BL_LENGTH)])
        data -= int(baseline)
        return data

    def read_scr(self, channel: str) -> npt.NDArray[np.uint8]:
        """reads screen data for the channel"""

        self.__osc.write(':WAV:SOUR ' + channel) # type: ignore
        self.__osc.write(':WAV:MODE NORM') # type: ignore
        self.__osc.write(':WAV:FORM BYTE') # type: ignore
        self.__osc.write(':WAV:STAR 1') # type: ignore
        self.__osc.write(':WAV:STOP ' + str(self.MAX_SCR_POINTS)) # type: ignore
        self.__osc.write(':WAV:DATA?') # type: ignore
        data_chunk = np.frombuffer(self.__osc.read_raw(), # type: ignore
                                    dtype=np.uint8) 
        
        return data_chunk[self.HEADER_LEN:].astype(np.uint8)

    def measure_scr(self, 
                    read_ch1: bool=True, #read channel 1
                    read_ch2: bool=True, #read channel 2
                    correct_bl: bool=True, #correct baseline
                    smooth: bool=True, #smooth data
                    ) -> None:
        """Measure data from screen"""

        #reset bad read flag
        self.bad_read = False

        self.set_preamble()

        if read_ch1:
            self.ch1_scr_data_raw = self.read_scr('CHAN1')
            if self._ok_read(self.MAX_SCR_POINTS,
                             self.ch1_data_raw):
                if smooth:
                    self.ch1_scr_data_raw = self.rolling_average(self.ch1_scr_data_raw)
                if correct_bl:
                    self.ch1_scr_data_raw = self.baseline_correction(self.ch1_scr_data_raw)
                self.ch1_scr_data = self.to_volts_scr(self.ch1_data_raw)
                self.ch1_scr_amp = abs(np.amax(self.ch1_scr_data)
                                       - np.amin(self.ch1_scr_data))

        if read_ch2:
            self.ch2_scr_data_raw = self.read_scr('CHAN2')
            if self._ok_read(self.MAX_SCR_POINTS,
                             self.ch2_data_raw):
                if smooth:
                    self.ch2_scr_data_raw = self.rolling_average(self.ch2_scr_data_raw)
                if correct_bl:
                    self.ch2_scr_data_raw = self.baseline_correction(self.ch2_scr_data_raw)
                self.ch2_scr_data = self.to_volts_scr(self.ch2_data_raw)
                self.ch2_scr_amp = abs(np.amax(self.ch2_scr_data)
                                       - np.amin(self.ch2_scr_data))

    def measure(self,
                read_ch1: bool=True, #read channel 1
                read_ch2: bool=True, #read channel 2
                correct_bl: bool=True, #correct baseline
                smooth: bool=True, #smooth data
                ) -> None:
        """Measure data from memory"""

        #reset bad_read flag
        self.bad_read = False

        #wait for ready to read
        while int(self.__osc.query(':TRIG:POS?'))<0: # type: ignore
            time.sleep(0.1)

        # data from memory can be read only in STOP mode
        self.__osc.write(':STOP') # type: ignore
        if read_ch1:
            self.read_data('CHAN1')
            if not self.bad_read:
                if smooth:
                    self.ch1_data_raw = self.rolling_average(
                        self.ch1_data_raw)
                if correct_bl:
                    self.ch1_data_raw = self.baseline_correction(
                        self.ch1_data_raw)
                self.ch1_data = self.to_volts(self.ch1_data_raw)
                self.ch1_amp = abs(np.amax(self.ch1_data)
                                    - np.amin(self.ch1_data))

        if read_ch2:
            self.read_data('CHAN2')
            if not self.bad_read:
                if smooth:
                    self.ch2_data_raw = self.rolling_average(
                        self.ch2_data_raw)
                if correct_bl:
                    self.ch2_data_raw = self.baseline_correction(
                        self.ch2_data_raw)

                self.ch2_data = self.to_volts(self.ch2_data_raw)
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
                                          self.osc.xincrement)

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