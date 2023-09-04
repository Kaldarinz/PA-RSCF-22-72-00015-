"""
Oscilloscope based devices
"""
from doctest import debug
from tkinter import NO
from typing import List, Optional, cast
import logging
import time

import pyvisa as pv
import numpy as np
import numpy.typing as npt
import pint

import modules.exceptions as exceptions
from . import ureg

logger = logging.getLogger(__name__)

class Oscilloscope:
    """Rigol MSO1000Z/DS1000Z
    
    General usage procedure:
    1. Create class instance
    2. Call initialize
    3. Call measure or measure_scr
    4. check bad_read flag
    5. read data from data or data_raw list
    """  

    #defaults
    MAX_MEMORY_READ = 250000 #max read data points from osc memory
    MAX_SCR_POINTS = 1200 # fixed length of screen data
    HEADER_LEN = 12 # length of header in read data
    SMOOTH_LEN_FACTOR = 10 # determines minimum len of data for smoothing
    BL_LENGTH = 0.02 #fraction of datapoints for calculation of baseline
    OSC_ID = 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' # osc name
    CHANNELS = 2 # amount of channels
    CH_IDS = ['CHAN1', 'CHAN2'] # channel names

    def __init__(self) -> None:
        """oscilloscope class for Rigol MSO1000Z/DS1000Z device.
        Intended to be used as a module in other scripts.
        Call 'initialize' before working with Oscilloscope.
        """

        #attributes
        self.sample_rate: pint.Quantity
        self.format: int # 0 - BYTE, 1 - WORD, 2 - ASC 
        self.read_type: int # 0 - NORMal, 1 - MAXimum, 2 RAW
        self.points: int # between 1 and 240000000
        self.averages: int # number of averages in average mode, 1 in other modes
        self.xincrement: pint.Quantity # time diff between points
        self.xorigin: pint.Quantity # start time of the data
        self.xreference: pint.Quantity # reference time of data
        self.yincrement: float # the waveform increment in the Y direction
        self.yorigin: float # vertical offset relative to the yreference
        self.yreference: float # vertical reference position in the Y direction

        #channels attributes
        #
        # time before trig to save data
        self.pre_t: List[Optional[pint.Quantity]] = [None]*self.CHANNELS
        # time after trigger
        self.post_t: List[Optional[pint.Quantity]] = [None]*self.CHANNELS
        # duration of data
        self.dur_t: List[Optional[pint.Quantity]] = [None]*self.CHANNELS
        # same in points
        self.pre_p: List[Optional[int]] = [None]*self.CHANNELS
        self.post_p: List[Optional[int]] =[None]*self.CHANNELS
        self.dur_p: List[Optional[int]] = [None]*self.CHANNELS
        self.data: List[Optional[List[pint.Quantity]]] = [None]*self.CHANNELS
        self.data_raw: List[Optional[npt.NDArray[np.uint8|np.int16]]] = [None]*self.CHANNELS
        # amplitude of data
        self.amp: List[Optional[pint.Quantity]] = [None]*self.CHANNELS
        self.scr_data: List[Optional[List[pint.Quantity]]] = [None]*self.CHANNELS
        self.scr_data_raw: List[Optional[npt.NDArray[np.uint8|np.int16]]] = [None]*self.CHANNELS
        self.scr_amp: List[Optional[pint.Quantity]] = [None]*self.CHANNELS

        # data smoothing parameters
        self.ra_kernel: int# kernel size for rolling average smoothing
        
        self.bad_read = False # flag for indication of error during read
        self.not_found = True # flag for state of osc
        self.read_chunks: int# amount of reads required for a chan

        logger.debug('Oscilloscope class instantiated!')
        
    def initialize(self,
                 chan1_pre: pint.Quantity=ureg('150us'),
                 chan1_post: pint.Quantity=ureg('2500us'),
                 chan2_pre: pint.Quantity=ureg('100us'),
                 chan2_post: pint.Quantity=ureg('150us'),
                 ra_kernel_size: int=20 #smoothing by rolling average
                 ) -> None:
        """Oscilloscope initializator.

        <chan_pre> and <chan_post> are time intervals before and after
        trigger for saving data from corresponding channels.
        """
        
        logger.debug('Starting actual initialization of an oscilloscope...')
        rm = pv.ResourceManager()
        
        logger.debug('Searching for VISA devices')
        all_instruments = rm.list_resources()
        logger.debug(f'{len(all_instruments)} VISA devices found')
        if self.OSC_ID not in all_instruments:
            logger.debug('...Terminating. Oscilloscope was not found among VISA '
                         + 'devices. Init failed')
            raise exceptions.OscilloscopeError('Oscilloscope was not found')
        else:
            self.__osc: pv.resources.USBInstrument
            self.__osc = rm.open_resource(self.OSC_ID) # type: ignore
            logger.debug('Oscilloscope device found!')

        self.set_preamble()
        self.set_sample_rate()

        #smoothing parameters
        self.ra_kernel = ra_kernel_size

        #set time intervals for channels
        self.pre_t[0] = chan1_pre
        self.post_t[0] = chan1_post
        self.dur_t[0] = chan1_pre + chan1_post # type: ignore

        self.pre_t[1] = chan2_pre
        self.post_t[1] = chan2_post
        self.dur_t[1] = chan2_pre + chan2_post # type: ignore

        #update time intervals for both channels in points
        self.ch_points()
        
        self.not_found = False
        logger.debug('...Finishing. Success.')

    def connection_check(self) -> None:
        """Check connection to the oscilloscope.

        Set <not_found> flag.
        Never raises exceptions.
        """

        logger.debug('Starting connection check...')
        try:
            session = self.__osc.session
            logger.debug(f'...Finishing. Success. Session ID={session}')
            self.not_found = False
        except Exception as err:
            logger.debug(f'Operation failed with error {type(err)}')
            self.not_found = True

    def query(self, message: str) -> str:
        """Send a querry to the oscilloscope."""

        logger.debug(f'Sending query: {message}')
        return self.__osc.query(message)
        
    def set_preamble(self) -> None:
        """Set osc params."""

        logger.debug('Starting...')
        query_results = self.query(':WAV:PRE?')
        preamble_raw = query_results.split(',')
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
        logger.debug(f'...Finishing. Success. {len(preamble_raw)} '
                     + 'parameters obtained and set.')

    def set_sample_rate(self) -> None:
        """Update sample rate."""

        self.sample_rate = (float(self.query(':ACQ:SRAT?'))*ureg('hertz'))
        logger.debug(f'Sample rate updated to {self.sample_rate=}')

    def time_to_points (self, duration: pint.Quantity) -> int:
        """Convert duration into amount of data points."""
        
        logger.debug('Starting...')
        logger.debug(f'Calculating amount of points for {duration=}')
        points = int((duration*self.sample_rate).magnitude) + 1
        logger.debug(f'...Finishing. Amount of calculated points: {points}')
        return points

    def ch_points(self) -> None:
        """Update len of pre, post and dur points for all channels."""

        logger.debug('Starting set amount of data points for each channel...')
        self.set_sample_rate()
        for i in range(self.CHANNELS):
            logger.debug(f'Setting points for channel {i+1}')
            pre_t = self.pre_t[i]
            if pre_t is not None:
                self.pre_p[i] = self.time_to_points(pre_t)
            else:
                logger.debug(f'Warning! "pre time" for CHAN{i+1} is not set.')
            post_t = self.post_t[i]
            if post_t is not None:
                self.post_p[i] = self.time_to_points(post_t)
            else:
                logger.debug(f'Warning! "post time" for CHAN{i+1} is not set.')
            dur_t = self.dur_t[i]
            if dur_t is not None:
                self.dur_p[i] = self.time_to_points(dur_t)
            else:
                logger.debug(f'Warning! "dur time" for CHAN{i+1} is not set.')
            logger.debug('...Finishing.')

    def rolling_average(self,
                        data: npt.NDArray[np.uint8]
                        ) -> npt.NDArray[np.uint8]:
        """Smooth data using rolling average method."""
        
        logger.debug('Starting rolling_average smoothing...')
        min_signal_size = int(self.SMOOTH_LEN_FACTOR*self.ra_kernel)
        if len(data)<min_signal_size:
            logger.debug(f'...Terminating. Signal has only {len(data)} data points, '
                         +f'but at least {min_signal_size} is required')
            raise exceptions.OscilloscopeError('Data too small for smoothing')
        
        logger.debug(f'Kernel size is {self.ra_kernel}')
        kernel = np.ones(self.ra_kernel)/self.ra_kernel
        tmp_array = np.zeros(len(data))
        border = int(self.ra_kernel/2)

        tmp_array[border:-(border-1)] = np.convolve(data,kernel,mode='valid')
        
        #leave edges unfiltered
        tmp_array[:border] = tmp_array[border]
        tmp_array[-(border):] = tmp_array[-border]
        logger.debug(f'...Finishing. Success.')
        return tmp_array.astype(np.uint8)

    def _one_chunk_read(self,
                        start: int,
                        dur: int) -> npt.NDArray[np.uint8]:
        """read from memory in single chunk.
        
        Returns data without header.
        """

        logger.debug('Starting _one_chunk_read...')
        logger.debug(f'Start point: {start+1}, stop point: {dur + start}.')
        
        msg = ':WAV:STAR ' + str(start + 1)
        self.__osc.write(msg)
        msg = ':WAV:STOP ' + str(dur + start)
        self.__osc.write(msg)
        msg = ':WAV:DATA?'
        self.__osc.write(msg)

        data = np.frombuffer(self.__osc.read_raw(),
                             dtype=np.uint8)[self.HEADER_LEN:]
        logger.debug(f'...Finishing. Signal with {len(data)} data points read.')
        return data.astype(np.uint8)

    def _multi_chunk_read(self,
                          start: int,
                          dur: int) -> npt.NDArray[np.uint8]:
        """Read from memory in multiple chunks.
        
        Return data without header.
        """

        logger.debug('Starting _multi_chunk_read...')
        logger.debug(f'Start point: {start+1}, duartion point: {dur}.')
        data = np.zeros(dur).astype(np.uint8)
        data_frames = int(dur/self.MAX_MEMORY_READ) + 1
        logger.debug(f'{data_frames} reads are required.')

        for i in range(data_frames):
            start_i = i*self.MAX_MEMORY_READ
            stop_i = (i+1)*self.MAX_MEMORY_READ

            command = ':WAV:STAR ' + str(start + start_i + 1)
            logger.debug(f'Writing {command}')
            self.__osc.write(command)
            
            if (dur - stop_i) > 0:
                command = (':WAV:STOP ' + str(start + stop_i))
            else:
                command = (':WAV:STOP ' + str(start + dur - 1))
            logger.debug(f'Writing {command}')
            self.__osc.write(command)
            
            command = ':WAV:DATA?'
            logger.debug(f'Writing {command}')
            self.__osc.write(command)
            
            data_chunk = np.frombuffer(self.__osc.read_raw(),
                                       dtype=np.uint8)[self.HEADER_LEN:]
            logger.debug(f'Chunk with {len(data_chunk)} data points read')

            logger.debug('Start verification of read chunk length')
            if (dur - stop_i) > 0:
                if self._ok_read(self.MAX_MEMORY_READ,data_chunk):
                    data[start_i:stop_i] = data_chunk.copy()
            else:
                if self._ok_read(len(data[start_i:-1]), data_chunk):
                    data[start_i:-1] = data_chunk.copy()

        logger.debug(f'...Finishing. Signal with {len(data)} datapoints read')
        return data

    def to_volts(self,
                 data: npt.NDArray[np.uint8|np.int16]) -> List[pint.Quantity]:
        """Converts data to volts."""

        logger.debug('Starting data conversion to volts...')
        result = (data.astype(np.float64)
                - self.xreference.magnitude
                - self.yorigin)*self.yincrement*ureg('volt')
        logger.debug(f'...Finishing. Max val={result.max()}, min val={result.min()}')
        return result
    
    def to_volts_scr(self,
                     data: npt.NDArray[np.uint8|np.int16]) -> List[pint.Quantity]:
        """Convert screen data to volts."""

        logger.debug('Starting screen data conversion to volts...')
        dy = self.yincrement*ureg.V
        dy = cast(pint.Quantity, dy)
        logger.debug(f'One step is {dy}.')
        result = dy*data.astype(np.float64)
        logger.debug(f'...Finishing. Max val={result.max()}, min val={result.min()}') #type:ignore
        return result

    def _ok_read(self, dur: int,
                 data_chunk: npt.NDArray[np.uint8]) -> bool:
        """Verify that read data have necessary size."""

        logger.debug('Starting read verification...')
        if dur == len(data_chunk):
            logger.debug('...Finishing. Data length is OK.')
            return True
        else:
            logger.debug('...Finishing. Data length is wrong, '
                         + f'{dur} is required, '
                         + f'actual length is {len(data_chunk)}')

            self.bad_read = True
            return False

    def read_data(self, ch_id: int) -> npt.NDArray[np.uint8]:
        """Read data from the specified channel.
        
        Sets data to ch_data_raw attribute.
        """

        logger.debug(f'Starting read from {self.CH_IDS[ch_id]}.')
        cmd = ':WAV:SOUR ' + self.CH_IDS[ch_id]
        self.__osc.write(cmd)
        cmd = ':WAV:MODE RAW'
        self.__osc.write(cmd)
        cmd = ':WAV:FORM BYTE'
        self.__osc.write(cmd)
        
        #update preamble, sample rate and len of channel points
        self.set_preamble()
        self.ch_points()
 
        # trigger is in the mid of osc memory
        pre_points = self.pre_p[ch_id]
        if pre_points is not None:
            data_start = (int(self.points/2) - pre_points)
            logger.debug(f'position of trigger is {data_start}')
        else:
            logger.warning(f'...Terminating. Pre points for CHAN{ch_id+1} not set.')
            return np.array([])
        
        #if one can read the whole data in 1 read
        dur_points = self.dur_p[ch_id]
        if dur_points is not None:
            if self.MAX_MEMORY_READ > dur_points:
                data_chunk = self._one_chunk_read(data_start, dur_points)
            else:
                data_chunk = self._multi_chunk_read(data_start, dur_points)
        else:
            logger.warning(f'...Terminating. Duration points for CHAN{ch_id+1} not set.')
            return np.array([])

        if self._ok_read(dur_points, data_chunk):
            logger.debug(f'...Finishing. Data read from {self.CH_IDS[ch_id]} is OK.')
            return data_chunk
        else:
            logger.debug('...Terminating. Problem during read from '
                         + f'{self.CH_IDS[ch_id]}. Returning empty array.')
            return np.array([])

    def baseline_correction(self,
                            data: npt.NDArray[np.uint8]
                            ) -> npt.NDArray[np.int16]:
        """Correct baseline for the data.
        
        Assumes that baseline as at the start of measured signal.
        """

        bl_points = int(len(data)*self.BL_LENGTH)
        logger.debug('Starting baseline correction on signal with '
                     + f'{len(data)} data points...')
        logger.debug(f'Baseline length is {bl_points}.')
        baseline = np.average(data[:bl_points])
        data_tmp = data.astype(np.int16)
        data_tmp -= int(baseline)
        logger.debug(f'...Finishing. Max val = {data_tmp.max()}, '
                     + f'min val = {data_tmp.min()}')
        return data_tmp

    def read_scr(self, ch_id: int) -> npt.NDArray[np.uint8]:
        """Read screen data for the channel."""

        chan = self.CH_IDS[ch_id]
        logger.debug(f'Reading screen data from {chan}')
        self.__osc.write(':WAV:SOUR ' + chan)
        self.__osc.write(':WAV:MODE NORM')
        self.__osc.write(':WAV:FORM BYTE')
        self.__osc.write(':WAV:STAR 1')
        self.__osc.write(':WAV:STOP ' + str(self.MAX_SCR_POINTS))
        self.__osc.write(':WAV:DATA?')
        data_chunk = np.frombuffer(self.__osc.read_raw(),
                                    dtype=np.uint8)[self.HEADER_LEN:]
        
        if self._ok_read(self.MAX_SCR_POINTS, data_chunk):
            logger.debug(f'...Finishing. Max val = {data_chunk.max()}, '
                         + f'min val = {data_chunk.min()}')
            return data_chunk.astype(np.uint8)
        else:
            logger.debug('...Terminating. Returning zeros.')
            return np.zeros(self.MAX_SCR_POINTS).astype(np.uint8)

    def measure_scr(self, 
                    read_ch1: bool=True, #read channel 1
                    read_ch2: bool=True, #read channel 2
                    correct_bl: bool=True, #correct baseline
                    smooth: bool=True, #smooth data
                    ) -> None:
        """Measure data from screen."""

        logger.debug('Starting measure signal from oscilloscope '
                     + 'screen.')
        self.bad_read = False
        self.set_preamble()

        for i, read_flag in enumerate([read_ch1, read_ch2]):
            if read_flag:
                data_raw = self.read_scr(i)
                if smooth:
                    data_raw = self.rolling_average(data_raw)
                if correct_bl:
                    data_raw = self.baseline_correction(data_raw)
                data = self.to_volts_scr(data_raw)
                self.scr_amp[i] = data.ptp() #type: ignore
                self.scr_data_raw[i] = data_raw
                self.scr_data[i] = data
                logger.debug(f'Screen data set for channel {self.CH_IDS[i]} set')
                logger.debug(f'Signal amplitude is {self.scr_amp[i]}')
                logger.debug('...Finishing.')

    def measure(self,
                read_ch1: bool=True, #read channel 1
                read_ch2: bool=True, #read channel 2
                correct_bl: bool=True, #correct baseline
                smooth: bool=True, #smooth data
                ) -> None:
        """Measure data from memory."""

        logger.debug('Starting measure signal from oscilloscope '
                     + 'memory.')
        self.bad_read = False
        logger.debug('Waiting for trigger to set...')
        while int(self.__osc.query(':TRIG:POS?'))<0:
            time.sleep(0.1)
        logger.debug('Trigger is ready')

        logger.debug('Writing :STOP to enable reading from memory')
        self.__osc.write(':STOP')
        for i, read_flag in enumerate([read_ch1, read_ch2]):
            if read_flag:
                data_raw = self.read_data(i)
                if smooth:
                    data_raw = self.rolling_average(data_raw)
                if correct_bl:
                    data_raw = self.baseline_correction(data_raw)
                data = self.to_volts(data_raw)
                self.data[i] = data
                self.data_raw[i] = data_raw
                self.amp[i] = data.ptp() # type: ignore
                logger.debug(f'Data set for channel {self.CH_IDS[i]} set')
                logger.debug(f'Signal amplitude is {self.amp[i]}')
        
        # run the oscilloscope again
        logger.debug('Writing :RUN to enable oscilloscope')
        self.__osc.write(':RUN')
        logger.debug('...Finishing.')

class PowerMeter:
    
    ###DEFAULTS###
    SENS = 210.4 #scalar coef to convert integral readings into [uJ]

    ch: int # channel ID number
    osc: Oscilloscope
    threshold: float # fraction of max amp for signal start
    
    data: List[pint.Quantity]
    laser_amp: pint.Quantity
    #start and stop indexes of the measured signal
    start_ind: int = 0
    stop_ind: int = 1

    def __init__(self, 
                 osc: Oscilloscope,
                 ch_id: int=1,
                 threshold: float=0.05) -> None:
        """PowerMeter class for working with
        Thorlabs ES111C pyroelectric detector.
        osc is as instance of Oscilloscope class, 
        which is used for reading data.
        ch_id is number of channel (starting from 0) 
        to which the detector is connected.
        """

        logger.debug('Instantiating PowerMeter connected to '
                     + f'{osc.CH_IDS[ch_id]}...')
        self.osc = osc
        self.ch = ch_id
        self.threshold = threshold
        logger.debug('...Finishing')

    def get_energy_scr(self) -> pint.Quantity:
        """Measure energy from screen (fast)"""

        logger.debug('Starting fast energy measuring...')
        if self.osc.not_found:
            logger.debug('...Terminating. not_found flag in osc instance is set.')
            msg = 'Oscilloscope not found'
            raise exceptions.OscilloscopeError(msg)
        
        self.osc.measure_scr()
        #return 0, if read data was not successfull
        #Oscilloscope class will promt warning
        if self.osc.bad_read:
            logger.warning('Energy measurement failed. 0 is returned')
            logger.debug('...Terminating')
            return 0*ureg('J')
        
        logger.debug('PowerMeter response obtained')
        data = self.osc.scr_data[self.ch]
        if data is not None:
            self.data = data
            laser_amp = self.energy_from_data(self.data,
                                            self.osc.xincrement)
            logger.debug(f'...Finishing. {laser_amp=}')
            return laser_amp
        else:
            logger.debug('...Terminating. Data not not accessible.')
            return 0*ureg.J


    def energy_from_data(self,
                         data: List[pint.Quantity],
                         step: pint.Quantity) -> pint.Quantity:
        """Calculate laser energy from data.

        <Step> is time step for the data.
        Data must be baseline corrected.
        """

        logger.debug('Starting convertion of raw signal to energy...')
        if len(data) < 10:
            logger.warning(f'data length is too small ({len(data)})')
            logger.debug('...Terminating.')
            msg = 'data too small for laser energy calc'
            raise exceptions.OscilloscopeError(msg)

        max_amp = np.max(data)
        logger.debug(f'Max value in signal is {max_amp}')
        thr = max_amp*self.threshold # type: ignore
        logger.debug(f'Signal starts when amplitude exceeds {thr}')
        try:
            str_ind = np.where(data>(thr))[0][0]
            self.start_ind = str_ind
            logger.debug(f'Position of signal start is {str_ind}/'
                         + f'{len(data)}')
        except IndexError:
            msg = 'Problem in set_laser_amp start_index'
            logger.warning(msg)
            logger.debug('...Terminating')
            return 0*ureg.J

        try:
            logger.debug('Starting search for signal end...')
            neg_data = np.where(data[self.start_ind:].magnitude < 0)[0] # type: ignore
            stop_ind_arr = neg_data + self.start_ind
            logger.debug(f'{len(stop_ind_arr)} points with negative '
                         + 'values found after signal start')
            stop_index = 0
            #check interval. If data at index + check_int less than zero
            #then we assume that the laser pulse is really finished
            check_ind = int(len(data)/100)
            for x in stop_ind_arr:
                if (x+check_ind) < len(data) and data[x+check_ind] < 0:
                    stop_index = x
                    break
            if not stop_index:
                msg = 'End of laser impulse was not found'
                logger.warning(msg)
                logger.debug('...Terminating .')
                return 0*ureg.J
            logger.debug(f'Position of signal start is {stop_index}/'
                         + f'{len(data)}')
            self.stop_ind = stop_index
        except IndexError:
            msg = 'Problem in set_laser_amp stop_index'
            logger.warning(msg)
            logger.debug('...Terminating .')
            return 0*ureg.J

        laser_amp = np.sum(
            data[self.start_ind:self.stop_ind])*step.to('s').m*self.SENS

        laser_amp = laser_amp.m*ureg.mJ
        logger.debug(f'...Finishing. {laser_amp=}')
        return laser_amp
    
    def set_channel(
            self,
            chan: int,
            pre_time: pint.Quantity,
            post_time: pint.Quantity) -> None:
        """Sets read channel.
        
        <chan> is index, i.e. for <chan>=0 for CHAN1."""

        logger.debug(f'PowerMeter channel set to {self.osc.CH_IDS[chan]}')
        self.ch = chan
        self.osc.pre_t[chan] = pre_time
        self.osc.post_t[chan] = post_time
        self.osc.dur_t[chan] = pre_time + post_time #type:ignore

class PhotoAcousticSensOlymp:
    

    ch: int # channel ID number
    osc: Oscilloscope

    def __init__(self,
                 osc: Oscilloscope,
                 ch_id: int=0) -> None:
        """PA sensor class for working with
        oscilloscope based 1-channel Olympus US transducer.
        """

        logger.debug('Instantiating PA sensor connected to '
                     + f'{osc.CH_IDS[ch_id]}.')
        self._osc = osc
        self.ch = ch_id

    def set_channel(self, 
                    chan: int,
                    pre_time: pint.Quantity,
                    post_time: pint.Quantity,) -> None:
        """Sets read channel"""

        self.ch = chan
        self._osc.pre_t[chan] = pre_time
        self._osc.post_t[chan] = post_time
        self._osc.dur_t[chan] = pre_time + post_time #type:ignore
        logger.debug(f'PA sensor channel set to {self._osc.CH_IDS[chan]}')
        
