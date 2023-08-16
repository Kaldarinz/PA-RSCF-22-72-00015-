"""
Oscilloscope based devices
"""
from typing import List
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

    #channels attributes
    pre_t: List[pint.Quantity] # time before trig to save data
    post_t: List[pint.Quantity] # time after trigger
    dur_t: List[pint.Quantity] # duration of data
    pre_p: List[int] # same in points
    post_p: List[int]
    dur_p: List[int]
    data: List[List[pint.Quantity]]
    data_raw: List[npt.NDArray[np.uint8]]
    amp: List[pint.Quantity] # amplitude of data
    scr_data: List[List[pint.Quantity]]
    scr_data_raw: List[npt.NDArray[np.uint8]]
    scr_amp: List[pint.Quantity]

    # data smoothing parameters
    ra_kernel: int# kernel size for rolling average smoothing
    
    bad_read = False # flag for indication of error during read
    not_found = True # flag for state of osc
    read_chunks: int# amount of reads required for a chan

    def __init__(self) -> None:
        """oscilloscope class for Rigol MSO1000Z/DS1000Z device.
        Intended to be used as a module in other scripts.
        Call 'initialize' before working with Oscilloscope.
        """

        logger.debug('Oscilloscope class instantiated!')
        
    def initialize(self,
                 chan1_pre: pint.Quantity=ureg('150us'),
                 chan1_post: pint.Quantity=ureg('2500us'),
                 chan2_pre: pint.Quantity=ureg('100us'),
                 chan2_post: pint.Quantity=ureg('150us'),
                 ra_kernel_size: int=20 #smoothing by rolling average
                 ) -> None:
        """Oscilloscope initializator.
        chan_pre and chan_post are time intervals before and after
        trigger for saving data from corresponding channels.
        """
        
        logger.debug('Starting actual initialization of an oscilloscope...')
        rm = pv.ResourceManager()
        
        logger.debug('Searching for VISA devices')
        all_instruments = rm.list_resources()
        logger.debug(f'{len(all_instruments)} VISA devices found')
        instrument_name = list(filter(lambda x: self.OSC_ID in x,
                                    all_instruments))
        if len(instrument_name) == 0:
            logger.debug('Oscilloscope was not found among VISA '
                         + 'devices. Init failed')
            raise exceptions.OscilloscopeError('Oscilloscope was not found')
        else:
            self.__osc = rm.open_resource(instrument_name[0])
            logger.debug('Oscilloscope device found')

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

    def connection_check(self) -> None:
        """Checks connection to the oscilloscope.
        Sets not_found flag.
        Never raises exceptions.
        """

        logger.debug('Trying to read and write from the oscilloscope')
        try:
            self.__osc.write(':SYST:GAM?') # type: ignore
            np.frombuffer(self.__osc.read_raw(), dtype=np.uint8) # type: ignore
            logger.debug('Operations complete')
            self.not_found = False
        except Exception as err:
            logger.debug(f'Operation failed with error {type(err)}')
            logger.warning('Oscilloscope is not connected')
            self.not_found = True

    def query(self, message: str) -> str:
        """Sends a querry to the oscilloscope"""

        logger.debug(f'Sending query: {message}')
        return self.__osc.query(message) # type: ignore
        
    def set_preamble(self) -> None:
        """Sets osc params."""

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
        logger.debug(f'{len(preamble_raw)} parameters obtained and set')

    def set_sample_rate(self) -> None:
        """Updates sample rate."""

        self.sample_rate = (float(self.query(':ACQ:SRAT?'))*ureg('hertz'))
        logger.debug(f'Sample rate updated to {self.sample_rate}')

    def time_to_points (self, duration: pint.Quantity) -> int:
        """Convert duration into amount of data points."""
        
        logger.debug(f'Calculating amount of points for {duration}')
        points = int((duration*self.sample_rate).magnitude) + 1
        logger.debug(f'Amount of calculated points: {points}')
        return points

    def ch_points(self) -> None:
        """Updates len of pre, post and dur points for all channels."""

        logger.debug('Setting amount of data points for each channel...')
        self.set_sample_rate()
        for i in range(self.CHANNELS):
            logger.debug(f'Setting pre_points for channel {i+1}')
            self.pre_p[i] = self.time_to_points(self.pre_t[i])

            logger.debug(f'Setting post_points for channel {i+1}')
            self.post_p[i] = self.time_to_points(self.post_t[i])

            logger.debug(f'Setting total_dur_points for channel {i+1}')
            self.dur_p[i] = self.time_to_points(self.dur_t[i])

    def rolling_average(self,
                        data: npt.NDArray[np.uint8]
                        ) -> npt.NDArray[np.uint8]:
        """Smooth data using rolling average method."""
        
        logger.debug('Starting rolling_average smoothing...')
        min_signal_size = int(self.SMOOTH_LEN_FACTOR*self.ra_kernel)
        if len(data)<min_signal_size:
            logger.debug(f'Signal has only {len(data)} data points, '
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
        logger.debug(f'signal with {len(data)} data points smoothed')
        return tmp_array.astype(np.uint8)

    def _one_chunk_read(self,
                        start: int,
                        dur: int) -> npt.NDArray[np.uint8]:
        """read from memory in single chunk.
        
        Returns data without header.
        """

        logger.debug('Starting _one_chunk_read...')
        
        msg = ':WAV:STAR ' + str(start + 1)
        logger.debug(f'Writing {msg}')
        self.__osc.write(msg) # type: ignore

        msg = ':WAV:STOP ' + str(dur + start)
        logger.debug(f'Writing {msg}')
        self.__osc.write(msg) # type: ignore

        msg = ':WAV:DATA?'
        logger.debug(f'Writing {msg}')
        self.__osc.write(msg) # type: ignore

        data = np.frombuffer(self.__osc.read_raw(), # type: ignore
                             dtype=np.uint8)[self.HEADER_LEN:]
        logger.debug(f'Signal with {len(data)} data points read')

        return data.astype(np.uint8)

    def _multi_chunk_read(self,
                          start: int,
                          dur: int) -> npt.NDArray[np.uint8]:
        """reads from memory in multiple chunks.
        
        Returns data without header.
        """

        logger.debug('Starting _multi_chunk_read...')
        data = np.zeros(dur).astype(np.uint8)
        data_frames = int(dur/self.MAX_MEMORY_READ) + 1
        logger.debug(f'{data_frames} reads are required')

        for i in range(data_frames):
            start_i = i*self.MAX_MEMORY_READ
            stop_i = (i+1)*self.MAX_MEMORY_READ

            command = ':WAV:STAR ' + str(start + start_i + 1)
            logger.debug(f'Writing {command}')
            self.__osc.write(command) # type: ignore
            
            if (dur - stop_i) > 0:
                command = (':WAV:STOP ' + str(start + stop_i))
            else:
                command = (':WAV:STOP ' + str(start + dur - 1))
            logger.debug(f'Writing {command}')
            self.__osc.write(command) # type: ignore
            
            command = ':WAV:DATA?'
            logger.debug(f'Writing {command}')
            self.__osc.write(command) # type: ignore
            
            data_chunk = np.frombuffer(self.__osc.read_raw(), # type: ignore
                                       dtype=np.uint8)[self.HEADER_LEN:]
            logger.debug(f'Chunk with {len(data_chunk)} data points read')

            logger.debug('Start verification of read chunk length')
            if (dur - stop_i) > 0:
                if self._ok_read(self.MAX_MEMORY_READ,data_chunk):
                    data[start_i:stop_i] = data_chunk.copy()
            else:
                if self._ok_read(len(data[start_i:-1]), data_chunk):
                    data[start_i:-1] = data_chunk.copy()

        logger.debug(f'Signal with {len(data)} datapoints read')
        return data

    def to_volts(self,
                 data: npt.NDArray[np.uint8]) -> List[pint.Quantity]:
        """converts data to volts"""

        logger.debug('Converting data to volts')
        return (data.astype(np.float64)
                - self.xreference.magnitude
                - self.yorigin)*self.yincrement*ureg('volt')
    
    def to_volts_scr(self,
                     data: npt.NDArray[np.uint8]) -> List[pint.Quantity]:
        """converts screen data to volts"""

        dy = self.yincrement
        logger.debug('Converting screen data to volts. '
                     + f'one step is {dy:.7f} V')
        return data.astype(np.float64)*dy*ureg('volt')

    def _ok_read(self, dur: int,
                 data_chunk: npt.NDArray[np.uint8]) -> bool:
        """verify that read data have necessary size"""

        if dur == len(data_chunk):
            logger.debug('Data length is OK')
            return True
        else:
            logger.debug('Data length is wrong, '
                         + f'{dur} is required, '
                         + f'actual length is {len(data_chunk)}')

            self.bad_read = True
            return False

    def read_data(self, ch_id: int) -> npt.NDArray[np.uint8]:
        """Reads data from the specified channel.
        
        Sets data to ch_data_raw attribute.
        """

        logger.debug(f'Start reading from {self.CH_IDS[ch_id]} method')

        cmd = ':WAV:SOUR ' + self.CH_IDS[ch_id]
        logger.debug(f'Writing {cmd}')
        self.__osc.write(cmd) # type: ignore

        cmd = ':WAV:MODE RAW'
        logger.debug(f'Writing {cmd}')
        self.__osc.write(cmd) # type: ignore

        cmd = ':WAV:FORM BYTE'
        logger.debug(f'Writing {cmd}')
        self.__osc.write(cmd) # type: ignore
        
        #update preamble, sample rate and len of channel points
        self.set_preamble()
        self.ch_points()
 
        # trigger is in the mid of osc memory
        data_start = (int(self.points/2) - self.pre_p[ch_id])
        logger.debug(f'position of trigger is {data_start}')
        
        #if one can read the whole data in 1 read
        logger.debug('Starting actual data acquisition procedures')
        if self.MAX_MEMORY_READ > self.dur_p[ch_id]:
            data_chunk = self._one_chunk_read(data_start,
                                                self.dur_p[ch_id])
        else:
            data_chunk = self._multi_chunk_read(data_start,
                                                self.dur_p[ch_id])

        logger.debug('Validating length of the read data')
        if self._ok_read(self.dur_p[ch_id], data_chunk):
            logger.debug(f'Data read from {self.CH_IDS[ch_id]} is OK')
            return data_chunk
        else:
            logger.debug(f'Problem during read from {self.CH_IDS[ch_id]}, '
                         + 'returning array with zeros')
            return np.zeros(self.dur_p[ch_id]).astype(np.uint8)

    def baseline_correction(self,
                            data: npt.NDArray[np.uint8]
                            ) -> npt.NDArray[np.uint8]:
        """Corrects baseline for the data.
        
        Assumes that baseline as at the start of measured signal.
        """

        bl_points = int(len(data)*self.BL_LENGTH)
        logger.debug('Starting baseline correction on signal with '
                     + f'{len(data)} data points. '
                     + f'Baseline length is {bl_points}.')
        baseline = np.average(data[:bl_points])
        data -= int(baseline)
        return data

    def read_scr(self, ch_id: int) -> npt.NDArray[np.uint8]:
        """reads screen data for the channel."""

        self.__osc.write(':WAV:SOUR ' + self.CH_IDS[ch_id]) # type: ignore
        self.__osc.write(':WAV:MODE NORM') # type: ignore
        self.__osc.write(':WAV:FORM BYTE') # type: ignore
        self.__osc.write(':WAV:STAR 1') # type: ignore
        self.__osc.write(':WAV:STOP ' + str(self.MAX_SCR_POINTS)) # type: ignore
        self.__osc.write(':WAV:DATA?') # type: ignore
        data_chunk = np.frombuffer(self.__osc.read_raw(), # type: ignore
                                    dtype=np.uint8)[self.HEADER_LEN:]
        
        if self._ok_read(self.MAX_SCR_POINTS, data_chunk):
            return data_chunk.astype(np.uint8)
        else:
            return np.zeros(self.MAX_SCR_POINTS).astype(np.uint8)

    def measure_scr(self, 
                    read_ch1: bool=True, #read channel 1
                    read_ch2: bool=True, #read channel 2
                    correct_bl: bool=True, #correct baseline
                    smooth: bool=True, #smooth data
                    ) -> None:
        """Measure data from screen."""

        logger.debug('Start measuring of signal from oscilloscope '
                     + 'screen.')
        
        logger.debug('Resetting bad_read flag to False')
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
                self.scr_amp[i] = abs(np.amax(data) - np.amin(data))
                self.scr_data_raw[i] = data_raw
                self.scr_data[i] = data
                logger.debug(f'Screen data set for channel {self.CH_IDS[i]} set')
                logger.debug(f'Signal amplitude is {self.scr_amp[i]}')

    def measure(self,
                read_ch1: bool=True, #read channel 1
                read_ch2: bool=True, #read channel 2
                correct_bl: bool=True, #correct baseline
                smooth: bool=True, #smooth data
                ) -> None:
        """Measure data from memory."""

        logger.debug('Start measuring of signal from oscilloscope '
                     + 'memory.')
        
        logger.debug('Resetting bad_read flag to False')
        self.bad_read = False

        logger.debug('Waiting for trigger to set...')
        while int(self.__osc.query(':TRIG:POS?'))<0: # type: ignore
            time.sleep(0.1)
        logger.debug('Trigger is ready')

        logger.debug('Writing :STOP to enable reading from memory')
        self.__osc.write(':STOP') # type: ignore
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
                self.amp[i] = abs(np.amax(data) - np.amin(data))
                logger.debug(f'Data set for channel {self.CH_IDS[i]} set')
                logger.debug(f'Signal amplitude is {self.amp[i]}')
        # run the oscilloscope again

        logger.debug('Writing :RUN to enable oscilloscope')
        self.__osc.write(':RUN') # type: ignore

class PowerMeter:
    
    ###DEFAULTS###
    SENS = 2630000 #scalar coef to convert integral readings into [uJ]

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
                     + f'{osc.CH_IDS[ch_id]}.')
        self.osc = osc
        self.ch = ch_id
        self.threshold = threshold

    def get_energy_scr(self) -> pint.Quantity:
        """Measure energy from screen (fast)"""

        logger.debug('Starting fast energy measuring')
        logger.debug('Checking if oscilloscope is available')
        if self.osc.not_found:
            logger.debug('not_found flag in osc instance is set. Terminating energy measurement.')
            msg = 'Oscilloscope not found'
            raise exceptions.OscilloscopeError(msg)
        
        self.osc.measure_scr()
        #return 0, if read data was not successfull
        #Oscilloscope class will promt warning
        if self.osc.bad_read:
            logger.warning('Energy measurement failed. 0 is returned')
            return 0*ureg('J')
        
        logger.debug('PowerMeter response obtained')
        self.data = self.osc.scr_data[self.ch]
        laser_amp = self.energy_from_data(self.data,
                                          self.osc.xincrement)

        return laser_amp
    
    def energy_from_data(self,
                         data: List[pint.Quantity],
                         step: pint.Quantity) -> pint.Quantity:
        """Calculate laser energy from data.

        <Step> is time step for the data.
        Data must be baseline corrected.
        """

        logger.debug('Starting convertion of raw signal to energy')
        if len(data) < 10:
            logger.warning(f'data length is too small ({len(data)})')
            msg = 'data too small for laser energy calc'
            raise exceptions.OscilloscopeError(msg)

        max_amp = np.amax(data)
        logger.debug(f'Max value in signal is {max_amp}')
        thr = max_amp*self.threshold
        logger.debug(f'Signal starts when amplitude exceeds {thr}')
        try:
            str_ind = np.where(data>(thr))[0][0]
            self.start_ind = str_ind
            logger.debug(f'Position of signal start is {str_ind}/'
                         + f'{len(data)}')
        except IndexError:
            msg = 'Problem in set_laser_amp start_index'
            logger.warning(msg)
            raise exceptions.OscilloscopeError(msg)

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
                raise exceptions.OscilloscopeError(msg)
            logger.debug(f'Position of signal start is {stop_index}/'
                         + f'{len(data)}')
            self.stop_ind = stop_index
        except IndexError:
            msg = 'Problem in set_laser_amp stop_index'
            logger.debug(msg)
            raise exceptions.OscilloscopeError(msg)

        laser_amp = np.sum(
            data[self.start_ind:self.stop_ind])*step.to('s').m*self.SENS

        logger.debug(f'Calculated value of energy is {laser_amp}')
        return laser_amp
    
    def set_channel(self, chan: int) -> None:
        """Sets read channel"""

        logger.debug(f'PowerMeter channel set to {self.osc.CH_IDS[chan]}')
        self.ch = chan

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
        self.osc = osc
        self.ch = ch_id

    def set_channel(self, chan: int) -> None:
        """Sets read channel"""

        logger.debug(f'PowerMeter channel set to {self.osc.CH_IDS[chan]}')
        self.ch = chan
