"""
Oscilloscope based devices
"""

from typing import List, Optional, Tuple, cast
import logging
import time
from unittest import result

import pyvisa as pv
import numpy as np
import numpy.typing as npt
from pint.facets.plain.quantity import PlainQuantity

import modules.exceptions as exceptions
from . import Q_, ureg

logger = logging.getLogger(__name__)

class Oscilloscope:
    """Rigol MSO1000Z/DS1000Z
    
    General usage procedure:
    1. Create class instance
    2. Call initialize
    3. Use public API methods
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
        self.sample_rate: PlainQuantity
        self.format: int # 0 - BYTE, 1 - WORD, 2 - ASC 
        self.read_type: int # 0 - NORMal, 1 - MAXimum, 2 RAW
        self.points: int # between 1 and 240000000
        self.averages: int # number of averages in average mode, 1 in other modes
        self.xincrement: PlainQuantity # time diff between points
        self.xorigin: PlainQuantity # start time of the data
        self.xreference: PlainQuantity # reference time of data
        self.yincrement: float # the waveform increment in the Y direction
        self.yorigin: float # vertical offset relative to the yreference
        self.yreference: float # vertical reference position in the Y direction

        #channels attributes
        # time before trig to save data
        self.pre_t: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        # time after trigger
        self.post_t: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        # duration of data
        self.dur_t: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        # same in points
        self.pre_p: List[Optional[int]] = [None]*self.CHANNELS
        self.post_p: List[Optional[int]] =[None]*self.CHANNELS
        self.dur_p: List[Optional[int]] = [None]*self.CHANNELS
        self.data: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        self.data_raw: List[Optional[npt.NDArray[np.int8]]] = [None]*self.CHANNELS
        # amplitude of data
        self.amp: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        self.scr_data: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        self.scr_data_raw: List[Optional[npt.NDArray[np.int8]]] = [None]*self.CHANNELS
        self.scr_amp: List[Optional[PlainQuantity]] = [None]*self.CHANNELS

        # data smoothing parameters
        self.ra_kernel: int# kernel size for rolling average smoothing
        
        self.not_found = True # flag for state of osc
        self.read_chunks: int# amount of reads required for a chan

        logger.debug('Oscilloscope class instantiated!')
        
    def initialize(self,
                 ra_kernel_size: int=20 #smoothing by rolling average
                 ) -> bool:
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
            return False
        else:
            try:
                self.__osc.close()
            except:
                pass
            self.__osc: pv.resources.USBInstrument
            self.__osc = rm.open_resource(self.OSC_ID) # type: ignore
            logger.debug('Oscilloscope device found!')
        if not self._set_preamble():
            logger.debug('...Terminating. Preamble cannot be set.')
            return False
        if not self._set_sample_rate():
            logger.debug('...Terminating. Sample rate cannot be set.')
            return False

        #smoothing parameters
        self.ra_kernel = ra_kernel_size
        
        self.not_found = False
        logger.debug('...Finishing. Osc fully initiated.')
        return True

    def connection_check(self) -> Tuple[bool,bool]:
        """Check connection to the oscilloscope.

        Set <not_found> flag.
        Return method execution status, followed by connection status.
        """

        logger.debug('Starting connection check...')
        if self.not_found:
            logger.debug('...Finishing. not_found flag set')
            return True, False
        try:
            serial_number = self.__osc.serial_number
            if serial_number == 'NI-VISA-0':
                self.not_found = True
                logger.debug('...Finishing. Not found.')
                return True, False
            else:
                logger.debug(f'...Finishing. Success. {serial_number=}')
                return True, True
        except Exception as err:
            logger.debug(f'Operation failed with error {type(err)}')
            self.not_found = True
            return False, False

    def measure(self,
                read_ch1: bool=True, #flag for channel 1 read
                read_ch2: bool=True, #flag for channel 1 read
                correct_bl: bool=True, #correct baseline
                smooth: bool=True, #smooth data
        ) -> Tuple[bool,list[PlainQuantity|None],list[npt.NDArray[np.int8]|None]]:
        """Measure data from memory.
        
        Data is saved to <data> and <data_raw> attributes.
        Return tuple, where first element is execution flag,
        second and third are <data> and <data_raw> attributes.
        """

        logger.debug('Starting measure signal from oscilloscope '
                     + 'memory.')
        logger.debug('Resetting data and data_raw attributes.')
        self.data = [None]*self.CHANNELS
        self.data_raw = [None]*self.CHANNELS
        if self._wait_trig() is None:
            logger.debug('...Terminating measure. Trigger problem.')
            result = (False, self.data.copy(), self.data_raw.copy())
            return result
        
        logger.debug('Writing :STOP to enable reading from memory')
        written = self._write([':STOP'])
        if written is None:
            logger.debug('...Terminating measure. Error during writing to osc.')
            result = (False, self.data.copy(), self.data_raw.copy())
            return result
        logger.debug(f'{written} bytes written.')
        
        for i, read_flag in enumerate([read_ch1, read_ch2]):
            if read_flag:
                data_raw = self._read_data(i)
                if data_raw is None:
                    logger.debug('...Terminating measure. ' 
                                 + 'Error during writing to osc.')
                    result = (False, self.data.copy(), self.data_raw.copy())
                    return result
                if smooth:
                    data_raw = self._rolling_average(data_raw)
                    if data_raw is None:
                        logger.debug('...Terminating measure. ' 
                                 + 'Error during writing to osc.')
                        result = (False, 
                                  self.data.copy(), 
                                  self.data_raw.copy())
                        return result
                if correct_bl:
                    data_raw = self._baseline_correction(data_raw)
                data = self._to_volts(data_raw)
                self.data[i] = data
                self.data_raw[i] = data_raw
                self.amp[i] = data.ptp() # type: ignore
                logger.debug(f'Data for channel {self.CH_IDS[i]} set.')
                logger.debug(f'Signal amplitude is {self.amp[i]}')
        
        logger.debug('Writing :RUN to enable oscilloscope')
        written = self._write([':RUN'])
        if written is None:
            logger.debug('...Terminating measure. Error during writing to osc.')
            result = (False, self.data.copy(), self.data_raw.copy())
            return result
        logger.debug('...Finishing. Measure successfull.')
        result = (True, self.data.copy(), self.data_raw.copy())
        return result

    def measure_scr(self, 
                    read_ch1: bool=True, #read channel 1
                    read_ch2: bool=True, #read channel 2
                    correct_bl: bool=True, #correct baseline
                    smooth: bool=True, #smooth data
        ) -> Tuple[bool,list[PlainQuantity|None],list[npt.NDArray[np.int8]|None]]:
        """Measure data from screen."""

        logger.debug('Starting measure signal from oscilloscope '
                     + 'screen.')
        logger.debug('Resetting scr_data and scr_data_raw attributes.')
        self.scr_data = [None]*self.CHANNELS
        self.scr_data_raw = [None]*self.CHANNELS
        for i, read_flag in enumerate([read_ch1, read_ch2]):
            if read_flag:
                data_raw = self._read_scr_data(i)
                if data_raw is None:
                    logger.debug('...Terminating measure_scr. ' 
                                 + 'Error during writing to osc.')
                    result = (False,
                              self.scr_data.copy(),
                              self.scr_data_raw.copy())
                    return result
                if smooth:
                    data_raw = self._rolling_average(data_raw)
                    if data_raw is None:
                        logger.debug('...Terminating measure_scr. ' 
                                 + 'Error during writing to osc.')
                        result = (False, 
                                  self.scr_data.copy(),
                              self.scr_data_raw.copy())
                        return result
                if correct_bl:
                    data_raw = self._baseline_correction(data_raw)
                data = self._to_volts(data_raw)
                self.scr_amp[i] = data.ptp() #type: ignore
                self.scr_data_raw[i] = data_raw
                self.scr_data[i] = data
                logger.debug(f'Screen data for channel {self.CH_IDS[i]} set.')
                logger.debug(f'Signal amplitude is {self.scr_amp[i]}.')
        logger.debug('...Finishing. Screen measure successfull.')
        result = (True, self.scr_data.copy(), self.scr_data_raw.copy())
        return result

    def _wait_trig(self, timeout: int=5000) -> bool|None:
        """Wait for trigger to set.
        
        Return True, when trigger is set.
        Return None if timeout expired or error occured 
        communication with osc.
        """

        logger.debug('Waiting for trigger to set...')
        start = int(time.time()*1000)
        trig = self._query(':TRIG:POS?')
        if trig is None:
            logger.debug('...Terminating trigger set. Bad read from osc.')
            return None
        try:
            trig = int(trig)
        except ValueError:
            logger.debug('...Terminating trigger set. Bad read from osc.')
            return None
        while trig < 0:
            stop = int(time.time()*1000)
            if (stop-start) > timeout:
                logger.debug('...Terminating. Timeout reached.')
                return None
            time.sleep(0.1)
            trig = self._query(':TRIG:POS?')
            if trig is None:
                logger.debug('...Terminating trigger set. Bad read from osc.')
                return None
            try:
                trig = int(trig)
            except ValueError:
                logger.debug('...Terminating trigger set. Bad read from osc.')
                return None
        logger.debug('...Trigger set.')
        return True

    def _query(self, message: str) -> str|None:
        """Send a querry to the oscilloscope."""

        if self.not_found:
            logger.debug('Querry cannot be sent. Osc is not connected.')
            return None
        try:
            logger.debug(f'Sending query: {message}')
            answer = self.__osc.query(message)
            return answer
        except pv.errors.VisaIOError:
            self.not_found = True
            logger.warning('Querry to osc failed. Osc could be disconnected!')
            return None
        
    def _write(self, message: List[str]) -> int|None:
        """Send a querry to the oscilloscope.
        
        Return number of written bytes, or None if write was not performed.
        """

        if self.not_found:
            logger.debug('Write cannot be done. Osc is not connected.')
            return None
        try:
            written = 0
            for msg in message:
                written += self.__osc.write(msg)
            return written
        except pv.errors.VisaIOError:
            self.not_found = True
            logger.warning('Write to osc failed. Osc could be disconnected!')
            return None

    def _read(self, 
              cut_header: bool=True
        ) -> npt.NDArray[np.int8]|None:
        """Read data from osc buffer.
        
        Return data without header if cut_header is set.
        """

        if self.not_found:
            logger.debug('Read cannot be done. Osc is not connected.')
            return None
        try:
            raw_data = self.__osc.read_raw()
        except pv.errors.VisaIOError:
            self.not_found = True
            logger.warning('Read from osc failed. Osc could be disconnected!')
            return None
        data = np.frombuffer(raw_data, dtype=np.uint8)
        data = self._raw_dtype_convert(data)
        if cut_header:
            return data[self.HEADER_LEN:]
        else:
            return data

    def _raw_dtype_convert(self, data: npt.NDArray[np.uint8]
                           ) -> npt.NDArray[np.int8]:
        """Convert data type for an array."""

        return (data-128).astype(np.int8)

    def _set_preamble(self) -> bool:
        """Set osc params.
        
        Return execution status.
        """

        logger.debug('Starting osc params set...')
        query_results = self._query(':WAV:PRE?')
        if query_results is not None:
            try:
                preamble_raw = query_results.split(',')
                self.format = int(preamble_raw[0]) 
                self.read_type = int(preamble_raw[1])
                self.points = int(preamble_raw[2])
                self.averages = int(preamble_raw[3])
                self.xincrement = float(preamble_raw[4])*ureg.s
                self.xorigin = float(preamble_raw[5])*ureg.s
                self.xreference = float(preamble_raw[6])*ureg.s
                self.yincrement = float(preamble_raw[7])
                self.yorigin = float(preamble_raw[8])
                self.yreference = float(preamble_raw[9])
                logger.debug(f'...Finishing. Success. {len(preamble_raw)} '
                            + 'parameters read and set.')
                return True
            except IndexError:
                logger.debug('...Terminating osc params set. '
                             + 'Wrong amount of params read.')
                return False
            except ValueError:
                logger.debug('...Terminating osc params set. '
                             + 'Bad params read.')
                return False
        else:
            logger.debug('...Terminating osc params set. '
                         + 'params cannot be obtained')
            return False

    def _set_sample_rate(self) -> bool:
        """Update sample rate.
        
        Return execution status.
        """
        sample_rate = self._query(':ACQ:SRAT?')
        if sample_rate is not None:
            try:
                sample_rate = float(sample_rate)
            except ValueError:
                logger.debug('Sample_rate cannot be set. Bad value read.')
                return False
            self.sample_rate = Q_(sample_rate, 'Hz')
            logger.debug(f'Sample rate updated to {self.sample_rate}')
            return True
        else:
            logger.debug('Sample rate cannot be obtained.')
            return False

    def _time_to_points (self, duration: PlainQuantity) -> int:
        """Convert duration into amount of data points."""
        
        points = int((duration*self.sample_rate).magnitude) + 1
        logger.debug(f'{duration} converted to {points} data points.')
        return points

    def _ch_points(self) -> bool:
        """Update len of pre, post and dur points for all channels.
        
        Automatically update sample rate before calculation.
        Return execution status.
        """

        logger.debug('Starting set amount of data points for each channel...')
        if not self._set_sample_rate():
            logger.debug('...Terminating. Sample rate is not available.')
            return False
        
        all_upd = True
        for i in range(self.CHANNELS):
            logger.debug(f'Setting points for channel {i+1}')
            pre_t = self.pre_t[i]
            if pre_t is not None:
                self.pre_p[i] = self._time_to_points(pre_t)
            else:
                logger.debug(f'Warning! "pre time" for CHAN{i+1} is not set.')
                all_upd = False
            post_t = self.post_t[i]
            if post_t is not None:
                self.post_p[i] = self._time_to_points(post_t)
            else:
                logger.debug(f'Warning! "post time" for CHAN{i+1} is not set.')
                all_upd = False
            dur_t = self.dur_t[i]
            if dur_t is not None:
                self.dur_p[i] = self._time_to_points(dur_t)
            else:
                logger.debug(f'Warning! "dur time" for CHAN{i+1} is not set.')
                all_upd = False
            logger.debug('...Finishing.')

        if all_upd:
            return True
        else:
            return False
            
    def _rolling_average(self,
                        data: npt.NDArray[np.int8]
                        ) -> npt.NDArray[np.int8]|None:
        """Smooth data using rolling average method."""
        
        logger.debug('Starting _rolling_average smoothing...')
        min_signal_size = int(self.SMOOTH_LEN_FACTOR*self.ra_kernel)
        if len(data) < min_signal_size:
            logger.debug(f'...Terminating. Signal has only {len(data)} data points, '
                         + f'but at least {min_signal_size} is required.')
            return None
        kernel = np.ones(self.ra_kernel)/self.ra_kernel
        tmp_array = np.zeros(len(data))
        border = int(self.ra_kernel/2)
        tmp_array[border:-(border-1)] = np.convolve(data,kernel,mode='valid')
        #leave edges unfiltered
        tmp_array[:border] = tmp_array[border]
        tmp_array[-(border):] = tmp_array[-border]
        result = tmp_array.astype(np.int8)
        logger.debug(f'...Finishing. Success. '
                     +f'max val = {result.max()}, min val = {result.min()}.')
        return result

    def _read_chunk(self,
                        start: int,
                        dur: int) -> npt.NDArray[np.int8]|None:
        """Read a single chunk from memory.
        
        Return data without header.
        """

        logger.debug('Starting _read_chunk...')
        logger.debug(f'Start point: {start+1}, stop point: {dur + start}.')
        
        msg = []
        msg.append(':WAV:STAR ' + str(start+1))
        msg.append(':WAV:STOP ' + str(dur+start))
        msg.append(':WAV:DATA?')
        written = self._write(msg)
        if written is None:
            logger.debug('...Terminating _read_chunk. '
                         + 'Error during writing data to osc.')
            return None
        logger.debug(f'{written} bytes written to osc.')
        data = self._read()
        if data is None:
            logger.debug('...Terminating _read_chunk. Read from osc failed.')
            return None
        if self._ok_read(dur, data):
            logger.debug(f'...Finishing. Signal with {len(data)} data points read. '
                     + f'max val = {data.max()}, min val = {data.min()}')
            return data
        else:
            logger.debug('...Terminating. Wrong length of data read.')
            return None

    def _multi_chunk_read(self,
                          start: int,
                          dur: int) -> npt.NDArray[np.int8]|None:
        """Read from memory in multiple chunks.
        
        Return data without header.
        """

        logger.debug('Starting _multi_chunk_read...')
        logger.debug(f'Start point: {start+1}, duartion point: {dur}.')
        data_frames = int(dur/self.MAX_MEMORY_READ) + 1
        logger.debug(f'{data_frames} reads are required.')
        data = np.empty(dur, dtype=np.int8)
        for i in range(data_frames):
            start_i = i*self.MAX_MEMORY_READ
            stop_i = (i+1)*self.MAX_MEMORY_READ
            if (dur - stop_i) > 0:
                dur_i = self.MAX_MEMORY_READ
            else:
                dur_i = dur - 1 - start_i
            data_chunk = self._read_chunk(start + start_i, dur_i)
            if data_chunk is None:
                logger.debug('...Terminating _multi_chunk_read. read_chunk failed.')
                return None
            data[start_i:start_i+dur_i] = data_chunk.copy()
        logger.debug(f'...Finishing. Signal with {len(data)} data points read. '
                     + f'max val = {data.max()}, min val = {data.min()}')
        return data

    def _to_volts(self, data: npt.NDArray[np.int8]) -> PlainQuantity:
        """Converts data to volts."""

        result = data*self.yincrement
        result = Q_(result, 'volt')
        logger.debug(f'...Finishing. Max val={result.max()}, ' #type: ignore
                     + f'min val={result.min()}') #type: ignore
        return result

    def _ok_read(self, dur: int,
                 data_chunk: npt.NDArray) -> bool:
        """Verify that read data have necessary size."""

        if dur == len(data_chunk):
            logger.debug('Data length is OK.')
            return True
        else:
            logger.debug('Data length is wrong, '
                         + f'{dur} is required, '
                         + f'actual length is {len(data_chunk)}')
            return False

    def _read_data(self, ch_id: int) -> npt.NDArray[np.int8]|None:
        """Read data from the specified channel.
        
        Return read data or None.
        """

        logger.debug(f'Starting read from {self.CH_IDS[ch_id]}.')
        cmd = []
        cmd.append(':WAV:SOUR ' + self.CH_IDS[ch_id])
        cmd.append(':WAV:MODE RAW')
        cmd.append(':WAV:FORM BYTE')
        written = self._write(cmd)
        if written is None:
            logger.debug('...Terminating _read_data. '
                         + 'Error during writing data to osc.')
            return None
        logger.debug(f'{written} bytes written to osc.')
        if not self._set_preamble():
            logger.debug('...Terminating _read_data. Preamble cannot be set.')
            return None
        if not self._ch_points():
            logger.debug('...Terminating _read_data. Points for channel. '
                         + 'cannot be calculated.')
            return None
        # _ch_points ensured that values are calculated
        pre_points = cast(int, self.pre_p[ch_id])
        dur_points = cast(int, self.dur_p[ch_id])
        # trigger is in the mid of osc memory
        data_start = (int(self.points/2) - pre_points)
        
        #if one can read the whole data in 1 read
        if self.MAX_MEMORY_READ > dur_points:
            data = self._read_chunk(data_start, dur_points)
        else:
            data = self._multi_chunk_read(data_start, dur_points)
        if data is None:
            logger.debug('...Terminating _read_data. Data cannot be read.')
            return None
        return data

    def _baseline_correction(self,
                            data: npt.NDArray[np.int8]
                            ) -> npt.NDArray[np.int8]:
        """Correct baseline for the data.
        
        Assumes that baseline is at the start of measured signal.
        """

        bl_points = int(len(data)*self.BL_LENGTH)
        logger.debug('Starting baseline correction on signal with '
                     + f'{len(data)} data points...')
        logger.debug(f'Baseline length is {bl_points}.')
        baseline = np.average(data[:bl_points])
        data_tmp = data.astype(np.int8)
        data_tmp -= int(baseline)
        logger.debug(f'...Finishing. Max val = {data_tmp.max()}, '
                     + f'min val = {data_tmp.min()}')
        return data_tmp

    def _read_scr_data(self, ch_id: int) -> npt.NDArray[np.int8]|None:
        """Read screen data for the channel."""

        chan = self.CH_IDS[ch_id]
        logger.debug(f'Starting screen read from {chan}')
        if not self._set_preamble():
            logger.debug('...Terminating _read_scr_data. '
                         + 'Preamble cannot be set.')
            return None
        cmd = []
        cmd.append(':WAV:SOUR ' + chan)
        cmd.append(':WAV:MODE NORM')
        cmd.append(':WAV:FORM BYTE')
        cmd.append(':WAV:STAR 1')
        cmd.append(':WAV:STOP ' + str(self.MAX_SCR_POINTS))
        cmd.append(':WAV:DATA?')
        written = self._write(cmd)
        if written is None:
            logger.debug('...Terminating _read_scr_data. '
                         + 'Error during writing data to osc.')
            return None
        logger.debug(f'{written} bytes written to osc.')
        data = self._read()
        if data is None:
            logger.debug('...Terminating _read_scr_data. Read from osc failed.')
            return None
        if self._ok_read(self.MAX_SCR_POINTS, data):
            logger.debug(f'...Finishing. Signal with {len(data)} data points read. '
                     + f'max val = {data.max()}, min val = {data.min()}')
            return data
        else:
            logger.debug('...Terminating. Wrong length of data read.')
            return None


class PowerMeter:
    
    ###DEFAULTS###
    SENS = 210.4 #scalar coef to convert integral readings into [uJ]

    ch: int # channel ID number
    osc: Oscilloscope
    threshold: float # fraction of max amp for signal start
    
    data: PlainQuantity
    laser_amp: PlainQuantity
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

    def get_energy_scr(self) -> Optional[PlainQuantity]:
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
            return None
        
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
            return None


    def energy_from_data(self,
                         data: PlainQuantity,
                         step: PlainQuantity) -> Optional[PlainQuantity]:
        """Calculate laser energy from data.

        <Step> is time step for the data.
        Data must be baseline corrected.
        """

        logger.debug('Starting convertion of raw signal to energy...')
        if len(data) < 10:
            logger.warning(f'data length is too small ({len(data)})')
            logger.debug('...Terminating.')
            msg = 'data too small for laser energy calc'
            return None

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
            return None

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
                return None
            logger.debug(f'Position of signal start is {stop_index}/'
                         + f'{len(data)}')
            self.stop_ind = stop_index
        except IndexError:
            msg = 'Problem in set_laser_amp stop_index'
            logger.warning(msg)
            logger.debug('...Terminating .')
            return None

        laser_amp = np.sum(
            data[self.start_ind:self.stop_ind])*step.to('s').m*self.SENS

        laser_amp = laser_amp.m*ureg.mJ
        logger.debug(f'...Finishing. {laser_amp=}')
        return laser_amp
    
    def set_channel(
            self,
            chan: int,
            pre_time: PlainQuantity,
            post_time: PlainQuantity) -> None:
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
                    pre_time: PlainQuantity,
                    post_time: PlainQuantity,) -> None:
        """Sets read channel"""

        self.ch = chan
        self._osc.pre_t[chan] = pre_time
        self._osc.post_t[chan] = post_time
        self._osc.dur_t[chan] = pre_time + post_time #type:ignore
        logger.debug(f'PA sensor channel set to {self._osc.CH_IDS[chan]}')