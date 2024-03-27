"""
Oscilloscope based devices.

Public API functions raise following kinds of osc_device related exceptions:
1. OscConnectError indicates that not_found flag is set, which means
that the last low level communiation with osc device failed and therefore
the osc device may need reinitialization.
2. OscIOError indicates that there was an error during communication with
osc, but the error is not critical and you could try to call the function again.
3. OscValueError indicated errors during processing data.

Implementation details.
1. Calls to private methods should assume that correct results are returned.
2. Osc exceptions should be handled by a public caller.
3. <connection_check> and <initialize> do not raise exceptions.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from typing import List, Optional, Tuple, cast
from collections import abc
import logging
import time
import math
from datetime import datetime

import pyvisa as pv
import numpy as np
import numpy.typing as npt
from scipy.signal import decimate
from pint.facets.plain.quantity import PlainQuantity

from ..exceptions import (
    OscConnectError,
    OscIOError,
    OscValueError
)
from ..data_classes import (
    EnergyMeasurement,
    OscMeasurement
)
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
        self.trig: int # trigger position in points

        #channels attributes
        # time before trig to save data
        self.pre_t: list[PlainQuantity | None] = [None]*self.CHANNELS
        # time after trigger
        self.post_t: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        # duration of data
        self.dur_t: List[Optional[PlainQuantity]] = [None]*self.CHANNELS
        # same in points
        self.pre_p: List[Optional[int]] = [None]*self.CHANNELS
        self.post_p: List[Optional[int]] =[None]*self.CHANNELS
        self.dur_p: List[Optional[int]] = [None]*self.CHANNELS
        self.data_raw: List[Optional[npt.NDArray[np.int16]]] = [None]*self.CHANNELS
        self.scr_data_raw: List[Optional[npt.NDArray[np.int16]]] = [None]*self.CHANNELS
        self.scale: list[PlainQuantity | None] = [None]*self.CHANNELS
        self.scr_scale: list[PlainQuantity | None] = [None]*self.CHANNELS

        # data smoothing parameters
        self.ra_kernel: int # kernel size for rolling average smoothing
        
        self.not_found = True # flag for state of osc
        self.read_chunks: int# amount of reads required for a chan

        logger.debug('Oscilloscope class instantiated!')
        
    def initialize(
            self,
            ra_kernel_size: int=6 #smoothing by rolling average, should be even
        ) -> bool:
        """Oscilloscope initializator."""
        
        logger.debug('Starting actual initialization of an oscilloscope...')
        rm = pv.ResourceManager()
        
        logger.debug('Searching for VISA devices')
        all_instruments = rm.list_resources()
        logger.debug(f'{len(all_instruments)} VISA devices found')
        if self.OSC_ID not in all_instruments:
            logger.debug(f'Found devices: {[dev for dev in all_instruments]}')
            logger.debug('...Terminating. Oscilloscope was not found among VISA '
                         + 'devices. Init failed')
            return False
        else:
            try:
                self.__osc.close()
            except:
                pass
            logger.debug('Trying to open resource')
            self.__osc: pv.resources.USBInstrument
            self.__osc = rm.open_resource(self.OSC_ID) # type: ignore
            self.not_found = False
            logger.debug('Oscilloscope device found!')
        try:
            self._set_preamble()
            self._set_sample_rate()
            self.set_measurement()
            self.run_normal()
        except (OscConnectError, OscIOError, OscValueError) as err:
            logger.debug(f'...Terminating. {err.value}')
            return False
        #smoothing parameters
        self.ra_kernel = ra_kernel_size
        logger.debug('...Finishing. Osc fully initiated.')
        return True

    def connection_check(self) -> bool:
        """Check connection to the oscilloscope.

        Set <not_found> flag if check fails.
        Return connection status.
        Do not raise exceptions.
        """

        logger.debug('Starting connection check...')
        if self.not_found:
            logger.debug('...Finishing. not_found flag set.')
            return False
        try:
            serial_number = self.__osc.serial_number
            if serial_number == 'NI-VISA-0':
                self.not_found = True
                logger.debug('...Finishing. Not found.')
                return False
            else:
                logger.debug(f'...Finishing. Success. {serial_number=}')
                return True
        except Exception as err:
            logger.debug(f'Communiation failed with error {type(err)}')
            self.not_found = True
            return False

    def run_normal(self) -> None:
        """Force osc to run normally."""

        self._write(
            [
                ':TRIG:SWE NORM',
                ':RUN'
            ]
        )

    def set_measurement(
            self,
            srat: PlainQuantity = Q_(25, 'MHz'),
            duration: PlainQuantity = Q_(2.4, 'ms'),
            trig_offset: PlainQuantity = Q_(200, 'us') 
        ) -> None:
        """
        Set measurement parameters.
        
        * Set sample rate, duration measured signal and position of
        trigger.
        * Set Y scale (`self.scale`) for both channels.
        * Verify and correct `pre_t` and `post_t` for both channels. 

        WARNING
        -------
        In current implementation this method does not accept any
        parameters and work only with default values.\n
        Internal logic of the osc makes it hard to support arbitrary
        values.
        """

        logger.debug('Start setting measurement params.')
        cmd = []
        # Changes to time scale and memory depth cannot be applied in
        # Stop mode
        cmd.append(':RUN')
        # In current implementation measured signal duration is equal
        # to signal on display. :TIM:SCAL: command set length of single
        # div on the screen, which have 12 divs in horizontal axis.
        div_dur = f'{(duration/12).to("s").m:f}'
        cmd.append(':TIM:SCAL ' + div_dur)
        # Amount of points is just product of sample rate and signal duration
        pts = int(srat*duration)
        cmd.append(':ACQ:MDEP ' + str(pts))
        # Trigger offset is set from the middle of the screen
        trig_pos = f'{(duration/2 - trig_offset).to("s").m:f}'
        cmd.append(':TIM:OFFS ' + trig_pos)
        self._write(cmd)

        # Read sampel rate
        self._set_sample_rate()

        # Verify pre and post time
        for i in range(self.CHANNELS):
            if self.pre_t[i] > trig_offset:
                self.pre_t[i] = trig_offset
            if self.post_t[i] > (duration - trig_offset):
                self.post_t[i] = (duration - trig_offset)
        self._ch_points()
        
        # Set Y scales for both channels
        for i in range(self.CHANNELS):
            self._write([':WAV:SOUR ' + self.CH_IDS[i]])
            yinc = self._query(':WAV:YINC?')
            try:
                self.scale[i] = Q_(float(yinc), 'V')
            except ValueError:
                err_msg = f'Bad yincrement for {self.CH_IDS[i]} read.'
                logger.debug(err_msg)
                raise OscIOError(err_msg)

    def fast_measure(self,
                read_ch1: bool=True,
                read_ch2: bool=True,
                eq_limits: bool=False,
                correct_bl: bool=True,
                smooth: bool=False
        ) -> OscMeasurement:
        """
        Fast measurement of data from memory.

        Accept the same arguments as `measure` and functions similarly,
        but assume that oscilloscope parameters did not change since
        last `set_measurement` call and do not request any additional
        data.
        """
        start = time.time()
        logger.debug(
            'Starting fast measure signal from oscilloscope memory.'
        )
        result = self._measure(
            read_ch1, read_ch2, eq_limits, correct_bl, smooth)
        stop = time.time()
        delta = (stop-start)*1000
        logger.debug(
            f'...Finishing fast measure from memory in {delta:.1f} ms.'
        )
        return result

    def measure(self,
                read_ch1: bool=True,
                read_ch2: bool=True,
                eq_limits: bool=False,
                correct_bl: bool=True,
                smooth: bool=False
        ) -> OscMeasurement:
        """
        Measure data from memory.
        
        Data is saved to ``data_raw`` attribute.\n
        Attributes
        ----------
        `read_ch1`, `read_ch2` - flags for measuring channels.\n
        `eq_limits` - this flag is checked only if only one of channels
            is set to measure. In such a case, setting this flag to
            `True` will result in measure of BOTH channels, but within
            bounds of a channel which was set to measure. This can be
            usefull, if one want to measure short PA signal without
            measuring long PM signal. Measurement of PM signal within
            bounds of PA signal will allow one to correctly calculate
            time offset of the PA signal.\n
        `correct_bl` - flag to perform baseline correction of measered
            data.\n
        `smooth` - flag to apply signal smoothing (rolling average
            in current implementation).\n
        Return
        ------
        OscMeasurement instance.
        """

        start = time.time()
        logger.debug(
            'Starting measure signal from oscilloscope memory.'
        )
        self.set_measurement()
        result = self._measure(
            read_ch1, read_ch2, eq_limits, correct_bl, smooth)
        stop = time.time()
        delta = (stop-start)*1000
        logger.debug(
            f'...Finishing measure from memory in {delta:.1f} ms.'
        )
        return result

    def measure_scr(
            self, 
            read_ch1: bool=True,
            read_ch2: bool=True,
            correct_bl: bool=True,
            smooth: bool=True,
        ) -> OscMeasurement:
        """
        Measure data from screen.
        
        ``read_ch1`` and ``read_ch2`` determine which channels to read.\n
        ``correct_bl`` applies baseline correction to measured data.\n
        ``smooth`` applies smoothing (rolling average) to the measured data.
        """

        start = time.time()
        # logger.debug(
        #     'Starting measure signal from oscilloscope screen. '
        #     + f'{smooth=}, {correct_bl=}'
        # )
        self.run_normal()
        self.scr_data_raw = [None]*self.CHANNELS
        for i, read_flag in enumerate([read_ch1, read_ch2]):
            if read_flag:
                # logger.debug(f'Starting screen read from {self.CH_IDS[i]}')
                data_raw = self._read_scr_data(i)
                if smooth:
                    data_raw = self.rolling_average(data_raw)
                if correct_bl:
                    data_raw = self._baseline_correction(data_raw)
                data_raw = self._trail_correction(data_raw)
                self.scr_data_raw[i] = data_raw
                # logger.debug(
                #     f'Screen data for channel {self.CH_IDS[i]} set. '
                #     + f'min = {data_raw.min()}, max = {data_raw.max()}'
                # )  
        
        result = OscMeasurement(
            datetime = datetime.now(),
            data_raw = self.scr_data_raw.copy(),
            dt = self.xincrement,
            pre_t = [self.xincrement*self.MAX_SCR_POINTS/2]*2,
            yincrement = self.scr_scale.copy()
        )
        stop = time.time()
        logger.debug(
            f'Measure screen done in {(stop-start)*1000:.1f} ms.'
        )
        return result

    def _measure(self,
                read_ch1: bool=True,
                read_ch2: bool=True,
                eq_limits: bool=False,
                correct_bl: bool=True,
                smooth: bool=False,
        ) -> OscMeasurement:

        if read_ch1^read_ch2 and eq_limits:
            if read_ch1:
                main_ind = 0
                slave_ind = 1
                read_ch2 = True
            else:
                main_ind = 1
                slave_ind = 0
                read_ch1 = True
            # Save current values of bounds for other channel
            old_pre = self.pre_t[slave_ind]
            old_post = self.post_t[slave_ind]
            old_dur = self.dur_t[slave_ind]
            # Set main channel bounds for both channels
            self.pre_t[slave_ind] = self.pre_t[main_ind]
            self.post_t[slave_ind] = self.post_t[main_ind]
            self.dur_t[slave_ind] = self.dur_t[main_ind]
            logger.debug(
                f'Both channels will be measured using bounds of '
                + f'{self.CH_IDS[main_ind]}'
            )
        self._write([':SING'])
        self._wait_trig()
        scan_datatime = datetime.now()
        for i, read_flag in enumerate([read_ch1, read_ch2]):
            if read_flag:
                logger.debug(f'Starting memory read from {self.CH_IDS[i]}.')
                self.data_raw[i] = None
                data_raw = self._read_data(i)
                logger.debug(
                    f'min = {data_raw.min()}, max = {data_raw.max()}'
                )
                if smooth:
                    data_raw = self.rolling_average(data_raw)
                if correct_bl:
                    data_raw = self._baseline_correction(data_raw)
                data_raw = self._trail_correction(data_raw)
                self.data_raw[i] = data_raw
                logger.debug(
                    f'Memory data for channel {self.CH_IDS[i]} set. '
                    + f'min = {data_raw.min()}, max = {data_raw.max()}'
                )
        result = OscMeasurement(
            datetime = scan_datatime,
            data_raw = self.data_raw.copy(),
            dt = (1/self.sample_rate).to('us'),
            pre_t = self.pre_t.copy(),
            yincrement = self.scale.copy()
        )
        if read_ch1^read_ch2 and eq_limits:
            logger.debug(f'Restoring bounds for {self.CH_IDS[slave_ind]}')
            self.pre_t[slave_ind] = old_pre
            self.post_t[slave_ind] = old_post
            self.dur_t[slave_ind] = old_dur
        return result

    def _trail_correction(
            self,
            data: npt.NDArray[np.int16],
            w: int = 10
        ) -> npt.NDArray[np.int16]:
        """
        Correct trailing values of data.
        
        ``w`` - amount of trailing points to be corrected.
        """
        
        # logger.debug('Starting trail correction procedure.')
        if len(data) < 2*w:
            logger.warning(
                'Trail correction cannot be done. Data too short.'
            )
            return data
        
        # minimum increment of data
        dif = np.abs(np.diff(data))
        if dif.max() == 0:
            return data
        # min_inc = dif[dif>0].min()
        # # correction is required, when average value
        # # in the last Window differs from the previous Window
        # # for more than 2 minimum data increment
        # if abs(data[-2*w:-w].mean() - data[-w:].mean()) > 2 * min_inc:
        #     # data is corrected by filling last Window values 
        #     # with mean from previous Window
        fill = int(data[-2*w:-w].mean())
        data[-w:] = fill
        return data

    def _wait_trig(self, timeout: int=5000) -> bool:
        """Wait for trigger to set.
        
        Return True, when trigger is set.
        <timeout> in ms.
        """

        logger.debug('Waiting for trigger to set...')
        start = int(time.time()*1000)
        trig = self._query(':TRIG:POS?')
        try:
            trig = int(trig)
        except ValueError:
            err_msg = 'Trig cannot be calculated. Bad read from osc.'
            logger.debug(err_msg)
            raise OscIOError(err_msg)
        while trig < 0:
            stop = int(time.time()*1000)
            if (stop-start) > timeout:
                err_msg = 'Trigger timeout reached.'
                logger.debug(err_msg)
                raise OscIOError(err_msg)
            time.sleep(0.01)
            trig = self._query(':TRIG:POS?')
            try:
                trig = int(trig)
            except ValueError:
                err_msg = 'Trig cannot be calculated. Bad read from osc.'
                logger.debug(err_msg)
                raise OscIOError(err_msg)
        stop = int(time.time()*1000)
        self.trig = trig
        logger.debug(f'...Trigger set in {stop-start} ms.')
        return True

    def _query(self, message: str) -> str:
        """Send a querry to the oscilloscope."""

        if self.not_found:
            err_msg = 'Querry cannot be sent. Osc is not connected.'
            logger.debug(err_msg)
            raise OscConnectError(err_msg)
        try:
            start = time.time()
            answer = self.__osc.query(message)
            stop = time.time()
            # logger.debug(f'Query {message} took {(stop-start)*1000} ms.')
            return answer
        except pv.errors.VisaIOError:
            self.not_found = True
            err_msg = 'Querry to osc failed.'
            logger.warning(err_msg)
            raise OscConnectError(err_msg)
        
    def _write(self, message: List[str]) -> int:
        """Send a querry to the oscilloscope.
        
        Return number of written bytes.
        """

        if self.not_found:
            err_msg = 'Write cannot be done. Osc is not connected.'
            logger.debug(err_msg)
            raise OscConnectError(err_msg)
        try:
            written = 0
            start = time.time()
            for msg in message:
                written += self.__osc.write(msg)
            stop = time.time()
            delta = stop - start
            # logger.debug(f'{written} bytes written to osc in {delta*1000:.1f} ms.')
            return written
        except pv.errors.VisaIOError:
            self.not_found = True
            err_msg = 'Write to osc failed.'
            logger.debug(err_msg)
            raise OscConnectError(err_msg)

    def _read(self, cut_header: bool=True
        ) -> npt.NDArray[np.int16]:
        """Read data from osc buffer.
        
        Return data without header if cut_header is set.
        """

        if self.not_found:
            err_msg = 'Read cannot be done. Osc is not connected.'
            logger.debug(err_msg)
            raise OscConnectError(err_msg)
        try:
            start = time.time()
            raw_data = self.__osc.read_raw()
            stop = time.time()
            delta = stop - start
            # logger.debug(f'Reading from osc took: {delta*1000:.1f} ms.')
        except pv.errors.VisaIOError:
            self.not_found = True
            err_msg = 'Read from osc failed.'
            logger.debug(err_msg)
            raise OscConnectError(err_msg)
        data = np.frombuffer(raw_data, dtype=np.uint8)
        data = self._raw_dtype_convert(data)
        if cut_header:
            return data[self.HEADER_LEN:]
        else:
            return data

    def _raw_dtype_convert(self, data: npt.NDArray[np.uint8]
                           ) -> npt.NDArray[np.int16]:
        """Convert data type for an array."""

        data = data.astype(np.int16)
        return (data-128)

    def _set_preamble(self) -> None:
        """Set osc params for the current channel."""

        query_results = self._query(':WAV:PRE?')
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
        except IndexError:
            err_msg = 'Wrong amount of params read.'
            logger.debug(err_msg)
            raise OscIOError(err_msg)
        except ValueError:
            err_msg = 'Bad params read.'
            logger.debug(err_msg)
            raise OscIOError(err_msg)

    def _set_sample_rate(self) -> None:
        """Update sample rate.
        
        Return execution status.
        """
        sample_rate = self._query(':ACQ:SRAT?')
        try:
            sample_rate = float(sample_rate)
        except ValueError:
            err_msg = 'Sample_rate cannot be set. Bad value read.'
            logger.debug(err_msg)
            raise OscIOError(err_msg)
        self.sample_rate = Q_(sample_rate, 'Hz')
        # logger.debug(f'Sample rate updated to {self.sample_rate}')

    def _time_to_points (self, duration: PlainQuantity) -> int:
        """Convert duration into amount of data points."""
        
        points = int((duration*self.sample_rate)) + 1
        logger.debug(f'{duration} converted to {points} data points.')
        return points

    def _ch_points(self) -> bool:
        """Update len of pre, post and dur points for all channels.
        
        Automatically update sample rate before calculation.
        Return flag, which indicates that all values converted.
        """

        logger.debug('Starting set amount of data points for each channel...')
        all_upd = True
        for i in range(self.CHANNELS):
            logger.debug(f'Setting points for {self.CH_IDS[i]}')
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
            logger.debug('...Finishing data points calculation.')

        if all_upd:
            return True
        else:
            return False
            
    def rolling_average(
            self,
            data: npt.NDArray[np.int16],
            auto_kernel: bool = False
        ) -> npt.NDArray[np.int16]:
        """Smooth data using rolling average method.
        
        If `auto_kernel` is True, then BL_LENGTH is used for 
        calculation kernel.
        Does not modify any attributes.
        """
        
        # logger.debug('Starting _rolling_average smoothing...')
        if auto_kernel:
            kernel_size = int(len(data)*self.BL_LENGTH)   
        else:
            kernel_size = self.ra_kernel
        min_signal_size = int(self.SMOOTH_LEN_FACTOR*kernel_size)
        if len(data) < min_signal_size:
            err_msg = (f'Signal has only {len(data)} data points, '
                       + f'but at least {min_signal_size} is required.')
            logger.debug(err_msg)
            raise OscIOError(err_msg)
        kernel = np.ones(kernel_size)/kernel_size
        tmp_array = np.zeros(len(data))
        border = int(kernel_size/2)
        tmp_array[border:-(border-1)] = np.convolve(data,kernel,mode='valid')
        #leave edges unfiltered
        tmp_array[:border] = tmp_array[border]
        tmp_array[-(border):] = tmp_array[-border]
        result = tmp_array.astype(np.int16)
        # logger.debug(f'...Finishing smoothing. {result.min()=}, {result.max()=}')
        return result
        
    def _read_chunk(
            self,
            start: int,
            dur: int
        ) -> npt.NDArray[np.int16]:
        """Read a single chunk from memory.
        
        Return data without header.
        """

        logger.debug('Starting _read_chunk...'
                      + f'Start point: {start+1}, '
                      + f'stop point: {dur + start}.')
        msg = []
        msg.append(':WAV:STAR ' + str(start+1))
        msg.append(':WAV:STOP ' + str(dur+start))
        msg.append(':WAV:DATA?')
        self._write(msg)
        data = self._read()
        self._ok_read(dur, data)
        # logger.debug(f'...Finishing. Signal with {len(data)} '
        #                 + f'data points read. {data.min()=},'
        #                 + f'{data.max()=}')
        return data

    def _multi_chunk_read(self,
                          start: int,
                          dur: int) -> npt.NDArray[np.int16]:
        """Read from memory in multiple chunks.
        
        Return data without header.
        """

        logger.debug('Starting _multi_chunk_read... '
                     + f'Start point: {start+1}, '
                     + f'total required points: {dur}.')
        data_frames = int(dur/self.MAX_MEMORY_READ) + 1
        logger.debug(f'{data_frames} reads are required.')
        data = np.empty(dur, dtype=np.int16)
        for i in range(data_frames):
            start_i = i*self.MAX_MEMORY_READ
            stop_i = (i+1)*self.MAX_MEMORY_READ
            if (dur - stop_i) > 0:
                dur_i = self.MAX_MEMORY_READ
            else:
                dur_i = dur - 1 - start_i
            data_chunk = self._read_chunk(start + start_i, dur_i)
            data[start_i:start_i+dur_i] = data_chunk
        # logger.debug(f'...Finishing. Full signal with {len(data)} '
        #              + f'points read. {data.min()=}, {data.max()=}'
        # )
        return data

    def _to_volts(self, data: npt.NDArray[np.int16]) -> PlainQuantity:
        """Converts data to volts."""

        result = data*self.yincrement
        result = Q_(result, 'volt')
        logger.debug(f'...Finishing. Max val={result.max()}, ' #type: ignore
                     + f'min val={result.min()}') #type: ignore
        return result

    def _ok_read(self,
                dur: int,
                data_chunk: npt.NDArray,
                strict: bool=True
        ) -> bool:
        """Verify that read data have necessary size.
        
        If data length is wrong and `strict` is True, raise OscIOError.
        Otherwise just return False.
        """

        if dur == len(data_chunk):
            return True
        elif strict:
            err_msg = ('Data length is wrong, '
                       + f'{dur} is required, '
                       + f'actual length is {len(data_chunk)}')
            logger.debug(err_msg)
            raise OscIOError(err_msg)
        else:
            return False

    def _read_data(self, ch_id: int) -> npt.NDArray[np.int16]:
        """Read data from the specified channel.
        
        Return read data.
        """

        cmd = []
        cmd.append(':WAV:SOUR ' + self.CH_IDS[ch_id])
        cmd.append(':WAV:MODE RAW')
        cmd.append(':WAV:FORM BYTE')
        self._write(cmd)
        pre_points = cast(int, self.pre_p[ch_id])
        dur_points = cast(int, self.dur_p[ch_id])
        data_start = (self.trig - pre_points)
        if data_start < 1:
            data_start = 1
        #if one can read the whole data in 1 read
        if self.MAX_MEMORY_READ > dur_points:
            data = self._read_chunk(data_start, dur_points)
        else:
            data = self._multi_chunk_read(data_start, dur_points)
        return data

    def _baseline_correction(self,
                            data: npt.NDArray[np.int16]
                            ) -> npt.NDArray[np.int16]:
        """Correct baseline for the data.
        
        Assume that baseline is at the start of measured signal.
        """

        bl_points = int(len(data)*self.BL_LENGTH)
        # logger.debug('Starting baseline correction on signal with '
        #              + f'{len(data)} data points... '
        #              + f'Baseline length is {bl_points}.')
        baseline = np.average(data[:bl_points])
        data -= int(baseline)
        # logger.debug('...Finishing baseline correction., '
        #              + f'{data.min()=}, {data.max()=}')
        return data

    def _read_scr_data(self, ch_id: int) -> npt.NDArray[np.int16]:
        """Read screen data for the channel."""

        chan = self.CH_IDS[ch_id]
        self._set_preamble()
        self.scr_scale[ch_id] = Q_(self.yincrement, 'V')
        cmd = []
        cmd.append(':WAV:SOUR ' + chan)
        cmd.append(':WAV:MODE NORM')
        cmd.append(':WAV:FORM BYTE')
        cmd.append(':WAV:STAR 1')
        cmd.append(':WAV:STOP ' + str(self.MAX_SCR_POINTS))
        cmd.append(':WAV:DATA?')
        self._write(cmd)
        data = self._read()
        if not self._ok_read(self.MAX_SCR_POINTS, data, strict=False):
            return np.zeros(self.MAX_SCR_POINTS, np.int16)
        return data

class PowerMeter:
    
    ###DEFAULTS###
    SENS = 2888 #scalar coef to convert integral readings into [mJ]
    BL_LENGTH = Oscilloscope.BL_LENGTH

    ch: int # channel ID number
    osc: Oscilloscope
    
    data: PlainQuantity|None
    laser_amp: PlainQuantity
    start_ind: int = 0
    stop_ind: int = 1

    def __init__(self, 
                 osc: Oscilloscope,
                 ch_id: int=1,
                 threshold: float=0.01) -> None:
        """PowerMeter class for Thorlabs ES111C pyroelectric detector.

        ``osc`` is as instance of ``Oscilloscope`` class, 
        which is used for reading data.\n
        ``ch_id`` is index of channel (starting from 0) 
        to which the detector is connected.
        """

        logger.debug('Instantiating PowerMeter connected to '
                     + f'{osc.CH_IDS[ch_id]}...')
        self.osc = osc
        self.ch = ch_id
        logger.debug('...Finishing')

    def get_energy_scr(self) -> EnergyMeasurement:
        """
        Measure energy from screen (fast).
        
        Do not change any attributes.
        This operation is not thread safe and must be
        called only by Actor.
        """

        # logger.debug('Starting fast energy measuring...')
        meas_channel = self._build_chan_list()
        data = self.osc.measure_scr(
                read_ch1=meas_channel[0],
                read_ch2=meas_channel[1])
        # Get pysical quantity data for PM channel
        pm_data = data.data_raw[self.ch]*data.yincrement[self.ch] # type: ignore
        if pm_data is None:
            msg = 'Data cannot be read from osc.'
            logger.warning(msg)
            raise OscIOError(msg)
        # logger.debug('PowerMeter response obtained')
        result = PowerMeter.energy_from_data(
            pm_data,
            self.osc.xincrement
            )
        if result is None:
            result = EnergyMeasurement(datetime.now())
        return result

    @staticmethod
    def energy_from_data(
            data: PlainQuantity,
            step: PlainQuantity
        ) -> EnergyMeasurement|None:
        """
        Calculate laser energy from data.

        Attributes
        ----------
        `data` must be baseline corrected.\n
        ``Step`` - time step for the data.

        Algorithm
        ---------
        We assume that the signal is baseline corrected.\n
        First, we calculate average value of NEGATIVE values among
        points, used for baseline correction (this value is 0 if no
        negative values was found).\n
        Then we sum all values, which are higher than 2 folds the average.\n
        Note that we cannot simply sum all positive values in data, as
        in such a case noise will have compounding contribution to the
        final value. We also cannot just sum all values, because for
        long enough sampling intervals signal tends to go below base line.\n
        Thread safe.
        """

        # logger.debug('Starting convertion of raw signal to energy...')
        if not data.is_compatible_with(ureg.V):
            logger.warning('PM energy error. Wrong units. Volts expected.')
            return None
        data = data.to('V')
        start_ind = PowerMeter.find_pm_start_ind(data)
        if start_ind is None:
            return None
        # minimum noise value on points, used for base line calculation
        bl_length = int(len(data)*PowerMeter.BL_LENGTH) # type: ignore
        noise_min = data[:bl_length].min() # type: ignore
        if noise_min < 0:
            noise_neg = data[:bl_length][data[:bl_length]<0].mean() # type: ignore
        else:
            noise_neg = 0
        signal_data = data[start_ind:] # type: ignore
        laser_amp = signal_data[signal_data>(2*noise_neg)].sum()*step.to('s').m*PowerMeter.SENS # type: ignore
        laser_amp = Q_(laser_amp.m, 'mJ')
        result = EnergyMeasurement(
            datetime.now(),
            signal = data,
            dt = step.to('us'),
            istart= start_ind,
            energy = laser_amp
        )
        logger.debug(f'Laser energy {str(result.energy)}')
        return result
    
    @staticmethod
    def check_data(data: PlainQuantity) -> bool:
        """Check whether data is valid power meter data."""

        try:
            iter(data)
        except TypeError:
            err_msg = 'data is not iterable.'
            logger.debug(err_msg)
            return False
        if len(data) < 50: # type: ignore
            err_msg = ('data too small for laser energy '
                       + f'calc ({len(data.m)})')
            logger.debug(err_msg)
            return False
        return True

    def set_channel(
            self,
            chan: int,
            pre_time: PlainQuantity,
            post_time: PlainQuantity) -> None:
        """
        Set read channel.
        
        ``chan`` is index, i.e. <chan>=0 for CHAN1.\n
        Thread safe.
        """

        logger.debug(f'PowerMeter channel set to {self.osc.CH_IDS[chan]}')
        self.ch = chan
        self.osc.pre_t[chan] = pre_time
        self.osc.post_t[chan] = post_time
        self.osc.dur_t[chan] = pre_time + post_time #type:ignore

    def _build_chan_list(self) -> List[bool]:
        """
        Build a mask List for channels.
        
        Thread safe.
        """

        len = self.osc.CHANNELS
        result = []
        channel = self.ch
        for i in range(len):
            if channel == 0:
                result.append(True)
            else:
                result.append(False)
            channel -= 1
        return result

    @staticmethod
    def find_pm_start_ind(data: PlainQuantity) -> int|None:
        """
        Find index of power meter signal begining.
        
        Algorithm
        ---------
        We assume that the signal is baseline corrected.\n
        First, we calculate average value of POSITIVE values among
        points, used for baseline correction. We additionally check that
        there are at least 5 positive point (we expand search range if
        necessary).\n
        Then we find a point, which is at least 2 folds larger than
        calculated average AND among next 20 points at least 80% also
        comply to the value requirement.\n
        Thread safe.
        """

        # logger.debug('Starting search for power meter signal start...')
        if not PowerMeter.check_data(data):
            return None
        av_len = int(len(data)*PowerMeter.BL_LENGTH) # type: ignore
        av_span = data[:av_len] # type: ignore
        while np.count_nonzero(av_span[av_span > 0]) < 5:
            av_len += int(len(data)*PowerMeter.BL_LENGTH) # type: ignore
            av_span = data[:av_len] # type: ignore
            if av_len + 21 > len(data): # type: ignore
                logger.debug('Start ind error. Too few pos values.')
                return None
        aver = av_span[av_span >= 0].mean()
        ind = None
        for i in np.where(data>(aver*2))[0]:
            if np.count_nonzero(data[i:i+20][data[i:i+20] > aver]) > 16: # type: ignore
                ind = i
                break
        logger.debug(f'PM signal begins at index {ind}')
        return ind

    def pulse_offset(
            self,
            data: PlainQuantity,
            step: PlainQuantity
        ) -> PlainQuantity|None:
        """
        Calculate time offset of laser in PM data.
        
        Thread safe.
        """

        # logger.debug('Starting lase pulse offset calculation...')
        index = PowerMeter.find_pm_start_ind(data)
        if index is None:
            return None
        offset = (step*index).to('us')
        logger.debug(f'Laser pulse offset is {offset}')
        return offset

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
        """
        Set read channel.
        
        Thread safe.
        """

        self.ch = chan
        self._osc.pre_t[chan] = pre_time
        self._osc.post_t[chan] = post_time
        self._osc.dur_t[chan] = pre_time + post_time #type:ignore
        logger.debug(f'PA sensor channel set to {self._osc.CH_IDS[chan]}')