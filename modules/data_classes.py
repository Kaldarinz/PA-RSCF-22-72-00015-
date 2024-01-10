"""
Module with data classes.
"""

import sys
from typing import Callable, TypeVar, Any, Literal
from typing_extensions import ParamSpec, Self, Type
from dataclasses import dataclass, field, fields
import traceback
import logging
import heapq
import threading
from datetime import datetime as dt

from pint.facets.plain.quantity import PlainQuantity
from pint.errors import UndefinedUnitError as UnitError
import numpy.typing as npt
import numpy as np
from pylablib.devices.Thorlabs import KinesisMotor
from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
    QRunnable
)

from . import Q_
from .constants import (
    Priority
)

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

@dataclass
class FileMetadata:
    """General attributes of a file."""

    version: float = 0.0
    'version of data structure'
    measurements_count: int = 0
    'amount of measurements in the file'
    created: str = ''
    'date and time of file creation'
    updated: str = ''
    'date and time of last file update'
    notes: str = ''
    'description of the file'
    zoom_pre_time: PlainQuantity = Q_(2, 'us')
    'start time from the center of the PA data frame for zoom in data view'
    zoom_post_time: PlainQuantity = Q_(13, 'us')
    'end time from the center of the PA data frame for zoom in data view'

@dataclass
class BaseData:
    """Single raw PA data."""

    data: PlainQuantity
    'measured PA signal'
    data_raw: npt.NDArray[np.uint8]
    'measured PA signal in raw format'
    a: float
    'coef for coversion ``data_raw`` to ``data``: <data> = a*<data_raw> + b '
    b: float
    'coef for coversion ``data_raw`` to ``data``: <data> = a*<data_raw> + b '
    max_amp: PlainQuantity
    'max(data) - min(data)'
    x_var_step: PlainQuantity
    x_var_start: PlainQuantity
    x_var_stop: PlainQuantity
    x_var_name: str = 'Time'
    y_var_name: str = 'PhotoAcoustic Signal'
    
@dataclass
class ProcessedData:
    """Single processed PA data."""

    data: PlainQuantity
    'processed PA signal'
    x_var_step: PlainQuantity
    x_var_start: PlainQuantity
    x_var_stop: PlainQuantity
    x_var_name: str
    y_var_name: str
    max_amp: PlainQuantity
    'max(data) - min(data)'

@dataclass
class PointMetadata:
    """General attributes of a single PA measurement."""

    pm_en: PlainQuantity
    'laser energy measured by power meter in glass reflection'
    sample_en: PlainQuantity
    'laser energy at sample'
    param_val: list[PlainQuantity] = field(default_factory=list)
    'value of independent parameters'

@dataclass
class DataPoint:
    """Single PA measurement for storage."""

    attrs: PointMetadata
    raw_data: BaseData
    filt_data: ProcessedData
    freq_data: ProcessedData

@dataclass
class MeasurementMetadata:
    """General attributes of a measurement."""

    measurement_dims: int
    'dimensionality of the measurement'
    parameter_name: list[str] = field(default_factory = list)
    'independent parameter, changed between measured PA signals'
    data_points: int = 0
    'amount of datapoints in the measurement'
    created: str = ''
    'date and time of file creation'
    updated: str = ''
    'date and time of last file update'
    notes: str = ''
    'description of the measurement'
    max_len: int = 0
    'maximum amount of samples in a single datapoint in this measurement'

@dataclass
class Measurement:
    """A Photoacoustic measurement with metadata."""

    attrs: MeasurementMetadata
    "MetaData of the measurement."
    data: dict[str, DataPoint] = field(default_factory = dict)

@dataclass
class Coordinate:
    
    x: PlainQuantity|None = None
    y: PlainQuantity|None = None
    z: PlainQuantity|None = None

    @classmethod
    def from_tuples(
        cls: Type[Self],
        coords: list[tuple[str, PlainQuantity]],
        default: PlainQuantity|None = None
        ) -> Self:
        """
        Alternative constructor from list of tuples with (Axis, value).
        
        ``default`` is default value for mising axes.
        """

        pos = cls()
        for coord in coords:
            if coord[0] in dir(pos):
                setattr(pos, coord[0], coord[1])
            else:
                raise TypeError(
                    f'Trying to set non-existing axis = {coord[0]}'
                )
        # Set default values for missing attributes
        set_flds = set(coord[0] for coord in coords)
        all_flds = set(fld.name for fld in fields(pos))
        missing = all_flds - set_flds
        for fld in missing:
            setattr(pos, fld, default)
        return pos
    
    def add(self, added: Self, strict: bool = True) -> None:
        """
        Add another coordinate to current coordinate.
        
        If ``strict`` is ``True``, then  if axis value of any operand
        is None, the result is None. Otherwise if axis value of
        any operand is not None, the result is existing value.
        """

        if not isinstance(added, type(self)):
            return
        
        # Iterate over all fields
        for fld in fields(self):
            attr1 = getattr(self, fld.name)
            attr2 = getattr(added, fld.name)
            # If both fields are None, then skip it
            if attr1 is None and attr2 is None:
                continue
            # If both values exist, sum them
            elif None not in [attr1, attr2]:
                setattr(self, fld.name, attr1 + attr2)
            else:
                # if only one value axist and adding is not strict, 
                # set existing value
                if not strict:
                    if attr1 is None:
                        setattr(self, fld.name, attr2)

    def dist(self, point: Self) -> PlainQuantity:
        """
        Calculate distance to ``point``.
        
        Only both non None axes are calculated.\n
        If there no such pairs, magnitude of returned value is np.nan.
        """

        dist = Q_(np.nan, 'm**2')
        for axis in ['x', 'y', 'z']:
            ax1 = getattr(self, axis)
            ax2 = getattr(point, axis)
            if None not in [ax1, ax2]:
                dist = np.nan_to_num(dist) + (ax2-ax1)**2

        return np.sqrt(dist)

    def direction(self, point: Self) -> Self|None:
        """Get unit vector in the direction of ``point``."""

        dist = self.dist(point)
        # Return None if distance cannot be calculated
        if dist.m is np.nan:
            return None
        return (point - self)/dist
        
    def __add__(self, added: Self) -> Self:
        """If value of any axis is None, result for the axis is None."""
        if not isinstance(added, type(self)):
            return NotImplemented
        
        new_coord = type(self)()
        for fld in fields(self):
            attr1 = getattr(self, fld.name)
            attr2 = getattr(added, fld.name)
            if None not in [attr1, attr2]:
                setattr(new_coord, fld.name, attr1 + attr2)

        return new_coord
    
    def __sub__(self, subed: Self) -> Self:
        if not isinstance(subed, type(self)):
            return NotImplemented
        return self + (-subed)
    
    def __truediv__(self, val: PlainQuantity|float) -> Self:
        result = Coordinate()
        for axis in ['x', 'y', 'z']:
            ax1 = getattr(self, axis)
            if ax1 is not None:
                setattr(result, axis, ax1/val)
        return result

    def __mul__(self, val: PlainQuantity|float) -> Self:
        result = Coordinate()
        for axis in ['x', 'y', 'z']:
            ax1 = getattr(self, axis)
            if ax1 is not None:
                setattr(result, axis, ax1*val)
        return result

    def __neg__(self) -> Self:
        new_coord = type(self)()
        for fld in fields(self):
            attr = getattr(self, fld.name)
            if attr is not None:
                setattr(new_coord, fld.name, -attr)
        return new_coord

def def_pos() -> Coordinate:
    return Coordinate()

def empty_ndarray():
    return np.empty(0, dtype=np.int8)

def empty_arr_quantity() -> PlainQuantity:
    return Q_(np.empty(0), 'uJ')

def empty_arr_quantity_s() -> PlainQuantity:
    return Q_(np.empty(0), 's')

@dataclass
class OscMeasurement:
    """
    Oscilloscope measurement.
    
    Could contain data for all channels.
    """

    datetime: dt = field(compare=False)
    data_raw: list[npt.NDArray|None] = field(
        default_factory=list
    )
    "List with raw data from osc channels."
    dt: PlainQuantity = Q_(np.nan, 'us')
    "Time step of ``data_raw``"
    pre_t: list[PlainQuantity] = field(default_factory=list)
    "List with time intervals from start of sampling to trigger."
    yincrement: PlainQuantity = Q_(np.nan, 'V')
    "Data = yincrement*data_raw."

@dataclass
class EnergyMeasurement:
    """Energy measurement from power meter."""

    datetime: dt = field(compare=False)
    signal: PlainQuantity = field(
        default_factory=empty_arr_quantity,
        compare = False
    )
    "Measured PM signal (full data)."
    dt: PlainQuantity = Q_(np.nan, 'us')
    istart: int = -1
    "Index of laser pulse start."
    istop: int = -1
    "Index of laser pulse end."
    energy: PlainQuantity = Q_(np.nan, 'uJ')
    "Last measured laser energy."

@dataclass(init=False)
class PaEnergyMeasurement(EnergyMeasurement):
    """Energy information for PA measurement."""

    def __init__(self, en_info: EnergyMeasurement, sample_en: PlainQuantity):
        super().__init__(
            datetime=en_info.datetime,
            signal=en_info.signal,
            dt=en_info.dt,
            istart=en_info.istart,
            istop=en_info.istop,
            energy=en_info.energy
        )

        self.sample_en = sample_en
        "Energy at sample."

@dataclass
class MeasuredPoint:
    """Single PA measurement."""

    dt_pm: PlainQuantity
    "Sampling interval for PM data, could differ from dt due to downsampling of PM data."
    pa_signal: PlainQuantity = Q_(np.empty(0), 'V/mJ')
    "Sampled PA signal in physical units"
    pm_signal: PlainQuantity = Q_(np.empty(0), 'V')
    "Sampled power meter signal in volts"
    max_amp: PlainQuantity = Q_(np.nan, 'V/uJ')
    "Maximum PA signal amplitude"
    start_time: PlainQuantity = Q_(np.nan, 'us')
    "Start of PA signal sampling interval relative to begining of laser pulse"
    stop_time: PlainQuantity = Q_(np.nan, 'us')
    "Stop of PA signal sampling interval relative to begining of laser pulse"

    def __init__(
            self,
            data: OscMeasurement,
            energy_info: PaEnergyMeasurement,
            wavelength: PlainQuantity,
            pa_ch_ind: int = 0,
            pm_ch_ind: int = 1,
            pos: Coordinate = Coordinate()
        ) -> None:

        # Direct attribute initiation
        self.datetime = data.datetime
        "Date and time of measurement"
        self.pa_signal_raw = data.data_raw[pa_ch_ind]
        "Sampled PA signal in int8 format"
        self.pm_signal_raw = data.data_raw[pm_ch_ind]
        "Sampled PA signal in int8 format"
        self.dt = data.dt
        "Sampling interval for PA data"
        self.wavelength = wavelength
        "Excitation laser wavelength"
        self.pos = pos
        "Coordinate of the measured point"
        self.yincrement: PlainQuantity = Q_(data.yincrement, 'V')
        "Scaling factor to convert raw data to volts"
        self.pm_info = energy_info
        "General information laser energy."
        self.pm_energy = energy_info.energy
        "Energy at power meter in physical units"
        self.sample_en = energy_info.sample_en
        "Energy at sample in physical units"
        self._pm_start = data.pre_t[pm_ch_ind]
        self._pa_start = data.pre_t[pa_ch_ind]

        # Init pm data
        self._set_pm_data()

        # Calculate energy 
        self._set_energy()

        # Set boundary conditions for PA signal relative to start of laser pulse
        self._set_pa_offset()

    def _set_pm_data(self) -> None:
        """Set ``dt_pm`` and ``pm_signal`` attributes."""

        # Downsample power meter data if it is too long
        if len(self.pm_signal_raw) > len(self.pa_signal_raw):
            pm_signal_raw, pm_decim_factor = self.decimate_data(
                self.pm_signal_raw,
                int(len(self.pm_signal_raw)/len(self.pa_signal_raw) + 1)
            )
        else:
            pm_signal_raw = self.pm_signal_raw
            pm_decim_factor = 1
        # Calculate dt for downsampled data
        self.dt_pm = self.dt*pm_decim_factor
        # Convert downsampled signal to volts
        self.pm_signal = pm_signal_raw * self.yincrement

    def _set_energy(self) -> None:
        """Set energy attributes.
        
        Actually set:``max_amp`` and ``pa_signal``.
        """

        # PA signal in volts
        pa_signal_v = self.pa_signal_raw*self.yincrement
        # Set ``pa_signal``
        self.pa_signal = pa_signal_v/self.sample_en
        # Set ``max_amp``
        self.max_amp = pa_signal_v.ptp()/self.sample_en

    def _set_pa_offset(self) -> None:

        # Calculate time from start of pm_signal to trigger position
        pm_offset = self.pm_info.dt*self.pm_info.istart
        if pm_offset is not None:
            # Start time of PA data relative to start of laser pulse
            self.start_time = (self._pm_start - pm_offset) - self._pa_start
            # Stop time of PA data relative to start of laser pulse
            self.stop_time = self.dt*(len(self.pa_signal_raw)-1) + self.start_time

    @staticmethod
    def decimate_data(
            data: np.ndarray[np.int8],
            target: int = 10_000,
        ) -> tuple[np.ndarray, int]:
        """Downsample data to <target> size.
        
        Does not guarantee size of output array.
        Return decimated data and decimation factor.
        """

        logger.debug(f'Starting decimation data with {len(data)} size.')
        factor = int(len(data)/target)
        if factor == 0:
            logger.debug('...Terminatin. Decimation is not required.')
            return (data, 1)
        logger.debug(f'Decimation factor = {factor}')
        iterations = int(math.log10(factor))
        rem = factor//10**iterations
        decim_factor = 1
        for _ in range(iterations):
            data = decimate(data, 10) # type: ignore
            decim_factor *=10
        if rem > 1:
            data = decimate(data, rem) # type: ignore
            decim_factor *=rem
        logger.debug(f'...Finishing. Final size is {len(data)}')
        return data, decim_factor

@dataclass(init=False)
class ScanLine:
    """Single scan line."""

    ltype: str
    "Type of scan line"
    startp: Coordinate
    "Exact position of scan start."
    stopp: Coordinate
    "Exact position of scan stop."
    points: int
    "Amount of regular points."
    step: Coordinate
    "Vector step."
    raw_points: int
    "Amount of all measured points."
    raw_sig: list[MeasuredPoint]
    "List with raw measurements."
    raw_pos: list[tuple[dt, Coordinate]]
    "List with positions, measured along scan line with timestamps."
    data: list[MeasuredPoint]
    "List with data points along exact scan line geometry."
    raw_data: list[MeasuredPoint]
    "List with data points at their measured positions."

    def __init__(
            self,
            startp: Coordinate,
            stopp: Coordinate,
            points: int,
            raw_sig: list[OscMeasurement] = [],
            raw_pos: list[tuple[dt, Coordinate]] = [],
            ltype: Literal['straight line'] = 'straight line'   
        ) -> None:
        """Default scan line constructor.
        
        ``startp`` - Exact position of scan start.\n
        ``stopp`` - Exact position of scan stop.\n
        ``raw_sig`` - optional list with measured osc signals.\n
        ``raw_pos`` - optional list with measured scan positions.
        """
        
        self.startp = startp
        self.stopp = stopp
        self.points = points
        self.raw_sig = raw_sig
        self.raw_pos = raw_pos
        self.ltype = ltype
        self.raw_data = []
        self.data = []

        # Set step
        dist = self.stopp.dist(self.startp)
        # Unit vector in direction from start point to stop point
        unit = self.startp.direction(self.stopp)
        self.step = unit*dist/(self.points - 1)

    def add_signal_points(self, osc_data: list[MeasuredPoint]) -> None:
        """Append OscMeasurements to ``raw_sig`` list."""

        self.raw_sig += osc_data

    def add_pos_point(self, pos: Coordinate|None) -> None:
        """Append position along scan line."""

        if pos is not None:
            self.raw_pos.append((dt.now(),pos))

    def set_raw_data(self) -> None:
        """Calculate ``raw_data`` attribute based on ``raw_sig`` and ``raw_pos``."""

        self.raw_data = self.calc_scan_coord(
            signals = self.raw_sig,
            poses = self.raw_pos,
            start = self.startp,
            stop = self.stopp
        )

    def calc_grid(self) -> None:
        """
        Calculate regulary arranged data.
        
        Automatically set raw data, if was not set.
        """

        logger.warning('calc_grid not implemented!')
        return

    @staticmethod
    def calc_scan_coord(
            signals: list[MeasuredPoint],
            poses: list[tuple[dt,Coordinate]],
            start: Coordinate|None = None,
            stop: Coordinate|None = None,
            trim_edges: bool = True
        ) -> list[MeasuredPoint]:
        """
        Calculate positions of OscMeasurements.
        
        Assume that during scanning signals and coordinates were measured
        with arbitrary time intervals (no periodicity, no parcticular
        dependence between measurements of signal and positions).
        For each OscMeasurement the closest positions, measured before 
        and after the OscMeasurement are taken and position of 
        OscMeasurement is calculated by linear interpolation between those points.
        ``signals`` - OscMeasurements which positions should be determined.\n
        ``poses`` - list of tuples of positions and timestamps, when
        they were measured.\n
        Both lists are sorted by timestamp before calculations.\n
        Signals can be measured before position measurements were started,
        and their measurements can lasts longer than position. To handle 
        this case following optional params can be used. Their values 
        will be used as start and stop position accordingly.\n
        ``start`` - start coordinate of scan line.\n
        ``stop`` - stop coordinate of scan line.\n
        ``trim_edges`` - if True, leave only 1 measurement with 
         positions equal to ``start`` and ``stop``.
        """

        logger.debug(
            f'Starting calculation signal position for {len(signals)} '
            + f'MeasuredPoints based on {len(poses)} coordinate measurements.'
        )
        # Sort data
        result = sorted(signals, key = lambda x: x.datetime)
        poses.sort(key = lambda x: x[0])
        # Reference point for time delta is time of the first measured position
        t0 = poses[0][0]
        # Array with time points relative to ref of signal measurements
        en_time = np.array([(x.datetime - t0).total_seconds() for x in result])
        # Array with time points relative to ref of position
        pos_time = np.array([(x[0] - t0).total_seconds() for x in poses])

        for fld in fields(Coordinate):
            # Assume that all positions have the same non None fields
            # and iterate only on those fields.
            if getattr(poses[1], fld.name) is not None:
                # Array with measured axes
                coord = np.array([getattr(x, fld.name).to('m').m for x in poses[1]])
                # Array with calculated axes
                en_coord = np.interp(
                    x = en_time,
                    xp = pos_time,
                    fp = coord,
                    left = getattr(start, fld.name).to('m').m,
                    right = getattr(stop, fld.name).to('m').m
                )
                # Set axes values to result
                for i, res in enumerate(result):
                    setattr(res.pos, fld.name, Q_(en_coord[i], 'm'))

        # Remove duplicated edge values
        dups = 0
        if trim_edges:
            istart = 0
            while result[istart + 1].pos == start:
                istart += 1
                dups += 1
            istop = len(result) - 1
            while result[istop - 1].pos == stop:
                istop -= 1
                dups += 1
        logger.debug(f'{dups} duplicate edge values trimmed.')
        return result[istart:(istop + 1)]

    @property
    def raw_points(self) -> int:
        return len(self.raw_data)


class MapData:

    def __init__(
            self,
            center: Coordinate,
            width: PlainQuantity,
            height: PlainQuantity,
            hpoints: int,
            vpoints: int,
            scan_plane: Literal['XY', 'YZ', 'ZX'],
            scan_dir: str = 'HLB'
        ) -> None:

        self.startp = center
        "Center point of scan."
        self.width = width
        "Horizontal size of scan."
        self.height = height
        "Vertical size of scan."
        self.data: list[ScanLine] = []
        "All scanned lines."
        self.hpoints = hpoints
        "Amount of points along horizontal axis."
        self.vpoints = vpoints
        "Amount of points along vertical axis."
        self._hstep: PlainQuantity = Q_(np.nan, 'mm')
        self._vstep: PlainQuantity = Q_(np.nan, 'mm')
        # Update step size
        self._upd_steps()
        self._scan_dir = ''
        self.scan_dir = scan_dir

        self._scan_plane = ''
        self.scan_plane = scan_plane
        self._haxis: str = 'x'
        self._vaxis: str = 'y'
        
        
        
        xunits: str = ''
        yunits: str = ''
        units: str = ''

    def _upd_steps(self) -> None:
        """
        Recalculate steps along axes.
        
        Method is internally called, when scan size or points are changed
        """

    @property
    def haxis(self) -> str:
        """Actual title of horizontal axis."""
        return self._haxis
    @haxis.setter
    def haxis(self, val: str = '') -> None:
        self._haxis = val.lower()

    @property
    def haxis(self) -> str:
        """Actual title of vertical axis."""
        return self._haxis
    @haxis.setter
    def haxis(self, val: str) -> None:
        self._haxis = val.lower()

    @property
    def scan_dir(self) -> str:
        """
        Determine direction and starting point of scan.
        """
        return self._scan_dir
    @scan_dir.setter
    def scan_dir(self, val: str) -> None:
        self._scan_dir = val
        # Update some stuff
        pass

    @property
    def scan_plane(self) -> str:
        """
        Pair of axis along which scan is done.
        
        First letter is horizontal axis, second is vertical.
        """
        return self._scan_plane
    @scan_plane.setter
    def scan_plane(self, val: str) -> None:
        self._scan_plane = val
        # Update x and y axis
        self.haxis = val[0]
        self.vaxis = val[1]

@dataclass
class StagesStatus:

    x_open: bool|None = None
    y_open: bool|None = None
    z_open: bool|None = None
    x_status: list[str] = field(default_factory = list)
    y_status: list[str] = field(default_factory = list)
    z_status: list[str] = field(default_factory = list)

    def has_status(self, status: str) -> bool:
        """Return True if any stage is in requested status."""

        statuses = set(self.x_status + self.y_status + self.z_status)
        return True if status.lower() in statuses else False

### Threading objects ###

class Signals:
    """Object for communication between threads."""

    def __init__(self, is_running:bool = True) -> None:
        
        self.is_running = is_running
        self.count = 0
        self.progress = threading.Event()

class WorkerSignals(QObject):
    """
    Signals available from a running worker thred.
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progess = Signal(object)

class Worker(QRunnable):

    def __init__(self, func: Callable, *args, **kwargs) -> None:
        """
        Generic working thread for backend functions.

        ``func`` - any callable. It should at least accept following 
        argurments: ``signals``: WorkerSignals, which contains callback
        functions and ``flags``: Dict, which contains flags for 
        communication with the running callable.\n
        ``*args`` and ``**kwargs`` are additional arguments,
        which will passed to the callable. 
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        #allow signals emit from func
        self.kwargs['signals'] = self.signals

        #for sending data to running func
        self.kwargs['flags'] = {
            'is_running': True,
            'pause': False
        }

    @Slot()
    def run(self) -> None:
        """Actually run a func."""

        try:
            self.result = self.func(*self.args, **self.kwargs)
        except:
            exctype, value = sys.exc_info()[:2]
            logger.warning(
                'An error occured while trying to launch '
                + f'{self.func.__name__}: {exctype}, {value}')
            self.signals.error.emit(
                (exctype, value, traceback.format_exc())
            )
        else:
            self.signals.result.emit(self.result)
        finally:
            self.signals.finished.emit()

class Result:
    """Result of actor call."""

    def __init__(self) -> None:
        
        self._event = threading.Event()
        self._result = None

    def set_result(self, value: object) -> None:
        
        self._result = value
        self._event.set()

    def result(self) -> object:
        
        self._event.wait()
        return self._result

class PriorityQueue:

    def __init__(self) -> None:
        self._queue = []
        self._count = 0
        "Item id."
        self._cv = threading.Condition()

    def put(
            self,
            item: tuple[Callable, tuple, dict, Result],
            priority: int
        ) -> bool:
        """
        Put an item into the queue.
        
        Return operation status.
        """

        if priority < 3 and len(self._queue) > 10:
            logger.debug('Too many calls, skipping low priority task.')
            return False
        with self._cv:
            heapq.heappush(
                self._queue, (-priority, self._count, item)
            )
            self._count += 1
            self._cv.notify()
            return True

    @property
    def nitems(self) -> int:
        return len(self._queue)

    def get(self) -> tuple[Callable, tuple, dict, Result]:
        with self._cv:
            while len(self._queue) == 0:
                self._cv.wait()
            return heapq.heappop(self._queue)[-1]
    
class ActorExit(Exception):
    
    pass

class ActorFail:
    """Sentinel value for failed actor call."""

    def __init__(self, reason:str = '') -> None:
        self.reason = reason

class Actor:
    """Object for serial processing of calls."""

    def __init__(self) -> None:
        self._mailbox = PriorityQueue()
        self.enabled = False

    @property 
    def nitems(self) -> int:
        return self._mailbox.nitems

    def _send(
            self,
            msg: tuple[Callable, Any, Any, Result],
            priority: int
        ) -> None:
        """Put a function to priority queue."""

        if not self.enabled:
            msg[3].set_result(ActorFail('Actor disabled'))
        elif not self._mailbox.put(msg, priority):
            msg[3].set_result(ActorFail('Queue is full'))

    def recv(self) -> tuple[Callable, Any, Any, Result]:
        msg = self._mailbox.get()
        if msg[0] is ActorExit:
            logger.debug(f'{self.nitems} will be cancelled.')
            self.flush()
            logger.debug('All calls cancelled. Finishing actor.')
            raise ActorExit()
        return msg
    
    def flush(self) -> None:
        """Cancel all calls from the queue."""

        while self.nitems:
            msg = self._mailbox.get()
            msg[3].set_result(ActorFail('Actor terminating'))

    def close(
            self,
            close_func: Callable|None = None
        ) -> None:
        """
        Close request.
        
        ``close_func`` - optional routine to close communication,
        which will be performed just before closing the communication.
        """
        if close_func is not None:
            self.submit(
                Priority.HIGHEST,
                close_func
            )
        r = Result()
        # Close request is sent with highest priority.
        self._send((ActorExit, tuple(), dict(), r), Priority.HIGHEST)
        # Stop accepting new submits
        self.enabled = False

    def reset(self) -> None:
        """Reset call stack."""

        self._mailbox = PriorityQueue()

    def start(self):
        self._terminated = threading.Event()
        t = threading.Thread(target = self._bootstrap)
        t.daemon = True
        t.start()
        
    def _bootstrap(self):
        try:
            self._run()
        except ActorExit:
            pass
        finally:
            self._terminated.set()

    def join(self):
        self._terminated.wait()

    def submit(
            self,
            priority: int,
            func: Callable[P, T],
            *args: Any,
            **kwargs: Any
            ) -> T|ActorFail:
        """
        Submit a function for serail processing.
        
        Priority can have values from 0 (lowest) to 10 (highest).\n
        """

        r = Result()
        logger.debug(f'Function {func.__name__} submitted to actor')
        self._send((func, args, kwargs, r), priority)
        res = r.result()
        if isinstance(res, ActorFail):
            if not res.reason == 'Queue is full':
                logger.warning(res.reason)
        return res # type: ignore

    def _run(self):
        """Main execution loop."""

        self.enabled = True
        while True:
            func, args, kwargs, r = self.recv()
            logger.debug(f'Actor Starting {func.__name__}')
            try:
                r.set_result(func(*args, **kwargs))
            except:
                exctype, value = sys.exc_info()[:2]
                msg = f'Error in call: {exctype}, {value}'
                r.set_result(ActorFail(msg))
            logger.debug(f'{func.__name__} ready')