"""
Module with data classes.
"""

import sys
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from dataclasses import dataclass, field
import traceback
import logging
import heapq
import threading

from pint.facets.plain.quantity import PlainQuantity
import numpy.typing as npt
import numpy as np
from pylablib.devices.Thorlabs import KinesisMotor
from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
    QRunnable
)

from .osc_devices import Oscilloscope, PowerMeter, PhotoAcousticSensOlymp
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
    """Single PA measurement."""

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

def empty_ndarray():
    return np.empty(0, dtype=np.int8)

def empty_arr_quantity() -> PlainQuantity:
    return Q_(np.empty(0), 'uJ')

@dataclass
class MeasuredPoint:
    """Single PA measurement."""

    pa_signal_raw: npt.NDArray[np.uint8] = field(default_factory=empty_ndarray) # type: ignore
    "Sampled PA signal in int8 format"
    dt: PlainQuantity = Q_(0,'s')
    "Sampling interval for PA data"
    dt_pm: PlainQuantity = Q_(0, 's')
    "Sampling interval for PM data, could differ from dt due to downsampling of PM data."
    pa_signal: PlainQuantity = Q_(np.empty(0), 'V/uJ')
    "Sampled PA signal in physical units"
    pm_signal: PlainQuantity = Q_(np.empty(0), 'V')
    "Sampled power meter signal in volts"
    start_time: PlainQuantity = Q_(0, 'us')
    "Start of sampling interval relative to begining of laser pulse"
    stop_time: PlainQuantity = Q_(0, 'us')
    "Stop of sampling interval relative to begining of laser pulse"
    pm_energy: PlainQuantity = Q_(0, 'uJ')
    "Energy at power meter in physical units"
    sample_energy: PlainQuantity = Q_(0, 'uJ')
    "Energy at sample in physical units"
    max_amp: PlainQuantity = Q_(0, 'V/uJ')
    "Maximum PA signal amplitude"
    wavelength: PlainQuantity = Q_(0, 'nm')
    "Excitation laser wavelength"

@dataclass
class EnergyMeasurement:
    """Energy measurement from power meter."""

    data: PlainQuantity = field(default_factory=empty_arr_quantity)
    "History of energy values."
    signal: PlainQuantity = field(default_factory=empty_arr_quantity)
    "Measured PM signal (full data)."
    sbx: int = 0
    "Index of laser pulse start."
    sex: int = 0
    "Index of laser pulse end."
    energy: PlainQuantity = Q_(0, 'uJ')
    "Last measured laser energy."
    aver: PlainQuantity = Q_(0, 'uJ')
    "Average laser energy."
    std: PlainQuantity = Q_(0, 'uJ')
    "Standard deviation of laser energy."

@dataclass
class Coordinate:
    
    x: PlainQuantity|None = None
    y: PlainQuantity|None = None
    z: PlainQuantity|None = None

@dataclass
class StagesStatus:

    x_open: bool|None = None
    y_open: bool|None = None
    z_open: bool|None = None
    x_status: list[str] = field(default_factory = list)
    y_status: list[str] = field(default_factory = list)
    z_status: list[str] = field(default_factory = list)

class Hardware():
    """Class for hardware references."""
    
    def __init__(self):
        self.power_meter: PowerMeter | None = None
        self.pa_sens: PhotoAcousticSensOlymp | None = None
        self.stages: dict[str, KinesisMotor] = {}
        self.motor_axes: int = -1
        self.axes_titles = ('X','Y','Z')
        self.osc: Oscilloscope = Oscilloscope()
        self.config: dict = {}

hardware = Hardware()

### Threading objects ###

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
            logger.warning(f'An error occured: {exctype}, {value}')
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
        ) -> None:
        with self._cv:
            heapq.heappush(
                self._queue, (-priority, self._count, item)
            )

    def get(self) -> tuple[Callable, tuple, dict, Result]:
        with self._cv:
            while len(self._queue) == 0:
                self._cv.wait()
            return heapq.heappop(self._queue)[-1]
    
class ActorExit(Exception):
    
    pass

class Actor:
    """Object for serial processing of calls."""

    def __init__(self) -> None:
        self._mailbox = PriorityQueue()

    def _send(
            self,
            msg: tuple[Callable, tuple, dict, Result],
            priority: int
        ) -> None:
        """Put a function to priority queue."""
        self._mailbox.put(msg, priority)

    def recv(self) -> tuple[Callable, tuple, dict, Result]:
        msg = self._mailbox.get()
        if msg is ActorExit:
            raise ActorExit()
        return msg
    
    def close(self) -> None:
        """Close request."""
        
        r = Result()
        # Close request is sent with highest priority.
        self._send((ActorExit, tuple(), dict(), r), Priority.HIGHEST)

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
            *args: tuple,
            **kwargs: dict
            ) -> T:
        """
        Submit a function for serail processing.
        
        Priority can have values from 0 (lowest) to 10 (highest).
        """
        r = Result()
        self._send((func, args, kwargs, r), priority)
        return r.result() # type: ignore

    def _run(self):
        """Main execution loop."""

        while True:
            func, args, kwargs, r = self.recv()
            r.set_result(func(*args, **kwargs))