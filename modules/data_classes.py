"""
Module with data classes.
"""

import sys
from typing import TypedDict, List, Callable
from dataclasses import dataclass, field
import traceback
import logging

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

logger = logging.getLogger(__name__)

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
    data_raw: npt.NDArray[np.uint8|np.int16]
    'measured PA signal in raw format'
    a: float
    'coef for coversion ``data_raw`` to ``data``: <data> = a*<data_raw> + b '
    b: float
    'coef for coversion ``data_raw`` to ``data``: <data> = a*<data_raw> + b '
    x_var_step: PlainQuantity
    x_var_start: PlainQuantity
    x_var_stop: PlainQuantity
    x_var_name: str = 'Time'
    y_var_name: str = 'PhotoAcoustic Signal'
    max_amp: PlainQuantity
    'max(data) - min(data)'

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
    param_val: List[PlainQuantity] = field(default_factory=list)
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
    parameter_name: List[str]
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
    data: dict[str, DataPoint] = field(default_factory=dict)

def empty_ndarray():
    return np.empty(0, dtype=np.int8)

@dataclass
class MeasuredPoint:
    """Single PA measurement."""

    pa_signal_raw: npt.NDArray[np.uint8|np.int16] = field(default_factory=empty_ndarray)
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

class Hardware():
    """Class for hardware references."""
    
    def __init__(self):
        self.power_meter: PowerMeter | None = None
        self.pa_sens: PhotoAcousticSensOlymp | None = None
        self.stages: List[KinesisMotor] = []
        self.motor_axes: int = -1
        self.axes_titles = ('X','Y','Z')
        self.osc: Oscilloscope = Oscilloscope()
        self.config: dict = {}

hardware = Hardware()

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
            'is_running': True
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
