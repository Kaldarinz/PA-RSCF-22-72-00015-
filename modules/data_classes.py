"""
Module with data classes.
"""

from typing import TypedDict, List
from dataclasses import dataclass, field

import pint
from pint.facets.plain.quantity import PlainQuantity
import numpy.typing as npt
import numpy as np
from pylablib.devices.Thorlabs import KinesisMotor

from .osc_devices import Oscilloscope, PowerMeter, PhotoAcousticSensOlymp
from . import ureg, Q_


class BaseMetadata(TypedDict):
    """Typed dict for general metadata."""

    version: float
    measurement_dims: int
    parameter_name: List[str]
    data_points: int
    created: str
    updated: str
    filename: str
    zoom_pre_time: pint.Quantity
    zoom_post_time: pint.Quantity

class RawMetadata(TypedDict):
    """Typed dict for raw_data metadata."""

    max_len: int
    x_var_name: str
    y_var_name: str

class FiltMetadata(TypedDict):
    """Typed dict for filt_data metadata."""

    x_var_name: str
    y_var_name: str

class RawData(TypedDict):
    """Typed dict for a data point."""

    data: pint.Quantity
    data_raw: npt.NDArray[np.uint8|np.int16]
    param_val: List[pint.Quantity]
    x_var_step: pint.Quantity
    x_var_start: pint.Quantity
    x_var_stop: pint.Quantity
    pm_en: pint.Quantity
    sample_en: pint.Quantity
    max_amp: pint.Quantity

class FreqData(TypedDict):
    """Typed dict for a frequency data."""

    data: pint.Quantity
    x_var_step: pint.Quantity
    x_var_start: pint.Quantity
    x_var_stop: pint.Quantity
    max_amp: pint.Quantity

def empty_ndarray():

    return np.empty(0, dtype=np.int8)

@dataclass
class DataPoint:
    """Single PA measurement."""

    pa_signal_raw: npt.NDArray[np.uint8|np.int16] = field(default_factory=empty_ndarray)
    "Sampled PA signal in int8 format"
    dt: PlainQuantity = Q_(0,'s')
    "Sampling interval"
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

tst = DataPoint()
tst.dt