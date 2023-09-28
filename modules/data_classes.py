"""
Module with data classes.
"""

from typing import TypedDict, List

import pint
import numpy.typing as npt
import numpy as np
from pylablib.devices.Thorlabs import KinesisMotor

from .osc_devices import Oscilloscope, PowerMeter, PhotoAcousticSensOlymp


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

class Data_point(TypedDict):
    """Single PA measurement"""

    dt: pint.Quantity
    pa_signal: pint.Quantity
    pa_signal_raw: npt.NDArray[np.uint8|np.int16]
    pm_signal: pint.Quantity
    start_time: pint.Quantity
    stop_time: pint.Quantity
    pm_energy: pint.Quantity
    sample_energy: pint.Quantity
    max_amp: pint.Quantity
    wavelength: pint.Quantity

class Hardware():
    """Class for hardware references."""
    
    def __init__(self):
        self.power_meter: PowerMeter | None = None
        self.pa_sens: PhotoAcousticSensOlymp | None = None
        self.stages: List[KinesisMotor]
        self.motor_axes: int
        self.osc: Oscilloscope
        self.config: dict

hardware = Hardware()