"""
Module with data classes.
"""

from typing import TypedDict, List

import pint
import numpy.typing as npt
import numpy as np
from pylablib.devices import Thorlabs

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

    data: List[pint.Quantity]
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

    data: List[pint.Quantity]
    x_var_step: pint.Quantity
    x_var_start: pint.Quantity
    x_var_stop: pint.Quantity
    max_amp: pint.Quantity

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware."""

    stage_x: Thorlabs.KinesisMotor
    stage_y: Thorlabs.KinesisMotor
    osc: Oscilloscope
    config: dict

class Data_point(TypedDict):
    """Single PA measurement"""

    dt: pint.Quantity
    pa_signal: List[pint.Quantity]
    pa_signal_raw: npt.NDArray[np.uint8|np.int16]
    pm_signal: List[pint.Quantity]
    start_time: pint.Quantity
    stop_time: pint.Quantity
    pm_energy: pint.Quantity
    sample_energy: pint.Quantity
    max_amp: pint.Quantity
    wavelength: pint.Quantity

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware."""
    
    power_meter: PowerMeter
    pa_sens: PhotoAcousticSensOlymp
    stage_z: Thorlabs.KinesisMotor
