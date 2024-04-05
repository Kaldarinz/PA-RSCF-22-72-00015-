"""
Module with data classes.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

import sys
from typing import (
    Callable,
    TypeVar,
    Any,
    Literal,
    cast,
    TypedDict,
    NamedTuple,
    Iterable,
    Sequence
)
from typing_extensions import ParamSpec, Self, Type
from dataclasses import dataclass, field, fields, asdict
from pprint import pprint
import traceback
import logging
import heapq
import threading
import math
from datetime import datetime as dt
import copy
import time

import pint
from pint.facets.plain.quantity import PlainQuantity
from pint.errors import UndefinedUnitError as UnitError
import numpy.typing as npt
import numpy as np
from scipy.signal import decimate
from scipy.interpolate import griddata
from pylablib.devices.Thorlabs import KinesisMotor
from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
    QRunnable,
    QThreadPool
)

from . import Q_
from .constants import (
    Priority
)

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

Plottable = npt.NDArray | PlainQuantity | Sequence | float | int

class PlotData(NamedTuple):
    ydata: Iterable
    yerr: Iterable | None = None
    xdata: Iterable | None = None
    xerr: Iterable | None = None
    ylabel: str | None = None
    xlabel: str | None = None

class QuantRect(NamedTuple):
    """Rectangle with quantity values of its bottom left corner, width and height."""
    left: PlainQuantity
    bottom: PlainQuantity
    width: PlainQuantity
    height: PlainQuantity

class PointIndex(NamedTuple):
    line_ind: int
    point_ind: int

class Direction:
    """
    Abstract unitless direction from one point to another.
    
    Has unit numerical 'length'.
    """

    _FIELDS = ('x', 'y', 'z')
    x: float | None = None
    y: float | None = None
    z: float | None = None

    def __init__(
            self,
            a: 'Position',
            b: 'Position | None'=None
        ) -> None:
        """ Default class constructor.
        
        When initialized with single argument, results point along this
        position. If two arguments are supplied, the result points from
        the first position to the second.
        """

        # Single attribute case
        if b is None:
            length = a.value()
            vect = a
        # Two attributes case
        else:
            length = a.dist(b)
            # Return None if distance cannot be calculated
            if length.m is np.nan:
                return None
            vect = b - a

        for fld in self._FIELDS:
            if (coord:=getattr(vect, fld)) is not None:
                dir_val = (coord/length).to_base_units().m
                setattr(self, fld, dir_val)

    def __mul__(self, val: PlainQuantity) -> 'Position':
        """Produce a vector with `val` length."""

        res = Position()
        for fld in self._FIELDS:
            if (coord:=getattr(self, fld)) is not None:
                coord_val = coord*val
                setattr(res, fld, coord_val)
        return res
    
class Position:
    """Cartesian coordinate representation of a vector."""
    _FIELDS = ('x', 'y', 'z')

    def __init__(
            self,
            x: PlainQuantity | None=None,
            y: PlainQuantity | None=None,
            z: PlainQuantity | None=None,
        ) -> None:
        
        self.x = x
        self.y = y 
        self.z = z

    @classmethod
    def from_tuples(
        cls: Type[Self],
        coords: list[tuple[str, PlainQuantity]],
        default: PlainQuantity | None=None
        ) -> Self:
        """
        Alternative constructor from list of tuples with (Axis, value).
        
        ``default`` is default value for mising axes.
        """

        pos = cls()
        for coord in coords:
            if coord[0].lower() in cls._FIELDS:
                setattr(pos, coord[0].lower(), coord[1])
            else:
                raise TypeError(
                    f'Trying to set non-existing axis = {coord[0]}'
                )
        # Set default values for missing attributes
        set_flds = set(coord[0].lower() for coord in coords)
        missing = set(cls._FIELDS) - set_flds # type: ignore
        for fld in missing:
            setattr(pos, fld, default)
        return pos
    
    @classmethod
    def from_dict(
        cls: Type[Self],
        data: dict,
        default: PlainQuantity | None=None,
        prefix: str | None=None
        ) -> Self:
        """
        Alternative constructor from dict of base types.
        
        Each coordinate is defined by 2 values: one for magnitude,
        and another for units. Example {'x': 1, 'x_u': 'mm'} define
        position with ``x = 1 mm``.\n
        ``default`` - optional default value for mising axes.\n
        `prefix` - optional prefix name to be added before each
        coordinate.\n
        Example: `prefix = 'pos'` -> {'pos_x': 1, 'pos_x_u': 'mm'}
        define position with ``x = 1 mm``.
        """

        pos = cls()
        set_flds = set()
        for coord in cls._FIELDS:
            units = coord + '_u'
            try:
                if prefix is not None:
                    val = Q_(
                        data[prefix + '_' + coord],
                        data[prefix + '_' + units])
                else:    
                    val = Q_(data[coord], data[units])
            except:
                pass
            else:
                setattr(pos, coord, val)
                set_flds.add(coord)
        # Set default values for missing attributes
        missing = set(cls._FIELDS) - set_flds
        for fld in missing:
            setattr(pos, fld, default)
        return pos

    def to(self, val: str) -> Self:
        """Return copy with all coordinates converted to new units."""

        new = type(self)()
        for axis in self._FIELDS:
            if (ax:=getattr(self, axis)) is not None:
                setattr(new, axis, ax.to(val))
        return new
    
    def value(self) -> PlainQuantity:
        """Sqrt of squares of non-None coordinates."""
        
        val = Q_(np.nan, 'm**2')
        # Flag for a valid value
        valid = False
        for axis in self._FIELDS:
            if (ax:=getattr(self, axis)) is not None:
                val = np.nan_to_num(val) + (ax)**2 # type: ignore
                valid = True
        if valid:
            return np.sqrt(val) # type: ignore
        else:
            raise ValueError

    def add(self, added: Self, strict: bool=True) -> None:
        """
        Add another coordinate to current coordinate.
        
        If ``strict`` is ``True``, then  if axis value of any operand
        is None, the result is None. Otherwise if axis value of
        any operand is not None, the result is existing value.
        """

        if not isinstance(added, type(self)):
            return
        
        # Iterate over all coordinates
        for fld in self._FIELDS:
            attr1 = getattr(self, fld)
            attr2 = getattr(added, fld)
            # If both fields are None, then skip it
            if attr1 is None and attr2 is None:
                continue
            # If both values exist, sum them
            elif None not in [attr1, attr2]:
                setattr(self, fld, attr1 + attr2)
            else:
                # if only one value axist and adding is not strict, 
                # set existing value
                if not strict:
                    if attr1 is None:
                        setattr(self, fld, attr2)

    def copy(self) -> Self:
        return copy.copy(self)

    def dist(self, point: Self) -> PlainQuantity:
        """
        Calculate distance to ``point``.
        
        Only both non None axes are calculated.\n
        If there no such pairs, magnitude of returned value is np.nan.
        """

        dist = Q_(np.nan, 'm**2')
        for axis in self._FIELDS:
            ax1 = getattr(self, axis)
            ax2 = getattr(point, axis)
            if None not in [ax1, ax2]:
                dist = np.nan_to_num(dist) + (ax2-ax1)**2 # type: ignore

        return np.sqrt(dist) # type: ignore

    def direction(self, point: Self) -> Direction:
        """Get unit vector in the direction of ``point``."""

        return Direction(self, point)

    def serialize(self, name: str | None=None) -> dict:
        """
        Serialize instance to base types.
        
        Only not None coordinates are saved.
        """

        result = {}
        
        for fld in self._FIELDS:
            attr = getattr(self, fld)
            if attr is not None:
                magnitude = attr.m
                units = str(attr.u)
                if name is not None:
                    result.update({name + '_' + fld: magnitude})
                    result.update({name + '_' + fld + '_u': units})
                else:
                    result.update({fld: magnitude})
                    result.update({fld + '_u': units})
        return result

    def dotprod(self, other: Self) -> PlainQuantity | None:
        """Calculate dot product with second vector."""

        result = Q_(np.nan, 'm**2')
        # Flag for a valid value
        valid = False
        for axis in self._FIELDS:
            coord1 = getattr(self, axis)
            coord2 = getattr(other, axis)
            if None not in [coord1, coord2]:
                result = np.nan_to_num(result) + coord1*coord2 # type: ignore
                valid = True
        if valid:
            return result # type: ignore
        else:
            return None

    def __add__(self, added: Self) -> Self:
        """If value of any axis is None, result for the axis is None."""
        if not isinstance(added, type(self)):
            return NotImplemented
        
        new_coord = type(self)()
        for fld in self._FIELDS:
            attr1 = getattr(self, fld)
            attr2 = getattr(added, fld)
            if None not in [attr1, attr2]:
                setattr(new_coord, fld, attr1 + attr2)

        return new_coord
    
    def __sub__(self, subed: Self) -> Self:
        if not isinstance(subed, type(self)):
            return NotImplemented
        return self + (-subed) # type: ignore
    
    def __truediv__(self, val: float | int) -> Self:
        result = type(self)()
        for axis in self._FIELDS:
            if (ax1:=getattr(self, axis)) is not None:
                setattr(result, axis, ax1/val)
        return result

    def __mul__(self, val: float | int) -> Self:
        result = type(self)()
        for axis in self._FIELDS:
            if (ax1:=getattr(self, axis)) is not None:
                setattr(result, axis, ax1*val)
        return result

    def __rmul__(self, val: float | int) -> Self:
        return self*val

    def __neg__(self) -> Self:
        new_coord = type(self)()
        for fld in self._FIELDS:
            attr = getattr(self, fld)
            if attr is not None:
                setattr(new_coord, fld, -attr)
        return new_coord

    def __repr__(self) -> str:

        reprs = [self.__class__.__name__ + ':']
        
        for fld in self._FIELDS:
            reprs.append(fld + ' =')
            if fld == 'z':
                if (val:=getattr(self, fld)) is not None:
                    reprs.append(str(val.to('mm')))
                else:
                    reprs.append(str(val))
            else:
                if (val:=getattr(self, fld)) is not None:
                    reprs.append(str(val.to('mm')) + ',')
                else:
                    reprs.append(str(val) + ',')
        return ' '.join(reprs)

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
    data_raw: npt.NDArray[np.int8]
    'measured PA signal in raw format'
    datetime: dt
    "Date and time of measurement."
    yincrement: PlainQuantity
    'Scaling factor to convert raw data to [V]: Data = yincrement*data_raw.'
    max_amp: PlainQuantity
    'max(data) - min(data)'
    x_var_step: PlainQuantity
    x_var_start: PlainQuantity
    x_var_stop: PlainQuantity
    x_var_name: str = 'Time'
    y_var_name: str = 'PhotoAcoustic Signal'
    pm_en: PlainQuantity = Q_(np.nan, 'uJ')
    'laser energy measured by power meter in glass reflection'
    sample_en: PlainQuantity = Q_(np.nan, 'uJ')
    'laser energy at sample'
    
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

    wavelength: PlainQuantity = Q_(np.nan, 'nm')
    'Excitation wavelength'
    pos: Position = field(default_factory=lambda: Position())
    'Position, at which point was measured'
    param_val: list[PlainQuantity] = field(default_factory=list)
    'value of independent parameters'
    repetitions: int = 1
    'Number of datapoint measurements for averaging'

@dataclass
class DataPoint:
    """Single PA measurement for storage."""

    attrs: PointMetadata
    filt_data: dict[str, ProcessedData] = field(default_factory = dict, compare=False, repr=False)
    freq_data: dict[str, ProcessedData] = field(default_factory = dict, compare=False, repr=False)
    raw_data: dict[str, BaseData] = field(default_factory = dict, compare=False, repr=False)

@dataclass
class MeasurementMetadata:
    """General attributes of a measurement."""

    measurement_dims: int
    'dimensionality of the measurement'
    parameter_name: list[str] = field(default_factory = list, compare=False)
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
    center: Position = field(default_factory=lambda: Position())
    'Center point of 2D measurement.'
    width: PlainQuantity = Q_(np.nan, 'mm')
    'Scan size of 2D measurement along horizontal axis.'
    height: PlainQuantity = Q_(np.nan, 'mm')
    'Scan size of 2D measurement along vertical axis.'
    hpoints: int = 0
    'Number of scan points of 2D measurement along horizontal axis.'
    vpoints: int = 0
    'Number of scan points of 2D measurement along vertical axis.'
    scan_dir: str = ''
    'Scan direction of 2D measurement.'
    scan_plane: str = ''
    'Scan plane of 2D measuremnt.'
    wavelength: PlainQuantity = Q_(np.nan, 'nm')
    'Excitation laser wavelength.'

@dataclass
class Measurement:
    """A Photoacoustic measurement with metadata."""

    attrs: MeasurementMetadata
    "MetaData of the measurement."
    data: dict[str, DataPoint] = field(default_factory = dict, compare=False, repr=False)

@dataclass
class OscMeasurement:
    """
    Oscilloscope measurement.
    
    Equality test only check `data_raw` attributes. 

    Attributes
    ----------
    `datetime`: `datetime` - Date and time of measurement.\n
    `data_raw`: `list[ndarray|None]` - list with raw data from osc.
    Its length is equal to amount of osc channels. Contain `None` for
    channels, which were not measured.\n
    `dt`: `PlainQuantity` - time step for measurements in `data_raw`.\n
    `pre_t`: `list[PlainQuantity]` - list with time offsets from start
    of sampling to trigger positions, where absolute time of all
    signals match. Length of this list is equal to length of `data_raw`.\n
    `yincrement`: `PlainQuantity` - Scaling factor to convert raw data
    to [V]: Data = yincrement*data_raw.
    """

    datetime: dt = field(compare=False)
    "Date and time of measurement."
    data_raw: list[npt.NDArray[np.int16] | None] = field(
        default_factory=list
    )
    """
    List with raw data from osc channels.
    Its length is equal to amount of osc channels. Contain `None` for
    channels, which were not measured.
    """
    dt: PlainQuantity = Q_(np.nan, 'us')
    "Time step for measurements in `data_raw`."
    pre_t: list[PlainQuantity | None] = field(default_factory=list)
    """
    List with time offsets from start
    of sampling to trigger positions, where absolute time of all
    signals match. Length of this list is equal to length of `data_raw`.
    """
    yincrement: list[PlainQuantity | None] = field(default_factory=list)
    """
    Scaling factor to convert raw data to [V]: Data = yincrement*data_raw.
    """

    def __eq__(self, __value: Type[Self]) -> bool:
        """Only check equality of `data_raw` attribute."""

        logger.debug('Starting equality check of OscMeasurement')
        # Check if some data is present in both operands
        if len(self.data_raw) and len(__value.data_raw):
            # Compare each measured signal
            cnt = 0
            for d1, d2 in zip(self.data_raw, __value.data_raw):
                if not (d1 is None or d2 is None):
                   if np.allclose(d1, d2):
                       cnt += 1
                elif d1 is None and d2 is None:
                    cnt += 1
            if cnt == len(self.data_raw):
                logger.debug('compare to True')
                return True
        logger.debug('Compare to False')
        return False

@dataclass
class EnergyMeasurement:
    """
    Energy measurement from power meter.
    
    Equality test does not check `datetime` attribute. 

    Attributes
    ----------
    `datetime`: `datetime` - Date and time of measurement.\n
    `signal`: `PlainQuantity[ndarray]` - Sampled PM signal in Volts.\n
    `dt`: `PlainQuantity` - time stamp for data in `signal`.\n
    `istart`: `int` - index of laser pulse start in `signal`.\n
    `istop`: `int` - index of laser pulse stop in `signal`.\n
    `energy`: `PlainQuantity` - laser pulse energy in [uJ]. 
    """

    datetime: dt = field(compare=False)
    "Date and time of measurement."
    signal: PlainQuantity = field(
        default_factory=lambda: Q_(np.empty(0), 'V'),
        compare = False
    )
    "Sampled PM signal in Volts."
    dt: PlainQuantity = Q_(np.nan, 'us')
    "Time stamp for data in `signal`."
    istart: int | None = -1
    "Index of laser pulse start."
    energy: PlainQuantity = Q_(np.nan, 'uJ')
    "Laser pulse energy in [uJ]."

@dataclass(init=False)
class PaEnergyMeasurement(EnergyMeasurement):
    """
    Energy information for PA measurement.
    
    Equality test does not check `datetime` attribute.

    Attributes
    ----------
    `datetime`: `datetime` - Date and time of measurement.\n
    `signal`: `PlainQuantity[ndarray]` - Sampled PM signal in Volts.\n
    `dt`: `PlainQuantity` - time stamp for data in `signal`.\n
    `istart`: `int` - index of laser pulse start in `signal`.\n
    `istop`: `int` - index of laser pulse stop in `signal`.\n
    `energy`: `PlainQuantity` - laser pulse energy [J] calculated
    from `signal`. This value is actual value at power meter, which can
    be located at some intermediate point.\n
    `sample_en`: `PlainQuantity` - laser pulse energy [J] at sample.
    """

    def __init__(
            self,
            en_info: EnergyMeasurement,
            sample_en: PlainQuantity
        ):
        super().__init__(
            datetime=en_info.datetime,
            signal=en_info.signal,
            dt=en_info.dt,
            istart=en_info.istart,
            energy=en_info.energy
        )

        self.sample_en = sample_en
        "Laser pulse energy [J] at sample."

class MeasuredPoint:
    """
    Single PA measurement.
    
    Attributes
    ----------
    `datetime`: `datetime` - Date and time of measurement.\n
    `pa_signal_raw`: ndarray - Raw PA signal.\n
    `pm_signal_raw`: ndarray - Raw PM signal.\n
    `pa_signal`: `PlainQuantity[array]` - PA signal in physical units.\n
    `pm_signal`: `PlainQuantity[array]` - PM signal in physical units.
    This signal is downsampled from `pm_signal_raw`.\n
    `dt`: `PlainQuantity` - Sampling interval for PA data.
    `dt_pm`: PlainQuantity - Sampling interval for PM data, could differ
    from `dt` due to downsampling of PM data.\n
    `start_time`: `PlainQuantity` - Start of PA signal sampling interval
    relative to the begining of laser pulse.\n
    `stop_time`: `PlainQuantity` - Stop of PA signal sampling interval
    relative to the begining of laser pulse.\n
    `yincrement`: `PlainQuantity` - Scaling factor to convert raw data to physical units.\n
    `pm_yin`: `PlainQuantity` - Scaling factor to convert PM raw data to physical units."
    `max_amp`: `PlainQuantity` - Maximum PA amplitude in physical units.\n
    `pm_energy`: `PlainQuantity` - Energy at PM in [J].\n
    `sample_en`: `PlainQuantity` - Energy at sample in [J].\n
    `wavelength`: `PlainQuantity` - Excitation laser wavelength.\n
    `pos`: `Position` - Coordinate of the measured point.
    """

    def __init__(self) -> None:
        """
        Create default measurement.
        
        This is not functional instance.\n
        Use one of classmethods strting with `MeasuredPoint.from_` to
        create a functional instance or set all attributes manually.
        """
        
        self.datetime = dt.now()
        "Date and time of measurement."
        self.pa_signal = Q_(np.empty(0), 'V/mJ')
        "Sampled PA signal in [V/mJ]."
        self.pm_signal = Q_(np.empty(0), 'V')
        "Sampled PM signal in [V]. This signal is downsampled from `pm_signal_raw`."
        self.max_amp = Q_(np.nan, 'V/uJ')
        "Maximum PA amplitude in [V/J]."
        self.start_time = Q_(0, 'us')
        "Start of PA signal sampling interval relative to begining of laser pulse."
        self.stop_time = Q_(np.nan, 'us')
        "Stop of PA signal sampling interval relative to begining of laser pulse."
        self.pa_signal_raw = np.empty(0, dtype=np.int8)
        "Raw PA signal."
        self.pm_signal_raw = np.empty(0, dtype=np.int8)
        "Raw PM signal."
        self.dt = Q_(np.nan, 'us')
        "Sampling interval for PA data."
        self.dt_pm: PlainQuantity = Q_(np.nan, 'us')
        """
        Sampling interval for PM data, could differ from `dt` due 
        to downsampling of PM data.
        """
        self.wavelength = Q_(np.nan, 'nm')
        "Excitation laser wavelength."
        self.pos = Position()
        "Coordinate of the measured point."
        self.yincrement = Q_(np.nan, 'V')
        "Scaling factor to convert PA raw data to [V]."
        self.pm_yin = Q_(np.nan, 'V')
        "Scaling factor to convert PM raw data to [V]."
        self.pm_energy = Q_(np.nan, 'uJ')
        "Energy at PM in [J]."
        self.sample_en = Q_(np.nan, 'uJ')
        "Energy at sample in [J]."
    
    @classmethod
    def from_msmnts(
            cls: Type[Self],
            data: OscMeasurement,
            energy_info: PaEnergyMeasurement | None,
            wavelength: PlainQuantity,
            pa_ch_ind: int=0,
            pm_ch_ind: int=1,
            pos: Position | None=None
        ) -> Self:
        """
        Create MeasuredPoint instance from measurements.
        """

        new = cls()
        # Init general data
        new.datetime = data.datetime
        new.wavelength = wavelength
        if pos is not None:
            new.pos = pos
        # Init PA data
        new.pa_signal_raw = data.data_raw[pa_ch_ind]
        new.yincrement = data.yincrement[pa_ch_ind]
        new.pa_signal = new.pa_signal_raw * new.yincrement # type: ignore
        new.max_amp = new.pa_signal.ptp()
        new.dt = data.dt
        new.stop_time = len(new.pa_signal_raw - 1)*new.dt #type: ignore

        # If PM data is present, include it
        if energy_info is not None:
            new.pm_signal_raw = data.data_raw[pm_ch_ind]
            new.pm_yin = data.yincrement[pm_ch_ind]
            new._set_pm_data()
            new.pm_energy = energy_info.energy
            new.sample_en = energy_info.sample_en
            # Normalize PA signal to sample energy
            new.pa_signal = new.pa_signal/new.sample_en
            new.max_amp = new.max_amp/new.sample_en
            new.yincrement = new.yincrement/new.sample_en
            # Calculate time from start of pm_signal to trigger position
            _pm_start = data.pre_t[pm_ch_ind]
            _pa_start = data.pre_t[pa_ch_ind]
            pm_offset = energy_info.dt*energy_info.istart
            new.start_time = (_pm_start - pm_offset) - _pa_start # type: ignore
            new.stop_time = new.dt*(len(new.pa_signal_raw)-1) + new.start_time # type: ignore
        return new

    @classmethod
    def from_datapoint(
        cls: Type[Self],
        dp: DataPoint
        ) -> Self:
        """
        Create class instance from `DataPoint`.
        
        Intended for loading data from file.
        """

        new = cls()
        # Data loaded from raw_data of the first sample
        sample = next(iter(getattr(dp, 'raw_data').values()))
        sample =cast(BaseData, sample)
        new.pa_signal = sample.data
        new.max_amp = sample.max_amp
        new.start_time = sample.x_var_start
        new.stop_time = sample.x_var_stop
        new.datetime = sample.datetime
        new.pa_signal_raw = sample.data_raw
        new.dt = sample.x_var_step
        new.wavelength = dp.attrs.wavelength
        new.pos = dp.attrs.pos.to('mm')
        new.yincrement = sample.yincrement
        new.pm_energy = sample.pm_en
        new.sample_en = sample.sample_en

        return new

    def _set_pm_data(self) -> None:
        """Set ``dt_pm`` and ``pm_signal`` attributes."""

        # Downsample power meter data if it is too long
        if len(self.pm_signal_raw) > len(self.pa_signal_raw):
            pm_signal_raw, pm_decim_factor = self.decimate_data(
                self.pm_signal_raw,
                1000
            )
        else:
            pm_signal_raw = self.pm_signal_raw
            pm_decim_factor = 1
        # Calculate dt for downsampled data
        self.dt_pm = self.dt*pm_decim_factor
        # Convert downsampled signal to volts
        self.pm_signal = pm_signal_raw * self.pm_yin

    @staticmethod
    def decimate_data(
            data: npt.NDArray[np.int8],
            target: int=10_000,
        ) -> tuple[np.ndarray, int]:
        """Downsample data to <target> size.
        
        Does not guarantee size of output array.
        Return decimated data and decimation factor.
        """

        # logger.debug(f'Starting decimation data with {len(data)} size.')
        factor = int(len(data)/target)
        if factor == 0:
            # logger.debug('...Terminatin. Decimation is not required.')
            return (data, 1)
        # logger.debug(f'Decimation factor = {factor}')
        iterations = int(math.log10(factor))
        rem = factor//10**iterations
        decim_factor = 1
        for i in range(iterations):
            data = decimate(data, 10) # type: ignore
            # logger.debug(f'{len(data)=} after decim step {i}')
            decim_factor *=10
            # logger.debug(f'Curr {decim_factor=}')
        if rem > 1:
            data = decimate(data, rem) # type: ignore
            decim_factor *=rem
        # logger.debug(f'...Finishing. Final size is {len(data)}')
        return data, decim_factor

class ScanLine:
    """
    Single scan line.
    
    Attributes
    ----------
    There are two kinds of attributes: `setted` and `measured`. The
    former are parameters defined at initialization, the latter are
    actually measured values.\n
    `startp`: `Position` - `setted` position of line start.\n
    `stopp`: `Position` - `setted` position of line stop.\n
    `raw_data`: `list[MeasuredPoint]` - `property`. List with `measured`
    points. Internally call `calc_scan_coord` method if `_raw_pos` and
    `_raw_sig` are not empty, which can be computationally costly.
    Should return correct data even if line was not fully scanned.
    If any of `_raw_pos` or `_raw_sig` is empty, return last setted value.\n
    `num_points`: `int` - `property`. Read only. Amount of `measured` points.\n
    `ltype`: `Literal['straight line']` - `setted` shape of the scan line.\n

    Methods
    -------
    `add_pos_point`\n
    `calc_grid`\n
    `calc_scan_coord`
    """

    def __init__(
            self,
            startp: Position,
            stopp: Position,
            ltype: Literal['straight line']='straight line'   
        ) -> None:
        """Default scan line constructor.
        
        Attributes
        ----------
        ``startp`` - `setted` position of line start.\n
        ``stopp`` - `setted` position of line stop.\n
        ``points`` - `setted` amount of regular line points.\n
        ``ltype`` - optional shape of the scan line.\n
        """
        
        self.startp = startp
        "Exact position of scan start."
        self.stopp = stopp
        "Exact position of scan stop."
        self._raw_sig = []
        "List with raw measurements."
        self._raw_pos = []
        "List with positions, measured along scan line with timestamps."
        self.ltype = ltype
        "Type of scan line"
        self._raw_data: list[MeasuredPoint] = []

    @classmethod
    def from_list(
        cls: Type[Self],
        lst: list[MeasuredPoint]
        ) -> Self | None:
        """
        Create ScanLine from list of measured points.
        
        Intended to be used for loading data from file.
        Starting point is position of the first datapoint, stop point
        is position of the last data point. `lst` should contain at
        least 2 points.\n
        Does not check whether all points are on the same line.
        """

        if len(lst) < 3:
            logger.warning('Scan line cannot be created. Too few data.')
            return
        # Start and stop points are positions of the first and last
        # points in the list.
        startp = lst[0].pos
        stopp = lst[-1].pos

        new_line = cls(startp, stopp)
        new_line.add_measurements(lst)
        return new_line

    def add_measurements(self, points: list[MeasuredPoint]) -> None:
        """Append MeasuredPoints to scan line."""
        
        new_data = self.raw_data
        for point in points:
            new_data.append(point)
        self.raw_data = new_data

    def add_pos_point(self, pos: Position | None) -> None:
        """Append position along scan line."""

        if pos is not None:
            self._raw_pos.append((dt.now(),pos))

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
            poses: list[tuple[dt,Position]],
            start: Position | None=None,
            stop: Position | None=None,
            trim_edges: bool=True
        ) -> list[MeasuredPoint]:
        """
        Calculate positions of OscMeasurements.
        
        Assume that during scanning signals and coordinates were measured
        with arbitrary time intervals (no periodicity, no parcticular
        dependence between measurements of signal and positions).
        For each OscMeasurement the closest positions, measured before 
        and after the OscMeasurement are taken and position of 
        OscMeasurement is calculated by linear interpolation between those points.
        
        Attributes
        ----------
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
         
        Return
        ------
        New list with measured points having calcualted positions.
        """

        logger.debug(
            f'Starting calculation signal position for {len(signals)} '
            + f'MeasuredPoints based on {len(poses)} coordinate measurements.'
        )
        if not len(signals) or not len(poses):
            return []
        # Sort data
        result = sorted(signals, key = lambda x: x.datetime)
        poses.sort(key = lambda x: x[0])
        # Reference point for time delta is time of the first measured position
        t0 = poses[0][0]
        # Array with time points relative to ref of signal measurements
        en_time = np.array([(x.datetime - t0).total_seconds() for x in result])
        # Array with time points relative to ref of position
        pos_time = np.array([(x[0] - t0).total_seconds() for x in poses])
        for fld in Position._FIELDS:
            # Assume that all positions have the same non-None fields
            # and iterate only on those fields.
            if getattr(start, fld) is not None:
                # Values of measured coordinates
                coord = np.array([getattr(x[1], fld).to('mm').m for x in poses])
                # Array with calculated axes
                en_coord = np.interp(
                    x = en_time,
                    xp = pos_time,
                    fp = coord,
                    left = getattr(start, fld).to('mm').m,
                    right = getattr(stop, fld).to('mm').m
                )
                # Set axes values to result
                for i, res in enumerate(result):
                    setattr(res.pos, fld, Q_(en_coord[i], 'mm'))

        # Remove duplicated edge values
        dups = 0
        istart = 0
        istop = len(result) - 1
        if trim_edges:
            while result[istart + 1].pos == start:
                istart += 1
                dups += 1
            while result[istop - 1].pos == stop:
                istop -= 1
                dups += 1
        if dups:
            logger.debug(f'{dups} duplicate edge values trimmed.')
        return result[istart:(istop + 1)]

    @property
    def raw_data(self) -> list[MeasuredPoint]:
        """List with data points at their measured positions."""

        # if positions and signals were added, then calculate results
        # otherwise it is supposed that MeasuredPoints were setted.
        if len(self._raw_pos) and len(self._raw_sig):
            if not len(self._raw_data):
                self._raw_data = self.calc_scan_coord(
                signals = self._raw_sig,
                poses = self._raw_pos,
                start = self.startp,
                stop = self.stopp
            )
        return self._raw_data
    @raw_data.setter
    def raw_data(self, val: list[MeasuredPoint]) -> None:
        self._raw_data = val

    @property
    def num_points(self) -> int:
        """
        Amount of measured points.

        Read only.
        """
        return len(self.raw_data)
    @num_points.setter
    def num_points(self, val: Any) -> None:
        logger.warning(
            'Amount of raw points along scan line is read only.'
        )

    def __str__(self) -> str:
        
        reprs = [self.__class__.__name__ + ':']
        reprs.append('Points with pos =')
        reprs.append(str(self.num_points))
        reprs.append('Measured PA signals =')
        reprs.append(str(len(self._raw_sig)))
        reprs.append('Pos points =')
        reprs.append(str(len(self._raw_pos)))
        return ' '.join(reprs)

class MapData:
    """"
    2D PA data.
    
    Measured points irregularly spaced along one axis (fast scan axis).

    Attributes
    ----------
    There are two kinds of attributes: `setted` and `measured`. The
    former are parameters defined at initialization, the latter are
    actually measured values.\n
    `lines`:         `list[ScanLine]` - All `measured` scan lines.\n
    `points`:       `list[MeasuredPoint]` - All `measured point.\n
    `wavelength`:   `PlainQuantity` - `setted` excitation wavelength.
    `num_points`:   `int` - `Property`. Read only `measured` number
                    of scanned points.

    `centp`:        `Position` - `setted` center point of scan in
                    absolute coordinates.\n
    `startp`:       `Position` - `Property`. Read only. Calculated
                    `setted` starting position of scan in absolute
                    coordinates.\n
    `blp`:          `Position` - `Property`. Read only. Calculated 
                    `setted` position of bottom-left corner of scan
                    in absolute coordinates.

    `width`:        `PlainQuantity` - `setted` scan size along 
                    `horizontal` axis.\n
    `height`:       `PlainQuantity` - `setted` scan size along 
                    `vertical` axis.\n
    `hpoints`:      `int` - `setted` number of scan points along
                    `horizontal` axis.\n
    `vpoints`:      `int` - `setted` number of scan points along
                    `vertical` axis.\n
    `hstep`:        `PlainQuantity` - `Property`. Read only. Calculated
                    `setted` step along `horizontal` scan axis.\n
    `vstep`:        `PlainQuantity` - `Property`. Read only. Calculated
                    `setted` step along `vertical` scan axis.\n
    `haxis`:        `str` - `Property`. Read only. Title of `horizontal`
                    scan axis.\n
    `vaxis`:        `str` - `Property`. Read only. Title of `vertical`
                    scan axis.

    `fstep`:        `Position` - `Property`. Read only. Calculated
                    `setted` step along `fast` scan axis.\n
    `sstep`:        `Position` - `Property`. Read only. Calculated
                    `setted` step along `slow` scan axis.\n
    `fsize`:        `PlainQuantity` - `Property`. Read only. Calcualted
                    `setted` scan size along `fast` scan axis.\n
    `fpoints`:      `int` - `Property`. Read only. Calcualted `setted`
                    number of scan points along `fast` scan axis.\n
    `fp_raw_max`:   `int` - `Property`. Read only. Maximum number
                    of points in measured scan lines.\n
    `spoints`:      `int` - `Property`. Read only. Calcualted `setted`
                    number of scan points along `slow` scan axis.\n
    `faxis`:        `str` - `Property`. Read only. Title of `fast` scan
                    axis.\n
    `saxis`:        `str` - `Property`. Read only. Title of `slow` scan
                    axis.

    `scan_dir`:     `str` - `Property`. `setted` 3-letter direction and
                    starting point of scan. All letters are
                    automatically converted to upper case. First letter
                    ['H'|'V"] - Horizontal or Vertical direction of
                    fast scan axis. Second letter ['L'|'R'] -
                    horizontal position of starting point (Left or
                    Right). Third letter ['B'|'T'] - vertical position
                    of starting point (Bottom or Top).\n
    `scan_plane`:   `Literal['XY', 'YZ', 'ZX']` - `Property`. `setted`
                    Pair of axis along which scan is done. First letter
                    is horizontal axis, second is vertical.\n
    `mode`:         `Literal['Normal', 'Fast']` - `setted` PA signal
                    measurement method. `Fast` method implies 
                    measurement of PA signal only, therefore no energy
                    information available in this mode. 
    """
    
    def __init__(
            self,
            center: Position,
            width: PlainQuantity,
            height: PlainQuantity,
            hpoints: int,
            vpoints: int,
            scan_plane: Literal['XY', 'YZ', 'ZX']='XY',
            scan_dir: str='HLB',
            mode: str='Normal',
            wavelength: PlainQuantity=Q_(100, 'nm'),
            **kwargs
        ) -> None:

        self.centp = center
        "`setted` center point of scan in absolute coordinates."
        self.width = width
        "`setted` scan size along `horizontal` axis."
        self.height = height
        "`setted` scan size along `vertical` axis."
        self.hpoints = hpoints
        "`setted` number of scan points along `horizontal` axis."
        self.vpoints = vpoints
        "`setted` number of scan points along `vertical` axis."
        self.scan_dir = scan_dir
        self.scan_plane = scan_plane
        self.mode = mode
        "`setted` signal measurement method."
        self.wavelength = wavelength
        "`setted` excitation wavelength."
        self.lines: list[ScanLine] = []
        "All `measured` scan lines."
        self._raw_points_sig: list[list[MeasuredPoint]] = []
        "Cashed lines for signal."
        self._raw_points_coord: list[list[MeasuredPoint]] = []
        "Cashed lines for coordinates."
        self._plot_coords: list[list[Position]] = []
        self._plot_sigs: list[list[MeasuredPoint]] = []

    @classmethod
    def from_measmd(cls: Type[Self], md: MeasurementMetadata) -> Self:
        """
        Initialize MapData from measurement metadata.
        
        Intended to be used in data loading from file.
        """
        return cls(**asdict(md))

    def create_line(self) -> ScanLine|None:
        """
        Create empty line.
        
        Assume that scan pattern is snake-like, i.e. start and stop
        points alternate between consequent lines.

        Return
        ------
        Reference to the created scan line.
        """

        logger.debug(f'Starting addition of scan line number {len(self.lines)}')
        if (lines:=len(self.lines)) >= self.spoints:
            logger.warning(
                'Cannot add new line. Maximum scan lines reached.'
            )
            return
        # Calculate displacement from scan starting point to
        # to starting and ending points of new line
        delta_start = (lines + 0.5)*self.sstep + Direction(self.fstep)*((lines%2)*self.fsize)
        logger.debug(f'{delta_start=}')
        delta_stop = (lines + 0.5)*self.sstep + Direction(self.fstep)*(((lines + 1)%2)*self.fsize)
        # Generate start and stop points of new line
        line_start = self.startp + delta_start
        line_stop = self.startp + delta_stop

        logger.debug(f'New line created with start {line_start}')

        new_line = ScanLine(
            startp = line_start,
            stopp = line_stop
        )
        return new_line 

    def add_line(self, line: ScanLine) -> None:
        """
        Add `line` to scan.
        
        Line is actually appended to `data` attribute.
        """

        self.lines.append(line)

    def get_plot_data(
            self, signal: str
        ) -> tuple[PlainQuantity, PlainQuantity, PlainQuantity]:
        """
        Prepare data for plotting.
        
        Indexing: data[x,y].\n
        Return
        ------
        Tuple of 3 Pint quantities, which are 2D arrays: first 
        value contains position of scan points along horizontal axis,
        second - positions along vertical axis, third values of required
        signal.\n
        Size of the third array is one less for each dimension.
        Extra point for coordinates are duplicated fast axis coordinates
        of the first scan line and duplicated slow axis coordinates of 
        first point in each scanned line.
        """

        tstart = time.time()
        coord_data = self.get_plot_coords()
        signal_data = self.get_plot_points()
        logger.debug(f'{self.blp=}')
        def get_coord(axis: Literal['h', 'v']):
            coords = np.zeros_like(coord_data)
            if axis == 'h':
                units = None
                for index, pos in np.ndenumerate(coord_data):
                    pos = cast(Position, pos)
                    pos = pos - self.blp
                    coords[index] = getattr(pos, self.haxis).m
                    if units is None:
                        units = getattr(pos, self.haxis).u
            else:
                units = None
                for index, pos in np.ndenumerate(coord_data):
                    pos = cast(Position, pos)
                    pos = pos - self.blp
                    coords[index] = getattr(pos, self.vaxis).m
                    if units is None:
                        units = getattr(pos, self.vaxis).u
            return Q_(coords, units)
        
        def get_signal(signal: str):
            arr = np.zeros_like(signal_data, dtype=np.float_)
            units = None
            for index, point in np.ndenumerate(signal_data):
                point = cast(MeasuredPoint|None, point)
                if point is not None:
                    # Assume that signal attribute is PlainQuantity
                    if units is None:
                        units = getattr(point, signal).u
                    arr[index] = getattr(point, signal).to(units).m
                    
            return Q_(arr, units)
        
        x = get_coord('h')
        y = get_coord('v')
        z = get_signal(signal)
        logger.info(f'get_plot_data executed in {(time.time() - tstart):.3}')
        return (x,y,z)
    
    def get_plot_coords(self) -> np.ndarray:
        """
        Get positions for uneven pixel corners.

        Signals are centered in pixels.
        """

        tstart = time.time()
        # Amount of cashed scan lines
        icashed = int(len(self._plot_coords)/2)
        # Create a deep copy of _new_ (not cashed) data
        data = copy.deepcopy(self.lines[icashed:])
        # New lines can have more points than old,
        # therefore copy last point necessary amount of times in cashed data
        max_fpoints = self.fp_raw_max
        for line in self._plot_coords:
            while (j:=max_fpoints - len(line) + 1) > 0:
                line.append(copy.copy(line[-1]))
                j -= 1
        # Fill array with existing data
        for line in data:
            line_pos = [point.pos for point in line.raw_data]
            # To match line size add required amoun of line stop points
            while (j:=max_fpoints - len(line_pos)) > 0:
                line_pos.append(copy.copy(line.stopp))
                j -= 1
            # Add start point in the begining
            line_pos.insert(0, line.startp)
            # Shift position for fast axis so that the first coordinate
            # in line's starting point, the last coordinate is the
            # line's stop point, all other coordinates are half way
            # between measured points.
            for i, pos in enumerate(line_pos):
                if i:
                    if (i < len(line_pos) - 1) and pos != line.stopp:
                        line_pos[i] += (line_pos[i+1] - pos)/2
                    elif i == len(line_pos) - 1:
                        line_pos[i] = copy.copy(line.stopp)
            # Sort data along fast scan axes.
            # The same sorting should be applied in get_plot_points
            line_pos.sort(key = lambda x: getattr(x, self.faxis).to_base_units())
            # For lines of position should be added for each scanned line
            # Lines are shifted by half of slow scan step from real line
            line_pos = [pos - self.sstep/2 for pos in line_pos]
            self._plot_coords.append(copy.deepcopy(line_pos))
            line_pos = [pos + self.sstep for pos in line_pos]
            self._plot_coords.append(line_pos)
        # Transpose data if fast scan axis is vertical
        if self.scan_dir[0] == 'V':
            res = [list(x) for x in zip(*copy.deepcopy(self._plot_coords))]
        else:
            res = copy.deepcopy(self._plot_coords)

        tot = time.time() - tstart
        logger.debug(f'Done get_plot_coords in {(tot):.3}')
        logger.debug(f'{np.array(res, dtype=object).shape=}')
        return np.array(res, dtype=object)

    def get_plot_points(self, add_zero_points: bool=False) -> np.ndarray:
        """
        2D array containing deep copy of measured points.

        The first index is vertical axis, the second is horizontal.
        Indexing starts ar left bottom corner of scan.
        As amount of points along fast scan axis can vary from line
        to line, size of the array in this direction corresponds to
        the longest scan line in ``data``. Size along slow scan axis
        is ``spoints``. Default value of not scanned points is 
        ``None``.\n

        Attributes
        ----------
        `add_zero_points` if `True` duplicate initial scan points so that
        size of the resulting array will be increased by one for each
        dimension.
        """

        tstart = time.time()
        if len(self._plot_sigs) == 0:
            # Do not forget to sort data in the first line
            line_points = copy.deepcopy(self.lines[0].raw_data)
            line_points.sort(key = lambda x: getattr(x.pos, self.faxis))
            self._plot_sigs.append(line_points)
        # Amount of cashed scan lines
        icashed = int((len(self._plot_sigs) + 1)/2)
        # Create a deep copy of new data
        data = copy.deepcopy(self.lines[icashed:])
        # New lines can have more points than old,
        # therefore copy last point necessary amount of times
        max_fpoints = self.fp_raw_max
        for line in self._plot_sigs:
            while (j:=max_fpoints- len(line)) > 0:
                line.append(copy.deepcopy(line[-1]))
                j -= 1
        # Fill array with existing data
        for line in data:
            line_points = line.raw_data
            # To match line size last pos in each line is duplicated
            # required times
            while (j:=max_fpoints - len(line_points)) > 0:
                line_points.append(copy.deepcopy(line_points[-1]))
                j -= 1
            # The same sorting should be applied in get_plot_coords
            line_points.sort(key = lambda x: getattr(x.pos, self.faxis))
            self._plot_sigs.append(line_points)
            self._plot_sigs.append(line_points)
        if self.scan_dir[0] == 'V':
            res = [list(x) for x in zip(*copy.deepcopy(self._plot_sigs))]
        else:
            res = copy.deepcopy(self._plot_sigs)

        tot = time.time() - tstart
        logger.debug(f'Done get_plot_points in {(tot):.3}')
        logger.debug(f'{np.array(res, dtype=object).shape=}')
        return np.array(res, dtype=object)

    def get_scanned_pos(self) -> np.ndarray:
        """
        Coordinates of all scanned points as 2D array of floats in 'mm'.
        """
        res = []
        start = time.time()
        for line in self.lines:
            for point in line.raw_data:
                pos = point.pos
                res.append(
                    [
                        getattr(pos, self.haxis).m,
                        getattr(pos, self.vaxis).m
                    ]
                )
        logger.debug(f'get_scanned_pos took {int((time.time()-start)*1000)} ms')
        return np.array(res)

    def get_bounded_rect(self, dpm: int) -> tuple[np.ndarray,np.ndarray]:
        """Calculate coordinates of area for data interpolation"""

        start = time.time()
        for line_n, line in enumerate(self.lines):
            f_pnt = line.raw_data[0]
            l_pnt = line.raw_data[-1]
            f_coord0 = getattr(f_pnt.pos, self.faxis).m
            f_coord1 = getattr(l_pnt.pos, self.faxis).m
            f_coord0, f_coord1 = sorted([f_coord0, f_coord1])
            if line_n == 0:
                # Save initial min and max along fast axis
                f_min = f_coord0
                f_max = f_coord1
                # Save min value for slow axis
                s_min = getattr(f_pnt.pos, self.saxis).m
            else:
                if f_coord0 > f_min:
                    f_min = f_coord0
                if f_coord1 < f_max:
                    f_max = f_coord1
            s_max = getattr(f_pnt.pos, self.saxis).m
        s_min, s_max = sorted([s_min, s_max])

        if self.scan_dir.startswith('H'):
            x1 = f_min
            x2 = f_max
            y1 = s_min
            y2 = s_max
        else:
            x1 = s_min
            x2 = s_max
            y1 = f_min
            y2 = f_max

        x_coords = np.linspace(x1, x2, int((x2-x1)*dpm)).reshape((1, -1))
        y_coords = np.linspace(y1, y2, int((y2-y1)*dpm)).reshape((-1, 1))
        logger.debug(f'get_bounded_rect took {int((time.time()-start)*1000)} ms')
        return (x_coords, y_coords)

    def get_image_data(
            self,
            signal: str,
            dpm: int=100,
            method: Literal['nearest', 'linear', 'cubic']='cubic',
            **kwargs
        ) -> np.ndarray:
        """
        Prepared smooth image of the scanned data.

        Perform interpolation using `griddata` method from scipy.

        Attributes
        ----------
        `signal` - valid attributes of MeasuredPoint, which will be
            shown.\n
        `dpm` - number of points per mm in the image.
        `method` - interpolation technique.

        Return
        ------
        ndarray with data.
        """

        strart = time.time()

        scanned_pos = self.get_scanned_pos()
        scanned_sig = self.get_scanned_sig(signal)
        x_coords, y_coords = self.get_bounded_rect(dpm)
        # Interpolation will fail if we have only 1 scanned line
        if len(self.lines) < 2:
            result = np.full(
                (x_coords.shape[1], y_coords.shape[0]),
                scanned_sig.mean()
            )
        else:
            result = griddata(
                points = scanned_pos,
                values = scanned_sig,
                xi = (x_coords, y_coords),
                method = method,
                fill_value = scanned_sig.mean()
            )
        delta = int((time.time() - strart)*1000)
        logger.info(f'Get_image_data done in {delta} ms.')
        return result

    def get_scanned_sig(self, signal: str) -> np.ndarray:
        """
        Signal from all scanned points.

        Attributes
        ----------
        `signal` should be valid attribute of measured points.

        Return
        ------
        1D ndarray with floats in preferred units.
        """

        res = []
        start = time.time()
        for line in self.lines:
            for point in line.raw_data:
                res.append(getattr(point, signal).m)
        logger.debug(f'get_scanned_sig took {int((time.time()-start)*1000)} ms')
        return np.array(res)
    
    def point_from_pos(self, pos: Position) -> MeasuredPoint | None:
        """
        Get Datapoint for a given position.
        
        Attributes
        ----------
        `pos` - Arbitrary position in relative coordinates within
        scanned data.

        Return
        ------
        A datapoint, which pixel representation on plot
        contains `pos`.\n
        """

        if self.lines is None:
            return
        
        sstep = self.sstep
        # Scan starting point in relative coords
        scan_startp = self.startp - self.blp
        # Vector from scan starting point to clicked_pos
        scan_rel_pos = pos - scan_startp
        # Distance to the pos from start point along slow axis as number of sstep's
        pos_sstep = (sstep).dotprod(scan_rel_pos)/sstep.value()**2
        line_no = int(pos_sstep)
        line = self.lines[line_no]
        # The needed datapoint is the closest to the clicked pos in the line.
        point = min(line.raw_data, key = lambda x: (x.pos - self.blp - pos).value())
        return point
    
    def point_index(self, point: MeasuredPoint) -> PointIndex:
        """Calculate PointIndex for the given point."""

        # Distance to the pos from start point along slow axis as number of sstep's
        pos_sstep = (self.sstep).dotprod(point.pos-self.startp)/self.sstep.value()**2
        
        line_ind = int(pos_sstep)
        point_ind = self.lines[line_ind].raw_data.index(point)
        
        return PointIndex(line_ind, point_ind)

    def point_rect(self, point: MeasuredPoint) -> QuantRect:
        """
        Calculate a rectangle, which bounds given point on plot.
        
        Relative coordinates.
        """

        index = self.point_index(point)

        # The idea is to calculate diagonal positions of the pixel
        diag1 = point.pos.copy()
        diag2 = point.pos.copy()
        # Calcultaion for fast axis requires special handling of
        # boundary points
        line = self.lines[index.line_ind]
        if not index.point_ind:
            diag1 = line.startp.copy()
            diag2 = (diag2 + line.raw_data[index.point_ind + 1].pos)/2
        elif index.point_ind + 1 == line.num_points:
            diag1 = (diag1 + line.raw_data[index.point_ind - 1].pos)/2
            diag2 = line.stopp.copy()
        else:
            diag1 = (diag1 + line.raw_data[index.point_ind - 1].pos)/2
            diag2 = (diag2 + line.raw_data[index.point_ind + 1].pos)/2
        # Calculate position along slow axis is straight forward
        diag1 += -self.sstep/2
        diag2 += self.sstep/2
        # Switch to relative coordinates
        diag1 += -self.blp
        diag2 += -self.blp
        # Get coordinates
        hcoords = [getattr(x, self.haxis) for x in [diag1, diag2]]
        vcoords = [getattr(x, self.vaxis) for x in [diag1, diag2]]

        left = min(hcoords)
        bottom = min(vcoords)
        width = max(hcoords) - min(hcoords)
        height = max(vcoords) - min(vcoords)
        result = QuantRect(left, bottom, width, height)
        return result

    @property
    def num_points(self) -> int:
        """Number of scanned points."""

        pts = 0
        for line in self.lines:
            pts += line.num_points
        return pts

    @property
    def startp(self) -> Position:
        """
        Starting position of scan in absolute coordinates.
        
        Read only.
        """
        startp = Position()

        if self.scan_dir[1] == 'L':
            f0 = getattr(self.centp, self.haxis) - self.width/2
            setattr(startp, self.haxis, f0)
        else:
            f0 = getattr(self.centp, self.haxis) + self.width/2
            setattr(startp, self.haxis, f0)
        if self.scan_dir[2] == 'B':
            s0 = getattr(self.centp, self.vaxis) - self.height/2
            setattr(startp, self.vaxis, s0)
        else:
            s0 = getattr(self.centp, self.vaxis) + self.height/2
            setattr(startp, self.vaxis, s0)

        return startp
    @startp.setter
    def startp(self, val: Any) -> None:
        logger.warning('start position is read only.')

    @property
    def blp(self) -> Position:
        """
        Position of bottom-left corner of scan in absolute coordinates.
        
        Read only.
        """
        blp = Position()
        h0 = getattr(self.centp, self.haxis) - self.width/2
        setattr(blp, self.haxis, h0)
        v0 = getattr(self.centp, self.vaxis) - self.height/2
        setattr(blp, self.vaxis, v0)
        return blp
    @blp.setter
    def blp(self, val: Any) -> None:
        logger.warning('Bottom-left position is read only.')

    @property
    def fstep(self) -> Position:
        """
        Calculated `setted` step along `fast` scan axis.
        
        Read only.
        """
        fstep = Position()
        if self.scan_dir.startswith('H'):
            setattr(fstep, self.vaxis, Q_(0, 'mm'))
            if self.scan_dir[1] == 'L':
                setattr(fstep, self.haxis, self.hstep)
            else:
                setattr(fstep, self.haxis, -1*self.hstep)
        else:
            setattr(fstep, self.haxis, Q_(0, 'mm'))
            if self.scan_dir[2] == 'B':
                setattr(fstep, self.vaxis, self.vstep)
            else:
                setattr(fstep, self.vaxis, -1*self.vstep)

        return fstep
    @fstep.setter
    def fstep(self, val: Any) -> None:
        logger.warning('Fast step is read only.')

    @property
    def sstep(self) -> Position:
        """
        Calculated `setted` step along `slow` scan axis.
        
        Read only.
        """
        sstep = Position()
        if self.scan_dir.startswith('H'):
            setattr(sstep, self.haxis, Q_(0, 'mm'))
            if self.scan_dir[2] == 'B':
                setattr(sstep, self.vaxis, self.vstep)
            else:
                setattr(sstep, self.vaxis, -1*self.vstep)
        else:
            setattr(sstep, self.vaxis, Q_(0, 'mm'))
            if self.scan_dir[1] == 'L':
                setattr(sstep, self.haxis, self.hstep)
            else:
                setattr(sstep, self.haxis, -1*self.hstep)

        return sstep
    @sstep.setter
    def sstep(self, val: Any) -> None:
        logger.warning('Slow step is read only.')

    @property
    def fsize(self) -> PlainQuantity:
        """
        Calcualted `setted` scan size along `fast` scan axis.

        Read only.
        """

        if self.scan_dir.startswith('H'):
            fsize = self.width
        else:
            fsize = self.height
        return fsize
    @fsize.setter
    def fsize(self, val: Any) -> None:
        logger.warning('fsize is read only.')

    @property
    def fpoints(self) -> int:
        """
        Calcualted `setted` number of scan points along `fast` scan axis.
        
        Read only.
        """

        if self.scan_dir.startswith('H'):
            fpoints = self.hpoints
        else:
            fpoints = self.vpoints
        return fpoints
    @fpoints.setter
    def fpoints(self, val: Any) -> None:
        logger.warning('fpoints is read only.')

    @property
    def fp_raw_max(self) -> int:
        """
        Maximum number of points in measured scan lines.
        
        Read only.
        """
        points = 0
        for line in self.lines:
            if line.num_points > points:
                points = line.num_points
        return points
    @fp_raw_max.setter
    def fp_raw_max(self, val: Any) -> None:
        logger.warning('fpoints_raw_max is read only.')

    @property
    def spoints(self) -> int:
        """
        Calcualted `setted` number of scan points along `slow` scan axis.
        
        Read only.
        """

        if self.scan_dir.startswith('H'):
            spoints = self.vpoints
        else:
            spoints = self.hpoints
        return spoints
    @spoints.setter
    def spoints(self, val: Any) -> None:
        logger.warning('spoints is read only.')

    @property
    def saxis(self) -> str:
        """
        Title of `slow` scan axis.
        
        Read only.
        """

        if self.scan_dir.startswith('H'):
            saxis = self.vaxis
        else:
            saxis = self.haxis
        return saxis
    @saxis.setter
    def saxis(self, val: Any) -> None:
        logger.warning('saxis is read only.')
    
    @property
    def faxis(self) -> str:
        """
        Title of `fast` scan axis.
        
        Read only.
        """

        if self.scan_dir.startswith('H'):
            faxis = self.haxis
        else:
            faxis = self.vaxis
        return faxis
    @faxis.setter
    def faxis(self, val: Any) -> None:
        logger.warning('faxis is read only.')

    @property
    def hstep(self) -> PlainQuantity:
        """
        Calculated `setted` step along `horizontal` scan axis.
        
        Read only.
        """
        if self.width.m is not np.nan and self.hpoints > 1:
            hstep = self.width/(self.hpoints)
        else:
            hstep = Q_(np.nan, 'mm')
        return hstep
    @hstep.setter
    def hstep(self, val: Any) -> None:
        logger.warning('hstep is read only attributes and cannot be set.')

    @property
    def vstep(self) -> PlainQuantity:
        """
        Calculated `setted` step along `vertical` scan axis.
        
        Read only.
        """
        if self.height.m is not np.nan and self.vpoints > 1:
            vstep = self.height/(self.vpoints)
        else:
            vstep = Q_(np.nan, 'mm')
        return vstep
    @vstep.setter
    def vstep(self, val: Any) -> None:
        logger.warning('vstep is read only attributes and cannot be set.')

    @property
    def haxis(self) -> str:
        """
        Title of `horizontal` scan axis.
        
        Read only.\n
        Automatically calculated from ``scan_plane``.
        """
        return self.scan_plane[0].lower()
    @haxis.setter
    def haxis(self, val: str = '') -> None:
        logger.warning('haxis is read only.')

    @property
    def vaxis(self) -> str:
        """
        Title of `vertical` scan axis.
        
        Read only.\n
        Automatically calculated from ``scan_plane``.
        """
        return self.scan_plane[1].lower()
    @vaxis.setter
    def vaxis(self, val: str) -> None:
        logger.warning('vaxis is read only.')

    @property
    def scan_dir(self) -> str:
        """
        3-letter direction and starting point of scan.

        All letters are automatically converted to upper case.\n
        First letter ['H'|'V"] - horizontal or vertical direction of
        fast scan axis.\n
        Second letter ['L'|'R'] - horizontal position of starting point
        (left or right).\n
        Third letter ['B'|'T'] - vertical position of starting point
        (bottom or top).
        """
        return self._scan_dir
    @scan_dir.setter
    def scan_dir(self, val: str) -> None:
        self._scan_dir = val.upper()

    @property
    def scan_plane(self) -> str:
        """
        Pair of axis along which scan is done.
        
        First letter is horizontal axis, second is vertical.
        """
        return self._scan_plane
    @scan_plane.setter
    def scan_plane(self, val: str) -> None:
        if val.upper() not in ['XY', 'YZ', 'XZ']:
            raise ValueError
        self._scan_plane = val.upper()

    def __str__(self):
        return (f'Center={self.centp}. Scanned points={sum(map(lambda x: x.num_points, self.lines))}')

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

class WorkerFlags(TypedDict):
    is_running: bool
    pause: bool

class ThreadSignals:
    """
    Object for communication between threads.
    
    It is intended to be an argument for a function, which run infinite
    loop in a separate thread.

    Attributes
    ----------
    `is_running`: `bool` - flag to stop execution of the function.\n
    `count`: `int` - counter of successfull loop executions.\n
    `progress`: `threading.Event` - an event, which is set after each
    successfull loop execution. 
    """

    def __init__(self, is_running: bool=True) -> None:
        
        self.is_running = is_running
        "Flag to stop a thread from outside."
        self.count: int = 0
        "Counter of successfull loop executions."
        self.progress = threading.Event()
        "Event, indicating successfull loop execution."

class WorkerSignals(QObject):
    """
    Signals available from a running worker thred.
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progess = Signal(object)

    def disconnect_all(self):
        try:
            self.finished.disconnect()
            self.error.disconnect()
            self.result.disconnect()
            self.progess.disconnect()
        except:
            logger.debug('Unable to disconect some signals.')

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
        self.flags = WorkerFlags(
                {
                'is_running': True,
                'pause': False
            }
        )

        #allow signals emit from func
        self.kwargs['signals'] = self.signals

        #for sending data to running func
        self.kwargs['flags'] = self.flags

    @Slot()
    def run(self) -> None:
        """Actually run a func."""

        try:
            self.result = self.func(*self.args, **self.kwargs)
        except:
            exctype, value = sys.exc_info()[:2]
            logger.error(
                'An error occured while trying to launch worker'
                + f' {self.func.__name__}: {exctype}, {value}\n'
                + f'{traceback.format_exc()}'
            )
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

        if priority < 3 and self.nitems > 3:
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
            close_func: Callable | None=None
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
            ) -> T | ActorFail:
        """
        Submit a function for serail processing.
        
        Priority can have values from 0 (lowest) to 10 (highest).\n
        """

        r = Result()
        # logger.debug(f'Function {func.__name__} submitted to actor')
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
            # logger.debug(f'Actor Starting: {func.__name__}')
            try:
                r.set_result(func(*args, **kwargs))
            except:
                exctype, value = sys.exc_info()[:2]
                msg = (f'Error in call: {exctype}, {value}'
                + f'{traceback.format_exc()}')
                logger.error(msg)
                r.set_result(ActorFail(msg))
            # logger.debug(f'{func.__name__} is ready')

    def show_tasks(self):

        logger.info(pprint(self._mailbox._queue))