"""
Operations with PA data.
Requires Python > 3.10

Workflow with PaData:
1. init class instance.
2. Create measuremnt.
3. Add data points to measurement.

Data structure of PaData class:
PaData:
|--attrs: FileMetadata
|  |--version: float - version of data structure
|  |--measurements_count: int - amount of measurements in the file
|  |--created: str - date and time of file creation
|  |--updated: str - date and time of last file update
|  |--notes: str - description of the file
|  |--zoom_pre_time: Quantity - start time from the center of the PA data frame for zoom in data view
|  |--zoom_post_time: Quantity - end time from the center of the PA data frame for zoom in data view
|--measurements: dict[str, Measurement]
|  |--measurement001: Measurement  
|  |  |--attrs: MeasurementMetadata
|  |  |  |--data_points: int - amount of datapoints in the measurement
|  |  |  |--parameter_name: List[str] - independent parameter, changed between measured PA signals
|  |  |  |--measurement_dims: int - dimensionality of the measurement
|  |  |  |--max_len: int - maximum amount of samples in a single datapoint in this measurement
|  |  |  |--created: str - date and time of file creation
|  |  |  |--updated: str - date and time of last file update
|  |  |  |--notes: str - description of the measurement
|  |  |--data: dict[str,DataPoint]
|  |  |  |--point001
|  |  |  |  |--attrs: PointMetadata
|  |  |  |  |  |--pm_en: Quantity - laser energy measured by power meter in glass reflection
|  |  |  |  |  |--sample_en: Quantity - laser energy at sample
|  |  |  |  |  |--param_val: List[Quantity] - value of independent parameters
|  |  |  |  |--raw_data: BaseData
|  |  |  |  |  |--data: Quantity - measured PA signal
|  |  |  |  |  |--data_raw: NDArray[int8] - measured PA signal in raw format
|  |  |  |  |  |--a: float - coef for coversion ``data_raw`` to ``data``: <data> = a*<data_raw> + b 
|  |  |  |  |  |--b: float - coef for coversion ``data_raw`` to ``data``: <data> = a*<data_raw> + b
|  |  |  |  |  |--x_var_step: Quantity - single step for x data
|  |  |  |  |  |--x_var_start: Quantity - start value for x data
|  |  |  |  |  |--x_var_stop: Quantity - stop value for x data
|  |  |  |  |  |--x var name: str - name of the x variable
|  |  |  |  |  |--y var name: str - name of the Y variable
|  |  |  |  |  |--max_amp: Quantity - max(data) - min(data)
|  |  |  |  |--filt_data: ProcessedData
|  |  |  |  |  |--data: Quantity - filtered PA signal
|  |  |  |  |  |--x_var_step: Quantity - single step for x data
|  |  |  |  |  |--x_var_start: Quantity - start value for x data
|  |  |  |  |  |--x_var_stop: Quantity - stop value for x data
|  |  |  |  |  |--x var name: str - name of the x variable
|  |  |  |  |  |--y var name: str - name of the Y variable
|  |  |  |  |  |--max_amp: Quantity - max(data) - min(data)
|  |  |  |  |--freq_data: ProcessedData
|  |  |  |  |  |--data: Quantity - Fourier transform of PA signal
|  |  |  |  |  |--x_var_step: Quantity - single step for x data
|  |  |  |  |  |--x_var_start: Quantity - start value for x data
|  |  |  |  |  |--x_var_stop: Quantity - stop value for x data
|  |  |  |  |  |--x var name: str - name of the x variable
|  |  |  |  |  |--y var name: str - name of the Y variable
|  |  |  |  |  |--max_amp: Quantity - max(data) - min(data)
|  |  |  |--point002
|  |  |  |  |--attrs: PointMetadata
|  |  |  |  ...
|  |--measurement001: Measurement  
|  |  |--attrs: MeasurementMetadata
|  ...


------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from typing import Iterable, Any, Tuple, TypeVar, Type, Self
from dataclasses import fields, field
from datetime import datetime
import time
import math
import os, os.path
import logging
from itertools import zip_longest

from scipy.fftpack import rfft, irfft, fftfreq
import pint
from pint.facets.plain.quantity import PlainQuantity
import h5py
import numpy as np
import numpy.typing as npt

from . import Q_
from .data_classes import (
    FileMetadata,
    BaseData,
    ProcessedData,
    PointMetadata,
    DataPoint,
    MeasurementMetadata,
    Measurement,
    Position,
    OscMeasurement,
    PaEnergyMeasurement,
    MeasuredPoint,
    MapData,
    ScanLine,
    PlotData,
    PointIndex
)
from modules.exceptions import (
    PlotError
)

logger = logging.getLogger(__name__)

T =TypeVar('T')

class PaData:
    """Class for PA data storage and manipulations"""

    VERSION = 1.4

    def __init__(self) -> None:

        #general metadata
        self.attrs = FileMetadata(
            version=self.VERSION,
            created=self._get_cur_time(),
            updated=self._get_cur_time(),
        )
        self.measurements: dict[str, Measurement] = {}
        self.maps: dict[str, MapData] = {}
        self.changed = False
        "Data changed flag"
        logger.debug('PaData instance created')

    @staticmethod
    def create_measurement(
            dims: int,
            params: list[str]
    ) -> Measurement:
        """Create measurement, but do not add it to data."""

        metadata = MeasurementMetadata(
            measurement_dims = dims,
            parameter_name = params.copy(),
            created = PaData._get_cur_time(),
            updated = PaData._get_cur_time(),
        )
        return Measurement(metadata)

    def append_measurement(
            self,
            msmnt: Measurement
        ) -> tuple[str, Measurement]:
        """Append msmnt to data."""

        title = self._build_name(
            self.attrs.measurements_count + 1,
            'measurement'
        )
        self.measurements.update({title: msmnt})
        self.attrs.updated = self._get_cur_time()
        self.attrs.measurements_count += 1
        self.changed = True
        logger.debug(f'{title} created.')
        return (title, msmnt)

    def add_measurement(
            self,
            dims: int,
            params: list[str]
        ) -> tuple[str, Measurement]|None:
        """
        Create a measurement, where data points will be stored.
        
        ``dims`` - dimensionality of the measurements.\n
        ``params`` - Variable title for each dimension.\n
        Created measurement is added to ``measurements`` attribute
        and returned.
        """

        measurement = self.create_measurement(dims, params)
        if measurement is None:
            return
        return self.append_measurement(measurement)

    def add_map(
            self,
            map: MapData
        ) -> None:
        """Add map data."""

        # Create measurement
        title, msmnt = self.add_measurement(2, ['line', 'point']) # type: ignore
        msmnt.attrs.center = map.centp
        msmnt.attrs.width = map.width
        msmnt.attrs.height = map.height
        msmnt.attrs.hpoints = map.hpoints
        msmnt.attrs.vpoints = map.vpoints
        msmnt.attrs.scan_dir = map.scan_dir
        msmnt.attrs.scan_plane = map.scan_plane
        msmnt.attrs.wavelength = map.wavelength
        # Add scan to list of scans
        self.maps[title] = map
        # Add all scanned points
        for lno, line in enumerate(map.data):
            for pno, point in enumerate(line.raw_data):
                self.add_point(
                    measurement = msmnt,
                    data = [point],
                    param_val = [Q_(lno, ''), Q_(pno, '')]
                )
                # Set correct created time for measurement
                if lno == 0 and pno == 0:
                    msmnt.attrs.created = self._get_time(point.datetime)

    @staticmethod
    def add_point(
            measurement: Measurement,
            data: list[MeasuredPoint],
            param_val: list[PlainQuantity]
        ) -> None:
        """
        Add datapoint to measurement.
        
        Replace already measured point with similar `param_val`
        """

        logger.debug('Starting datapoint addition to measurement...')
        if (exist_point:=PaData.point_by_param(measurement, param_val)) is not None:
            for key, val in measurement.data.items():
                if val == exist_point:
                    title = key
                    new_point = False
                    break
        else:
            title = PaData._build_name(measurement.attrs.data_points + 1)
            new_point = True
        if not title:
            logger.error('Max data_points reached! Data cannot be added!')
            return None
        logger.debug(f'{title=}')
        # ensure that units of parameter and data for new point
        # is the same as units of already present points
        if measurement.attrs.data_points:
            # get paramater value from any existing datapoint
            exist_params = next(iter((measurement.data.values()))).attrs.param_val
            # in 0D case param value can be missing
            if exist_params:
                param_val = [x.to(y.u) for x,y in zip(param_val,exist_params)]
            # get data from any existing datapoint
            exist_dp = next(iter(measurement.data.values()))
            exist_data =next(iter(exist_dp.raw_data.values())).data
            for dp in data:
                if exist_data.u != dp.pa_signal.u:
                    logger.debug(f'Changing units of data from {dp.pa_signal.u} '
                                +f'to {exist_data.u}')
                    dp.pa_signal = dp.pa_signal.to(exist_data.u)
        
        metadata = PointMetadata(
            wavelength = data[0].wavelength,
            pos = data[0].pos,
            param_val = param_val.copy(),
            repetitions = len(data)
        )
        datapoint = DataPoint(attrs = metadata)
        for ind, dp in enumerate(data):
            dp_title = PaData._build_name(ind, 'sample')
            rawdata = BaseData(
                data = dp.pa_signal.copy(), # type: ignore
                data_raw = dp.pa_signal_raw.copy(),
                pm_en = dp.pm_energy,
                sample_en = dp.sample_en,
                datetime = dp.datetime,
                yincrement = dp.yincrement,
                max_amp = dp.pa_signal.ptp(), # type: ignore
                x_var_step = dp.dt,
                x_var_start = dp.start_time,
                x_var_stop = dp.stop_time,
            )
            datapoint.raw_data.update({dp_title: rawdata})
            filtdata, freqdata = PaData.bp_filter(rawdata)
            datapoint.filt_data.update({dp_title: filtdata})
            datapoint.freq_data.update({dp_title: freqdata})

        measurement.data.update({title: datapoint})
        # update some technical attributes
        if len(rawdata.data) > measurement.attrs.max_len: # type: ignore
            measurement.attrs.max_len = len(rawdata.data) # type: ignore
        if new_point:
            measurement.attrs.data_points += 1
        measurement.attrs.updated = PaData._get_cur_time()
        logger.debug('...Finishing data point addition to measurement.')

    @staticmethod
    def _get_cur_time() -> str:
        """Return timestamp of current time."""
        
        cur_time = time.time()
        date_time = datetime.fromtimestamp(cur_time)
        return PaData._get_time(date_time)

    @staticmethod
    def _get_time(date_time: datetime) -> str:
        return date_time.strftime("%d-%m-%Y, %H:%M:%S")

    def save(self, filename: str) -> None:
        """Save data to file."""

        self._flush(filename)

    def save_tmp(self) -> None:
        """Save current data to TmpData.hdf5."""

        logger.debug('Starting tmp save...')
        base = os.getcwd()
        filename = os.path.join(base, 'measuring results', 'TmpData.hdf5')
        self._flush(filename)
        logger.debug('...Finishing tmp save.')

    def _flush(self, filename: str) -> None:
        """
        Write data to disk.
        
        Processed data (filt and freq) is not saved.
        """

        logger.debug(f'Start writing data to {filename}')
        with h5py.File(filename, 'w') as file:
            # set file metada
            file.attrs.update(self._to_dict(self.attrs))
            # set all measurements
            for msmnt_title, msmnt in self.measurements.items():
                s_msmnt = file.create_group(msmnt_title)
                # set measurement metadata
                s_msmnt.attrs.update(self._to_dict(msmnt.attrs))
                # save datapoints in each measurement
                for ds_name, ds in msmnt.data.items():
                    s_datapoint = s_msmnt.create_group(ds_name)
                    # set point metadata
                    s_datapoint.attrs.update(self._to_dict(ds.attrs))
                    # set raw data
                    for sample_t, sample in ds.raw_data.items():
                        new_dataset = s_datapoint.create_dataset(
                            sample_t,
                            data = sample.data_raw
                        )
                        new_dataset.attrs.update(self._to_dict(sample))
                        # additionaly save units of data
                        new_dataset.attrs.update({'y_var_units': str(sample.data.u)})
        self.changed = False
        logger.debug('Data saved to disk')

    def _to_dict(self, obj: object) -> dict:
        """"
        Convert dataclass to dict of basic types.
        
        ``data`` and ``data_raw`` attributes are excluded.
        """

        result = {}
        for fld in fields(obj): # type: ignore
            if fld.name not in ('data','data_raw'):
                value = getattr(obj, fld.name)
                # each quantity is saved as magnitude and unit
                if fld.type == PlainQuantity:
                    result.update(
                        {
                            fld.name: value.m,
                            fld.name + '_u': str(value.u)
                        }
                    )
                # lists must be unpacked, as they can contain quantities
                elif fld.type == list[PlainQuantity]:
                    # save only non-empty list
                    if len(value):
                        mags = [x.m for x in value]
                        units = [str(x.u) for x in value]
                        result.update(
                            {
                                fld.name: mags,
                                fld.name + '_u': units
                            }
                        )
                # Spetial case for Position
                elif fld.type == Position:
                    result.update(value.serialize(fld.name))
                # Special case for datetime
                elif fld.type == datetime:
                    result.update({
                        fld.name: str(getattr(obj, fld.name))
                        })
                # Otherwise it should be basic type
                else:
                    result.update({fld.name: value})
        return result

    def _from_dict(self, dtype: Type[T], source: dict) -> T:
        """
        Generate dataclass from a dictionary.
        
        Intended for use in loading data from file.\n
        Attributes
        ----------
        ``dtype`` - dataclass, which instance should be created.
        ``source`` - dictionary with the data to load.
        """

        init_vals = {}
        for fld in fields(dtype): # type: ignore
            value = None
            # Iterate through possible types. Special cases should be
            # handled, when a field is serialized into several records. 
            if fld.type == list[PlainQuantity]:
                # Try to construct quantities only for non-empty lists
                if len(source.get(fld.name, [])):
                    mags = source.get(fld.name, None)
                    units = source.get(fld.name + '_u', None)
                    if mags is not None and units is not None:
                        value = [Q_(m, u) for m, u in zip(mags, units)]
            elif fld.type == PlainQuantity:
                try:
                    value = Q_(source[fld.name], source[fld.name + '_u'])
                except:
                    logger.debug(f'Error in data loading for {fld.name}.')
            # Special case for Position
            elif fld.type == Position:
                value = Position.from_dict(data = source, prefix = fld.name)
            # Special case for datetime
            elif fld.type == datetime:
                iso_format = source.get(fld.name, None)
                if iso_format is not None:
                    value = datetime.fromisoformat(iso_format)
            # Otherwise values is one of basic types
            else:
                value = source.get(fld.name, None)
            if value is not None:
                init_vals.update({fld.name: value})
        return dtype(**init_vals)

    def _load_basedata(
            self,
            data_raw: npt.NDArray[np.uint8],
            source: dict
        ) -> BaseData:
        """"Load BaseData."""

        init_vals: dict[str, Any] = {
            'data_raw': data_raw
        }
        for fld in fields(BaseData):
            # this condition check is different from others
            if fld.name not in ['data', 'data_raw']:
                value = None
                if fld.type == list[PlainQuantity]:
                    mags = source[fld.name]
                    units = source[fld.name + '_u']
                    value = [Q_(m, u) for m, u in zip(mags, units)]
                elif fld.type == PlainQuantity:
                    try:
                        value = Q_(source[fld.name], source[fld.name + '_u'])
                    except:
                        logger.debug(f'Error in data loading for {fld.name}.')
                elif fld.type == datetime:
                    iso_format = source.get(fld.name, None)
                    if iso_format is not None:
                        value = datetime.fromisoformat(iso_format)
                else:
                    value = source[fld.name]
                if value is not None:
                    init_vals.update({fld.name: value})
                    # Support for file version < 1.4
        if init_vals.get('datetime', None) is None:
            init_vals['datetime'] = datetime.fromtimestamp(0)
        init_vals['data'] = (
            data_raw*init_vals['yincrement']/init_vals['sample_en']
        )
        return BaseData(**init_vals)

    def _calc_data_fit(self,
                       data: PlainQuantity,
                       data_raw: npt.NDArray[np.uint8],
                       points: int=20
        ) -> Tuple[float,float]:
        """
        Calculate coefs to convert ``data_raw`` to ``data``.
        
        Return (a,b), where ``data`` = a*``data_raw`` + b.\n
        ``points`` - amount of points used for fitting.
        """

        DIFF_VALS = 5

        #check if data_raw contain at least DIFF_VALS values
        if len(np.unique(data_raw)) < DIFF_VALS:
            logger.warning(f'Datapoint has less than {DIFF_VALS}'
                           + 'different values!')
            return (0,0)

        #in the begining of data there could be long baseline
        #therefore start searching for meaningfull data from maximum
        max_ind = np.flatnonzero(data==data.max())[0] # type: ignore
        x = data_raw[max_ind:max_ind+points]
        if len(np.unique(x))< DIFF_VALS:
            i = 0
            for i in range(len(data_raw)):
                stop_ind = max_ind+points+i
                x = data_raw[max_ind:stop_ind].copy()
                if len(np.unique(x)) == DIFF_VALS:
                    break
            logger.debug(f'{i} additional points added to find '
                        + f'{DIFF_VALS} unique values.')
        y = [quantity.m for quantity, _ in zip(data[max_ind:], x)] # type: ignore
        coef = np.polyfit(x, y, 1)
        return coef # type: ignore

    def load(self, filename: str, *args, **kwargs) -> None:
        """Load data from file."""

        logger.debug('load procedure is starting...')
        logger.debug(f'{filename=}')
        with h5py.File(filename, 'r') as file:
            #load file metadata
            general = file.attrs
            if ('version' not in general or
                (general.get('version', 0)) < 1.2):
                logger.warning('File has old structure and cannot be loaded.')
                return

            self.attrs = self._from_dict(
                FileMetadata,
                dict(file.attrs.items())
            )
            # load all measurements from the file
            self.measurements = {}
            for msmnt_title, msmnt in file.items():
                msm_attrs = self._from_dict(
                    MeasurementMetadata,
                    dict(msmnt.attrs.items())
                )
                measurement = Measurement(attrs = msm_attrs, data = {})
                # load all datapoints from the measurement
                for datapoint_title, datapoint in msmnt.items():
                    point_attrs = self._from_dict(
                        PointMetadata,
                        dict(datapoint.attrs.items())
                    )
                    raw_data = {}
                    filt_data = {}
                    freq_data = {}
                    for sample_t, sample in datapoint.items():
                        data_raw = sample[...]
                        # restore data
                        base_attrs = dict(sample.attrs.items())
                        basedata = self._load_basedata(
                            data_raw,
                            base_attrs
                        )
                        filtdata, freqdata = self.bp_filter(basedata)
                        raw_data.update({sample_t: basedata})
                        filt_data.update({sample_t: filtdata})
                        freq_data.update({sample_t: freqdata})
                    dp = DataPoint(
                        attrs = point_attrs,
                        raw_data = raw_data,
                        freq_data = freq_data,
                        filt_data = freq_data
                    )
                    measurement.data.update({datapoint_title: dp})
                self.measurements.update({msmnt_title: measurement})
                # For scan data
                if measurement.attrs.measurement_dims == 2:
                    scan = MapData.from_measmd(measurement.attrs)
                    line_no = 0
                    line_points: list[MeasuredPoint] = []
                    for _, point in measurement.data.items():
                        cur_lino_no = point.attrs.param_val[0]
                        if cur_lino_no == line_no:
                            line_points.append(
                                MeasuredPoint.from_datapoint(point)
                            )
                        else:
                            line = scan.create_line()
                            if line is not None:
                                line.add_measurements(line_points)
                                scan.add_line(line)
                            line_no += 1
                            line_points = [MeasuredPoint.from_datapoint(point)]
                    # add last line
                    line = scan.create_line()
                    if line is not None:
                        line.add_measurements(line_points)
                        scan.add_line(line)
                    self.maps.update({msmnt_title: scan})
     
    @staticmethod
    def _build_name(n: int, name: str='point') -> str:
        """
        Build and return name.
        
        ``n`` - index.\n
        ``name`` - base name.
        """
        
        return name + f'{n:05d}'
    
    @staticmethod
    def param_data_plot(
            msmnt: Measurement,
            dtype: str='filt_data',
            value: str='max_amp'
        ) -> PlotData:
        """Get main data for plotting.
        
        ``msmnt`` - measurement, from which to plot data.\n
        ``type`` - type of data to return can be 'filt_data' or 'raw_data'\n
        ``value`` - property of the data to be represented.\n
        Return a tuple, which contain [ydata, xdata, ylabel, xlabel].
        """
        
        xdata = PaData.get_dependance(
            msmnt=msmnt,
            data_type=dtype,
            value = 'param_val')
        if xdata is None:
            err_msg = 'Cannot get param_val for all datapoints.'
            logger.debug(err_msg)
            raise PlotError(err_msg)

        ydata = PaData.get_dependance(
            msmnt=msmnt,
            data_type=dtype,
            value = value)
        if ydata is None:
            err_msg = f'Cant get {value} of {dtype}.'
            logger.debug(err_msg)
            raise PlotError(err_msg)
        
        x_label = msmnt.attrs.parameter_name[0]

        point = next(iter(msmnt.data.values()))
        sample = next(iter(getattr(point, dtype).values())) 
        y_label = sample.y_var_name
        result = PlotData(
            ydata = ydata[0],
            yerr = ydata[1].to(ydata[0].u),
            xdata = xdata[0],
            xerr = xdata[1].to(xdata[0].u),
            ylabel = y_label,
            xlabel = x_label
        )
        return result

    @staticmethod
    def point_data_plot(
            point: DataPoint,
            dtype: str,
            sample: str
        ) -> PlotData:
        """
        Get point data for plotting.
        
        Attributes
        ----------
        ``index`` - index of data to get the data point.\n
        ``dtype`` - type of data to be returned.\n
        ``dstart`` and ``dstop`` limit plot range.\n
        ``sample`` and ``dstop`` limit plot range.\n
        Return a tuple, which contain [ydata, xdata, ylabel, xlabel].
        """

        dtype_dct = getattr(point, dtype)
        ds = dtype_dct.get(sample, None)
        if ds is None:
            logger.warning(f'There is no {sample} for {dtype}')
        result = PaData._point_data(ds)
        return result

    @staticmethod
    def _point_data(
            ds: BaseData|ProcessedData
        ) -> PlotData:
        """
        Get point data for plotting.
        
        ``ds`` - dataset for plotting.\n
        ``attrs`` - dict with attributes of group containing ``ds``.\n
        ``dstart`` and ``dstop`` limit plot range.\n
        Return a tuple, which contain [ydata, ydata, ylabel, xlabel].
        """
        
        start = ds.x_var_start
        stop = ds.x_var_stop
        step = ds.x_var_step
        num = len(ds.data) # type: ignore
        time_data = Q_(np.linspace(start.m,stop.m,num), start.u)

        x_label = ds.x_var_name
        y_label = ds.y_var_name
        return PlotData(
            xdata = time_data,
            ydata = ds.data,
            xlabel = x_label,
            ylabel = y_label
        )

    @staticmethod
    def get_dependance(
            msmnt: Measurement,
            data_type: str, 
            value: str
        ) -> tuple[PlainQuantity, PlainQuantity] | None:
        """
        Get mean and str of ``value`` for DataPoints in measurement.
        
        Attributes
        ----------
        ``msmnt`` - measurement, from which to read the data.\n
        ``data_type`` - type of data, can have 3 values 
        'raw_data'|'filt_data|'freq_data'.\n
        ``value`` - name of the attribute.

        Return
        ------
        A tuple with arrays of mean and std values.
        """

        logger.debug(f'Start building array of {value}.')
        mean_vals = []
        std_vals = []
        if not msmnt.attrs.data_points:
            logger.error(f'Measurement contains no data points.')
            return None
        
        for i in range(msmnt.attrs.data_points):
            # explicitly call datapoint by names to preserve order
            dp_name = PaData._build_name(i+1)
            dp = msmnt.data.get(dp_name)
            if dp is None:
                logger.error(f'Datapoint {dp_name} is missing')
                return
            # First check if requested data is in PointMetaData
            if value in [fld.name for fld in fields(PointMetadata)]:
                quant = getattr(dp.attrs, value)
                # This is bad. We imply here that the only list is
                # param_val, and it's value is only relevant for curve.
                if isinstance(quant, list):
                    quant = quant[0]
                mean_vals.append(quant)
                std_vals.append(Q_(0, quant.u))
            # Otherwise requested value should be calculated from samples
            else:
                groupd = getattr(dp, data_type) # dict
                all_vals = []
                for repeat in groupd.values():
                    all_vals.append(getattr(repeat, value))
                all_vals = pint.Quantity.from_list(all_vals)
                mean_vals.append(all_vals.mean()) # type: ignore
                std_vals.append(all_vals.std()) # type: ignore
        return (
            pint.Quantity.from_list(mean_vals),
            pint.Quantity.from_list(std_vals)
        )

    @staticmethod
    def point_by_param(
        msmnt: Measurement,
        param: list | PointIndex
        ) -> DataPoint | None:
        """Get datapoint from a measurement by its param value."""

        if isinstance(param, PointIndex):
            param = list(param)
        for point in msmnt.data.values():
            if point.attrs.param_val == param:
                return point
        logger.debug(f'Point for param val {param} not found.')
        return None
    
    @staticmethod
    def bp_filter(
            data: BaseData,
            low: PlainQuantity=Q_(1, 'kHz'),
            high: PlainQuantity=Q_(10, 'MHz')
        ) -> tuple[ProcessedData, ProcessedData]:
        """
        Perform bandpass filtration.
        
        ``data`` - structure, containing data to filter along with metadata.\n
        ``low`` and ``high`` - frequency ranges for the bandpass filtration.\n
        Return tuple, containing (filt_data, freq_data).
        """

        logger.debug(f'Starting FFT ' 
                     + f'with bp filter ({low}:{high})...')
        dt = data.x_var_step.to('s').m
        low = low.to('Hz').m
        high = high.to('Hz').m
        logger.debug(f'{dt=}')
        # array with freqs
        W = fftfreq(len(data.data.m), dt)
        # signal in f-space
        f_signal = rfft(data.data.m)

        filtered_f_signal = f_signal.copy()
        # kind of bad high pass filtering
        filtered_f_signal[(W<low)] = 0

        if high > 1/(2.5*dt): # Nyquist frequency check
            filtered_f_signal[(W>1/(2.5*dt))] = 0 
        else:
            filtered_f_signal[(W>high)] = 0

        #pass frequencies
        filtered_freq = W[(W>low)*(W<high)]
        filtered_data = f_signal[(W>low)*(W<high)]
        freq_data = ProcessedData(
            data = Q_(filtered_data, data.data.u),
            x_var_step = Q_((filtered_freq[1]-filtered_freq[0]), 'Hz'),
            x_var_start = Q_(filtered_freq.min(), 'Hz'),
            x_var_stop = Q_(filtered_freq.max(), 'Hz'),
            x_var_name= 'Frequency',
            y_var_name= 'FFT amplitude',
            max_amp = filtered_data.ptp()
        )
        logger.debug(f'freq step: {freq_data.x_var_step}')
        logger.debug(f'FFT amplitude: {freq_data.max_amp}')

        filt_signal = Q_(irfft(filtered_f_signal), data.data.u)
        filt_data = ProcessedData(
            data = filt_signal,
            x_var_step = data.x_var_step,
            x_var_start = data.x_var_start,
            x_var_stop = data.x_var_stop,
            x_var_name= data.x_var_name,
            y_var_name= 'Filtered photoAcoustic signal',
            max_amp = filt_signal.ptp() # type: ignore
        )
        logger.debug(f'Filtered signal amplitude = {filt_data.max_amp}')
        return (filt_data,freq_data)
    
    def __eq__(self, __value: Self) -> bool:
        """Compare creation time."""
        if self.attrs.created == __value.attrs.created:
            return True
        else:
            return False
    
    def __ne__(self, __value: Self) -> bool:
        """Compare creation time."""
        return not self.__eq__(__value)