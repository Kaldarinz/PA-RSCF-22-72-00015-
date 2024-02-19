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
    ScanLine
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
                    data = point,
                    param_val = [Q_(lno, ''), Q_(pno, '')]
                )
                # Set correct created time for measurement
                if lno == 0 and pno == 0:
                    msmnt.attrs.created = self._get_time(point.datetime)

    @staticmethod
    def add_point(
            measurement: Measurement,
            data: MeasuredPoint,
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
            exist_data = next(iter(measurement.data.values())).raw_data.data
            if exist_data.u != data.pa_signal.u:
                logger.debug(f'Changing units of data from {data.pa_signal.u} '
                             +f'to {exist_data.u}')
                data.pa_signal = data.pa_signal.to(exist_data.u)
        
        metadata = PointMetadata(
            pm_en = data.pm_energy,
            sample_en = data.sample_en,
            datetime = data.datetime,
            wavelength = data.wavelength,
            pos = data.pos,
            param_val = param_val.copy()
        )
        
        rawdata = BaseData(
            data = data.pa_signal.copy(), # type: ignore
            data_raw = data.pa_signal_raw.copy(),
            yincrement = data.yincrement,
            max_amp = data.pa_signal.ptp(), # type: ignore
            x_var_step = data.dt,
            x_var_start = data.start_time,
            x_var_stop = data.stop_time,
        )
        filtdata, freqdata = PaData.bp_filter(rawdata)
        datapoint = DataPoint(
            attrs = metadata,
            raw_data = rawdata,
            filt_data = filtdata,
            freq_data = freqdata
        )
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

    def export_txt(
            self,
            group_name: str,
            filename: str,
            full: bool=False) -> None:
        """Export all data points to txt format.
        
        <group_name> is 'raw_data'|'filt_data'|'freq_data'.
        Export waveforms from <group_name> to <filname> file in 
        'measuring results' folder.
        The whole waveform is exported if <fool> flag is set,
        otherwise only zoomed data will be exported.
        """

        logger.debug(f'Start export procedure from {group_name}.')

        group = getattr(self, group_name, None)
        if group is None:
            logger.warning(f'{group_name} is missing in PaData class.')
            return
        if len(self.attrs['parameter_name']) > 1:
            logger.warning('Export to txt is supported only for 0D and 1D data.')
            return
        data = []
        header = ('# 1st line: name of data. 2nd line: units of data.'
                  + '3rd line: paramter name. 4th line: parameter units.'
                  + '5th line: excitation energy units. 6th line: '
                  + 'parameter value. 7th line: excitation energy value.'
                  + 'following lines: data.\n')
        for ds_name, datapoint in group.items():
            if ds_name == 'attrs':
                continue
            logger.debug(f'Building list with x data for {ds_name}')
            col_x = []
            col_x.append(group['attrs']['x_var_name'])
            col_x.append(datapoint['x_var_step'].u)
            param_name = self.attrs['parameter_name'][0]
            col_x.append(param_name)
            param_units = datapoint['param_val'][0].u
            param_val = datapoint['param_val'][0].m
            col_x.append(param_units)
            energy_units = datapoint['sample_en'].u
            energy_val = datapoint['sample_en'].m
            col_x.append(energy_units)
            col_x.append(param_val)
            col_x.append(energy_val)
            
            col_y = []
            col_y.append(group['attrs']['y_var_name'])
            col_y.append(datapoint['data'][0].u)
            col_y.append(param_name)
            col_y.append(param_units)
            col_y.append(energy_units)
            col_y.append(param_val)
            col_y.append(energy_val)

            if group_name in ('raw_data', 'filt_data') and not full:
                max_y = np.amax(datapoint['data'])
                max_ind = np.flatnonzero(datapoint['data']==max_y)[0]
                x_step = datapoint['x_var_step']
                pre_time = self.attrs['zoom_pre_time']
                post_time = self.attrs['zoom_post_time']
                pre_points = int(pre_time.to(x_step.u).m/x_step.m)
                post_points = int(post_time.to(x_step.u).m/x_step.m)
                start_zoom_ind = max_ind-pre_points
                if start_zoom_ind < 0:
                    start_zoom_ind = 0
                t_points = int((datapoint['x_var_stop']
                                - datapoint['x_var_start'])/x_step)
                stop_zoom_ind = max_ind + post_points
                if stop_zoom_ind > t_points:
                    stop_zoom_ind = t_points
                for i in range(start_zoom_ind, stop_zoom_ind):
                    col_x.append((datapoint['x_var_start']+i*x_step).m)
                    col_y.append(datapoint['data'][i].m)
            else:
                x = datapoint['x_var_start'].m
                col_x.append(x)
                while x < (datapoint['x_var_stop'].m - datapoint['x_var_step'].m):
                    x += datapoint['x_var_step'].m
                    col_x.append(x)
                col_y.extend((val.m for val in datapoint['data']))

            data.append(col_x)
            data.append(col_y)
        
        with open(filename, 'w') as file:
            file.write(header)
            for row in zip_longest(*data, fillvalue=''):
                file.write(';'.join(map(str,row)) + '\n')
    
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
                    new_dataset = s_datapoint.create_dataset(
                        'data_raw',
                        data = ds.raw_data.data_raw
                    )
                    new_dataset.attrs.update(self._to_dict(ds.raw_data))
                    
                    # additionaly save units of data
                    new_dataset.attrs.update({'y_var_units': str(ds.raw_data.data.u)})
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
                if len(source.get(fld.name, None)):
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
            # Support for file version < 1.4
            if dtype is PointMetadata and init_vals.get('datetime', None) is None:
                init_vals['datetime'] = datetime.fromtimestamp(0)
        return dtype(**init_vals)

    def _load_basedata(
            self,
            data: PlainQuantity,
            data_raw: npt.NDArray[np.uint8],
            source: dict
        ) -> BaseData:
        """"Load BaseData."""

        init_vals = {
            'data': data,
            'data_raw': data_raw
        }
        for key, val in BaseData.__annotations__.items():
            # this condition check is different from others
            if key not in ['data', 'data_raw']:
                # load old data format
                if self.attrs.version < 1.4:
                    if key == 'yincrement':
                        item = Q_(np.nan, 'V')
                        init_vals.update({key: item})
                        continue
                if val == list[PlainQuantity]:
                    mags = source[key]
                    units = source[key + '_u']
                    item = [Q_(m, u) for m, u in zip(mags, units)]
                elif val is PlainQuantity:
                    item = Q_(source[key], source[key + '_u'])
                else:
                    item = source[key]
                init_vals.update({key: item})
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
                    ds = datapoint['data_raw']
                    data_raw = ds[...]
                    y_var_units = ds.attrs['y_var_units']
                    # restore data
                    if general.get('version', 0) < 1.4:
                        p = np.poly1d([ds.attrs['a'], ds.attrs['b']])
                        data = Q_(p(data_raw), y_var_units)
                    else:
                        data = data_raw * ds.attrs['yincrement'] / point_attrs.sample_en
                    basedata = self._load_basedata(
                        data,
                        data_raw,
                        dict(ds.attrs.items())
                    )
                    filtdata, freqdata = self.bp_filter(basedata)
                    dp = DataPoint(
                        point_attrs,
                        basedata,
                        filtdata,
                        freqdata
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

    def _load_old(self, file: h5py.File) -> None:
        """Load file with version < 1.2."""

        self.attrs = FileMetadata(
            version = file.attrs['version'], # type: ignore
            measurements_count = 1,
            created = file.attrs['created'], # type: ignore
            updated = file.attrs['updated'], # type: ignore
            notes = ''
        )
        self.measurements = {}

        raw_data = file['raw_data']
        msmnt_md = MeasurementMetadata(
            measurement_dims = file.attrs['measurement_dims'], # type: ignore
            parameter_name = file.attrs['parameter_name'], # type: ignore
            data_points = file.attrs['data_points'], # type: ignore
            max_len = raw_data.attrs['max_len'] # type: ignore
        )
        msmnt = Measurement(msmnt_md)
        self.measurements.update({'measurement001': msmnt})
        for ds_name, ds in raw_data.items(): # type: ignore
            data_raw = ds[...]
            y_var_units = raw_data.attrs['y_var_u']
            p = np.poly1d([ds.attrs['a']], ds.attrs['b'])
            data = Q_(p(data_raw), y_var_units) # type: ignore
            basedata = BaseData(
                data = data,
                data_raw = data_raw,
                a = ds.attrs['a'],
                b = ds.attrs['b'],
                max_amp = Q_(
                    ds.attrs['max_amp'],
                    ds.attrs['max_amp_u']
                ),
                x_var_step = Q_(
                    ds.attrs['x_var_step'],
                    raw_data.attrs['x_var_u'] # type: ignore
                ), # type: ignore
                x_var_start = Q_(
                    ds.attrs['x_var_start'],
                    raw_data.attrs['x_var_u'] # type: ignore
                ), # type: ignore
                x_var_stop = Q_(
                    ds.attrs['x_var_stop'],
                    raw_data.attrs['x_var_u'] # type: ignore
                ) # type: ignore
            )
            fildata, freqdata = self.bp_filter(basedata)
            p_md = PointMetadata(
                pm_en = Q_(ds.attrs['pm_en'], ds.attrs['pm_en_u']),
                sample_en = Q_(
                    ds.attrs['sample_en'],
                    ds.attrs['sample_en_u']
                ),
                param_val= [Q_(x,y) for x,y in zip(
                    ds.attrs['param_val'], file.attrs['parameter_u'])] # type: ignore
            )
            dp = DataPoint(
                p_md,
                basedata,
                fildata,
                freqdata
            )
            msmnt.data.update({ds_name:dp})
        
    @staticmethod
    def _build_name(n: int, name: str='point') -> str:
        """
        Build and return name.
        
        ``n`` - index.\n
        ``name`` - base name.
        """
        
        if n <10:
            n_str = '00' + str(n)
        elif n<100:
            n_str = '0' + str(n)
        elif n<1000:
            n_str = str(n)
        else:
            return ''
        return name + n_str
    
    @staticmethod
    def param_data_plot(
            msmnt: Measurement,
            dtype: str='filt_data',
            value: str='max_amp'
        ) -> tuple[npt.NDArray, npt.NDArray, str, str]:
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
        
        x_label = (msmnt.attrs.parameter_name[0] 
                + ', ['
                + f'{xdata.u:~.2gP}'
                + ']')
        y_label = (msmnt.data['point001'].raw_data.y_var_name
                + ', ['
                + f'{ydata.u:~.2gP}'
                + ']')

        return (ydata.m, xdata.m, y_label, x_label)

    @staticmethod
    def point_data_plot(
            msmnt: Measurement,
            index: int,
            dtype: str,
            dstart: PlainQuantity | None=None,
            dstop: PlainQuantity | None=None
        ) -> tuple[npt.NDArray, npt.NDArray, str, str]:
        """
        Get point data for plotting.
        
        ``index`` - index of data to get the data point.\n
        ``dtype`` - type of data to be returned.\n
        ``dstart`` and ``dstop`` limit plot range.\n
        Return a tuple, which contain [ydata, xdata, ylabel, xlabel].
        """

        empty_data = (np.array([]),np.array([]),'','')
        result = empty_data
        ds_name = PaData._build_name(index+1)

        ds = getattr(msmnt.data[ds_name], dtype)
        result = PaData._point_data(ds, dstart, dstop)
        return result

    @staticmethod
    def _point_data(
            ds: BaseData|ProcessedData,
            dstart: PlainQuantity | None=None,
            dstop: PlainQuantity | None=None
        ) -> tuple[npt.NDArray, npt.NDArray, str, str]:
        """
        Get point data for plotting.
        
        ``ds`` - dataset for plotting.\n
        ``attrs`` - dict with attributes of group containing ``ds``.\n
        ``dstart`` and ``dstop`` limit plot range.\n
        Return a tuple, which contain [ydata, ydata, ylabel, xlabel].
        """
        
        start = ds.x_var_start
        ### Need to finish it on real data.
        if dstart is not None and dstart.is_compatible_with(start):
            pass
        stop = ds.x_var_stop
        step = ds.x_var_step
        num = len(ds.data) # type: ignore
        time_data = Q_(np.linspace(start.m,stop.m,num), start.u)

        x_label = (ds.x_var_name
                   + ', ['
                   + str(start.u)
                   + ']')
        y_label = (ds.y_var_name
                   + ', ['
                   + str(ds.data.u)
                   + ']')
        return (ds.data.m, time_data.m, y_label, x_label)

        if zoom_ax is not None:
            #marker for max value
            max_val = np.amax(ds['data'])
            max_ind = np.flatnonzero(ds['data']==max_val)[0]
            max_t = time_data[max_ind] #type:ignore
            ax.plot(max_t.m, max_val.m, **self._marker_style)
            
            #marker for min value
            min_val = np.amin(ds['data'])
            min_ind = np.flatnonzero(ds['data']==min_val)[0]
            min_t = time_data[min_ind] #type:ignore
            ax.plot(min_t.m, min_val.m, **self._marker_style)
        
            #marker for zoomed area
            pre_time = self.attrs['zoom_pre_time']
            post_time = self.attrs['zoom_post_time']
            pre_points = int(pre_time.to(step.u).m/step.m)
            post_points = int(post_time.to(step.u).m/step.m)
            start_zoom_ind = max_ind-pre_points
            if start_zoom_ind < 0:
                start_zoom_ind = 0
            stop_zoom_ind = min_ind + post_points
            if stop_zoom_ind > (len(time_data.m) - 1):
                stop_zoom_ind = len(time_data.m) - 1
            ax.fill_betweenx(
                [min_val.m, max_val.m],
                time_data[start_zoom_ind].m, #type:ignore
                time_data[stop_zoom_ind].m, #type:ignore
                **self._fill_style
            )

            #plot zoomed
            zoom_ax.clear()
            # uncomment to change into absolute x_axis for zoomed adrea
            # z_time_off = time_data[start_zoom_ind].m #type:ignore
            # zoom_ax.plot(
            #     time_data[start_zoom_ind:stop_zoom_ind+1].m-z_time_off, #type:ignore
            #     ds['data'][start_zoom_ind:stop_zoom_ind+1].m
            # )
            zoom_ax.plot(
                time_data[start_zoom_ind:stop_zoom_ind+1].m,
                ds['data'][start_zoom_ind:stop_zoom_ind+1].m
            )
            zoom_ax.set_xlabel(x_label)
            zoom_ax.set_ylabel(y_label)
            zoom_ax.set_title('Zoom of ' + title)

    @staticmethod
    def get_dependance(
            msmnt: Measurement,
            data_type: str, 
            value: str
        ) -> PlainQuantity | None:
        """
        Return ``value`` for each dataPoint in a given measurement.
        
        ``msmnt`` - measurement, from which to read the data.\n
        ``data_type`` - type of data, can have 3 values 
        'raw_data'|'filt_data|'freq_data'.\n
        ``value`` - name of the attribute.
        """

        logger.debug(f'Start building array of {value}.')
        dep = [] #array for return values
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
            if value in PointMetadata.__annotations__.keys():
                if PointMetadata.__annotations__[value] == list[PlainQuantity]:
                    # temporary solution, fix later
                    dep.append(getattr(dp.attrs, value)[0])
                else:
                    dep.append(getattr(dp.attrs, value))
            else:
                groupd = getattr(dp, data_type)
                dep.append(getattr(groupd, value))
        # explicit units are required at least for 'nm'
        return pint.Quantity.from_list(dep, f'{dep[0].u:~}')

    @staticmethod
    def point_by_param(
        msmnt: Measurement,
        param: list
        ) -> DataPoint | None:
        """Get datapoint from a measurement by its param value."""

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