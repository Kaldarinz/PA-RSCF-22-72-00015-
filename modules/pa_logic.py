"""
PA backend.

All calls to hardware should be performed via corresponding actors (_stage_call, _osc_call).
"""

from typing import List, Tuple, Optional, cast, Literal
from dataclasses import fields
import logging
import os
from itertools import combinations
from collections import deque
import math
from threading import Thread
from dataclasses import replace
import time
from datetime import datetime as dt

import yaml
import pint
from pint.facets.plain.quantity import PlainQuantity
import numpy as np
import numpy.typing as npt
from pylablib.devices.Thorlabs import KinesisMotor
from pylablib.devices import Thorlabs

from .osc_devices import (
    Oscilloscope,
    PowerMeter,
    PhotoAcousticSensOlymp
)
from modules.exceptions import (
    OscConnectError,
    OscIOError,
    OscValueError,
    StageError
    )

from .data_classes import (
    WorkerSignals,
    MeasuredPoint,
    EnergyMeasurement,
    Coordinate,
    Actor,
    ActorFail,
    StagesStatus,
    Signals
)
from .constants import (
    Priority
)
from . import ureg, Q_

logger = logging.getLogger(__name__)

class Hardware():
    """Class for hardware references."""
    
    def __init__(self):
        self.osc: Oscilloscope = Oscilloscope()
        self.power_meter: PowerMeter | None = None
        self.pa_sens: PhotoAcousticSensOlymp | None = None
        self.stages: dict[str, KinesisMotor] = {}
        self.motor_axes: int = -1
        self.config: dict = {}

hardware = Hardware()

def _init_call() -> None:
    """Start actors for serial communication with hardware."""

    global _stage_call
    _stage_call = Actor()
    "Serial communication with stages."
    _stage_call.start()

    global _osc_call
    _osc_call = Actor()
    "Serial communication with oscilloscope."
    _osc_call.start()

def init_hardware(**kwargs) -> bool:
    """Initialize all hardware.
    
    Load hardware config from rsc/config.yaml if it was not done.\n
    Thread safe.
    """

    logger.info('Starting hardware initialization...')
    
    # Start Actors to communicate with hardware
    _init_call()
    # Try init oscilloscope.
    if not init_osc():
        logger.warning('Oscilloscope cannot be loaded!')
    else:
        logger.info('Oscilloscope initiated.')
    
    osc = hardware.osc
    # Try init stages
    if not init_stages():
        logger.warning('Stages cannot be loaded!')
    
    config = hardware.config
    pm = hardware.power_meter
    if pm is not None:
        #save pm channel to apply it after init
        pm_chan = pm.ch
        pre_time = float(config['power_meter']['pre_time'])*ureg.us
        post_time = float(config['power_meter']['post_time'])*ureg.us
        pm = PowerMeter(osc)
        pm.set_channel(pm_chan, pre_time, post_time)
        hardware.power_meter = pm
        logger.debug('Power meter reinitiated on the same channel')

    pa = hardware.pa_sens
    if pa is not None:
        #save pa channel to apply it after init
        pa_chan = pa.ch
        pre_time = float(config['pa_sensor']['pre_time'])*ureg.us
        post_time = float(config['pa_sensor']['post_time'])*ureg.us
        pa = PhotoAcousticSensOlymp(osc)
        pa.set_channel(pa_chan, pre_time, post_time)
        hardware.pa_sens = pa
        logger.debug('PA sensor reinitiated on the same channel')
                
    logger.debug('...Finishing hardware initialization.')
    return True

def close_hardware(**kwargs) -> None:
    """
    Cloase all hardware.
    
    Thread safe.
    """

    _stage_call.close(__close_stages)
    _osc_call.close()
    _stage_call.join()
    _osc_call.join()
    logger.info('Hardware communication terminated.')

def __close_stages() -> None:
    """
    Close request for all stages.
    
    Intended to be called by actor only.
    """

    for stage in hardware.stages.values():
        stage.close()

def load_config() -> dict:
    """Load configuration.

    Configuration is loaded from rsc/config.yaml.\n
    Additionally add all optional devices to hardware dict.\n
    Thread safe.
    """

    logger.debug('Starting loading config...')
    base_path = os.path.dirname(os.path.dirname(__name__))
    sub_dir = 'rsc'
    filename = 'config.yaml'
    full_path = os.path.join(base_path, sub_dir, filename)
    try:
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)['hardware']
    except:
        logger.warning('...Terminatin. Connfig cannot be properly loaded')
        return {}
    
    hardware.config = config
    # We need to know amount of axes before initiation of stages.
    axes = int(config['stages']['axes'])
    hardware.motor_axes = axes
    logger.debug('Config loaded into hardware["config"]')
    
    osc = hardware.osc
    if bool(config['power_meter']['connected']):
        pm = PowerMeter(osc)
        hardware.power_meter = pm
        pm_chan = int(config['power_meter']['connected_chan']) - 1
        pre_time = float(config['power_meter']['pre_time'])*ureg.us
        post_time = float(config['power_meter']['post_time'])*ureg.us
        pm.set_channel(pm_chan, pre_time, post_time)
        logger.debug(f'Power meter added to hardare list at CHAN{pm_chan+1}')
    if bool(config['pa_sensor']['connected']):
        pa = PhotoAcousticSensOlymp(osc)
        hardware.pa_sens = pa
        pa_chan = int(config['pa_sensor']['connected_chan']) - 1
        pre_time = float(config['pa_sensor']['pre_time'])*ureg.us
        post_time = float(config['pa_sensor']['post_time'])*ureg.us
        pa.set_channel(pa_chan, pre_time, post_time)
        logger.debug(f'PA sensor added to hardare list at CHAN{pa_chan+1}')
    logger.debug(f'...Finishing. Config file read.')
    return config

def init_stages() -> bool:
    """
    Initiate Thorlabs KDC based stages.
    
    Thread safe.\n
    Priority is high.
    """

    logger.debug(f'Starting...')
    logger.debug('Checking if connection to stages is '
                +'already estblished')

    if stages_open():
        logger.info('...Finishing. Connection to all stages already established!')
        return True

    logger.debug('Searching for Thorlabs kinsesis devices (stages)')
    stages = Thorlabs.list_kinesis_devices()
    logger.debug(f'{len(stages)} devices found')
    axes_count = hardware.motor_axes
    if len(stages) < axes_count:
        msg = f'Less than {axes_count} kinesis devices found!'
        logger.error(msg)
        logger.debug('...Terminating.')
        return False

    connected = True
    for stage_id, id, axes in zip(stages, range(axes_count), ['x', 'y', 'z']):
        #motor units [m]
        stage = Thorlabs.KinesisMotor(stage_id[0], scale='stage')
        logger.debug('Trying to call is_opened')
        test_con = _stage_call.submit(
            Priority.HIGH,
            stage.is_opened
        )
        if isinstance(test_con, ActorFail):
            msg = f'Failed attempt to coomunicate with stage {axes}'
            logger.error(msg)
            connected = False
        else:
            hardware.stages.update({axes: stage})
            logger.info(f'Stage {axes} with ID={stage_id} is initiated')
    
    if connected:
        logger.info('Stages initiated.')
    else:
        logger.warning('Stages are not initiated')
    
    logger.debug(f'...Finishing. Stages {connected=}')
    return connected

def osc_open(priority:int = Priority.NORMAL) -> bool|None:
    """
    Check connection to oscilloscope.
    
    Thread safe.
    """

    is_open = _osc_call.submit(
        priority = priority,
        func = hardware.osc.connection_check
    )
    if isinstance(is_open, ActorFail):
        return None
    return is_open

def init_osc(priority:int = Priority.NORMAL) -> bool:
    """Initialize oscilloscope.

    Return true if connection is already established or
    initialization is successfull.\n
    Thread safe.
    """
    osc = hardware.osc
    logger.debug('Starting init_osc...')
    is_connected = _osc_call.submit(
        priority = priority,
        func = osc.connection_check
    )
    if not isinstance(is_connected,ActorFail) and is_connected:
        logger.info('Connection to oscilloscope is already established!')
        logger.debug('...Finishing init_osc.')
        return True

    logger.debug('No connection found. Trying to establish connection')
    init = _osc_call.submit(
        priority = priority,
        func = osc.initialize
    )
    if not isinstance(init, ActorFail) and osc.initialize():
        logger.debug('...Finishing. Oscilloscope initialization complete')
        return True
    else:
        logger.warning(f'Attempt to initialize osc failed.')
        logger.debug('...Terminating.')
        return False       

def stages_open() -> bool:
    """Return True if all stages are responding and open.
    
    Never raise exceptions.\n
    Thread safe.\n
    Priority is high.
    """

    logger.debug('Starting connection check to stages...')
    connected = True

    if not len(hardware.stages):
        logger.debug('...Finishing. Stages are not initialized.')
        return False
    for axes, stage in hardware.stages.items():
        if stage is not None:
            is_open = _stage_call.submit(
                Priority.HIGH,
                stage.is_opened
            )
            if isinstance(is_open, ActorFail) or not is_open:
                logger.debug(f'stage {axes} is not open')
                connected = False
            else:
                logger.debug(f'stage {axes} is open')
    if connected:
        logger.debug('All stages are connected and open')
    logger.debug(f'...Finishing. stages {connected=}')
    return connected

def stages_status(**kwargs) -> StagesStatus:
    """
    Return status of all stages.
    
    Thread safe.
    Have low priority.
    """

    status = StagesStatus()
    for axes, stage in hardware.stages.items():
        status_lst = _stage_call.submit(
            Priority.LOW,
            stage.get_status
        )
        if not isinstance(status_lst, ActorFail):
            setattr(status, axes + '_status', status_lst)
        is_open = _stage_call.submit(
            Priority.LOW,
            stage.is_opened
        )
        if not isinstance(is_open, ActorFail):
            setattr(status, axes + '_open', is_open)
    return status

def stages_position(**kwargs) -> Coordinate:
    """
    Get position of all stages.
    
    Thread safe.
    Have low priority.
    """

    coord = Coordinate()
    for axes, stage in hardware.stages.items():
        pos = _stage_call.submit(
            Priority.LOW,
            stage.get_position
        )
        if not isinstance(pos, ActorFail):
            setattr(coord, axes, Q_(pos, 'm'))
    return coord

def pm_open() -> bool:
    """Return true if power meter is configured."""

    logger.debug('Starting power meter connection check...')

    if hardware.power_meter is None:
        logger.warning('Power meter is off in config file.')
        connected = False
    else:
        connected = hardware.osc.connection_check()
        logger.debug(f'...Finishing. Power meter {connected=}')
    return connected

def stage_jog(
        axes: Literal['x', 'y', 'z'],
        direction: Literal['+','-'],
        **kwargs,
    ) -> None:
    """
    Jog given axis in given direction.
    
    Thread safe.\n
    Priority is normal.
    """

    stage = hardware.stages.get(axes, None)
    if stage is None:
        logger.warning(f'Invalid axes ({axes}) for jogging.')
        return
    _stage_call.submit(
        Priority.NORMAL,
        stage.jog,
        direction
    )

def stage_stop(
        axes: Literal['x', 'y', 'z'],
        priority: int = Priority.NORMAL,
        **kwargs
    ) -> None:
    """
    Stop movement along given axes.
    
    Thread safe.\n
    Priority is normal.
    """

    stage = hardware.stages.get(axes, None)
    if stage is None:
        logger.warning(f'Invalid axes ({axes}) for stop.')
        return
    _stage_call.submit(
        priority,
        stage.stop,
        sync = False
    )

def break_all_stages(**kwargs) -> None:
    """
    Stop move for all axes and reset call stack for stages.
    
    Thread safe.\n
    Highest priority.
    """

    for axes in hardware.stages.keys():
        stage_stop(axes, Priority.HIGHEST) # type: ignore
    _stage_call.reset()

def move_to(new_pos: Coordinate, **kwargs) -> None:
    """Send motors to new position.
    
    Thread safe.\n
    Priority is normal.
    """

    logger.debug('Starting move_to procedure...')
    for axes, stage in hardware.stages.items():
        coord = getattr(new_pos, axes)
        if coord is not None:
            coord = coord.to('m').m
            _stage_call.submit(
                Priority.NORMAL,
                stage.move_to,
                position = coord
            )

def home(**kwargs) -> None:
    """
    Home all stages.
    
    Thread safe.\n
    Priority is normal.
    """

    logger.debug('Starting homing...')
    for axes, stage in hardware.stages.items():
        _stage_call.submit(
            Priority.NORMAL,
            stage.home,
            sync = False,
            force = True
        )

def en_meas_fast(
        priority:int = Priority.LOW,
        **kwargs
    ) -> EnergyMeasurement:
    """
    Measure laser energy.
    
    Thread safe.
    """

    pm = hardware.power_meter
    if pm is None:
        msg = 'Power meter is not connected in config.'
        logger.warning(msg)
        return EnergyMeasurement(dt.now())
    
    result = _osc_call.submit(
        priority,
        pm.get_energy_scr
    )
    if isinstance(result, ActorFail):
        return EnergyMeasurement(dt.now())
    logger.debug('Returning measured energy. '
                    +f'signal len = {len(result.signal.m)}')
    return result

def en_meas_fast_cont(
        signals: WorkerSignals,
        flags: dict[str, bool],
        priority: int = Priority.NORMAL,
        **kwargs
    ) -> list[EnergyMeasurement]:
    """
    Get screen information non-stop.

    Thread safe.
    """

    # Object for communication with lower level fucntion
    comm = Signals()
    # List with results
    result = []
    tkwargs = {
        'priority': priority,
        'func': __en_meas_fast_cont,
        'comm': comm,
        'result': result
    }
    t = Thread(target = _osc_call.submit, kwargs = tkwargs)
    t.start()
    while flags.get('is_running'):
        time.sleep(0.1)
    comm.is_running = False
    t.join()

    return result


def __en_meas_fast_cont(
        comm: Signals,
        result: list,
        timeout: float = 20,
    ) -> None:
    """
    Get screen information non-stop.
    
    ``comm`` - object for interthread communication.\n
    Measuring continue untill ``comm.is_running`` set to False,
    or ``timeout`` expires.\n
    Intended to be called by actor only.
    """

    pm = hardware.power_meter
    if pm is None:
        return
    start = time.time()
    # execution loop
    while comm.is_running:
        # exit by timeout
        if (time.time() - start) > timeout:
            break
        msmnt = pm.get_energy_scr()
        # Skip bad read
        if msmnt == EnergyMeasurement(dt.now()):
            continue
        # Add only unique measurements
        if len(result):
            if result[-1] != msmnt:
                result.append(msmnt)
        # Add first measurement
        else:
            result.append(msmnt)

def glass_calculator(
        wavelength: PlainQuantity,
        current_energy_pm: PlainQuantity,
        target_energy: PlainQuantity,
        max_combinations: int,
        threshold: float = 2.5,
        results: int = 5,
    ) -> dict:
    """Calculate filter combination to get required energy on sample.
    
    Return a dict with up to <results> filter combinations, 
    having the closest transmissions,
    which are higher than required but not more than <threshold> folds higher.
    Accept only wavelengthes, which are present in rsc/ColorGlass.txt.
    """

    logger.debug('Starting calculation of filter combinations...')
    #file with filter's properties
    sub_folder = 'rsc'
    filename = 'ColorGlass.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        data = np.loadtxt(filename,skiprows=1)
        header = open(filename).readline()
    except FileNotFoundError:
        logger.error('File with color glass data not found!')
        logger.debug('...Terminating.')
        return {}
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return {}
    
    _glass_rm_zeros(data)
    glass_calc_od(data)
    filter_titles = header.split('\n')[0].split('\t')[2:]

    try:
        wl_index = np.where(data[1:,0] == wavelength)[0][0] + 1
    except IndexError:
        logger.error('Target WL is missing in color glass data table!')
        logger.debug('...Terminating')
        return {}
    # calculated laser energy at sample
    laser_energy = current_energy_pm/data[wl_index,1]*100
    if laser_energy == 0:
        logger.error('Laser radiation is not detected!')
        logger.debug('...Terminating')
        return {}
    #required total transmission of filters
    target_transm = target_energy/laser_energy
    
    logger.info(f'Target sample energy = {target_energy}')
    logger.info(f'Current sample energy = {laser_energy}')
    logger.info(f'Target transmission = {target_transm*100:.1f} %')

    #build filter_dict = {filter_name: OD at wavelength}
    #find index of wavelength
    filters = {}
    for key, value in zip(filter_titles,data[wl_index,2:]):
        filters.update({key:value})

    filter_combinations = glass_combinations(filters, max_combinations)
    result_comb = glass_limit_comb(
        filter_combinations,
        target_transm,
        results,
        threshold
    )
    logger.info('Valid filter combinations:')
    for comb_name, transmission in result_comb.items():
        logger.info(f'{comb_name}: {transmission*100:.1f} %')

    logger.debug('...Finishing. Filter combinations calculated.')
    return result_comb

def _glass_rm_zeros(data: np.ndarray) -> None:
    """Replaces zeros in filters data by linear fit from nearest values."""

    logger.debug('Starting replacement of zeros in filter data...')
    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            if data[i+1,j+2] == 0:
                if i == 0:
                    if data[i+2,j+2] == 0 or data[i+3,j+2] == 0:
                        logger.warning('missing value for the smallest WL cannot be calculated!')
                    else:
                        data[i+1,j+2] = 2*data[i+2,j+2] - data[i+3,j+2]
                elif i == data.shape[0]-2:
                    if data[i,j+2] == 0 or data[i-1,j+2] == 0:
                        logger.warning('missing value for the smallest WL cannot be calculated!')
                    else:
                        data[i+1,j+2] = 2*data[i,j+2] - data[i-1,j+2]
                else:
                    if data[i,j+2] == 0 or data[i+2,j+2] == 0:
                        logger.warning('adjacent zeros in filter data are not supported!')
                    else:
                        data[i+1,j+2] = (data[i,j+2] + data[i+2,j+2])/2
    logger.debug('...Finishing. Zeros removed from filter data.')

def glass_calc_od(data: np.ndarray) -> None:
    """Calculate OD using thickness of filters."""

    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            data[i+1,j+2] = data[i+1,j+2]*data[0,j+2]

def glass_combinations(filters: dict, max_comb: int) -> dict:
    """Calc transmission of combinations up to <max_comb> filters."""

    filter_combinations = {}
    for i in range(max_comb):
        for comb in combinations(filters.items(),i+1):
            key = ''
            value = 0
            for k,v in comb:
                key +=k
                value+=v
            filter_combinations.update({key:math.pow(10,-value)})

    return filter_combinations

def glass_limit_comb(
        filter_combs: dict,
        target_transm: float,
        results: int,
        threshold: float
    ) -> dict:
    """Limit combinations of filters.
    
    Up to <results> with transmission higher than <target_transm>,
    but not higher than <target_transm>*<threshold> will be returned."""

    result = {}
    filter_combs = dict(sorted(
        filter_combs.items(),
        key=lambda item: item[1]))
    
    i=0
    for key, value in filter_combs.items():
        if (value-target_transm) > 0 and value/target_transm < threshold:
            result.update({key: value})
            i += 1
            if i == results:
                break

    return result

def glass_reflection(wl: PlainQuantity) -> Optional[float]:
    """Get reflection (fraction) from glass at given wavelength.
    
    pm_energy/sample_energy = glass_reflection."""

    logger.debug('Starting calculation of glass reflection...')
     #file with filter's properties
    sub_folder = 'rsc'
    filename = 'ColorGlass.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        data = np.loadtxt(filename,skiprows=1)
    except FileNotFoundError:
        logger.error('File with color glass data not found!')
        logger.debug('...Terminating.')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return
    
    try:
        wl_nm = wl.to('nm')
        wl_index = np.where(data[1:,0] == wl_nm.m)[0][0] + 1
    except IndexError:
        logger.warning('Target WL is missing in color glass data table!')
        logger.debug('...Terminating')
        return
    
    logger.debug('...Finishing. Glass reflection calculated')
    return data[wl_index,1]/100

def glan_calc_reverse(
        target_energy: PlainQuantity,
        fit_order: int=1
    ) -> PlainQuantity|None:
    """Calculate energy at power meter for given sample energy.
    
    It is assumed that power meter measures laser energy
    reflected from a thin glass.
    Calculation is based on callibration data from 'rsc/GlanCalibr'.
    <fit_order> is a ploynom order used for fitting callibration data.
    """

    logger.debug('Starting calculation of power meter '
                 + 'energy from sample energy...')
    sub_folder = 'rsc'
    filename = 'GlanCalibr.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        calibr_data = np.loadtxt(filename, dtype=np.float64)
    except FileNotFoundError:
        logger.error('File with glan callibration not found!')
        logger.debug('...Terminating')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return

    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    if fit_order == 1:
        # target_energy = coef[0]*energy + coef[1]
        energy = (target_energy.to('uJ').m - coef[1])/coef[0]*ureg.uJ
    else:
        logger.warning('Reverse Glan calculation for nonlinear fit is not '
                       'realized! Linear fit used instead!')
        energy = (target_energy.to('uJ').m - coef[1])/coef[0]*ureg.uJ
    
    logger.debug('...Finishing. Power meter energy calculated')
    return energy

def glan_calc(
        energy: PlainQuantity,
        fit_order: int=1
    ) -> PlainQuantity|None:
    """
    Calculates energy at sample for a given power meter energy.
    
    Do not interact with hardware.
    """

    logger.debug('Starting sample energy calculation...')
    sub_folder = 'rsc'
    filename = 'GlanCalibr.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        calibr_data = np.loadtxt(filename, dtype=np.float64)
    except FileNotFoundError:
        logger.error('File with glan callibration not found!')
        logger.debug('...Terminating')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return

    #get coefficients which fit calibration data with fit_order polynom
    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    #init polynom with the coefficients
    fit = np.poly1d(coef)

    #return the value of polynom at energy
    sample_en = fit(energy.to('uJ').m)*ureg.uJ
    logger.debug('...Finishing. Sample energy calculated.')
    return sample_en

def measure_point(
        wavelength: PlainQuantity,
        priority: int = Priority.NORMAL,
        **kwargs
    ) -> MeasuredPoint|None:
    """
    Measure single PA data point.
    
    Thread safe.
    """

    logger.debug('Starting PA point measurement...')
    osc = hardware.osc
    pm = hardware.power_meter
    config = hardware.config
    if pm is None:
        logger.warning('...Terminating. Power meter is off in config.')
        return None
    pa_ch_id = int(config['pa_sensor']['connected_chan']) - 1
    pm_ch_id = int(config['power_meter']['connected_chan']) - 1
    data = _osc_call.submit(
        priority = priority,
        func = osc.measure
    )
    # Check if operation was successfull
    if isinstance(data, ActorFail):
        logger.error('Error in measure point function.')
        return None

    # Downsample power meter data
    pm_signal_raw, pm_decim_factor = Oscilloscope.decimate_data(
        data.data_raw[pm_ch_id]
    )
    # Calculate dt for downsampled data
    dt_pm = data.dt*pm_decim_factor
    # Convert downsampled signal to volts
    pm_signal = Q_(pm_signal_raw*data.yincrement, 'V')
    
    # Calculate energy information
    pm_data = pm.energy_from_data(pm_signal, dt_pm)
    if pm_data is None:
        logger.error('Energy data cannot be calculated.')
        return None
    # Calculate time from start of pm_signal to trigger position
    pm_offset = pm.pulse_offset(pm_signal, dt_pm)
    if pm_offset is None:
        logger.error('Time offset calculation for PM data failed.')
        return None

    # PA signal in volts
    pa_signal_v = Q_(data.data_raw[pa_ch_id]*data.yincrement, 'V')
    # Amplitude of PA signal in volts
    pa_amp_v = pa_signal_v.ptp() # type: ignore

    pm_start = data.pre_t[pm_ch_id]
    pa_start = data.pre_t[pa_ch_id]
    # Start time of PA data relative to start of laser pulse
    start_time = (pm_start - pm_offset) - pa_start
    # Stop time of PA data relative to start of laser pulse
    stop_time = data.dt*(len(pa_signal_v)-1) + start_time # type: ignore

    if config['power_control'] == 'Filters':
        reflection = glass_reflection(wavelength)
        if reflection is not None and reflection !=0:
            sample_energy = pm_data.energy/reflection
            pa_signal = pa_signal_v/sample_energy
            pa_amp = pa_amp_v/sample_energy
        else:
            sample_energy = Q_(0, 'J')
            pa_amp = Q_(0, 'V/J')
            pa_signal = pa_signal_v
            logger.warning('Sample energy cannot be '
                            +'calculated')
    elif config['power_control'] == 'Glan prism':
        sample_energy = glan_calc(pm_data.energy)
        if sample_energy is not None and sample_energy:
            pa_signal = pa_signal_v/sample_energy
            pa_amp = pa_amp_v/sample_energy
        else:
            logger.warning(f'Sample energy = {sample_energy}')
            pa_amp = Q_(0, 'uJ')
            sample_energy = Q_(0, 'uJ')
            pa_signal = pa_signal_v
    else:
        logger.error('Unknown power control method! '
                    + 'Measurements terminated!')
        logger.debug('...Terminating.')
        return None
    measurement = MeasuredPoint(
        pa_signal_raw = data.data_raw[pa_ch_id].copy(),
        dt = data.dt,
        dt_pm = dt_pm,
        pa_signal = pa_signal,
        pm_signal = pm_data.signal,
        start_time = start_time,
        stop_time = stop_time, # type: ignore
        pm_energy = pm_data.energy,
        sample_energy = sample_energy,
        max_amp = pa_amp,
        wavelength = wavelength 
    )
    logger.debug(f'{sample_energy=}')
    logger.debug(f'PA data has {len(pa_signal)} points ' # type: ignore
                    + f'with max value={pa_signal.max():.2D}') #type: ignore
    logger.debug(f'{pa_amp=}')

    logger.debug('...Finishing PA point measurement.')
    return measurement

def aver_measurements(measurements: List[MeasuredPoint]) -> MeasuredPoint:
    """Calculate average measurement from a given list of measurements.
    
    Actually only amplitude values are averaged, in other cases data
    from the last measurement from the <measurements> is used."""

    logger.debug('Starting measurement averaging...')
    result = MeasuredPoint()
    total = len(measurements)
    for measurement in measurements:
        result.dt = measurement.dt
        result.pa_signal = measurement.pa_signal
        result.pa_signal_raw = measurement.pa_signal_raw
        result.pm_signal = measurement.pm_signal
        result.start_time = measurement.start_time
        result.stop_time = measurement.stop_time
        result.wavelength = measurement.wavelength
        
        result.pm_energy += measurement.pm_energy
        result.sample_energy += measurement.sample_energy
        result.max_amp += measurement.max_amp

    if total:
        result.pm_energy = result.pm_energy/total
        result.sample_energy = result.sample_energy/total
        result.max_amp = result.max_amp/total
    else:
        result.pm_energy = Q_(0, 'J')
        result.sample_energy = Q_(0, 'J')
        result.max_amp = Q_(0, 'V/J')

    logger.info(f'Average power meter energy {result.pm_energy}')
    logger.info(f'Average energy at {result.sample_energy}')
    logger.info(f'Average PA signal amp {result.max_amp}')
    
    logger.debug('...Finishing averaging of measurements.')
    return result

