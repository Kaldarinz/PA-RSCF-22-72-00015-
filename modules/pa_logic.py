"""
PA backend.

All calls to hardware should be performed via corresponding actors (_stage_call, _osc_call).

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from typing import (
    cast,
    Literal,
    Iterable,
    ParamSpec,
    TypeVar,
    Callable
) 
import logging
import os
from threading import Thread
import time
from datetime import datetime as dt

import yaml
from pint.facets.plain.quantity import PlainQuantity
import numpy as np
from pylablib.devices.Thorlabs import KinesisMotor
from pylablib.devices import Thorlabs

from .hardware.osc_devices import (
    Oscilloscope,
    PowerMeter,
    PhotoAcousticSensOlymp
)
from .hardware.glob import hardware
from .hardware.utils import (
    calc_sample_en
)
from .data_classes import (
    WorkerSignals,
    EnergyMeasurement,
    OscMeasurement,
    PaEnergyMeasurement,
    Position,
    Actor,
    ActorFail,
    StagesStatus,
    ThreadSignals,
    MapData,
    ScanLine,
    MeasuredPoint,
    WorkerFlags
)
from .constants import (
    Priority,
    SCAN_MODES
)
from . import ureg, Q_

logger = logging.getLogger(__name__)
rng = np.random.default_rng()

P = ParamSpec('P')
T = TypeVar('T')

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
    result = True
    # Start Actors to communicate with hardware
    _init_call()
    # Try init oscilloscope.
    if not init_osc():
        logger.warning('Oscilloscope cannot be loaded!')
        result = False
    else:
        logger.info('Oscilloscope initiated.')
    
    osc = hardware.osc
    # Try init stages
    if not init_stages():
        logger.warning('Stages cannot be loaded!')
        result = False
    
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
    return result

def close_hardware(**kwargs) -> None:
    """
    Cloase all hardware.
    
    Thread safe.
    """

    try:
        _stage_call.close(__close_stages)
        _osc_call.close()
        _stage_call.join()
        _osc_call.join()
        logger.info('Hardware communication terminated.')
    except:
        logger.info('No hardawe communication was start during run time.')

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

def osc_open(priority: int=Priority.NORMAL) -> bool|None:
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

def init_osc(priority: int=Priority.NORMAL) -> bool:
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

def _stage_status(
        stage: KinesisMotor,
        priority: int=Priority.NORMAL
    ) -> list[str]:
    """
    Get status of a given stage.
    
    Thread safe.
    """

    status_lst = _stage_call.submit(
            priority = priority,
            func = stage.get_status
        )
    if not isinstance(status_lst, ActorFail):
        return status_lst
    else:
        return ['ActorFail']
    
def _stage_pos(
        stage: KinesisMotor,
        priority: int=Priority.NORMAL
    ) -> PlainQuantity|None:
    """
    Get position of a given stage.
    
    Thread safe.
    """

    pos = _stage_call.submit(
            priority = priority,
            func = stage.get_position
        )
    if not isinstance(pos, ActorFail):
        result = Q_(pos, 'm')
        return result
    else:
        return None

def set_stages_speed(
        speed: PlainQuantity,
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> PlainQuantity:

    logger.debug(f'Start setting stage speed to {speed}')
    set_speed = Q_(np.nan, 'm/s')
    for axes, stage in hardware.stages.items():
        vel_params = _stage_call.submit(
            priority,
            stage.setup_velocity,
            max_velocity = speed.to('m/s').m
        )
        if isinstance(vel_params, ActorFail):
            logger.info(f'Velocity cannot be set to {axes} stage.')
            return Q_(np.nan, 'm/s')
        set_speed = Q_(vel_params.max_velocity, 'm/s')
        logger.debug(f'New {axes} stage speed is {set_speed}.')
    return set_speed

def stages_status(
        priority: int=Priority.LOW,
        **kwargs
    ) -> StagesStatus:
    """
    Return status of all stages.
    
    Thread safe.
    """

    status = StagesStatus()
    for axes, stage in hardware.stages.items():
        status_lst = _stage_call.submit(
            priority,
            stage.get_status
        )
        if not isinstance(status_lst, ActorFail):
            setattr(status, axes + '_status', status_lst)
        is_open = _stage_call.submit(
            priority,
            stage.is_opened
        )
        if not isinstance(is_open, ActorFail):
            setattr(status, axes + '_open', is_open)
    return status

def stages_position(
        priority: int=Priority.LOW,
        **kwargs) -> Position:
    """
    Get position of all stages.
    
    Thread safe.
    Have low priority.
    """

    coord = Position()
    for axes, stage in hardware.stages.items():
        pos = _stage_call.submit(
            priority,
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
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> None:
    """
    Stop movement along given axes.
    
    Thread safe.
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

def wait_stage(
        axes: Literal['x', 'y', 'z'],
        timeout: int | None=100,
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> None:
    """
    Wait untill given axis stage stops.
    
    Thread safe.
    """

    stage = hardware.stages.get(axes, None)
    if stage is None:
        logger.warning(f'Invalid axes ({axes}) for waiting.')
        return
    wait = _stage_call.submit(
        priority,
        stage.wait_for_stop,
        timeout = timeout
    )

def wait_all_stages(
        priority: int=Priority.NORMAL,
        timeout: int | None=100,
        **kwargs
    ) -> None:
    """Wait untill all stages stop."""

    for axes in hardware.stages.keys():
        wait_stage(axes, timeout, priority) # type: ignore
    logger.debug('All stages stopped.')

def move_to(
        new_pos: Position,
        priority: int=Priority.NORMAL,
        **kwargs) -> None:
    """Send motors to new position.
    
    Thread safe.\n
    """

    logger.debug('Starting move_to procedure...')
    for axes, stage in hardware.stages.items():
        coord = getattr(new_pos, axes)
        if coord is not None:
            coord = coord.to('m').m
            _stage_call.submit(
                priority = priority,
                func = stage.move_to,
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

def show_stage_tasks(
        **kwargs
):
    _stage_call.show_tasks()

def en_meas_fast(
        priority: int=Priority.LOW,
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

def meas_cont(
        data: Literal['en_scr', 'pa_scr', 'pa_short', 'pa'],
        signals: WorkerSignals,
        flags: WorkerFlags,
        priority: int=Priority.NORMAL,
        max_count: int | None=None,
        timeout: float | None=100,
        **kwargs
    ) -> list[EnergyMeasurement]:
    """
    Non-stop measurement of required information.

    Intended to be called from GUI.\n
    Thread safe.\n
    Attributes
    ----------
    ``data`` - determines, which information should be measured.\n
    ``signals`` and ``flags`` are used to interthread communication
    with GUI and are automatically supplied.\n
    ``priority`` set priority of the call for Actor.\n
    ``max_count`` optional maximum amount of measurements.\n
    ``timeout`` optional timeout in seconds.\n
    Return
    ------
    List with measured signals.
    """

    if hardware.power_meter is None:
        logger.warning('Power meter is not connected')
        return []
    comm, result, tkwargs = set_cont(
        dtype = data,
        priority = priority,
        max_count = max_count,
        timeout = timeout
    )
    t = Thread(target = _osc_call.submit, kwargs = tkwargs)
    t.start()
    while flags['is_running']:
        comm.progress.wait()
        # Emit progress with last measured object
        signals.progess.emit(result[-1])
        if comm.progress.is_set():
            comm.progress.clear()
        # Stop if max measurements was set and the value was reached
        if max_count is not None and comm.count >= max_count:
            break
    comm.is_running = False
    t.join()

    return result

def osc_run_normal(
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> None:
    """Force oscilloscope to run normally."""

    _osc_call.submit(
        priority = priority,
        func = hardware.osc.run_normal
    )

def set_cont(
        dtype: str,
        priority: int,
        max_count: int | None=None,
        timeout: float | None=None
    ) -> tuple[ThreadSignals, list, dict]:
    """
    Prepare objects for continous measurement of data.

    Return
    ------
    A tuple with 3 objects:
    * ThreadSignals object for interthread communication.
    * List, which will contain results of measurement.
    * Dict, whith all necessary information to start measurement in a 
    new thread.
    """

    # Funstions, which provide necessary information
    funcs = {
        'en_scr': hardware.power_meter.get_energy_scr,
        'pa_scr': hardware.osc.measure_scr,
        'pa_short': hardware.osc.fast_measure,
        'pa': hardware.osc.fast_measure
    }
    # Additional arguments for the functions
    ch_mask = [False, False]
    pa_ch_id = int(hardware.config['pa_sensor']['connected_chan']) - 1
    ch_mask[pa_ch_id] = True
    kwargs = {
        'en_scr': {},
        'pa_scr': {},
        'pa_short': {
            'read_ch1': ch_mask[0],
            'read_ch2': ch_mask[1],
            'eq_limits': False
        },
        'pa': {}
    }
    # Object for communication with lower level fucntion
    comm = ThreadSignals()
    # List with results
    result = []
    tkwargs = {
        'priority': priority,
        'func': __meas_cont,
        'called_func': funcs[dtype],
        'comm': comm,
        'max_count': max_count,
        'timeout': timeout,
        'result': result
    }
    tkwargs.update(kwargs[dtype])
    return (comm, result, tkwargs)

def scan_2d(
        scan: MapData,
        signals: WorkerSignals,
        flags: WorkerFlags,
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> MapData:
    """Make 2D spatial scan.
    
    Emit progress after each scanned line.
    Thread safe.
    """

    logger.info('Starting scanning procedure.')
    # Set osc params
    _osc_call.submit(
        priority = priority,
        func = hardware.osc.set_measurement
    )
    # Scan loop
    for line_ind in range(scan.spoints):
        # Create scan line
        line = scan.create_line()
        if line is None:
            logger.error('Unexpected end of scan.')
            return scan
        # move to line  starting point
        if not line_ind:
            logger.info('Moving to scan start position.')
        else:
            logger.info('Moving to next line.')
        move_to(line.startp)
        wait_all_stages()
        if not line_ind:
            logger.info('At scan start position.')
        else:
            logger.info('Line scan strating.')
        # Launch energy measurements
        comm_en, result_en, tkwargs = set_cont(
            dtype = SCAN_MODES[scan.mode],
            priority = priority
        )
        t_en = Thread(target = _osc_call.submit, kwargs = tkwargs)
        t_en.start()
        # Then send stages to scan the line
        move_to(line.stopp)
        # While stage is moving, measure its position.
        while stages_status(priority).has_status('active'):
            # If scan was cancelled
            if not flags['is_running']:
                # Stop stage movements
                break_all_stages()
                # Stop signal measurements
                comm_en.is_running = False
                t_en.join()
                osc_run_normal()
                # Return what was scanned so far
                return scan
            line.add_pos_point(stages_position(priority))
        # Cancel signal measurements
        comm_en.is_running = False
        t_en.join()
        logger.info('Line scanned. Start converting OscMeas to MeasPoint')
        # Convert OscMeasurements to MeasuredPoints
        meas_points = [
            meas_point_from_osc(x, scan.wavelength) for x in result_en
        ]
        # Add measured points to scan line
        line._raw_sig = meas_points
        # Add scanned line to scan
        scan.add_line(line)
        signals.progess.emit(line)
        logger.info(f'Scanned {line}.')
    
    osc_run_normal()
    return scan

def __meas_cont(
        called_func: Callable,
        comm: ThreadSignals,
        result: list,
        timeout: float | None=None,
        max_count: int | None=None,
        *args,
        **kwargs
    ) -> None:
    """
    Private function which call ``called_func`` non-stop.
    
    Produce list of ``called_func`` results, which are not ``None``.\n
    ``comm`` - object for interthread communication.\n
    Measuring continue untill ``comm.is_running`` set to False,
    or ``timeout`` expires or ``max_count`` reached.\n
    Intended to be called by actor only.
    """

    start = time.time()
    # execution loop
    while comm.is_running:
        # exit by timeout
        t = time.time() - start
        if timeout and t > timeout:
            logger.warning(
                'Timeout expired during fast PA signal cont measure.'
            )
            break
        logger.debug(f'Prepare to measure {comm.count} at {t=}')
        msmnt = called_func(*args, **kwargs)
        # Skip bad reads
        if msmnt is None:
            continue
        # Add only unique measurements
        if len(result):
            if not (result[-1] == msmnt):
                result.append(msmnt)
                # Inform about good measurement
                comm.progress.set()
            else:
                logger.warning('Duplicated measurement!')
        # Add first measurement
        else:
            result.append(msmnt)
            # Inform about good measurement
            comm.progress.set()
        # inform about current amount of measurements
        comm.count = len(result)
        # Stop if max_count is set and reached
        if max_count and comm.count == max_count:
            logger.debug(f'{max_count=} reached')
            break
    logger.debug('Finishing __meas_count...')

def measure_point(
        wavelength: PlainQuantity,
        priority: int=Priority.NORMAL,
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
    
    data = _osc_call.submit(
        priority = priority,
        func = osc.measure
    )
    # Check if operation was successfull
    if isinstance(data, ActorFail):
        logger.error('Error in measure point function.')
        return None
    measurement = meas_point_from_osc(data, wavelength)
    logger.debug('...Finishing PA point measurement.')
    return measurement

def meas_point_from_osc(
        msmnt: OscMeasurement,
        wl: PlainQuantity
    ) -> MeasuredPoint | None:
    """Make MeasuredPoint from OscMeasurement and wavelength."""

    logger.debug('Start creating MeasuredPoint...')
    pm = hardware.power_meter
    pm_ch_id = int(hardware.config['power_meter']['connected_chan']) - 1
    pm_raw_data = msmnt.data_raw[pm_ch_id]
    pm_yinc = msmnt.yincrement[pm_ch_id]
    en_info = None
    if pm_raw_data is not None and pm_yinc is not None:
        # TODO This can potentially produce wrong energy values when
        # PM signal was not fully measured. Calculation of energy from
        # derivative of PM signal should be considered.
        pm_energy = pm.energy_from_data(
            pm_raw_data*pm_yinc, msmnt.dt
        )
        if pm_energy is not None:
            sample_en = calc_sample_en(wl, pm_energy.energy)
        else:
            sample_en = None
        if sample_en is not None:
            en_info = PaEnergyMeasurement(pm_energy, sample_en) # type: ignore
    measurement = MeasuredPoint.from_msmnts(
        data = msmnt,
        energy_info = en_info,
        wavelength = wl,
        pa_ch_ind = int(not bool(pm_ch_id)),
        pm_ch_ind = pm_ch_id
    )
    logger.debug('...MeasuredPoint created.')
    return measurement

### Emulation functions

def measure_point_emul(
        wavelength: PlainQuantity,
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> MeasuredPoint | None:
    """
    Emulate point measurement.
    
    Thread safe.
    """

    logger.info('Starting EMULATION of point measurement')
    result_en: list[OscMeasurement] = []
    tkwargs_en = {
        'step': 0.5,
        'comm': ThreadSignals(),
        'result': result_en,
        'max_count': 1
    }
    t_en = Thread(target = pa_fast_cont_emul, kwargs = tkwargs_en)
    t_en.start()
    t_en.join()
    msmnt = meas_point_from_osc(result_en[0], wavelength)
    return msmnt

def scan_2d_emul(
        scan: MapData,
        speed: PlainQuantity,
        signals: WorkerSignals,
        flags: WorkerFlags,
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> MapData:
    """Emulate 2D spatial scan.
    
    Emit progress after each scanned line.
    Thread safe.
    """

    logger.info('Starting emulation of scanning procedure.')

    # Scan loop
    for _ in range(scan.spoints):
        # move to line  starting point
        #logger.info('Moving to scan start position.')
        time.sleep((scan.sstep.value()/speed).to('s').m)
        #logger.info('At scan start position.')
        # Create scan line
        line = scan.create_line()
        if line is None:
            logger.error('Unexpected end of scan.')
            return scan
        
        # First launch energy measurements in a separate thread

        # Object for communication with lower level fucntion
        comm_en = ThreadSignals()
        # List with results
        result_en: list[OscMeasurement] = []
        tkwargs_en = {
            'step': 0.5,
            'comm': comm_en,
            'result': result_en
        }
        t_en = Thread(target = pa_fast_cont_emul, kwargs = tkwargs_en)
        t_en.start()

        # Give some time to obtain first energy measurement
        time.sleep(0.01)
        # Then send stages to scan the line
        scan_dist = line.stopp.dist(line.startp)
        scan_time = (scan_dist/speed).to('s').m
        logger.info(f'line {scan_time=}')
        # While stage is moving, measure its position.
        t0 = time.time()
        while t_en.is_alive():
            time.sleep(0.1*rng.random() + 0.01)
            if (cur_t:=time.time()-t0) > scan_time:
                if comm_en.is_running:
                    comm_en.is_running = False
            if not flags['is_running']:
                comm_en.is_running = False
                t_en.join()
                return scan
            # unit vector in direction of stop point
            unit = line.startp.direction(line.stopp)
            if unit is None:
                logger.error('Cannot get direction to stop point.')
                return scan
            cur_pos = line.startp + unit*scan_dist*cur_t/scan_time
            line.add_pos_point(cur_pos)
        logger.debug('Line scanned. Start converting OscMeas to MeasPoint')
        # Convert OscMeasurements to MeasuredPoints
        meas_points = [
            meas_point_from_osc(x, scan.wavelength) for x in result_en
        ]
        # Add measured points to scan line
        line._raw_sig = meas_points # type: ignore
        scan.add_line(line)
        signals.progess.emit(line)
        logger.info(f'Scanned {line}.')
    return scan

def pa_fast_cont_emul(
        step: float,
        comm: ThreadSignals,
        result: list[OscMeasurement],
        timeout: float | None=100,
        max_count: int | None=None
        ) -> list[OscMeasurement]:
    """
    Emulate measuring PA signal.
    
    Attributes
    ----------
    `step` - average time between measurements in s.
    """

    pm_ch_id = int(hardware.config['power_meter']['connected_chan']) - 1
    pa_ch_id = int(hardware.config['pa_sensor']['connected_chan']) - 1    

    # load emulation signals
    path = os.path.join(
        os.getcwd(),
        'rsc',
        'emulations'
    )
    pm_signal = np.loadtxt(os.path.join(path, 'pm_fast_norm.txt'))
    pa_signal = np.loadtxt(os.path.join(path, 'pa_fast_norm.txt'))
    
    start = time.time()
    time.sleep((rng.random()*0.2 + 0.9)*step/2)
    # execution loop
    while comm.is_running:
        # exit by timeout
        if timeout and (t := (time.time() - start)) > timeout:
            logger.warning(
                'Timeout expired during fast PA signal cont measure.'
            )
            break
        logger.debug(f'Prepare to emul measure {comm.count} at {t=}')
        
        # Generate Measurement
        raw_data: list[np.ndarray|None] = [None,None]
        raw_data[pm_ch_id] = (pm_signal[:,1]*(0.4*rng.random()+0.8)*100).astype(np.int16)
        raw_data[pa_ch_id] = (pa_signal[:,1]*(rng.random()+0.5)*(100/1.5)).astype(np.int16)
        msmnt = OscMeasurement(
            datetime = dt.now(),
            data_raw = raw_data,
            dt = Q_(10, 'us'),
            pre_t = [Q_(0, 'us'), Q_(0, 'us')],
            yincrement = [Q_(1, 'V'), Q_(1, 'V')]
        )

        # Add  measurement
        result.append(msmnt)
        # Inform about good measurement
        comm.progress.set()
        # inform about current amount of measurements
        comm.count = len(result)
        # Stop if max_count is set and reached
        if max_count and comm.count == max_count:
            logger.debug(f'{max_count=} reached')
            break
        time.sleep((rng.random()*0.2 + 0.9)*step)

    return result

def en_meas_fast_cont_emul(
        signals: WorkerSignals,
        flags: dict[str, bool],
        max_count: int | None=None,
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> list[EnergyMeasurement]:
    """
    Emulator of get screen information non-stop.

    Thread safe.
    """

    time_step = 0.25
    logger.info('Starting fast continous EnergyMeasurement emulation.')
    result = []
    total = 0
    while flags.get('is_running', False):
        delay = time_step*(0.9 + 0.2*rng.random())
        time.sleep(delay)
        msmnt = EnergyMeasurement(
            datetime=dt.now(),
            energy=Q_(rng.random(),'J')
        )
        result.append(msmnt)
        signals.progess.emit(msmnt)
        total += 1
        logger.debug('EnergyMeasurement generated.')
        if max_count is not None and total == max_count:
            break
    return result