"""
PA backend.

All calls to hardware should be performed via corresponding actors (_stage_call, _osc_call).
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
from datetime import timedelta
from datetime import datetime as dt
import random

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
    Signals,
    MapData,
    ScanLine,
    MeasuredPoint
)
from .constants import (
    Priority
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
        timeout: int | None=5,
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
    _stage_call.submit(
        priority,
        stage.wait_for_stop,
        timeout = timeout
    )

def wait_all_stages(
        priority: int=Priority.NORMAL,
        timeout: int | None=5,
        **kwargs
    ) -> None:
    """Wait untill all stages stop."""

    for axes in hardware.stages.keys():
        wait_stage(axes, timeout, priority)

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
        data: Literal['en_fast', 'pa_fast'],
        signals: WorkerSignals,
        flags: dict[str, bool],
        priority: int=Priority.NORMAL,
        max_count: int | None=None,
        timeout: float | None=100,
        **kwargs
    ) -> list[EnergyMeasurement]:
    """
    Non-stop measurement of required information.

    Intended to be called from GUI.\n
    ``data`` - determines, which information should be measured.\n
    ``signals`` and ``flags`` are used to interthread communication
    with GUI and are automatically supplied.\n
    ``priority`` set priority of the call for Actor.\n
    ``max_count`` is optional maximum amount of measurements.\n
    ``timeout`` is optional timeout in seconds.\n
    Thread safe.
    """

    # Funstions, which provide necessary information
    funcs = {
        'en_fast': hardware.power_meter.get_energy_scr,
        'pa_fast': hardware.osc.measure_scr
    }
    # Object for communication with lower level fucntion
    comm = Signals()
    # List with results
    result = []
    tkwargs = {
        'priority': priority,
        'func': __meas_cont,
        'called_func': funcs[data],
        'comm': comm,
        'max_count': max_count,
        'timeout': timeout,
        'result': result
    }
    t = Thread(target = _osc_call.submit, kwargs = tkwargs)
    t.start()
    while flags.get('is_running'):
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

    logger.info('Starting fast continous EnergyMeasurement emulation.')
    rng = np.random.default_rng()
    result = []
    total = 0
    while flags.get('is_running', False):
        delay = rng.random()/10.
        time.sleep(delay)
        msmnt = EnergyMeasurement(
            datetime=dt.now(),
            energy=Q_(rng.random(),'J')
        )
        result.append(msmnt)
        signals.progess.emit(msmnt)
        total += 1
        logger.info('EnergyMeasurement generated.')
        if max_count is not None and total == max_count:
            break
    return result

def scan_2d(
        scan: MapData,
        signals: WorkerSignals,
        flags: dict[str, bool],
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> MapData:
    """Make 2D spatial scan.
    
    Emit progress after each scanned line.
    Thread safe.
    """

    logger.info('Starting scanning procedure.')
    # Scan loop
    for _ in range(scan.spoints):
        # Create scan line
        line = scan.add_line()
        if line is None:
            logger.error('Unexpected end of scan.')
            return scan
        # move to line  starting point
        logger.info('Moving to scan start position.')
        move_to(line.startp)
        wait_all_stages()
        logger.info('At scan start position.')
        # First launch energy measurements
        # Object for communication with lower level fucntion
        comm_en = Signals()
        # List with results
        result_en: list[OscMeasurement] = []
        tkwargs_en = {
            'priority': priority,
            'func': __meas_cont,
            'called_func': hardware.osc.measure_scr,
            'comm': comm_en,
            'result': result_en
        }
        t_en = Thread(target = _osc_call.submit, kwargs = tkwargs_en)
        t_en.start()
        # Then send stages to scan the line
        move_to(line.stopp)
        # While stage is moving, measure its position.
        while stages_status(priority).has_status('active'):
            line.add_pos_point(stages_position(priority))
        # When stage stopped, cancel signal measurements
        comm_en.is_running = False
        t_en.join()
        logger.info('Line scanned. Start converting OscMeas to MeasPoint')
        # Convert OscMeasurements to MeasuredPoints
        meas_points = [
            meas_point_from_osc(x, scan.wavelength) for x in result_en
        ]
        # Add measured points to scan line
        line.raw_sig = meas_points
        signals.progess.emit(line)
        logger.info(f'Scanned {line}.')
    return scan

def __meas_cont(
        called_func: Callable[P,T],
        comm: Signals,
        result: list,
        timeout: float | None=100,
        max_count: int | None=None
    ) -> list[T]:
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
        if timeout and (t := (time.time() - start)) > timeout:
            logger.warning(
                'Timeout expired during fast PA signal cont measure.'
            )
            break
        logger.debug(f'Prepare to measure {comm.count} at {t=}')
        msmnt = called_func()
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
    measurement = meas_point_from_osc(data, wavelength, pm_ch_id)
    logger.debug(f'{measurement.max_amp=}')
    logger.debug('...Finishing PA point measurement.')
    return measurement

def meas_point_from_osc(
        msmnt: OscMeasurement,
        wl: PlainQuantity
    ) -> MeasuredPoint | None:
    """Make MeasurePoint from OscMeasurement and wavelength."""

    pm = hardware.power_meter
    pm_ch_id = int(hardware.config['power_meter']['connected_chan']) - 1
    if pm is None:
        logger.error('Measure point cannot be calculated. Pm is not connected.')
        return None
    pm_energy = pm.energy_from_data(
        msmnt.data_raw[pm_ch_id]*msmnt.yincrement, msmnt.dt
    )
    if pm_energy is None:
        logger.error('Power meter energy cannot be obtained.')
        return None
    sample_en = calc_sample_en(wl, pm_energy.energy)
    if sample_en is None:
        logger.error('Sample energy cannot be calculated.')
        return None
    en_info = PaEnergyMeasurement(pm_energy, sample_en)
    measurement = MeasuredPoint(
        data = msmnt,
        energy_info = en_info,
        wavelength = wl,
        pa_ch_ind = int(not bool(pm_ch_id)),
        pm_ch_ind = pm_ch_id
    )
    return measurement

def aver_measurements(measurements: list[MeasuredPoint]) -> MeasuredPoint:
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

### Emulation functions

def scan_2d_emul(
        scan: MapData,
        signals: WorkerSignals,
        flags: dict[str, bool],
        priority: int=Priority.NORMAL,
        **kwargs
    ) -> MapData:
    """Emulate 2D spatial scan.
    
    Emit progress after each scanned line.
    Thread safe.
    """

    logger.info('Starting emulation of scanning procedure.')

    # Scan speed. Currently it is hardwritten here.
    speed = Q_(1, 'mm/s')

    # Scan loop
    for _ in range(scan.spoints):
        # Create scan line
        line = scan.add_line()
        if line is None:
            logger.error('Unexpected end of scan.')
            return scan
        # move to line  starting point
        logger.info('Moving to scan start position.')
        time.sleep(rng.random()/2)
        logger.info('At scan start position.')
        
        # First launch energy measurements in a separate thread

        # Object for communication with lower level fucntion
        comm_en = Signals()
        # List with results
        result_en: list[OscMeasurement] = []
        tkwargs_en = {
            'step': 0.2,
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
        # While stage is moving, measure its position.
        t0 = time.time()
        while (cur_t:=time.time()-t0) < scan_time:
            # unit vector in direction of stop point
            unit = line.startp.direction(line.stopp)
            if unit is None:
                logger.error('Cannot get direction to stop point.')
                return scan
            cur_pos = line.startp + unit*scan_dist*cur_t/scan_time
            line.add_pos_point(cur_pos)
            time.sleep(0.1*rng.random() + 0.01)
        # When stage stopped, cancel signal measurements
        comm_en.is_running = False
        t_en.join()
        logger.info('Line scanned. Start converting OscMeas to MeasPoint')
        # Convert OscMeasurements to MeasuredPoints
        meas_points = [
            meas_point_from_osc(x, scan.wavelength) for x in result_en
        ]
        # Add measured points to scan line
        line.raw_sig = meas_points # type: ignore
        signals.progess.emit(line)
        logger.info(f'Scanned {line}.')
    return scan

def pa_fast_cont_emul(
        step: float,
        comm: Signals,
        result: list[OscMeasurement],
        timeout: float | None=100,
        max_count: int | None=None
        ) -> list[OscMeasurement]:
    """
    Emulate measuring PA signal.
    
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
    pm_signal = np.loadtxt(os.path.join(path, 'pm_fast.txt'))
    pa_signal = np.loadtxt(os.path.join(path, 'pa_fast_norm.txt'))
    
    start = time.time()
    # execution loop
    while comm.is_running:
        # exit by timeout
        if timeout and (t := (time.time() - start)) > timeout:
            logger.warning(
                'Timeout expired during fast PA signal cont measure.'
            )
            break
        logger.debug(f'Prepare to emul measure {comm.count} at {t=}')
        time.sleep(rng.random()*2*step)
        
        # Generate Measurement
        raw_data: list[np.ndarray|None] = [None,None]
        raw_data[pm_ch_id] = pm_signal[:,1]*(0.4*rng.random()+0.8)
        raw_data[pa_ch_id] = pa_signal[:,1]*rng.random()
        msmnt = OscMeasurement(
            datetime = dt.now(),
            data_raw = raw_data,
            dt = Q_(10, 'us'),
            pre_t = [Q_(0, 'us'), Q_(0, 'us')],
            yincrement = Q_(1, 'V')  
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

    return result