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

import yaml
import pint
from pint.facets.plain.quantity import PlainQuantity
import numpy as np
import numpy.typing as npt
from InquirerPy import inquirer
from pylablib.devices.Thorlabs import KinesisMotor
from pylablib.devices import Thorlabs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keyboard

import modules.osc_devices as osc_devices
from .utils import confirm_action
from modules.exceptions import (
    OscConnectError,
    OscIOError,
    OscValueError,
    StageError
    )
from .pa_data import PaData
from .data_classes import (
    WorkerSignals,
    MeasuredPoint,
    EnergyMeasurement,
    Coordinate,
    hardware,
    Actor,
    StagesStatus
)
from .constants import (
    Priority
)
from . import ureg, Q_

logger = logging.getLogger(__name__)



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
    Thread safe for stages.
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
        pm = osc_devices.PowerMeter(osc)
        pm.set_channel(pm_chan, pre_time, post_time)
        hardware.power_meter = pm
        logger.debug('Power meter reinitiated on the same channel')

    pa = hardware.pa_sens
    if pa is not None:
        #save pa channel to apply it after init
        pa_chan = pa.ch
        pre_time = float(config['pa_sensor']['pre_time'])*ureg.us
        post_time = float(config['pa_sensor']['post_time'])*ureg.us
        pa = osc_devices.PhotoAcousticSensOlymp(osc)
        pa.set_channel(pa_chan, pre_time, post_time)
        hardware.pa_sens = pa
        logger.debug('PA sensor reinitiated on the same channel')
                
    logger.debug('...Finishing hardware initialization.')
    return True

def load_config() -> dict:
    """Load configuration.

    Configuration is loaded from rsc/config.yaml.
    Additionally add all optional devices to hardware dict.
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
        pm = osc_devices.PowerMeter(osc)
        hardware.power_meter = pm
        pm_chan = int(config['power_meter']['connected_chan']) - 1
        pre_time = float(config['power_meter']['pre_time'])*ureg.us
        post_time = float(config['power_meter']['post_time'])*ureg.us
        pm.set_channel(pm_chan, pre_time, post_time)
        logger.debug(f'Power meter added to hardare list at CHAN{pm_chan+1}')
    if bool(config['pa_sensor']['connected']):
        pa = osc_devices.PhotoAcousticSensOlymp(osc)
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
        try:
            connected = _stage_call.submit(
                Priority.HIGH,
                stage.is_opened
            )
        except:
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

def osc_open() -> bool:
    """Check connection to oscilloscope."""

    return hardware.osc.connection_check()

def init_osc() -> bool:
    """Initialize oscilloscope.

    Return true if connection is already established or
    initialization is successfull.
    """
    osc = hardware.osc
    logger.debug('Starting init_osc...')
    if osc.connection_check():
        logger.info('Connection to oscilloscope is already established!')
        logger.debug('...Finishing init_osc.')
        return True

    logger.debug('No connection found. Trying to establish connection')
    if osc.initialize():
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
        if not stage is None:
            is_open = _stage_call.submit(
                Priority.HIGH,
                stage.is_opened
            )
            if is_open:
                logger.debug(f'stage {axes} is open')
                continue
        logger.debug(f'stage {axes} is not open')
        connected = False
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
        try:
            status_lst = _stage_call.submit(
                Priority.LOW,
                stage.get_status
            )
            is_open = _stage_call.submit(
                Priority.LOW,
                stage.is_opened
            )
            setattr(status, axes + '_open', is_open)
            setattr(status, axes + '_status', status_lst)
        except:
            logger.warning('General exception caught in "stages_status", need to change.')

    return status

def stages_position(**kwargs) -> Coordinate:
    """
    Get position of all stages.
    
    Thread safe.
    Have low priority.
    """

    coord = Coordinate()
    for axes, stage in hardware.stages.items():
        try:
            pos = _stage_call.submit(
                Priority.LOW,
                stage.get_position
            )
        except:
            logger.warning('General exception caught in "Stages_position", nned to change.')
        else:
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
        priority: int|None = None,
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
    if priority is None:
        priority = Priority.NORMAL
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

def stage_ident(stage: KinesisMotor) -> Thread:
    """Identify a stage.
    
    Identification is done by stage vibration
    and stage controller blinking.
    """

    stage.blink()
    return_thread = asinc_mech_ident(stage)
    return return_thread

def asinc_mech_ident(
        stage: KinesisMotor,
        amp: PlainQuantity = Q_(1,'mm')
    ) -> Thread:
    """Asinc call of mech_ident."""

    vibration = Thread(target=mech_ident, args=(stage, amp))
    vibration.start()
    return vibration

def mech_ident(
        stage: KinesisMotor,
        amp: PlainQuantity = Q_(1,'mm')
    ) -> None:
    """Vibrate several times near current position.
    
    <amp> is vibration amplitude.
    """
    cycles: int = 1
    amp_val = amp.to('m').m
    for _ in range(cycles):
        stage.move_by(amp_val)
        stage.wait_for_stop()
        stage.move_by(-2*amp_val)
        stage.wait_for_stop()
        stage.move_by(amp_val)
        stage.wait_for_stop()

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

def track_power(
        signals: WorkerSignals,
        flags: dict,
        tune_width: int = 50,
        **kwargs
    ) -> EnergyMeasurement:
    """Measure laser energy.

    Run infinite loop and measure energy from power meter.\n
    To stop the measurements ``flags['is_running']`` should be set
    to FALSE, which can be done from outside.\n
    Execution can be halted and resumed by sending ``flags['is_running']``
    from outside.\n
    After each measurement fire ``signals.progress``, which return
    an EnergyMeasurement instance.
    """

    ### config parameters
    #Averaging for mean and std calculations
    aver = 10
    # ignore energy read if it is smaller than threshold*mean
    threshold = 0.01
    # time delay between measurements
    measure_delay = ureg('10ms')
    ###

    results = EnergyMeasurement()

    pm = hardware.power_meter
    if pm is None:
        msg = 'Power meter is off in config'
        logger.warning(msg)
        raise OscIOError(msg)
    
    #tune_width cannot be smaller than averaging
    if tune_width < aver:
        logger.warning(f'{tune_width=} is smaller than averaging='
                       + f'{aver}. tune_width set to {aver}.')
        tune_width = aver
    data = deque(maxlen=tune_width)   
    mean = 0*ureg('J')
    logger.debug('Entering measuring loop')
    while True:
        if not flags['is_running']:
            return results
        while flags['pause']:
            time.sleep(0.1)
        try:
            laser_amp = pm.get_energy_scr()
        except (OscValueError, OscIOError):
            logger.debug('Measurement failed. Trying again.')
            continue
        if len(data) and laser_amp < data[-1]*threshold:
            logger.debug('Measurement failed. Trying again.')
            continue
        data.append(laser_amp)
        
        #ndarray for up to last <aver> values
        tmp_data = pint.Quantity.from_list(
            [x for i,x in enumerate(reversed(data)) if i<aver])
        mean = tmp_data.mean() # type: ignore
        std = tmp_data.std() # type: ignore
        logger.debug(f'{laser_amp=}, {mean=}, {std=}')
        results = EnergyMeasurement(
            data = pint.Quantity.from_list(
                [x for x in data]),
            signal = pm.data, # type: ignore
            sbx = pm.start_ind,
            sex = pm.stop_ind,
            energy = laser_amp,
            aver = mean,
            std = std
        )
        signals.progess.emit(results)
        time.sleep(measure_delay.to('s').m)

def spectrum(
        wl: PlainQuantity,
        target_energy: PlainQuantity,
        averaging: int
    ) -> Optional[PaData]:
    """Measure dependence of PA signal on excitation wavelength.
    
    Measurements start at <start_wl> wavelength and are taken
    every <step> with the last measurement at <end_wl>,
    so that the last step could be smaller than others.
    Measurements are taken at <target_energy>.
    Each measurement will be averaged <averaging> times.
    """

    logger.info('Starting measuring spectra...')
    data = PaData(dims=1, params=['Wavelength'])

    measurement, proceed = _ameasure_point(averaging, wl)
    if measurement:
        data.add_point(measurement, [current_wl])
        data.save_tmp()
    if not proceed:
        logger.debug('...Terminating.')
        return data
    data.bp_filter()
    logger.info('...Finishing. Spectral scanning complete!')
    return data

def single_measure(
        wl: PlainQuantity,
        target_energy: PlainQuantity,
        averaging: int
    ) -> Optional[PaData]:
    """Measure single PA point.
    
    Measurement is taken at <target_energy>.
    Each measurement will be averaged <averaging> times.
    """

    logger.info('Starting single point measurement...')
    data = PaData(dims=0, params=['Wavelength'])

    if not set_energy(wl, target_energy):
        logger.debug('...Terminating single point measurement.')
        return None
    measurement, _ = _ameasure_point(averaging, wl)
    if measurement is not None:
        data.add_point(measurement, [wl])
        data.save_tmp()
        data.bp_filter()
    logger.info('...Finishing single point measurement!')
    return data

def set_energy(
    current_wl: PlainQuantity,
    target_energy: PlainQuantity
    ) -> bool:
    """Set laser energy for measurements.
    
    Set <target_energy> at <current_wl>.
    <repeated> is flag which indicates that this is not the first
    call of set_energy."""

    logger.debug('Starting setting laser energy...')
    config = hardware.config
    if config['power_control'] == 'Filters':
        logger.info('Please remove all filters and measure '
                    + 'energy at glass reflection.')
        #measure mean energy at glass reflection
        energy = track_power(50)
        logger.info(f'Power meter energy = {energy}')

        #find valid filters combinations for current parameters
        max_filter_comb = config['energy']['max_filters']
        logger.debug(f'{max_filter_comb=}')
        filters = glass_calculator(
            current_wl,
            energy,
            target_energy,
            max_filter_comb)
        if not len(filters):
            logger.warning(f'No valid filter combination for '
                    + f'{current_wl}')
            if not confirm_action('Do you want to continue?'):
                logger.warning('Spectral measurements terminated!')
                logger.debug('...Terminating.')
                return False
            
        reflection = glass_reflection(current_wl)
        if reflection is not None and reflection != 0:
            target_pm_value = target_energy*reflection
            logger.info(f'Target power meter energy is {target_pm_value}!')
            logger.info('Please set it using filters combination '
                        + ' from above. Additional adjustment by '
                        + 'laser software could be required.')
            track_power(50)
        else:
            logger.warning('Target power meter energy '
                            + 'cannot be calculated!')
            logger.debug('...Terminating.')
            return False
    
    elif config['power_control'] == 'Glan prism':
        target_pm_value = glan_calc_reverse(target_energy)
        logger.info(f'Target power meter energy is {target_pm_value}!') # type: ignore
        logger.info(f'Please set it using Glan prism.')
        track_power(50)
    else:
        logger.error('Unknown power control method! '
                        + 'Measurements terminated!')
        logger.debug('...Terminating.')
        return False
    logger.debug('...Finishing. Laser energy set.')
    return True
                
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
    ) -> Optional[PlainQuantity]:
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
    ) -> Optional[PlainQuantity]:
    """Calculates energy at sample for a given power meter energy"""

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

def _ameasure_point(
    averaging: int,
    current_wl: PlainQuantity
    ) -> Tuple[MeasuredPoint|None, bool]:
    """Measure single PA data point with averaging.
    
    second value in the returned tuple (bool) is a flag to 
    continue measurements.
    """

    logger.debug('Starting measuring PA data point with averaging...')
    counter = 0
    msmnts: List[MeasuredPoint]=[]
    while counter < averaging:
        logger.info(f'Signal at {current_wl} should be measured '
                + f'{averaging-counter} more times.')
        action = set_next_measure_action()
        if action == 'Tune power':
            track_power(40)
        elif action == 'Measure':
            tmp_measurement = _measure_point(current_wl)
            if verify_measurement(tmp_measurement):
                msmnts.append(tmp_measurement)
                counter += 1
                if counter == averaging:
                    measurement = aver_measurements(msmnts)
                    logger.debug('...Finishinng. Data point successfully measured!')
                    return measurement, True
       
        elif action == 'Stop measurements':
            if confirm_action():
                if len(msmnts):
                    measurement = aver_measurements(msmnts)
                else:
                    logger.debug('No data was measured')
                    measurement = None
                logger.warning('Spectral measurement terminated')
                logger.debug('...Terminating.')
                return measurement, False
        else:
            logger.warning('Unknown command in Spectral measure menu!')
    
    logger.warning('Unexpectedly passed after main measure sycle!')
    logger.debug('...Terminating with empty data point.')
    return MeasuredPoint(), True

def _measure_point(
        wavelength: PlainQuantity,
        signals: WorkerSignals,
        **kwargs
    ) -> MeasuredPoint:
    """Measure single PA data point."""

    logger.debug('Starting PA point measurement...')
    measurement = MeasuredPoint()
    osc = hardware.osc
    pm = hardware.power_meter
    config = hardware.config
    if pm is None:
        logger.warning('...Terminating. Power meter is off in config')
        return measurement
    pa_ch_id = int(config['pa_sensor']['connected_chan']) - 1
    pm_ch_id = int(config['power_meter']['connected_chan']) - 1
    try:
        data = osc.measure()
        data = cast(
            Tuple[List[PlainQuantity],List[npt.NDArray[np.int8]]],
            data
            )
    except OscConnectError as err:
        logger.warning('Oscilloscope disconnected!')
        return measurement
    except OscIOError as err:
        logger.warning(f'Error during PA measurement: {err.value}')
        return measurement
    dt = (1/osc.sample_rate).to('us')
    pm_start = osc.pre_t[pm_ch_id]
    pa_start = osc.pre_t[pa_ch_id]
    pa_signal_v = data[0][pa_ch_id]
    pm_signal_raw, pm_time_decim = osc.decimate_data(data[1][pm_ch_id])
    dt_pm = dt*pm_time_decim
    pm_signal = osc._to_volts(pm_signal_raw)
    pa_signal_raw = data[1][pa_ch_id]
    pa_amp_v = osc.amp[pa_ch_id]
    try:
        pm_energy = pm.energy_from_data(pm_signal, dt_pm)
        pm_offset = pm.pulse_offset(pm_signal, dt_pm)
    except OscValueError:
        logger.warning('Power meter energy cannot be measured!')
        pm_energy = Q_(0, 'uJ')
        logger.warning(f'Power meter energy set to {pm_energy}')
    start_time = (pm_start - pm_offset) - pa_start
    stop_time = dt*(len(pa_signal_v.m)-1) + start_time
    measurement = replace(
        measurement,                 
        **{
            'wavelength': wavelength,
            'dt': dt,
            'dt_pm': dt_pm,
            'pa_signal_raw': pa_signal_raw,
            'start_time': start_time,
            'stop_time': stop_time,
            'pm_signal': pm_signal,
            'pm_energy': pm_energy,
        }
    )

    if config['power_control'] == 'Filters':
        reflection = glass_reflection(wavelength)
        if reflection is not None and reflection !=0:
            sample_energy = pm_energy/reflection
            pa_signal = pa_signal_v/sample_energy
            pa_amp = pa_amp_v/sample_energy
        else:
            sample_energy = Q_(0, 'J')
            pa_amp = Q_(0, 'V/J')
            logger.warning('Sample energy cannot be '
                            +'calculated')
    elif config['power_control'] == 'Glan prism':
        sample_energy = glan_calc(pm_energy)
        if sample_energy is not None and sample_energy:
            pa_signal = pa_signal_v/sample_energy
            pa_amp = pa_amp_v/sample_energy
        else:
            logger.warning(f'Sample energy = {sample_energy}')
            pa_amp = Q_(0, 'uJ')
            sample_energy = Q_(0, 'uJ')
    else:
        logger.error('Unknown power control method! '
                    + 'Measurements terminated!')
        logger.debug('...Terminating.')
        return measurement
    measurement.sample_energy = sample_energy
    measurement.pa_signal = pa_signal
    measurement.max_amp = pa_amp
    logger.debug(f'{sample_energy=}')
    logger.debug(f'PA data has {len(pa_signal)} points '
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

def verify_measurement(
        measurement: MeasuredPoint
    ) -> bool:
    """Verify a PA measurement."""

    logger.debug('Starting measurement verification...')
    # update state of power meter
    pm = hardware.power_meter
    if pm is None:
        logger.warning('Power meter is off in config')
        return False
    pm_signal = measurement.pm_signal
    dt = measurement.dt
    dt_pm = measurement.dt_pm
    pa_signal = measurement.pa_signal

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1,2)
    ax_pm = fig.add_subplot(gs[0,0])
    pm_time = Q_(np.arange(len(pm_signal))*dt_pm.m, dt.u)
    ax_pm.plot(pm_time.m,pm_signal.m)
    ax_pm.set_xlabel(f'Time, [{pm_time.u}]')
    ax_pm.set_ylabel(f'Power meter signal, [{pm_signal.u}]')

    #add markers for data start and stop
    ax_pm.plot(
        pm.start_ind*dt_pm.m,
        pm_signal[pm.start_ind].m,
        'o',
        alpha=0.4,
        ms=12,
        color='green')
    ax_pm.plot(
        pm.stop_ind*dt_pm.m,
        pm_signal[pm.stop_ind].m,
        'o',
        alpha=0.4,
        ms=12,
        color='red')
    ax_pa = fig.add_subplot(gs[0,1])
    pa_time = Q_(np.arange(len(pa_signal))*dt.m,dt.u)
    ax_pa.plot(pa_time.m,pa_signal.m)
    ax_pa.set_xlabel(f'Time, [{pa_time.u}]')
    ax_pa.set_ylabel(f'PA signal, [{pa_signal.u}]')
    plt.show()

    good_data = confirm_action('Data looks good?')
    logger.debug('...Finishing measurement verification.')
    return good_data

def set_next_measure_action() -> str:
    """Choose the next action during PA measurement.
    
    Returned values = ['Tune power'|'Measure'|'Stop measurements'].
    """

    logger.debug('Starting...')
    # in future this function can have several implementations
    # depending on whether CLI or GUI mode is used
    measure_ans = inquirer.rawlist(
    message='Chose an action:',
    choices=['Tune power','Measure','Stop measurements']
    ).execute()
    logger.debug(f'...Finishing. {measure_ans=} was choosen')
    return measure_ans