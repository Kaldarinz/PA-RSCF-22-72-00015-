"""
PA backend
"""

from typing import Any, TypedDict, Union
import logging
import yaml
import os

from pylablib.devices import Thorlabs
import modules.osc_devices as osc_devices
import modules.exceptions as exceptions

logger = logging.getLogger(__name__)

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware"""

    stage_x: Thorlabs.KinesisMotor
    stage_y: Thorlabs.KinesisMotor
    osc: osc_devices.Oscilloscope
    config_loaded: bool

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware"""
    
    power_meter: osc_devices.PowerMeter
    pa_sens: osc_devices.PhotoAcousticSensOlymp
    stage_z: Thorlabs.KinesisMotor

def init_hardware(hardware: Hardware) -> bool:
    """Initialize all hardware"""

    logger.info('Starting hardware initialization...')
    
    if not init_osc(hardware):
        logger.warning('Oscilloscope cannot be loaded')
        return False
    osc = hardware['osc']

    if not hardware['config_loaded']:
        logger.debug('Hardware configuration is not loaded')
        config = load_config(hardware)
        if config:
            if not init_stages(hardware):
                logger.warning('Stages cannot be loaded')
                return False
            
            pm_chan = int(config['power_meter']['connected_chan'])
            pm = hardware.get('power_meter')
            if pm is not None:
                pm.set_channel(pm_chan-1)
                logger.info(f'Power meter channel set to {pm_chan}')

            pa_chan = int(config['pa_sensor']['connected_chan'])
            pa = hardware.get('pa_sens')
            if pa is not None:
                pa.set_channel(pa_chan-1)
                logger.info(f'Power meter channel set to {pa_chan}')
            hardware['config_loaded'] = True
        else:
            logger.warning('Hardware config cannot be loaded')
            return False
    else:
        #config already loaded try to reload what is present
        if not init_stages(hardware):
            logger.warning('Stages cannot be loaded')
            return False
        
        pm = hardware.get('power_meter')
        if pm is not None:
            pm_chan = pm.ch
            pm = osc_devices.PowerMeter(osc)
            pm.set_channel(pm_chan)
            hardware.update({'power_meter': pm})
            logger.debug('Power meter reinitiated on the same channel')

        pa = hardware.get('pa_sens')
        if pa is not None:
            pa_chan = pa.ch
            pa = osc_devices.PhotoAcousticSensOlymp(osc)
            pa.set_channel(pa_chan)
            hardware.update({'pa_sens': pa})
            logger.debug('PA sensor reinitiated on the same channel')
                
    logger.info('Hardware initialization complete')
    return True

def load_config(hardware: Hardware) -> dict:
    """Load hardware configuration.
    Adds keys for all optional devices to hardware"""

    logger.debug('Start loading config...')
    
    base_path = os.path.dirname(os.path.dirname(__name__))
    sub_dir = 'rsc'
    filename = 'config.yaml'
    full_path = os.path.join(base_path, sub_dir, filename)
    try:
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)['hardware']
    except:
        logger.warning('Connfig cannot be properly loaded')
        return {}
    
    if int(config['stages']['axes']) == 3:
        logger.debug('Stage_z added to hardware list')
        hardware.update({'stage_z':0}) # type: ignore
    
    osc = hardware['osc']
    if bool(config['power_meter']['connected']):
        hardware.update({'power_meter': osc_devices.PowerMeter(osc)})
        logger.debug('Power meter added to hardare list')
    if bool(config['pa_sensor']['connected']):
        hardware.update(
            {'pa_sens':osc_devices.PhotoAcousticSensOlymp(osc)})
    logger.debug(f'Config file read')
    return config

def init_stages(hardware: Hardware) -> bool:
    """Initiate Thorlabs KDC based stages"""

    logger.debug(f'Init_stages is starting...')
    logger.debug('Checking if connection to stages is '
                +'already estblished')

    if stages_open(hardware):
        logger.info('Connection to all stages already established!')
        return True

    logger.debug('Searching for Thorlabs kinsesis devices (stages)')
    stages = Thorlabs.list_kinesis_devices()
    logger.debug(f'{len(stages)} devices found')

    if hardware.get('stage_z', default=None) is None:
        amount = 3
    else:
        amount = 2

    if len(stages) < amount:
        msg = f'Less than {amount} kinesis devices found!'
        logger.error(msg)
        return False

    connected = True
    stage1_ID = stages.pop()[0]
    #motor units [m]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage')
    try:
        stage1.is_opened()
    except:
        msg = 'Failed attempt to coomunicate with stage1'
        logger.error(msg)
        if connected:
            connected = False
    else:
        hardware['stage_x'] = stage1
        logger.info(f'Stage X with ID={stage1_ID} is initiated')

    stage2_ID = stages.pop()[0]
    #motor units [m]
    stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage')
    try:
        stage2.is_opened()
    except:
        msg = 'Failed attempt to coomunicate with stage2'
        logger.error(msg)
        if connected:
            connected = False
    else:
        hardware['stage_y'] = stage2
        logger.info(f'Stage Y with ID={stage2_ID} is initiated')

    if amount == 3:
        stage3_ID = stages.pop()[0]
        #motor units [m]
        stage3 = Thorlabs.KinesisMotor(stage3_ID, scale='stage')
        try:
            stage3.is_opened()
        except:
            msg = 'Failed attempt to coomunicate with stage3'
            logger.error(msg)
            if connected:
                connected = False
        else:
            hardware['stage_z'] = stage3
            logger.info(f'Stage Z with ID={stage3_ID} is initiated')
    
    if connected:
        logger.info('Stage initiation is complete')
    else:
        logger.warning('Stages are not initiated')
    
    return connected

def init_osc(hardware: Hardware) -> bool:
    """Initialization of oscilloscope.
    Returns true if connection is already established or
    initialization is successfull"""
    
    logger.debug('Init_osc is starting...')
    logger.debug('Checking if connection is already estblished')

    if osc_open(hardware):
        logger.info('Connection to oscilloscope is already established!')
        return True

    logger.debug('No connection found. Trying to establish connection')
    try:
        hardware['osc'].initialize()
    except Exception as err:
        logger.error(f'Error {type(err)} while trying initialize osc')
        return False
    else:
        logger.debug('Oscilloscope initialization complete')
        return True

def stages_open(hardware: Hardware) -> bool:
    """Return True if all stages are responding and open.
    Never raises exceptions."""

    logger.debug('Starting connection check to stages')
    connected = True
    try:
        if not hardware['stage_x'].is_opened():
            logger.warning('Stage X is not open')
            connected = False
        else:
            logger.debug('Stage X is open')
        if not hardware['stage_y'].is_opened():
            logger.warning('Stage Y is not open')
            connected = False
        else:
            logger.debug('Stage Y is open')
        stage_z = hardware.get('stage_z', default=None)
        if not stage_z is None:
            if not hardware['stage_z'].is_opened():
                logger.warning('Stage Z is not open')
                connected = False
            else:
                logger.debug('Stage Z is open')
    except:
        logger.error('Error during stages connection check')
        connected = False
    
    if connected:
        logger.debug('All stages are connected and open')

    return connected

def osc_open(hardware: Hardware) -> bool:
    """Returns true if oscilloscope is connected."""

    logger.debug('Starting connection check to oscilloscope')
    hardware['osc'].connection_check()
    connected = not hardware['osc'].not_found
    logger.debug(f'Oscilloscope {connected=}')
    return connected

def pm_open(hardware: Hardware) -> bool:
    """Returns true if power meter is configured"""

    logger.debug('Starting power meter connection check')

    if hardware.get('power_meter') is None:
        logger.warning('Power meter is not initialized')
        connected = False
    else:
        connected = osc_open(hardware)
        logger.debug(f'Power meter {connected=}')
    return connected

def move_to(X: float, Y: float, hardware: Hardware) -> None:
    """Sends PA detector to (X,Y) position.
    Does not wait for stop moving.
    Coordinates are in mm."""
    
    x_dest_mm = X/1000
    y_dest_mm = Y/1000

    logger.debug(f'Sending X stage to {x_dest_mm} mm position')
    try:
        hardware['stage_x'].move_to(x_dest_mm)
    except:
        msg = 'Stage X move_to command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

    logger.debug(f'Sending Y stage to {y_dest_mm} mm position')
    try:
        hardware['stage_y'].move_to(y_dest_mm)
    except:
        msg = 'Stage Y move_to command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

def wait_stages_stop(hardware: Hardware) -> None:
    """Waits untill all (2) stages stop"""

    logger.debug('waiting untill stages complete moving')
    try:
        hardware['stage_x'].wait_for_stop()
        logger.debug('Stage X stopped')
    except:
        msg = 'Stage X wait_for_stop command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)
    
    try:
        hardware['stage_y'].wait_for_stop()
        logger.debug('Stage Y stopped')
    except:
        msg = 'Stage Y wait_for_stop command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

def home(hardware: Hardware) -> None:
    """Homes all (2) stages"""

    logger.debug('homing is starting...')
    try:
        logger.debug('homing stage X')
        hardware['stage_x'].home(sync=False,force=True)
    except:
        msg = 'Stage X homing command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

    try:
        logger.debug('homing stage Y')
        hardware['stage_y'].home(sync=False,force=True)
    except:
        msg = 'Stage Z homing command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)
    
    wait_stages_stop(hardware)