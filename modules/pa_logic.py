"""
PA backend
"""

from typing import Any, TypedDict, Union
import logging

from pylablib.devices import Thorlabs
import modules.oscilloscope as oscilloscope
import modules.exceptions as exceptions

logger = logging.getLogger(__name__)

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware"""

    stage_x: Thorlabs.KinesisMotor
    stage_y: Thorlabs.KinesisMotor
    osc: oscilloscope.Oscilloscope

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware"""
    
    power_meter: oscilloscope.PowerMeter
    stage_z: Thorlabs.KinesisMotor

def init_hardware(hardware: Hardware) -> None:
    """Initialize all hardware"""

    logger.info('Starting hardware initialization...')

    staes_amount = 2 #can be only 2 or 3
    try:
        init_stages(hardware, staes_amount)
    except exceptions.StageError as err:
        raise exceptions.HardwareError(str(err))

    #do not work
    osc = hardware['osc']
    try:
        osc.connection_check()
    except exceptions.OscilloscopeError:
        osc.initialize()
        hardware.update({'power_meter': oscilloscope.PowerMeter(osc)})

    logger.info('Hardware initialization complete')
    
def init_stages(hardware: Hardware, amount: int) -> None:
    """Initiate <amount> = 2|3 Thorlabs KDC based stages"""

    logger.debug(f'Init_stages is starting for {amount} stages...')
    connected = True
    logger.debug('Checking if connection to some stages is '
                +'already estblished')
    
    try:
        logger.debug('Trying to communicate with X stage')
        if not hardware['stage_x'].is_opened():
            msg = 'Stage X is not open'
            logger.debug(msg)
            connected = False
        else:
            logger.debug('Connection to stage X established!')

        logger.debug('Trying to communicate with Y stage')
        if not hardware['stage_y'].is_opened():
            msg = 'Stage Y is not open'
            logger.debug(msg)
            connected = False
        else:
            logger.debug('Connection to stage X established!')

        if amount == 3:
            logger.debug('Trying to communicate with Y stage')
            if not hardware['stage_z'].is_opened():
                msg = 'Stage Z is not open'
                logger.debug(msg)
                connected = False
            else:
                logger.debug('Connection to stage Z established!')
    except:
        logger.debug('Communication attempt with stages failed')
        connected = False
    
    if connected:
        logger.info('Connection to all stages already established!')
        return

    logger.debug('Searching for Thorlabs kinsesis devices (stages)')
    stages = Thorlabs.list_kinesis_devices()
    logger.debug(f'{len(stages)} devices found')

    if len(stages) < amount:
        msg = f'Less than {amount} stages found!'
        logger.error(msg)
        raise exceptions.StageError(msg)

    stage1_ID = stages.pop()[0]
    #motor units [m]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage')
    try:
        stage1.is_opened()
    except:
        msg = 'Failed attempt to coomunicate with stage1'
        logger.error(msg)
        raise exceptions.StageError(msg)
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
        raise exceptions.StageError(msg)
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
            raise exceptions.StageError(msg)
        hardware['stage_z'] = stage3
        logger.info(f'Stage Z with ID={stage3_ID} is initiated')
    
    logger.info('Stage initiation is complete')

def move_to(X: float, Y: float, hardware: Hardware) -> None:
    """Move PA detector to (X,Y) position.
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
    
def check_stage():
    pass