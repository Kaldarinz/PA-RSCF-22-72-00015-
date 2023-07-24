"""
PA backend
"""

from typing import Any, TypedDict
import logging

from pylablib.devices import Thorlabs
import modules.oscilloscope as oscilloscope
import modules.exceptions as exceptions

logger = logging.getLogger(__name__)

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware"""

    stage_x: Any
    stage_y: Any
    osc: oscilloscope.Oscilloscope

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware"""
    
    power_meter: oscilloscope.PowerMeter
    stage_z: Any

def init_hardware(hardware: Hardware) -> None:
    """Initialize all hardware"""

    staes_amount = 2 #can be only 2 or 3
    init_stages(hardware, staes_amount)

    #do not work
    try:
        hardware['osc'].connection_check()
    except exceptions.OscilloscopeError:
        hardware['osc'].initialize()
    
def init_stages(hardware: Hardware, amount: int) -> None:
    """Initiate <amount> Thorlabs KDC based stages"""

    logger.debug('Init_stages is starting...')
    connected = True
    try:
        logger.debug('Check if connection to some stages is '
                    +'already estblished')
        logger.debug('Trying to communicate with X stage')
        hardware['stage_x'].get_position()
        logger.debug('Connection established!')
        logger.debug('Trying to communicate with Y stage')
        hardware['stage_y'].get_position()
        logger.debug('Connection established!')
        if amount == 3:
            logger.debug('Trying to communicate with Y stage')
            hardware['stage_z'].get_position()
            logger.debug('Connection established!')
    except:
        logger.debug('Attempt failed')
        connected = False
    
    if connected:
        logger.debug('Connection to all stages already established!')
        return

    logger.debug('Searching for Thorlabs kinsesis devices (stages)')
    stages = Thorlabs.list_kinesis_devices() # type: ignore
    logger.debug(f'{len(stages)} devices found')

    if len(stages) < amount:
        raise exceptions.StageError(f'Less than {amount} stages found!')

    stage1_ID = stages.pop()[0]
    #motor units [m]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') # type: ignore
    hardware['stage_x'] = stage1
    logger.debug(f'Stage X with ID={stage1_ID} is initiated')

    stage2_ID = stages.pop()[0]
    #motor units [m]
    stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') # type: ignore
    hardware['stage_y'] = stage2
    logger.debug(f'Stage Y with ID={stage2_ID} is initiated')

    if amount == 3:
        stage3_ID = stages.pop()[0]
        #motor units [m]
        stage3 = Thorlabs.KinesisMotor(stage3_ID, scale='stage') # type: ignore
        hardware['stage_z'] = stage3
        logger.debug(f'Stage Z with ID={stage3_ID} is initiated')
    
    logger.debug('Stage initiation is complete')

def move_to(X: float, Y: float, hardware: Hardware) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    
    x_dest_mm = X/1000
    y_dest_mm = Y/1000

    logger.debug(f'Sending X stage to {x_dest_mm} mm position')
    hardware['stage_x'].move_to(x_dest_mm)

    logger.debug(f'Sending Y stage to {y_dest_mm} mm position')
    hardware['stage_y'].move_to(y_dest_mm)

def wait_stages_stop(hardware: Hardware) -> None:
    """Waits untill all specified stages stop"""

    logger.debug('waiting untill stages complete moving')
    if hardware['stage_x']:
        hardware['stage_x'].wait_for_stop()
        logger.debug('Stage X stopped')
    
    if hardware['stage_y']:
        hardware['stage_y'].wait_for_stop()
        logger.debug('Stage Y stopped')

def home(hardware: Hardware) -> None:
    """Homes stages"""

    logger.debug('home is starting...')
    if hardware['stage_x'] and hardware['stage_y']:
        logger.debug('homing stage X')
        hardware['stage_x'].home(sync=False,force=True)

        logger.debug('homing stage Y')
        hardware['stage_y'].home(sync=False,force=True)
        wait_stages_stop(hardware)
    else:
        raise exceptions.StageError('Stages are not initialized')