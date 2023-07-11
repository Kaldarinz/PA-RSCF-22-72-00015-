"""
PA backend
"""

from typing import Any, TypedDict

from pylablib.devices import Thorlabs
import modules.oscilloscope as oscilloscope
import modules.bcolors as bcolors
import modules.exceptions as exceptions


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

    try:
        hardware['osc'].connection_check()
    except exceptions.OscilloscopeError:
        hardware['osc'].initialize()
    

def init_stages(hardware: Hardware, amount: int) -> None:
    """Initiate <amount> Thorlabs KDC based stages"""

    stages = Thorlabs.list_kinesis_devices() # type: ignore

    connected = True
    try:
        hardware['stage_x'].get_position()
        hardware['stage_y'].get_position()
        if amount == 3:
            hardware['stage_z'].get_position()
    except:
        connected = False

    if len(stages) < amount:
        raise exceptions.StageError(f'Less than {amount} stages found!')
    elif not connected:
        stage1_ID = stages.pop()[0]
        #motor units [m]
        stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') # type: ignore
        hardware['stage_x'] = stage1

        stage2_ID = stages.pop()[0]
        #motor units [m]
        stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') # type: ignore
        hardware['stage_y'] = stage2

        if amount == 3:
            stage3_ID = stages.pop()[0]
            #motor units [m]
            stage3 = Thorlabs.KinesisMotor(stage3_ID, scale='stage') # type: ignore
            hardware['stage_z'] = stage3

def move_to(X: float, Y: float, hardware: Hardware) -> None:
    """Move PA detector to (X,Y) position.
    Coordinates are in mm."""
    
    hardware['stage_x'].move_to(X/1000)
    hardware['stage_y'].move_to(Y/1000)

def wait_stages_stop(hardware: Hardware) -> None:
    """Waits untill all specified stages stop"""

    if hardware['stage_x']:
        hardware['stage_x'].wait_for_stop()
    
    if hardware['stage_y']:
        hardware['stage_y'].wait_for_stop()

def home(hardware: Hardware) -> None:
    """Homes stages"""

    if hardware['stage_x'] and hardware['stage_y']:
        hardware['stage_x'].home(sync=False,force=True)
        hardware['stage_y'].home(sync=False,force=True)
        wait_stages_stop(hardware)
    else:
        raise exceptions.StageError('Stages are not initialized')