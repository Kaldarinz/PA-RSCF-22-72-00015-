"""
PA backend
"""

from typing import Any, TypedDict

from pylablib.devices import Thorlabs
import modules.oscilloscope as oscilloscope
from modules.bcolors import bcolors
import modules.exceptions as exceptions

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware"""

    stage_x: Any
    stage_y: Any
    osc: oscilloscope.Oscilloscope

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware"""
    
    power_meter: oscilloscope.PowerMeter

def init_hardware(hardware: Hardware) -> None:
    """Initialize all hardware"""

    if not hardware['stage_x'] and not hardware['stage_y']:
        init_stages(hardware)

    if hardware['osc'].not_found:
        hardware['osc'].initialize()

def init_stages(hardware: Hardware) -> None:
    """Initiate Thorlabs KDC based stages."""

    stages = Thorlabs.list_kinesis_devices() # type: ignore

    if len(stages) < 2:
        raise exceptions.StageError('Less than 2 stages found!')
    else:
        stage1_ID = stages.pop()[0]
        #motor units [m]
        stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') # type: ignore
        hardware['stage_x'] = stage1

        stage2_ID = stages.pop()[0]
        #motor units [m]
        stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') # type: ignore
        hardware['stage_y'] = stage2

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