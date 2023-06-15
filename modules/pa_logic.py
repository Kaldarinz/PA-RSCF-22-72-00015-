"""
PA backend
"""

from typing import Any, TypedDict

from pylablib.devices import Thorlabs
import modules.oscilloscope as oscilloscope
from modules.bcolors import bcolors

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
    else:
        print(f'{bcolors.WARNING}Stages already initiated!{bcolors.ENDC}')

    if hardware['osc'].not_found:
        hardware['osc'].initialize()
    else:
        print(f'{bcolors.WARNING}Oscilloscope already initiated!{bcolors.ENDC}')

    if hardware['stage_x'] and hardware['stage_y'] and not hardware['osc'].not_found:
        hardware['power_meter'] = oscilloscope.PowerMeter(hardware['osc'])
        print(f'{bcolors.OKGREEN}Initialization complete!{bcolors.ENDC}')

def init_stages(hardware: Hardware) -> None:
    """Initiate stages."""

    print('Initializing stages...')
    stages = Thorlabs.list_kinesis_devices() # type: ignore

    if len(stages) < 2:
        print(f'{bcolors.WARNING}Less than 2 stages detected! Try again!{bcolors.ENDC}')

    else:
        stage1_ID = stages.pop()[0]
        #motor units [m]
        stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage') # type: ignore
        print(f'{bcolors.OKBLUE}Stage X{bcolors.ENDC} initiated. Stage X ID = {stage1_ID}')
        hardware['stage_x'] = stage1

        stage2_ID = stages.pop()[0]
        #motor units [m]
        stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage') # type: ignore
        print(f'{bcolors.OKBLUE}Stage Y{bcolors.ENDC} initiated. Stage X ID = {stage2_ID}')
        hardware['stage_y'] = stage2
