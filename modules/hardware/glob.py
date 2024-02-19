"""
Global objects.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

import logging

from pylablib.devices.Thorlabs import KinesisMotor

from .osc_devices import (
    Oscilloscope,
    PowerMeter,
    PhotoAcousticSensOlymp
)

logger = logging.getLogger(__name__)

class Hardware():
    """Class for hardware references."""
    
    def __init__(self):
        self.osc: Oscilloscope = Oscilloscope()
        self.power_meter: PowerMeter
        self.pa_sens: PhotoAcousticSensOlymp
        self.stages: dict[str, KinesisMotor] = {}
        self.motor_axes: int = -1
        self.config: dict = {}

hardware = Hardware()