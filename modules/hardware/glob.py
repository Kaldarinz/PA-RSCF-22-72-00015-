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
        self.power_meter: PowerMeter | None = None
        self.pa_sens: PhotoAcousticSensOlymp | None = None
        self.stages: dict[str, KinesisMotor] = {}
        self.motor_axes: int = -1
        self.config: dict = {}

hardware = Hardware()