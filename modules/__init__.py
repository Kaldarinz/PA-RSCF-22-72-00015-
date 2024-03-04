from pint import UnitRegistry
import pint
import logging

logger = logging.getLogger(__name__)

ureg = UnitRegistry()
ureg.default_preferred_units = [ # type: ignore
    ureg.m,
    ureg.s,
    ureg.J,
    ureg.V
] 
pint.set_application_registry(ureg)
ureg.default_format = '~.2gP'
Q_ = ureg.Quantity