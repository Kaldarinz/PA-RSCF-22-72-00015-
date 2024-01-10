from pint import UnitRegistry
import pint
import logging

logger = logging.getLogger(__name__)

ureg = UnitRegistry(auto_reduce_dimensions=True) # type: ignore
pint.set_application_registry(ureg)
ureg.default_format = '~.2gP'
Q_ = ureg.Quantity