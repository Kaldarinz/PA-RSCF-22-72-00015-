from pint import UnitRegistry
import logging

logger = logging.getLogger(__name__)

ureg = UnitRegistry(auto_reduce_dimensions=True) # type: ignore
ureg.default_format = '~.2gP'
ureg.setup_matplotlib(True)