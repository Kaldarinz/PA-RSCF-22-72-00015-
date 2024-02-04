from dataclasses import dataclass, field, fields
from pprint import pprint
from collections.abc import Iterable, Sequence
from datetime import datetime
import time, math
import typing
from pint import UnitRegistry
from pint.facets.plain.quantity import PlainQuantity
import pint
import numpy as np
import numpy.typing as npt
from enum import Enum
from modules.data_classes import Position
from modules.data_classes import (
    EnergyMeasurement,
    PaEnergyMeasurement,
    MeasuredPoint,
    OscMeasurement,
    Position,
    MapData,
    StagesStatus,
    ScanLine,
    PointMetadata,
    Direction
)
from modules.utils import (
    propvals
)
ureg = pint.get_application_registry()
Q_ = ureg.Quantity
rng = np.random.default_rng()

scan = MapData(
    center=Position(Q_(1,'m'),Q_(2,'m'),Q_(3,'m')),
    width=Q_(5,'m'),
    height=Q_(3,'m'),
    hpoints=10,
    vpoints=5
)

scan.add_line()
print(scan.get_raw_points())
#x,y,z = scan.get_plot_data('max_amp')