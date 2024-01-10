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
    ScanLine
)
from modules.utils import (
    propvals
)
ureg = pint.get_application_registry()
Q_ = ureg.Quantity
rng = np.random.default_rng()

scna = MapData(
    center=Position(Q_(4, 'mm'), Q_(6, 'mm'), Q_(10, 'mm')),
    width = Q_(3, 'mm'),
    height = Q_(5, 'mm'),
    hpoints = 10,
    vpoints = 20,
    scan_plane='YZ',
    scan_dir='VLB'
)


pprint(propvals(scna))
scna.fsize = Q_(10, 'mm')