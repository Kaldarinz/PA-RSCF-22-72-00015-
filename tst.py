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

def get_coord(axis: str):
    arr = np.empty(shape=raw_data.shape, dtype=object)
    arr[:] = None
    units = None
    for index in np.ndindex(raw_data.shape):
        point = raw_data[index]
        if point is not None:
            arr[index] = getattr(point, axis).to_base_units().m
            if units is None:
                units = getattr(point, axis).to_base_units().u
    return Q_(arr, units)

raw_data = np.empty((5,5), dtype=object)
for i in range(5):
    for j in range(5):
        raw_data[i,j] = Position(Q_(i, 'm'), Q_(j, 'mm'))

coords = get_coord('y')
print(coords.to())
# scna = MapData(
#     center=Position(Q_(4, 'mm'), Q_(6, 'mm'), Q_(10, 'mm')),
#     width = Q_(3, 'mm'),
#     height = Q_(5, 'mm'),
#     hpoints = 10,
#     vpoints = 20,
#     scan_plane='XY',
#     scan_dir='VLT'
# )

# pprint(propvals(scna))

# new_line = scna.add_line()
# print(scna.raw_data.shape)