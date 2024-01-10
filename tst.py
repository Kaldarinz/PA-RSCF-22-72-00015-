from dataclasses import dataclass, field, fields
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
from modules.data_classes import Coordinate
from modules.data_classes import (
    EnergyMeasurement,
    PaEnergyMeasurement,
    MeasuredPoint,
    OscMeasurement,
    Coordinate,
    MapData,
    StagesStatus,
    ScanLine
)
ureg = pint.get_application_registry()
Q_ = ureg.Quantity
rng = np.random.default_rng()

pos1 = Coordinate(Q_(0,'m'), Q_(0,'m'))
pos2 = Coordinate.from_tuples([('x', Q_(5,'m')), ('y', Q_(50,'m'))])
arr= [pos1,pos2]
print(arr)
arr[0].x = Q_(100,'m')
print(arr)
a = arr[0]
a.x = Q_(200, 'm')
print(arr)
a = Coordinate()
print(arr)
# @dataclass
# class Test:
#     lst: list[str]
#     atr: int = 0
#     atr2: PlainQuantity = Q_(1,'s')
#     data: int = 2
#     # dct: dict[str, int] = {'one': 2}

# def search(cls, val):
#     if cls.__annotations__[val] == list[str]:
#         print('It')

# search(Test, 'lst')

# cls = Test()
#print(Test.__annotations__.keys())
# init_dict = {}
# for key, val in Test.__annotations__.items():
#     print(type(val))
#     if val is int:
#         init_dict.update({key: 5})
#     if val is PlainQuantity:
#         init_dict.update({key: Q_(1,'s')})
#     if val == npt.NDArray[np.uint16]:
#         print('It!')
#         init_dict.update({key:[Q_(3,'s')]})
