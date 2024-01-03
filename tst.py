from dataclasses import dataclass, field, fields
from collections.abc import Iterable, Sequence
from datetime import datetime
import time
import typing
from pint import UnitRegistry
from pint.facets.plain.quantity import PlainQuantity
import pint
import numpy as np
import numpy.typing as npt
from enum import Enum
from modules.data_classes import Coordinate
from modules.constants import Priority


ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity
rng = np.random.default_rng()

class tst:
    a = Q_(np.empty(0), 's')

    def __init__(self, a: PlainQuantity|None = None) -> None:
        self.a = a

    def __repr__(self) -> str:
        return str(self.a)

cls1 = tst(Q_(1,'s'))
cls2 = tst(Q_(2,'s'))
cls3 = tst()
if cls3.a is None:
    print('ok')
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
