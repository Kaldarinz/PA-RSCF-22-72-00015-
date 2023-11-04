from dataclasses import dataclass, field
from typing import Iterable, Sized, List, Any, TypeVar
from pint import UnitRegistry
from pint.facets.plain.quantity import PlainQuantity
import numpy as np
import numpy.typing as npt
from enum import Enum

ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity


# dct = {'one': 1, 'two': 2}

# print(next(iter(dct.items())))



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

class DataGroupd(Enum):
    Raw = 'raw_data'

print(DataGroupd.)