from dataclasses import dataclass, field
from typing import Iterable, Sized, List, Any, TypeVar
from pint import UnitRegistry
from pint.facets.plain.quantity import PlainQuantity
import numpy as np
import numpy.typing as npt

ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity


t1 = ('cba', 5)
t2 = ('acb', 4)
t3 = ('bac_u', 3)
t4 = ('bac', 2)

lst = [Q_(1,'s'),Q_(2,'s')]

if isinstance(lst,list):
    print('hm')

# print(lst)
# lst.sort()
# print(lst)

# print(t3[0].replace('ac', ''))



@dataclass
class Test:
    lst: list[str]
    atr: int = 0
    atr2: PlainQuantity = Q_(1,'s')
    data: int = 2
    #dct: dict[str, int] = {'one': 2}

# cls = Test()
print(Test.__annotations__.keys())
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

