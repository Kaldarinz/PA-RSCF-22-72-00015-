from dataclasses import dataclass, field
from typing import Iterable
from pint import UnitRegistry
import numpy as np

ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity


@dataclass
class Test:
    a: Iterable

arr = np.array([1,2,3])
quant = Q_(arr,'s')

cls = Test(quant)
print(cls)
quant = Q_(10,'s')
print(cls)