from dataclasses import dataclass, field
from typing import Iterable, Sized
from pint import UnitRegistry
from pint.facets.plain.quantity import PlainQuantity
import numpy as np

ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity


class Test():
    def __init__(self) -> None:
        self.atr = 5
        self.atr2 = Q_(5,'s')
        self.data = 10

def to_dict(obj: object):

    result = {}
    for key, val in obj.__dict__.items():
        if key not in ('data', 'data_raw'):
            if isinstance(val, PlainQuantity):
                result.update(
                    {
                        key:val.m,
                        key + '_u': str(val.u)
                    }
                )
            else:
                result.update({key: val})
    return result
    
cls = Test()
print(to_dict(cls))
