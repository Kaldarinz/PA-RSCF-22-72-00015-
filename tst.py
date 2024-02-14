from dataclasses import dataclass, field, fields
from typing import TypedDict
from pprint import pprint
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
import time, math
import typing
import copy
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
    Direction,
    DataPoint,
    BaseData,
    MeasurementMetadata
)
from modules.utils import (
    propvals
)
ureg = pint.get_application_registry()
Q_ = ureg.Quantity
rng = np.random.default_rng()

a = np.array((1,2,3,4,5))
b = [2,3]
print(a[*b])