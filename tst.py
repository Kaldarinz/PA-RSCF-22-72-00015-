from pint.facets.plain.quantity import PlainQuantity
import pint
from modules import Q_
import numpy as np


start = Q_(0, 'nanometer')
stop = Q_(5, 'micrometer')
step = Q_(1, 'micrometer')

dep = pint.Quantity.from_list((start,step), f'{start.u:~}')

print(f'{dep.u:~}')

