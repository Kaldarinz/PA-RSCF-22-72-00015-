from pint.facets.plain.quantity import PlainQuantity
from modules import Q_
import numpy as np


start = Q_(0, 's')
stop = Q_(5, 's')
step = Q_(1, 's')

arr = np.arange(start,stop,step)
print(arr)

