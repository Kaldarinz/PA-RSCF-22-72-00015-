import pint
import numpy as np

ureg = pint.UnitRegistry(auto_reduce_dimensions=True)

arr = np.ones(50)
t = ureg('1ns')
arr2 = arr*t

print(arr2)

kernel = np.ones(4)
arr2 = np.convolve(arr2.magnitude,kernel)*arr2.units

print(type(arr2))