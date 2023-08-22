import pint
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
ureg.default_format = '~P'

a = 1*ureg.s
b = 2*ureg.s

print(a)

ulist = [a,b]
print(max(ulist))