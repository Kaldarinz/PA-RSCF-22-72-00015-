import pint

ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
ureg.default_format = '~P'

a = 1*ureg('ms')
b = 1*ureg('s')
c =a*b
a = -a
print(c)