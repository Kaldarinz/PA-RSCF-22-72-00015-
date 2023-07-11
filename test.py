import pint

ureg = pint.UnitRegistry()
ureg.default_format ='P'

accel = 1.3 * ureg('um/second**2')

def test_func(quant: pint.Quantity):
    print(f'{quant:~P}')

test_func(accel)