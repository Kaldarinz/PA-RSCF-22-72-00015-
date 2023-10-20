from pint.facets.plain.quantity import PlainQuantity
from modules import Q_

a = Q_([1,2],'V')

if isinstance(a,PlainQuantity):
    print('True')
else: 
    print('False')

