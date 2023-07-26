import matplotlib.pyplot as plt
import numpy as np
import pint
from collections import deque

ureg = pint.UnitRegistry()
q = deque(maxlen=5)

for i in range(7):
    q.appendleft(i*ureg('s'))
    #print(q)
    qlist = pint.Quantity.from_list([x for i,x in enumerate(q) if i<3])
    print(qlist)
