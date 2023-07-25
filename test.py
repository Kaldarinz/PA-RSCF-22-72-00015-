import matplotlib.pyplot as plt
import numpy as np
import pint

ureg = pint.UnitRegistry()
ureg.setup_matplotlib(True)

y = np.linspace(0, 30) * ureg.miles
x = np.linspace(0, 5) * ureg.hours

fig, ax = plt.subplots()
ax.plot(y)
plt.show()