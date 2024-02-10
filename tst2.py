from time import time, gmtime, strftime, sleep
from datetime import datetime

strt = datetime.now()
sleep(0.2)
delta = datetime.now() - strt
a, b = str(delta).split('.')
print(f'{a}.{b[:1]}')
