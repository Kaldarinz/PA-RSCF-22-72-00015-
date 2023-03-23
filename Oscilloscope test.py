import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt


def FindInstrument():
    instrument_name = list(filter(lambda x: 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' in x,
                                  all_instruments))  # USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR адрес прибора. Если используете другой прибор, то посмотрите вывод строки print(all_instruments)
    if len(instrument_name) == 0:
        print('Осциллограф не найден в списке устройств')
    else:
        return instrument_name[0]

rm = pv.ResourceManager()  # вызывает менеджер работы
all_instruments = rm.list_resources()  # показывает доступные порты передачи данных,имя которых по дефолту заканчивается на ::INSTR. USB RAW и TCPIP SOCKET не выводятся, но чтобы их посмотерть: '?*' в аргумент list_resources()
print(FindInstrument())  # вызываю функцию для теста подключения. Должна вернуть адрес USB
rigol = rm.open_resource(FindInstrument())


rigol.write(':WAV:MODE RAW')

timeoffset = float(rigol.query(':TIM:OFFS?'))
voltscale1 = float(rigol.query(':CHAN1:SCAL?'))
voltscale2 = float(rigol.query(':CHAN2:SCAL?'))
voltoffset1 = float(rigol.query(':CHAN1:OFFS?'))
voltoffset2 = float(rigol.query(':CHAN2:OFFS?'))

rigol.write(':WAV:STOP')
rigol.write(':WAV:SOUR CHAN1')  # Задание считывания данных с 1 канала
rigol.write(':WAV:DATA?')
rawdata1 = rigol.read_raw()  # Считывание информации с канала 1
rigol.write(':WAV:STARt')
print(rawdata1)
print(rigol.query(':WAV:MODE?'))
rawdata1 = rawdata1[10:]
data_size = len(rawdata1)
print(data_size)
data1 = np.frombuffer(rawdata1, 'B')
print(data1)
data1 = data1 + 255  # ИЗМЕНЕНО
data1 = ((data1 - 130.0 - voltoffset1 / voltscale1 * 28) / 25 * voltscale1)*1000
plt.plot(data1[1:-1])
plt.show()



