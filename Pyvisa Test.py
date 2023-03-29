import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os.path
#import earthpy as et
import statistics


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
rigol = rm.open_resource(FindInstrument())  # обозначение переменной устройства
print('Подключение установлено. Название устройства:', rigol.query('*IDN?'), end=' ')
print('Проверить: датчик для лазера на CH1, Датчик для ультразвука на СН2.')
rigol.write(':RUN')

timeoffset = float(rigol.query(':TIM:OFFS?')[0])
voltscale1 = float(rigol.query(':CHAN1:SCAL?')[0])
voltscale2 = float(rigol.query(':CHAN2:SCAL?')[0])
voltoffset1 = float(rigol.query(':CHAN1:OFFS?')[:-1])
voltoffset2 = float(rigol.query(':CHAN2:OFFS?')[:-1])

rigol.write(':WAV:POIN:MODE RAW')
rigol.write(':WAV:SOUR CHAN1')  # Задание считывания данных с 1 канала
rigol.write(':WAV:DATA? CHAN1')

rawdata1 = rigol.read_raw()  # Считывание информации с канала 1
rawdata1 = rawdata1[10:]
data_size = len(rawdata1)
data1 = np.frombuffer(rawdata1, 'B')
data1 = data1 + 255  # ИЗМЕНЕНО
data1 = (data1 - 130.0 - voltoffset1 / voltscale1 * 25) / 25 * voltscale1

channel2_array0 = []
channel2_array1 = []
channel2_array2 = []

rigol.write(':WAV:SOUR CHAN2')  # Задание считывания данных с 2 канала
rigol.write(':WAV:DATA? CHAN2')
rawdata2 = rigol.read_raw()  # Считывание информации с канала 2
rawdata2 = rawdata2[10:]
data_size = len(rawdata2)
data2 = np.frombuffer(rawdata2, 'B')
data2 = data2 + 255  # ИЗМЕНЕНО
data2 = (data2 - 130.0 - voltoffset2 / voltscale2 * 25) / 25 * voltscale2
channel2_array0.append(data2)

rigol.write(':WAV:SOUR CHAN2')  # Задание считывания данных с 2 канала
rigol.write(':WAV:DATA? CHAN2')
rawdata2 = rigol.read_raw()  # Считывание информации с канала 2
rawdata2 = rawdata2[10:]
data_size = len(rawdata2)
data2 = np.frombuffer(rawdata2, 'B')
data2 = data2 + 255  # ИЗМЕНЕНО
data2 = (data2 - 130.0 - voltoffset2 / voltscale2 * 25) / 25 * voltscale2
channel2_array1.append(data2)

rigol.write(':WAV:SOUR CHAN2')  # Задание считывания данных с 2 канала
rigol.write(':WAV:DATA? CHAN2')
rawdata2 = rigol.read_raw()  # Считывание информации с канала 2
rawdata2 = rawdata2[10:]
data_size = len(rawdata2)
data2 = np.frombuffer(rawdata2, 'B')
data2 = data2 + 255  # ИЗМЕНЕНО
data2 = (data2 - 130.0 - voltoffset2 / voltscale2 * 25) / 25 * voltscale2
channel2_array2.append(data2)

average_data2 = []
for i in range(len(channel2_array0)):
    average_data2.append((channel2_array0[i] + channel2_array1[i] + channel2_array2[i])/3)

average_data = np.array(average_data2[0])

plt.plot(average_data[1:-1], linewidth=0.3)
plt.xlabel('Время, мкс', fontsize=12)
plt.ylabel('Напряжение, мВ', fontsize=12)
plt.show()
plt.close()


fig, axc = plt.subplots(2, sharex = True)
fig.tight_layout()
axc[0].plot(data1[1:-1], 'tab:orange', linewidth=0.7)
axc[0].set_title('Сигнал от светового датчика', fontsize=12)
axc[0].set_ylabel('Напряжение, мВ', fontsize=11)
axc[1].plot(average_data[1:-1], 'tab:blue', linewidth=0.3)
axc[1].set_title('Сигнал от УЗ датчика', fontsize=12)
axc[1].set_ylabel('Напряжение, мВ', fontsize=11)
axc[1].set_xlabel('Время, мкс')

plt.show()

print('Файлы сохранены в директории:', os.path.join(et.io.HOME, 'Desktop', 'Фотоаккустика'))

