import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os.path
import earthpy as et
import statistics


def FindInstrument():
    instrument_name = list(filter(lambda x: 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' in x,
                                  all_instruments))  # USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR адрес прибора. Если используете другой прибор, то посмотрите вывод строки print(all_instruments)
    if len(instrument_name) == 0:
        print('Осциллограф не найден в списке устройств')
    else:
        return instrument_name[0]


def CreateDirectory():
    new_dir = os.path.join(et.io.HOME, 'Desktop', 'Фотоаккустика')
    dir_check = os.path.exists(new_dir)
    if dir_check == True:
        print('Директория сохранения файлов:', new_dir)
    else:
        os.mkdir(new_dir)
        print('Была создана новая дирректория для сохранения файлов:', new_dir)


rm = pv.ResourceManager()  # вызывает менеджер работы
all_instruments = rm.list_resources()  # показывает доступные порты передачи данных,имя которых по дефолту заканчивается на ::INSTR. USB RAW и TCPIP SOCKET не выводятся, но чтобы их посмотерть: '?*' в аргумент list_resources()
print(FindInstrument())  # вызываю функцию для теста подключения. Должна вернуть адрес USB
print(all_instruments)  # тест на список всех устройств. Убрать потом
rigol = rm.open_resource(FindInstrument())  # обозначение переменной устройства
print('Подключение установлено. Название устройства:', rigol.query('*IDN?'), end=' ')
print('Проверить: датчик для лазера на CH1, Датчик для ультразвука на СН2.')
rigol.write(':RUN')
print(rigol.query(':CHANnel1:BWLimit?'))  # запрос на ширину 1 канала (ТЕСТ)

CreateDirectory()

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

counter = 0
channel2_arrays = [[] for i in range(5)]
while counter <= 4:
    rigol.write(':WAV:SOUR CHAN2')  # Задание считывания данных с 2 канала
    rigol.write(':WAV:DATA? CHAN2')
    rawdata2 = rigol.read_raw()  # Считывание информации с канала 2
    rawdata2 = rawdata2[10:]
    data_size = len(rawdata2)
    data2 = np.frombuffer(rawdata2, 'B')
    data2 = data2 + 255  # ИЗМЕНЕНО
    data2 = (data2 - 130.0 - voltoffset2 / voltscale2 * 25) / 25 * voltscale2
    channel2_arrays[counter].append(data2)
    counter += 1

average_data2 = []
for i in range(len(channel2_arrays)):
    average_data2.append(statistics.mean(channel2_arrays[i]))

directory = os.path.join(et.io.HOME, 'Desktop', 'Фотоаккустика')
d = dt.datetime.now()
date_name = str(d.strftime("%H:%M:%S-%Y_%m_%d"))
print('Введите название файла')
user_given_name = str(input())
file_name = str('{}.txt'.format(user_given_name + '_' + date_name))
file_save = open(os.path.join(directory, file_name), 'w+')

data_content = str(average_data2)

for i in data_content:
    file.write(i)  # проверить, работает ли такая распаковка массива в документ, потому что до этого массив печается как [x, x, ..., x, x]
file.close()

plt.plot(average_data2[1:-1], linewidth=0.3)
plt.xlabel('Время, мкс', fontsize=12)
plt.ylabel('Напряжение, мВ', fontsize=12)
name_fig_1 = '{}.jpg'.format(user_given_name + '_Осциллограмма_УЗ_' + date_name)
plt.savefig(os.path.join(directory, name_fig_1))
plt.show()
plt.close()


fig, axc = plt.subplots(2, sharex=True)
fig.tight_layout()
axc[0].plot(data1[1:-1], 'tab:orange', linewidth=0.7)
axc[0].set_title('Сигнал от светового датчика', fontsize=12)
axc[0].set_ylabel('Напряжение, мВ', fontsize=11)
axc[1].plot(data2[1:-1], 'tab:blue', linewidth=0.3)
axc[1].set_title('Сигнал от УЗ датчика', fontsize=12)
axc[1].set_ylabel('Напряжение, мВ', fontsize=11)
axc[1].set_xlabel('Время, мкс')
name_fig_2 = '{}.jpg'.format(user_given_name + '_Совмещенная_Осциллограмма_' + name)
plt.savefig(os.path.join(directory, name_fig_2))

plt.show()

print('Файлы сохранены в директории:', os.path.join(et.io.HOME, 'Desktop', 'Фотоаккустика', file_name + '_' + name))

