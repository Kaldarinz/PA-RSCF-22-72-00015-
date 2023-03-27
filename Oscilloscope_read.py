import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt

####### Все численные параметры задаются здесь
data_points_amount = 240000 # задаём сколько точек считывается за раз (для типа BYTE максимум 240 000)
data_chunks_amount = 2 # сколько раз будут читаться данные
read_channel = 'CHAN1'
#######

def FindInstrument():
    instrument_name = list(filter(lambda x: 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' in x,
                                  all_instruments))  # USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR адрес прибора. Если используете другой прибор, то посмотрите вывод строки print(all_instruments)
    if len(instrument_name) == 0:
        print('Осциллограф не найден в списке устройств')
        exit()
    else:
        print('Осциллограф найден!')
        print('Адрес:', instrument_name[0])
        return instrument_name[0]

rm = pv.ResourceManager()  # вызывает менеджер работы
all_instruments = rm.list_resources()  # показывает доступные порты передачи данных,имя которых по дефолту заканчивается на ::INSTR. USB RAW и TCPIP SOCKET не выводятся, но чтобы их посмотерть: '?*' в аргумент list_resources()
rigol = rm.open_resource(FindInstrument())

rigol.write(':ACQ:MDEP 24000000')
rigol.write(':STOP')
rigol.write(':WAV:SOUR ' + read_channel)
rigol.write(':WAV:MODE RAW')
rigol.write('"WAV:FORM BYTE')

params_raw = rigol.query(':WAV:PRE?').split(',')
params = {
    'format': int(params_raw[0]), # 0 - BYTE, 1 - WORD, 2 - ASC 
    'type': int(params_raw[1]), # 0 - NORMal, 1 - MAXimum, 2 RAW
    'points': int(params_raw[2]), # between 1 and 240000000
    'count': int(params_raw[3]), # the number of averages in the average sample mode and 1 in other modes
    'xincrement': float(params_raw[4]), # the time difference brtween two neighboring points in the X direction
    'xorigin': float(params_raw[5]), # the start time of the waveform data in the X direction
    'xreference': float(params_raw[6]), # the reference time of the data point in the X direction
    'yincrement': float(params_raw[7]), # the waveform increment in the Y direction
    'yorigin': float(params_raw[8]), # the vertical offset relative to the "Vertical Reference Position" in the Y direction
    'yreference': float(params_raw[9]) #the vertical reference position in the Y direction
}

# по факту триггерный сигнал в середине сохранённого диапазона.
data_start = (int(params['points']/data_points_amount/2) - 1) * data_points_amount # выбираем начальную точку

sample_rate = float(rigol.query(':ACQ:SRAT?')) # шаг по времени в RAW 1/sample_rate
frame_duration = data_points_amount/sample_rate # длительность кадра считывания
total_duration = frame_duration * data_chunks_amount # общая длительность считывания

print('xincerement = ', params['xincrement'] * 1000000000, 'ns')
print('1/SampleRate = ', 1000000000/sample_rate, 'ns')

if (1/sample_rate) != params['xincrement']:
    print('Sample rate reading problem')

print('frame duration = ', frame_duration * 1000000, 'us')
print('total read duration = ', total_duration * 1000000, 'us')

data = np.zeros(data_chunks_amount*data_points_amount)

for i in range(data_chunks_amount):
    rigol.write(':WAV:STAR ' + str(i * data_points_amount + 1 + data_start))
    rigol.write(':WAV:STOP ' + str((i + 1) * data_points_amount + data_start))

    rigol.write(':WAV:DATA?')
    data_chunk = np.frombuffer(rigol.read_raw(), dtype=np.int8)

    data_chunk = (data_chunk - params['xreference'] - params['yorigin']) * params['yincrement']
    data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
    data[i*data_points_amount:data_points_amount*(i+1)] += data_chunk[12:]
rigol.write(':RUN')

x_data = np.arange(0, params['xincrement']*data[11:].size, params['xincrement'])

plt.plot(x_data, data[11:])
plt.show()