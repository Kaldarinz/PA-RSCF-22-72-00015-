import pyvisa as pv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import time

####### Все численные параметры задаются здесь

sample = 'Water'

pre_time = 10 # start time of data storage before trigger in micro seconds
frame_duration = 100 # whole duration of the stored frame
pm_response_time = 500 # response time of the power meter

read_channel1 = 'CHAN1'
read_channel2 = 'CHAN2'
averaging = 1

data_storage = 1 # 1 - Save data, 0 - do not save data 
#######

params = {}

def save_data(x_data, final_data) -> None:
    
    #make folder for data save if it does not exist
    Path('measuring results/').mkdir(parents=True, exist_ok=True)
    
    filename = 'measuring results/Sample_name-' + sample

    i = 1
    while (os.path.exists(filename + str(i) + '.txt')):
        i += 1
    filename = filename + str(i) + '.txt'
    
    np.savetxt(filename, final_data, header='X = time [s], Y = signal [V]')
    print('Data saved to ', filename)

def FindInstrument():
    rm = pv.ResourceManager()
    all_instruments = rm.list_resources()
    instrument_name = list(filter(lambda x: 'USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR' in x,
                                  all_instruments))  # USB0::0x1AB1::0x04CE::DS1ZD212100403::INSTR адрес прибора. Если используете другой прибор, то посмотрите вывод строки print(all_instruments)
    if len(instrument_name) == 0:
        print('Осциллограф не найден в списке устройств')
        exit()
    else:
        print('Осциллограф найден!')
        print('Адрес:', instrument_name[0], '\n')
        
    
    return rm.open_resource(instrument_name[0])

def read_data(data, channel, starting_point, points_number):
    rigol.write(':WAV:SOUR ' + channel)
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

    print('Channel = ', channel)
    print('Points amount', params['points'])
    # по факту триггерный сигнал в середине сохранённого диапазона.
    data_start = (int(params['points']/2) - starting_point) # выбираем начальную точку

    print('xincerement = ', params['xincrement'] * 1000000000, 'ns')

    for i in range(1): #мы скорей всего будем читать за 1 раз, но на всякий случай пока не удаляю код для считывания нескольких кадров
        rigol.write(':WAV:STAR ' + str(i * 240000 + data_start + 1))
        rigol.write(':WAV:STOP ' + str((i + 1) * points_number + data_start))
        rigol.write(':WAV:DATA?')
        data_chunk = np.frombuffer(rigol.read_raw(), dtype=np.int8)
        data_chunk = (data_chunk - params['xreference'] - params['yorigin']) * params['yincrement']
        data_chunk[-1] = data_chunk[-2] # убираем битый пиксель
        data[i*points_number:points_number*(i+1)] += data_chunk[12:]

    return data

def baseline_correction(data_array, points_number):

    baseline = np.average(data_array[:int(points_number/10)])
    return data_array-baseline

def pa_data_normalization(data_array, scale):
    """Divide array elements by scale"""
    return data_array/scale

def frame_size_calculation(frame_duration):
    """Calculate total points amount and starting point offset for storage"""

    sample_rate = float(rigol.query(':ACQ:SRAT?'))
    points = int(frame_duration*sample_rate/1000000) + 1
    return points

if __name__ == "__main__":

    rigol = FindInstrument()

    points_number_pm = frame_size_calculation(pm_response_time)
    data = np.zeros(points_number_pm)

    points_number_pa = frame_size_calculation(frame_duration)
    data_pa = np.zeros(points_number_pa)

    starting_point = frame_size_calculation(pre_time)

    for i in range(averaging):
        while int(rigol.query(':TRIG:POS?'))<0: #ждём пока можно будет снова читать данные
            time.sleep(0.1)
        rigol.write(':STOP')
        data += read_data(data, read_channel1, starting_point, points_number_pm)
        data_pa += read_data(data_pa, read_channel2, starting_point, points_number_pa)
        rigol.write(':RUN')
    data = data/averaging
    data_pa = data_pa/averaging

    data = baseline_correction(data, points_number_pm)
    data_pa = baseline_correction(data_pa, points_number_pa)

    laser_amplitude = data.max()
    #data_pa = pa_data_normalization(data_pa, laser_amplitude)
    print('Laser signal amplitude = ', laser_amplitude)

    rigol.write(':STOP')
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
    rigol.write(':RUN')
    x_data = np.arange(0, params['xincrement']*data_pa[11:].size, params['xincrement']) # generation of time points

    if data_storage == 1:
        save_data(x_data, np.stack((x_data, data[11:points_number_pa], data_pa[11:]), axis=1))

    fig, axc = plt.subplots(2, sharex = True)
    fig.tight_layout()
    axc[0].plot(x_data, data[11:points_number_pa], 'tab:orange', linewidth=0.7)
    axc[0].set_title('Channel 1', fontsize=12)
    axc[0].set_ylabel('Voltage, V', fontsize=11)
    axc[1].plot(x_data, data_pa[11:], 'tab:blue', linewidth=0.3)
    axc[1].set_title('Channel 2', fontsize=12)
    axc[1].set_ylabel('Voltage, V', fontsize=11)
    axc[1].set_xlabel('Time, s')
    plt.show()