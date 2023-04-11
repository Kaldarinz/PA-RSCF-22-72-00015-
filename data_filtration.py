import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq

### The first value of each filtered signal is the main frequence in Hz for this signal
### Parameters

low_cutof = 300000 # low cutoff frequency
high_cutof = 5000000 # high cutoff frequency

filename = 'measuring results/Sample_name-TiN-Full6.npy'
new_filename = filename.split('.')[0] + '_filtered.npy'

data = np.load(filename)
print('data loaded with shape', data.shape)

#dt = data[0,0,0]
dt = 1/50000000
print(dt)

x_data = np.arange(dt, data.shape[2]*dt, dt)

W = fftfreq(data.shape[2]-1, dt) # array with frequencies
f_signal = rfft(data[:,:,1:]) # signal in f-space


filtered_f_signal = f_signal.copy()

filtered_f_signal[:,:,(W<low_cutof)] = 0   # actual filtering procedure

if high_cutof > 1/(2.5*dt):
    filtered_f_signal[:,:,(W>1/(2.5*dt))] = 0 # Nyquist frequency check
else:
    filtered_f_signal[:,:,(W>high_cutof)] = 0  #

print('main_freq = ', W[filtered_f_signal[0,0,:].argmax()]/1000000, 'MHz') # надо подумать, что с этим делать

filtered_signal = np.zeros((data.shape))
filtered_signal[:,:,1:] = irfft(filtered_f_signal) # actual filtered signal

freq_array = W[filtered_f_signal.argmax(axis=-1)]/1000000
print('shape freq array', freq_array.shape)
filtered_signal[:,:,0] = W[filtered_f_signal.argmax(axis=-1)]/1000000

np.save(new_filename, filtered_signal)