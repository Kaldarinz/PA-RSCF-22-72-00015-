"""
Contain constants, used in the programm.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from enum import IntEnum

from . import Q_, ureg

POINT_SIGNALS = {
    'Raw': 'raw_data',
    'Filtered': 'filt_data',
    'FFT': 'freq_data'
}

MSMNTS_SIGNALS = {
    'Raw': 'raw_data',
    'Filtered': 'filt_data'
}
CURVE_PARAMS = {
    'Wavelength': {
        'start': Q_(740, 'nm'),
        'stop': Q_(750, 'nm'),
        'step': Q_(10, 'nm')
    },
    'Energy': {
        'start': Q_(100, 'uJ'),
        'stop': Q_(500, 'uJ'),
        'step': Q_(100, 'uJ')
    }
}

MSMNT_MODES = ['Single point', 'Curve', 'Map']

class Priority(IntEnum):
    HIGHEST = 10
    HIGH = 8
    NORMAL = 5
    LOW = 2
    LOWEST = 0