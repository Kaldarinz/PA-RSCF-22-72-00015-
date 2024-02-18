from enum import IntEnum

from . import Q_

POINT_SIGNALS = {
    'Raw': 'raw_data',
    'Filtered': 'filt_data',
    'Zoomed Raw': 'raw_data',
    'Zoomed_Filtered': 'filt_data',
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
class Priority(IntEnum):
    HIGHEST = 10
    HIGH = 8
    NORMAL = 5
    LOW = 2
    LOWEST = 0
