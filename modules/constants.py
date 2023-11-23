from enum import IntEnum

DetailedSignals = {
    'Raw': 'raw_data',
    'Filtered': 'filt_data',
    'Zoomed Raw': 'raw_data',
    'Zoomed_Filtered': 'filt_data',
    'FFT': 'freq_data'
}

ParamSignals = {
    'Raw': 'raw_data',
    'Filtered': 'filt_data'
}

class Priority(IntEnum):
    HIGHEST = 10
    HIGH = 8
    NORMAL = 5
    LOW = 2
    LOWEST = 0
