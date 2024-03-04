"""
Custom exceptions.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

class StageError(Exception):
    """Errors with mechanical stages."""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class OscIOError(Exception):
    """Input/output errors in communications with osc diveces."""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class OscValueError(Exception):
    """Errors during processing data in osc devices."""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class OscConnectError(Exception):
    """Oscilloscope is not found."""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class HardwareError(Exception):
    """General error with hardware."""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class PlotError(Exception):
    """Plotting error."""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))