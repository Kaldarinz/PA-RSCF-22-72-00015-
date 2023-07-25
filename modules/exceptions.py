"""
Custom exceptions
"""

class StageError(Exception):
    """Errors with mechanical stages"""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class OscilloscopeError(Exception):
    """Errors with Oscilloscope"""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))
    
class HardwareError(Exception):
    """General error with hardware"""

    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))