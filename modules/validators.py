"""
module with imput validators for CLI
"""

from prompt_toolkit.validation import ValidationError, Validator

cancel_option = '\n(press CTRL+Z to cancel)\n'

class ScanRangeValidator(Validator):
    def validate(self, document):
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message="Input should be a number!",
                cursor_position=document.cursor_position
            )
        if float(document.text) < 0:
            raise ValidationError(
                message='Coordinate cannot be less than 0',
                cursor_position=document.cursor_position
            )
        if float(document.text) > 25:
            raise ValidationError(
                message='coordinate cannot be more than 25',
                cursor_position=document.cursor_position
            )
        
class ScanPointsValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Input should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 2:
            raise ValidationError(
                message='Scan points cannot be less than 2',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 30:
            raise ValidationError(
                message='Scan points large than 30 are not supported',
                cursor_position=document.cursor_position
            )
        
class WavelengthValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Wavelength should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 690:
            raise ValidationError(
                message='Wavelengthes smaller than 690 are not supported',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 950:
            raise ValidationError(
                message='Wavelengthes larger than 690 are not supported',
                cursor_position=document.cursor_position
            )
        
class StepWlValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="step should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 10:
            raise ValidationError(
                message='Steps smaller than 10 nm are not supported',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 100:
            raise ValidationError(
                message='Steps larger than 100 nm are not supported',
                cursor_position=document.cursor_position
            )
        
class EnergyValidator(Validator):
    def validate(self, document):
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message="energy should be a number!",
                cursor_position=document.cursor_position
            )
        if float(document.text) < 0.01:
            raise ValidationError(
                message='Energies smaller than 10 uJ are not supported',
                cursor_position=document.cursor_position
            )
        if float(document.text) > 100:
            raise ValidationError(
                message='Energies larger than 100 mJ are not supported',
                cursor_position=document.cursor_position
            )
        
class FilterNumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Filter number should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 1:
            raise ValidationError(
                message='At least 1 filter should be used',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 5:
            raise ValidationError(
                message='More than 5 filters are not supported',
                cursor_position=document.cursor_position
            )
        
class AveragingValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Value should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 1:
            raise ValidationError(
                message='At least 1 measurement should be made',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 20:
            raise ValidationError(
                message='Too many measurements :(',
                cursor_position=document.cursor_position
            )
        
class FreqValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Frequency should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 1:
            raise ValidationError(
                message='Frequency should not be less than 1 Hz',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 50000000:
            raise ValidationError(
                message='Frequency cannot be higher than 50 MHz',
                cursor_position=document.cursor_position
            )
        
class DtValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Time step should be an integer!",
                cursor_position=document.cursor_position
            )
        if int(document.text) < 2:
            raise ValidationError(
                message='Time steps less than 2 ns are not supported',
                cursor_position=document.cursor_position
            )
        if int(document.text) > 10000:
            raise ValidationError(
                message='Time steps larger than 10 us are not supported',
                cursor_position=document.cursor_position
            )