"""
Latest developing version
"""

from __future__ import annotations

import os, os.path
import logging
import logging.config
from datetime import datetime
from typing import Union

from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import pint
import yaml

import modules.validators as vd
from modules.pa_data import PaData
import modules.osc_devices as osc_devices
import modules.pa_logic as pa_logic
import modules.exceptions as exceptions

MESSAGES = {
    'cancel_in': 'Input terminated!'
}

def init_logs() -> logging.Logger:
    """Initiate logging"""

    with open('rsc/log_config.yaml', 'r') as f:
        log_config = yaml.load(f, Loader=yaml.FullLoader)

    # adding current data to name of log files
    for i in (log_config["handlers"].keys()):
        # Check if filename in handler.
        # Console handlers might be present in config file
        if 'filename' in log_config['handlers'][i]:
            log_filename = log_config["handlers"][i]["filename"]
            base, extension = os.path.splitext(log_filename)
            today = datetime.today().strftime("_%Y_%m_%d")
            log_filename = f"{base}{today}{extension}"
            log_config["handlers"][i]["filename"] = log_filename

    logging.config.dictConfig(log_config)
    return logging.getLogger('pa_cli')

def init_hardware(hardware: pa_logic.Hardware) -> None:
    """CLI for hardware init"""

    try:
        pa_logic.init_hardware(hardware)
    except exceptions.HardwareError as err:
        logger.error('Hardware initialization failed')

def home(hardware: pa_logic.Hardware) -> None:
    """CLI for Homes stages"""

    try:
        pa_logic.home(hardware)
    except exceptions.StageError as err:
        logger.error('Homing command failed')

def print_status(hardware: pa_logic.Hardware) -> None:
    """Prints current status and position of stages and oscilloscope"""
    
    # todo: add information about channels to which PM and PA sensor are connected
    logger.debug('print_status called')
    stage_X = hardware['stage_x']
    stage_Y = hardware['stage_y']
    osc = hardware['osc']
    stages_connected = True
    try:
        logger.info(f'X stage'
              + f'homing status: {stage_X.is_homed()}, '
              + f'status: {stage_X.get_status()}, '
              + f'position: {stage_X.get_position()*1000:.2f} mm.')
        logger.info(f'Y stage '
              + f'homing status: {stage_Y.is_homed()}, '
              + f'status: {stage_Y.get_status()}, '
              + f'position: {stage_Y.get_position()*1000:.2f} mm.')
    except:
        logger.error('Stages are not responding!')
        stages_connected = False

    if not osc.not_found:
        logger.info('Oscilloscope is initiated!')
    else:
        logger.error('Oscilloscope is not initialized!')

    if stages_connected and not osc.not_found:
        logger.info('All hardware is initiated!')

def save_data(data: PaData) -> None:
    """"CLI for saving data to a file."""

    if not data.attrs['filename']:
        logger.debug('Filename is missing in data, '
                     + 'asking to set filename.')
        filename = inquirer.text(
            message='Enter Sample name' + vd.cancel_option,
            default='Unknown',
            mandatory=False
        ).execute()
        logger.debug(f'"{filename}" filename entered')
        if filename == None:
            logger.debug('Save terminated by user')
            return
        suffix = '.hdf5'
        cwd = os.path.abspath(os.getcwd())
        sub_folder = 'measuring results'
        full_name = os.path.join(cwd, sub_folder, filename+suffix)
    else:
        full_name = data.attrs['filename']

    logger.debug(f'Trying to save to {full_name}')
    if os.path.exists(full_name):
        logger.debug('File already exists')
        override = inquirer.confirm(
            message=('Do you want to override file '
            + str(full_name) + '?')
        ).execute()
        logger.debug(f'{override=} have chosen')
        
        if not override:
            i = 1
            full_name_tmp = full_name.split('.hdf5')[0] + str(i) + '.hdf5'
            while os.path.exists(full_name_tmp):
                i +=1
                full_name_tmp = full_name.split('.hdf5')[0] + str(i) + '.hdf5'
            full_name = full_name_tmp

    logger.debug(f'Data will be saved to {full_name}')
    data.save(full_name)
    filename = os.path.basename(full_name)
    logger.info(f'Data saved to {full_name}')

def load_data(old_data: PaData) -> PaData:
    """Return loaded data in the related format"""

    logger.debug('Start loading data...')
    cwd = os.path.abspath(os.getcwd())
    sub_folder = 'measuring results'
    home_path = os.path.join(cwd, sub_folder)
    logger.debug(f'Default folder for data file search: {home_path}')
    
    file_path = inquirer.filepath(
        message='Choose spectral file to load:' + vd.cancel_option,
        default=home_path,
        mandatory=False,
        validate=PathValidator(is_file=True, message='Input is not a file')
    ).execute()
    logger.debug(f'"{file_path}" entered for loading data')
    
    if file_path == None:
        logger.info('Data loading canceled by user')
        return old_data
    
    file_ext = os.path.splitext(file_path)[-1]
    if file_ext != '.hdf5':
        logger.warning(f'Wrong data format = "{file_ext}", while '
                       + '".hdf5" is required')
        return old_data
    
    new_data = PaData()
    new_data.load(file_path)
    logger.info(f'Data with {len(new_data.raw_data)-1} PA measurements loaded!')
    return new_data
          
def set_new_position(hardware: pa_logic.Hardware) -> None:
    """Queries new position and move PA detector to this position"""

    logger.debug('Setting new position...')
    if not pa_logic.stages_open(hardware):
        logger.error('Setting new position cannot start')
        return

    x_dest = inquirer.text(
        message='Enter X destination [mm] ' + vd.cancel_option,
        default='0.0',
        validate=vd.ScanRangeValidator(),
        mandatory=False
    ).execute()
    logger.debug(f'{x_dest=} set')
    if x_dest is None:
        logger.warning('Input terminated by user')
        return
    x_dest = float(x_dest)
    
    y_dest = inquirer.text(
        message='Enter Y destination [mm] ' + vd.cancel_option,
        default='0.0',
        validate=vd.ScanRangeValidator(),
        mandatory=False
    ).execute()
    logger.debug(f'{y_dest=} set')
    if y_dest is None:
        logger.warning('Input terminated by user')
        return
    y_dest = float(y_dest)

    logger.info(f'Moving to ({x_dest:.2f},{y_dest:.2f})...')
    try:
        pa_logic.move_to(x_dest, y_dest, hardware)
        pa_logic.wait_stages_stop(hardware)
        pos_x = hardware['stage_x'].get_position(scale=True)*1000
        pos_y = hardware['stage_y'].get_position(scale=True)*1000
    except:
        logger.error('Error during communicating with stages')
        return
    
    logger.info('Moving complete!')
    logger.info(f'Current position ({pos_x:.2f},{pos_y:.2f})')

def measure_0d(hardware: pa_logic.Hardware,
               old_data: PaData) -> PaData:
    """CLI for single PA measurement"""
    logger.warning('measure 0D not implemented')
    return old_data

def measure_1d(hardware: pa_logic.Hardware,
               old_data: PaData) -> PaData:
    """CLI for 1D PA measurements."""

    data = old_data
    while True:
        measure_ans = inquirer.rawlist(
                message='Choose measurement',
                choices=[
                    'Spectral scanning',
                    'Back'
                ]).execute()
        logger.debug(f'"{measure_ans}" menu option choosen')

        if measure_ans == 'Spectral scanning':
            new_data = spectra(hardware)
            if new_data is not None:
                data = new_data
        elif measure_ans == 'Back':
            break
        else:
            logger.warning(f'Unknown command "{measure_ans}" '
                           + 'in Measure 1D menu')

    return data

def measure_2d(hardware: pa_logic.Hardware,
               old_data: PaData) -> PaData:
    """CLI for 2D PA measurements"""
    logger.warning('measure 2D not implemented')
    return old_data

def bp_filter(data: PaData) -> None:
    """CLI for bandpass filtration."""

    low_cutof = inquirer.text(
        message='Enter low cutoff frequency [MHz]' + vd.cancel_option,
        default='1',
        mandatory=False,
        validate=vd.FreqValidator()
    ).execute()
    logger.debug(f'{low_cutof} MHz set.')
    if low_cutof is None:
        logger.warning('Input terminated.')
        return
    low_cutof = float(low_cutof)*ureg.MHz

    high_cutof = inquirer.text(
        message='Enter high cutoff frequency [MHz]' + vd.cancel_option,
        default='10',
        mandatory=False,
        validate=vd.FreqValidator()
    ).execute()
    logger.debug(f'{high_cutof} MHz set.')
    if high_cutof is None:
        logger.warning('Input terminated.')
        return
    high_cutof = float(high_cutof)*ureg.MHz

    data.bp_filter(low_cutof,high_cutof)

def spectra(hardware: pa_logic.Hardware) -> Union[PaData,None]:
    """CLI for PA spectral measurement"""

    if not pa_logic.osc_open(hardware) or not pa_logic.pm_open(hardware):
        logger.error('Error in spectral measure: hardware is not open.')
        return None
    
    #CLI to get measuring options
    power_control = inquirer.select(
        message='Choose method for laser energy control:',
        choices=hardware['config']['power_control'],
        mandatory=False
    ).execute()
    logger.debug(f'"{power_control}" set as laser energy control')
    if power_control is None:
        logger.warning(MESSAGES['cancel_in'])
        return None

    start_wl = inquirer.text(
        message='Set start wavelength, [nm]' + vd.cancel_option,
        default='950',
        mandatory=False,
        validate=vd.WavelengthValidator()
    ).execute()
    logger.debug(f'"{start_wl=}" nm')
    if start_wl is None:
        logger.warning(MESSAGES['cancel_in'])
        return None
    start_wl = int(start_wl)*ureg('nm')

    end_wl = inquirer.text(
        message='Set end wavelength, [nm]' + vd.cancel_option,
        default='690',
        mandatory=False,
        validate=vd.WavelengthValidator()
    ).execute()
    logger.debug(f'"{end_wl=}" nm')
    if end_wl is None:
        logger.warning(MESSAGES['cancel_in'])
        return None
    end_wl = int(end_wl)*ureg('nm')

    step = inquirer.text(
        message='Set wavelength step, [nm]' + vd.cancel_option,
        default='10',
        mandatory=False,
        validate=vd.StepWlValidator()
    ).execute()
    logger.debug(f'"{step=}" nm')
    if step is None:
        logger.warning(MESSAGES['cancel_in'])
        return None
    step = int(step)*ureg('nm')

    target_energy = inquirer.text(
        message='Set target energy in [mJ]' + vd.cancel_option,
        default='0.5',
        mandatory=False,
        validate=vd.EnergyValidator()
    ).execute()
    logger.debug(f'"{target_energy=}" mJ')
    if target_energy is None:
        logger.warning(MESSAGES['cancel_in'])
        return None
    target_energy = float(target_energy)*ureg('mJ')

    averaging = inquirer.text(
        message='Set averaging' + vd.cancel_option,
        default='5',
        mandatory=False,
        validate=vd.AveragingValidator()
    ).execute()
    logger.debug(f'"{averaging=}"')
    if averaging is None:
        logger.warning(MESSAGES['cancel_in'])
        return None
    averaging = int(averaging)  

    return pa_logic.spectrum(
        hardware,
        start_wl,
        end_wl,
        step,
        target_energy,
        averaging
    )

def calc_filters_for_energy(hardware: pa_logic.Hardware) -> None:
    """CLI to find a filter combination"""

    #max filters for calculation
    max_combinations = hardware['config']['energy']['max_filters']

    logger.debug('Start calculating filter combinations')
    wl = inquirer.text(
        message='Set wavelength, [nm]' + vd.cancel_option,
        default='750',
        mandatory=False,
        validate=vd.WavelengthValidator()
    ).execute()
    logger.debug(f'wavelength: {wl} nm set.')
    if wl == None:
        logger.warning('Intup terminated!')
        return
    else:
        wl = int(wl)*ureg.nm

    target_energy = inquirer.text(
        message='Set target energy in [mJ]' + vd.cancel_option,
        default='0.5',
        mandatory=False,
        validate=vd.EnergyValidator()
    ).execute()
    logger.debug(f'{target_energy=} mJ set.')
    if target_energy == None:
        logger.warning('Intup terminated!')
        return
    else:
        target_energy = float(target_energy)*ureg.mJ

    logger.info('Please remove all filters!')
    energy = pa_logic.track_power(hardware, 50)
    logger.info(f'Power meter energy = {energy}.')
    filters = pa_logic.glass_calculator(
        wl,
        energy,
        target_energy,
        max_combinations,
    )
    if len(filters):
        logger.warning('Valid filter combination were not found!')

    reflection = pa_logic.glass_reflection(wl)
    if reflection is None or reflection == 0:
        logger.warning('Target power meter energy cannot be calculated!')
    else:
        target_pm_value = energy/reflection
        logger.info(f'Target power meter energy is {target_pm_value}!')

def glan_check(hardware: pa_logic.Hardware) -> None:
    """Used to check glan performance."""

    # [uJ] maximum value, which does not damage PM
    damage_thr = hardware['config']['pa_sensor']['damage_thr']*ureg.uJ

    logger.info('Start procedure to check Glan performance')
    logger.warning(f'Do not try energies at sample large than {damage_thr}')
    
    while True:
        logger.info('Set some energy at glass reflection')
        energy = pa_logic.track_power(hardware, 50)
        target_en = pa_logic.glan_calc(energy)
        if target_en is None:
            logger.warning('Target energy was not calculated')
            continue
        elif target_en > damage_thr:
            logger.warning('Energy at sample will damage the PM, '
                           + 'set smaller energy!')
            continue
        logger.info(f'Energy at sample should be ~{target_en}. Check it!')
        pa_logic.track_power(hardware,50)

        option = inquirer.rawlist(
            message='Choose an action:',
            choices=[
                'Measure again',
                'Back'
            ]
        ).execute()
        logger.debug(f'{option=} was set.')

        if option == 'Measure again':
            continue
        elif option == 'Back':
            break
        else:
            logger.warning('Unknown command in Glan chack menu!')

def export_to_txt(data: PaData) -> None:
    """CLI for export data to txt."""

    export_type = inquirer.rawlist(
        message='Choose data to export:',
        choices=[
            'raw_data',
            'filt_data',
            'freq_data',
            'Spectral',
            'back'
        ]
    ).execute()
    logger.debug(f'{export_type} selected in data export menu.')
    if export_type == 'back':
        return
    elif export_type in ('raw_data', 'filt_data', 'freq_data'):
        #build full filename
        filename = data.attrs['filename']
        if not filename:
            filename = os.path.join(os.getcwd(),'measuring results','Unknown')
        else:
            filename = filename.split('.hdf5')[0]
        if export_type == 'Raw data':
            filename += '-raw.txt'
        elif export_type == 'Filtered data':
            filename += '-filt.txt'
        elif export_type == 'Freq data':
            filename += '-freq.txt'

        #if file already exists, ask to override it
        if os.path.exists(filename):
            override = inquirer.confirm(
                message=f'Do you want to override file {filename}?'
            ).execute()
            logger.debug(f'File override = {override}')
            if override:
                try:
                    os.remove(filename)
                except OSError:
                    logger.warning(f'{filename} cannot be deleted.')
            else:
                filename_tmp = filename.split('.txt')[0]
                i = 1
                while os.path.exists(filename_tmp + str(i) + '.txt'):
                    i += 1
                filename = filename_tmp + str(i) + '.txt'
        
        data.export_txt(export_type, filename)
    elif export_type == 'Spectral':
        logger.warning('Export of spectral data to txt is not implemented.')
    else:
        logger.warning('Unknown command in data export menu!')

if __name__ == "__main__":

    logger = init_logs()
    logger.info('Starting application')

    ureg = pint.UnitRegistry(auto_reduce_dimensions=True) # type: ignore
    ureg.default_format = '~P'
    logger.debug('Pint activated (measured values with units)')
    ureg.setup_matplotlib(True)
    logger.debug('MatplotLib handlers for pint activated')


    logger.debug('Creating a dict for storing hardware refs')
    hardware: pa_logic.Hardware = {
        'stage_x': 0,
        'stage_y': 0,
        'osc': osc_devices.Oscilloscope(),
        'config': {}
    } # type: ignore
    pa_logic.load_config(hardware)
    
    logger.debug('Initializing PaData class for storing data')
    data = PaData()

    logger.debug('Entering main CLI execution loop')
    while True:
        menu_ans = inquirer.rawlist(
            message='Choose an action',
            choices=[
                'Init and status',
                'Data',
                'Utils',
                'Measurements',
                'Exit'
            ],
            height=9).execute()
        logger.debug(f'"{menu_ans}" menu option choosen')

        if menu_ans == 'Init and status':
            while True:
                stat_ans = inquirer.rawlist(
                message='Choose an action',
                choices=[
                    'Init hardware',
                    'Get status',
                    'Home stages',
                    'Back'
                ]).execute()
                logger.debug(f'"{stat_ans}" menu option choosen')

                if stat_ans == 'Init hardware':
                    init_hardware(hardware)
                elif stat_ans == 'Home stages':
                    home(hardware)
                elif stat_ans == 'Get status':
                    print_status(hardware)
                elif stat_ans == 'Back':
                    break
                else:
                    logger.warning(f'Unknown command "{stat_ans}" '
                                   + 'in init and status menu!')

        elif menu_ans == 'Data':
            while True:
                data_ans = inquirer.rawlist(
                    message='Choose data action',
                    choices=[
                        'Load data',
                        'View data',
                        'Save data',
                        'FFT filtration',
                        'Export to txt',
                        'Back to main menu'
                    ]
                ).execute()
                logger.debug(f'"{data_ans}" menu option choosen')

                if data_ans == 'Load data':
                    data = load_data(data)
                elif data_ans == 'Save data':
                    save_data(data)
                elif data_ans == 'View data':
                    data.plot()
                elif data_ans == 'FFT filtration':
                    bp_filter(data)
                elif data_ans == 'Export to txt':
                    export_to_txt(data)
                elif data_ans == 'Back to main menu':
                        break         
                else:
                    logger.warning('Unknown command in Spectral scanning menu!')

        elif menu_ans == 'Utils':
            while True:
                utils_menu = inquirer.rawlist(
                    message='Choose an option',
                    choices = [
                        'Power meter',
                        'Glan check',
                        'Filter caclulation',
                        'Move detector',
                        'Back'
                    ]
                ).execute()
                logger.debug(f'"{utils_menu}" menu option choosen')

                if utils_menu == 'Power meter':
                    pa_logic.track_power(hardware, 100)
                elif utils_menu == 'Glan check':
                    glan_check(hardware)
                elif utils_menu == 'Filter caclulation':
                    calc_filters_for_energy(hardware)
                elif utils_menu == 'Move detector':
                    set_new_position(hardware)
                elif utils_menu == 'Back':
                    break
                else:
                    logger.warning('Unknown command in utils menu!')

        elif menu_ans == 'Measurements':
            while True:
                measure_ans = inquirer.rawlist(
                    message='Choose measurement action',
                    choices=[
                        'Measure 0D data',
                        'Measure 1D data',
                        'Measure 2D data',
                        'View measured data',
                        'Back to main menu'
                    ]
                ).execute()
                logger.debug(f'"{measure_ans}" menu option choosen')
                
                if measure_ans == 'Measure 0D data':
                    data = measure_0d(hardware, data)

                elif measure_ans == 'Measure 1D data':
                    data = measure_1d(hardware, data)

                elif measure_ans == 'Measure 2D data':
                    data = measure_2d(hardware, data)

                elif measure_ans == 'View measured data':
                    data.plot()

                elif measure_ans == 'Back to main menu':
                        break         

                else:
                    logger.warning(f'Unknown command "{measure_ans}" in measurement menu')

        elif menu_ans == 'Exit':
            exit_ans = inquirer.confirm(
                message='Do you really want to exit?'
                ).execute()
            if exit_ans:
                logger.info('Stopping application')
                if hardware['stage_x']:
                    hardware['stage_x'].close()
                if hardware['stage_y']:
                    hardware['stage_y'].close()
                exit()

        else:
            logger.warning(f'Unknown command "{menu_ans}" in the main menu!')