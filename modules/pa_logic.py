"""
PA backend
"""

from typing import Any, TypedDict, Union
import logging
import os

import yaml
import pint

from pylablib.devices import Thorlabs
import modules.osc_devices as osc_devices
import modules.exceptions as exceptions
from .pa_data import PaData
from . import ureg

logger = logging.getLogger(__name__)

class Hardware_base(TypedDict):
    """Base TypedDict for references to hardware."""

    stage_x: Thorlabs.KinesisMotor
    stage_y: Thorlabs.KinesisMotor
    osc: osc_devices.Oscilloscope
    config_loaded: bool
    power_control: list

class Hardware(Hardware_base, total=False):
    """TypedDict for refernces to hardware."""
    
    power_meter: osc_devices.PowerMeter
    pa_sens: osc_devices.PhotoAcousticSensOlymp
    stage_z: Thorlabs.KinesisMotor

def init_hardware(hardware: Hardware) -> bool:
    """Initialize all hardware.
    
    Load hardware config from rsc/config.yaml if it was not done.
    """

    logger.info('Starting hardware initialization...')
    
    if not init_osc(hardware):
        logger.warning('Oscilloscope cannot be loaded')
        return False
    osc = hardware['osc']

    if not hardware['config_loaded']:
        logger.debug('Hardware configuration is not loaded')
        config = load_config(hardware)
        if config:
            if not init_stages(hardware):
                logger.warning('Stages cannot be loaded')
                return False
            
            pm_chan = int(config['power_meter']['connected_chan'])
            pm = hardware.get('power_meter')
            if pm is not None:
                pm.set_channel(pm_chan-1)
                logger.info(f'Power meter channel set to {pm_chan}')

            pa_chan = int(config['pa_sensor']['connected_chan'])
            pa = hardware.get('pa_sens')
            if pa is not None:
                pa.set_channel(pa_chan-1)
                logger.info(f'Power meter channel set to {pa_chan}')
            hardware['config_loaded'] = True
        else:
            logger.warning('Hardware config cannot be loaded')
            return False
    else:
        #config already loaded try to reload what is present
        if not init_stages(hardware):
            logger.warning('Stages cannot be loaded')
            return False
        
        pm = hardware.get('power_meter')
        if pm is not None:
            #save pm channel to apply it after init
            pm_chan = pm.ch
            pm = osc_devices.PowerMeter(osc)
            pm.set_channel(pm_chan)
            hardware.update({'power_meter': pm})
            logger.debug('Power meter reinitiated on the same channel')

        pa = hardware.get('pa_sens')
        if pa is not None:
            #save pa channel to apply it after init
            pa_chan = pa.ch
            pa = osc_devices.PhotoAcousticSensOlymp(osc)
            pa.set_channel(pa_chan)
            hardware.update({'pa_sens': pa})
            logger.debug('PA sensor reinitiated on the same channel')
                
    logger.info('Hardware initialization complete')
    return True

def load_config(hardware: Hardware) -> dict:
    """Load hardware configuration and return it as dict.

    Configuration is loaded from rsc/config.yaml.
    Additionally add all optional devices to hardware dict.
    """

    logger.debug('Start loading config...')
    
    base_path = os.path.dirname(os.path.dirname(__name__))
    sub_dir = 'rsc'
    filename = 'config.yaml'
    full_path = os.path.join(base_path, sub_dir, filename)
    try:
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)['hardware']
    except:
        logger.warning('Connfig cannot be properly loaded')
        return {}
    
    hardware.update({'power_control':config['power_control']})
    logger.debug(f'Power control options loaded:{hardware["power_control"]}')

    if int(config['stages']['axes']) == 3:
        logger.debug('Stage_z added to hardware list')
        hardware.update({'stage_z':0}) # type: ignore
    
    osc = hardware['osc']
    if bool(config['power_meter']['connected']):
        hardware.update({'power_meter': osc_devices.PowerMeter(osc)})
        logger.debug('Power meter added to hardare list')
    if bool(config['pa_sensor']['connected']):
        hardware.update(
            {'pa_sens':osc_devices.PhotoAcousticSensOlymp(osc)})
    logger.debug(f'Config file read')
    return config

def init_stages(hardware: Hardware) -> bool:
    """Initiate Thorlabs KDC based stages."""

    logger.debug(f'Init_stages is starting...')
    logger.debug('Checking if connection to stages is '
                +'already estblished')

    if stages_open(hardware):
        logger.info('Connection to all stages already established!')
        return True

    logger.debug('Searching for Thorlabs kinsesis devices (stages)')
    stages = Thorlabs.list_kinesis_devices()
    logger.debug(f'{len(stages)} devices found')

    if hardware.get('stage_z', default=None) is None:
        amount = 3
    else:
        amount = 2

    if len(stages) < amount:
        msg = f'Less than {amount} kinesis devices found!'
        logger.error(msg)
        return False

    connected = True
    stage1_ID = stages.pop()[0]
    #motor units [m]
    stage1 = Thorlabs.KinesisMotor(stage1_ID, scale='stage')
    try:
        stage1.is_opened()
    except:
        msg = 'Failed attempt to coomunicate with stage1'
        logger.error(msg)
        if connected:
            connected = False
    else:
        hardware['stage_x'] = stage1
        logger.info(f'Stage X with ID={stage1_ID} is initiated')

    stage2_ID = stages.pop()[0]
    #motor units [m]
    stage2 = Thorlabs.KinesisMotor(stage2_ID, scale='stage')
    try:
        stage2.is_opened()
    except:
        msg = 'Failed attempt to coomunicate with stage2'
        logger.error(msg)
        if connected:
            connected = False
    else:
        hardware['stage_y'] = stage2
        logger.info(f'Stage Y with ID={stage2_ID} is initiated')

    if amount == 3:
        stage3_ID = stages.pop()[0]
        #motor units [m]
        stage3 = Thorlabs.KinesisMotor(stage3_ID, scale='stage')
        try:
            stage3.is_opened()
        except:
            msg = 'Failed attempt to coomunicate with stage3'
            logger.error(msg)
            if connected:
                connected = False
        else:
            hardware['stage_z'] = stage3
            logger.info(f'Stage Z with ID={stage3_ID} is initiated')
    
    if connected:
        logger.info('Stage initiation is complete')
    else:
        logger.warning('Stages are not initiated')
    
    return connected

def init_osc(hardware: Hardware) -> bool:
    """Initialize oscilloscope.

    Return true if connection is already established or
    initialization is successfull.
    """
    
    logger.debug('Init_osc is starting...')
    logger.debug('Checking if connection is already estblished')

    if osc_open(hardware):
        logger.info('Connection to oscilloscope is already established!')
        return True

    logger.debug('No connection found. Trying to establish connection')
    try:
        hardware['osc'].initialize()
    except Exception as err:
        logger.error(f'Error {type(err)} while trying initialize osc')
        return False
    else:
        logger.debug('Oscilloscope initialization complete')
        return True

def stages_open(hardware: Hardware) -> bool:
    """Return True if all stages are responding and open.
    
    Never raise exceptions.
    """

    logger.debug('Starting connection check to stages')
    connected = True
    try:
        if not hardware['stage_x'].is_opened():
            logger.warning('Stage X is not open')
            connected = False
        else:
            logger.debug('Stage X is open')
        if not hardware['stage_y'].is_opened():
            logger.warning('Stage Y is not open')
            connected = False
        else:
            logger.debug('Stage Y is open')
        stage_z = hardware.get('stage_z', default=None)
        if stage_z is not None:
            if not hardware['stage_z'].is_opened():
                logger.warning('Stage Z is not open')
                connected = False
            else:
                logger.debug('Stage Z is open')
    except:
        logger.error('Error during stages connection check')
        connected = False
    
    if connected:
        logger.debug('All stages are connected and open')

    return connected

def osc_open(hardware: Hardware) -> bool:
    """Return true if oscilloscope is connected."""

    logger.debug('Starting connection check to oscilloscope')
    hardware['osc'].connection_check()
    connected = not hardware['osc'].not_found
    logger.debug(f'Oscilloscope {connected=}')
    return connected

def pm_open(hardware: Hardware) -> bool:
    """Return true if power meter is configured."""

    logger.debug('Starting power meter connection check')

    if hardware.get('power_meter') is None:
        logger.warning('Power meter is not initialized')
        connected = False
    else:
        connected = osc_open(hardware)
        logger.debug(f'Power meter {connected=}')
    return connected

def move_to(X: float, Y: float, hardware: Hardware) -> None:
    """Send PA detector to (X,Y) position.
    
    Do not wait for stop moving.
    Coordinates are in mm.
    """
    
    x_dest_mm = X/1000
    y_dest_mm = Y/1000

    logger.debug(f'Sending X stage to {x_dest_mm} mm position')
    try:
        hardware['stage_x'].move_to(x_dest_mm)
    except:
        msg = 'Stage X move_to command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

    logger.debug(f'Sending Y stage to {y_dest_mm} mm position')
    try:
        hardware['stage_y'].move_to(y_dest_mm)
    except:
        msg = 'Stage Y move_to command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

def wait_stages_stop(hardware: Hardware) -> None:
    """Wait untill all (2) stages stop."""

    logger.debug('waiting untill stages complete moving')
    try:
        hardware['stage_x'].wait_for_stop()
        logger.debug('Stage X stopped')
    except:
        msg = 'Stage X wait_for_stop command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)
    
    try:
        hardware['stage_y'].wait_for_stop()
        logger.debug('Stage Y stopped')
    except:
        msg = 'Stage Y wait_for_stop command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

def home(hardware: Hardware) -> None:
    """Home all (2) stages."""

    logger.debug('homing is starting...')
    try:
        logger.debug('homing stage X')
        hardware['stage_x'].home(sync=False,force=True)
    except:
        msg = 'Stage X homing command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)

    try:
        logger.debug('homing stage Y')
        hardware['stage_y'].home(sync=False,force=True)
    except:
        msg = 'Stage Z homing command failed'
        logger.error(msg)
        raise exceptions.StageError(msg)
    
    wait_stages_stop(hardware)

def spectrum(
        hardware: Hardware,
        start_wl: pint.Quantity,
        end_wl: pint.Quantity,
        step: pint.Quantity,
        target_energy: pint.Quantity,
        averaging: int,
        max_filter_comb: int=0
) -> Union[PaData,None]:
    """Measure dependence of PA signal on excitation wavelength.
    
    Measurements start at 'start_wl' wavelength and are taken
    every 'step' with the last measurement at 'end_wl',
    so that the last step could be smaller than others.
    Measurements are taken at 'target_energy'.
    Each measurement will be averaged 'averaging' times.
    'max_filters_comb' is used calculations of filter's combinations
    when energy is controlled by the glass filters.
    """

    #make steps negative if going from long WLs to short
    if start_wl > end_wl:
        step = -step # type: ignore
        
    logger.info('Start measuring spectra!')

    #calculate amount of data points
    d_wl = end_wl-start_wl
    if d_wl%step:
        spectral_points = int(d_wl/step) + 2
    else:
        spectral_points = int(d_wl/step) + 1

    #create data class and set basic metadata
    data = PaData()
    data.set_metadata(data_group='general', metadata={
        'parameter name': 'wavelength',
        'parameter units': 'nm'
    })
    data.set_metadata(data_group='raw_data', metadata={
        'x var name': 'time',
        'x var units': 's',
        'y var name': 'PA signal',
        'y var units': 'V/uJ'
    })
    data.set_metadata(data_group='filt_data', metadata={
        'x var name': 'time',
        'x var units': 's',
        'y var name': 'PA signal',
        'y var units': 'V/uJ'
    })

    #main measurement cycle
    for i in range(spectral_points):
        
        #calculate current wavelength
        if abs(step*i) < abs(d_wl):
            current_wl = start_wl + step*i
        else:
            current_wl = end_wl

        # temp vars for averaging
        # should be reset in each sycle
        tmp_signal = 0
        tmp_laser = 0
        counter = 0

        print(f'\n{bcolors.HEADER}'
              + f'Start measuring point {(i+1)}'
              + f'{bcolors.ENDC}')
        print('Current wavelength is'
              + f'{bcolors.OKBLUE}{current_wl}{bcolors.ENDC}.'
              + 'Please set it!')
        
        #adjust laser energy with color glasses
        if power_control == 'Filters':
            print(f'{bcolors.UNDERLINE}'
                  + 'Please remove all filters!'
                  + f'{bcolors.ENDC}')
            #measure mean energy at glass reflection
            energy = track_power(hardware, 50)
            print(f'Power meter energy = {energy:.0f} [uJ]')

            #find valid filters combinations for current parameters
            filters,_,_ = glass_calculator(
                current_wl,
                energy,
                target_energy,
                max_combinations,
                no_print=True)
            if not len(filters):
                print(f'{bcolors.WARNING}'
                      + 'WARNING! No valid filter combination for '
                      + f'{current_wl} [nm]!{bcolors.ENDC}')
                cont_ans = inquirer.confirm(
                    message='Do you want to continue?').execute()
                if not cont_ans:
                    print(f'{bcolors.WARNING}'
                          + 'Spectral measurements terminated!'
                          + f'{bcolors.ENDC}')
                    return old_data

            #call glass_calculator again to print valid filter combinations
            _, target_pm_value,_ = glass_calculator(
                current_wl,
                energy,
                target_energy,
                max_combinations)
            
            print(f'Target power meter energy is {target_pm_value}!')
            print('Please set it using'
                  + f'{bcolors.UNDERLINE}laser software{bcolors.ENDC}')
        
        elif power_control == 'Glan prism':
            #for energy control by Glan prism target power meter
            #energy have to be calculated only once
            if i == 0:
                target_pm_value = glan_calc_reverse(target_energy*1000)
            print(f'Target power meter energy is {target_pm_value}!') #type: ignore
            print(f'Please set it using {bcolors.UNDERLINE}Glan prism{bcolors.ENDC}!')
            _ = track_power(hardware, 50)
        else:
            print(f'{bcolors.WARNING}'
                  + 'Unknown power control method! Measurements terminated!'
                  + f'{bcolors.ENDC}')
            return old_data

        #start measurements and averaging
        while counter < averaging:
            print('Signal at current WL should be measured '
                  + f'{averaging-counter} more times.')
            measure_ans = inquirer.rawlist(
                message='Chose an action:',
                choices=['Tune power','Measure','Stop measurements']
            ).execute()

            #adjust energy
            if measure_ans == 'Tune power':
                track_power(hardware, 40)

            #actual measurement
            elif measure_ans == 'Measure':
                
                #measure data on both channels
                osc.measure()
                dt = 1/osc.sample_rate

                #set signal data depending on channel assignment
                if chan_pa == 'CHAN1':
                    pa_signal = osc.ch1_data
                    pa_amp = osc.ch1_amp
                    pm_signal = osc.ch2_data
                elif chan_pa == 'CHAN2':
                    pa_signal = osc.ch2_data
                    pa_amp = osc.ch2_amp
                    pm_signal = osc.ch1_data
                else:
                    print(f'{bcolors.WARNING}'
                          + 'Uknown channel assignment in spectra'
                          + f'{bcolors.ENDC}')
                    return old_data
                
                #calculate laser energy at power meter
                cur_pm_laser = pm.energy_from_data(
                        pm_signal,
                        1/osc.sample_rate)
                #show measured data to visually check if data is OK
                fig = plt.figure(tight_layout=True)
                gs = gridspec.GridSpec(1,2)
                ax_pm = fig.add_subplot(gs[0,0])
                ax_pm.plot(pm_signal)
                #ax_pm.plot(signal.decimate(pm_signal, 100))
                #add markers for data start and stop
                ax_pm.plot(
                    pm.start_ind,
                    pm_signal[pm.start_ind],
                    'o',
                    alpha=0.4,
                    ms=12,
                    color='green')
                ax_pm.plot(
                    pm.stop_ind,
                    pm_signal[pm.stop_ind],
                    'o',
                    alpha=0.4,
                    ms=12,
                    color='red')
                ax_pa = fig.add_subplot(gs[0,1])
                ax_pa.plot(pa_signal)
                plt.show()

                #confirm that the data is OK
                good_data = inquirer.confirm(
                    message='Data looks good?').execute()
                if good_data:
                    #note that tmp_signal is only amplitude of signal for averaging
                    tmp_signal += pa_amp
                    tmp_laser += cur_pm_laser
                    counter += 1
                    if counter == averaging:
                        if power_control == 'Filters':
                            _, ___, sample_energy_aver = glass_calculator(
                                current_wl,
                                tmp_laser/averaging, #average energy
                                target_energy,
                                max_combinations,
                                no_print=True)

                            _, ___, sample_energy_cur = glass_calculator(
                                current_wl,
                                cur_pm_laser, #current energy
                                target_energy,
                                max_combinations,
                                no_print=True)

                            #add datapoint with metadata
                            max_amp = tmp_signal/(
                                averaging*sample_energy_aver)
                            data.add_measurement(
                                pa_signal/sample_energy_cur, #last mes
                                {'parameter value': current_wl,
                                'x var step': dt,
                                'x var start': 0,
                                'x var stop': dt*(len(pa_signal)-1),
                                'PM energy': tmp_laser/averaging,
                                'sample energy': sample_energy_aver,
                                'max amp': max_amp})
                            
                            print(f'{bcolors.OKBLUE}'
                                  + 'Average laser at sample = '
                                  + f'{sample_energy_aver:.0f} [uJ]')
                            print(f'Average PA signal = {max_amp:}'
                                  + f'{bcolors.ENDC}')
                            
                        elif power_control == 'Glan prism':
                            #add datapoint with metadata
                            sample_energy_aver = glan_calc(
                                tmp_laser/averaging)
                            max_amp = tmp_signal/(
                                averaging*sample_energy_aver)
                            data.add_measurement(
                                pa_signal/glan_calc(cur_pm_laser), #last mes
                                {'parameter value': current_wl,
                                'x var step': dt,
                                'x var start': 0,
                                'x var stop': dt*(len(pa_signal)-1),
                                'PM energy': tmp_laser/averaging,
                                'sample energy': sample_energy_aver,
                                'max amp': max_amp})
                            
                            print(f'{bcolors.OKBLUE}'
                                  + 'Average laser at sample = '
                                  + f'{sample_energy_aver:.0f} [uJ]')
                            print(f'Average PA signal = {max_amp:.3e}'
                                  + f'{bcolors.ENDC}')
                            
                        else:
                            print(f'{bcolors.WARNING}'
                                  + 'Unknown power control method in writing '
                                  + f'laser energy{bcolors.ENDC}')
                        data.save_tmp()
                        
            elif measure_ans == 'Stop measurements':
                confirm = inquirer.confirm(
                    message='Are you sure?'
                ).execute()
                if confirm:
                    print(f'{bcolors.WARNING}'
                          + 'Spectral measurements terminated!'
                          + f'{bcolors.ENDC}')
                    data.bp_filter()
                    return data
            else:
                print(f'{bcolors.WARNING}'
                      + 'Unknown command in Spectral measure menu!'
                      + f'{bcolors.ENDC}')
    
    print(f'{bcolors.OKGREEN}'
          + 'Spectral scanning complete!'
          + f'{bcolors.ENDC}')
    data.bp_filter()
    return data