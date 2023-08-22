"""
PA backend
"""

from doctest import debug
from typing import Any, TypedDict, Union, List, Tuple
import logging
import os
from itertools import combinations
from collections import deque
import time
import math

import yaml
import pint
import numpy as np
import numpy.typing as npt
from InquirerPy import inquirer
from pylablib.devices import Thorlabs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keyboard

import modules.osc_devices as osc_devices
import modules.exceptions as exceptions
from .pa_data import PaData
from .data_classes import *
from . import ureg

logger = logging.getLogger(__name__)

def new_data_point() -> Data_point:
    """Return default data point."""

    measurement: Data_point = {
                    'dt': 0*ureg.s,
                    'pa_signal': [],
                    'pa_signal_raw': np.empty(0, dtype=np.uint8),
                    'pm_signal': [],
                    'start_time': 0*ureg.s,
                    'stop_time': 0*ureg.s,
                    'pm_energy': 0*ureg.uJ,
                    'sample_energy': 0*ureg.uJ,
                    'max_amp': 0*ureg.uJ,
                    'wavelength': 0*ureg.nm
                }
    return measurement

def init_hardware(hardware: Hardware) -> bool:
    """Initialize all hardware.
    
    Load hardware config from rsc/config.yaml if it was not done.
    """

    logger.info('Starting hardware initialization...')
    
    if not init_osc(hardware):
        logger.warning('Oscilloscope cannot be loaded')
        return False
    osc = hardware['osc']
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
    """Load configuration.

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
    
    hardware.update({'config':config})
    logger.debug('Config loaded into hardware["config"]')

    if int(config['stages']['axes']) == 3:
        logger.debug('Stage_z added to hardware list')
        hardware.update({'stage_z':0}) # type: ignore
    
    osc = hardware['osc']
    if bool(config['power_meter']['connected']):
        hardware.update({'power_meter': osc_devices.PowerMeter(osc)})
        pm = hardware['power_meter'] # type: ignore
        pm_chan = config['power_meter']['connected_chan']
        pm.set_channel(pm_chan-1)
        logger.debug(f'Power meter added to hardare list at CHAN{pm_chan}')
    if bool(config['pa_sensor']['connected']):
        hardware.update(
            {'pa_sens':osc_devices.PhotoAcousticSensOlymp(osc)})
        pa = hardware['pa_sens'] # type: ignore
        pa_chan = config['pa_sensor']['connected_chan']
        pa.set_channel(pa_chan-1)
        logger.debug(f'PA sensor added to hardare list at CHAN{pa_chan}')
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

def track_power(hardware: Hardware, tune_width: int) -> pint.Quantity:
    """Build energy graph.
    Return averaged mean energy"""

    ### config parameters
    #Averaging for mean and std calculations
    aver = 10
    # ignore energy read if it is smaller than threshold*mean
    threshold = 0.01
    # time delay between measurements
    measure_delay = ureg('50ms')
    ###

    logger.debug('track_power is starting with config params: '
                 + f'{aver=}, {threshold=}, {measure_delay=}')
    pm = hardware['power_meter'] #type: ignore
    
    #tune_width cannot be smaller than averaging
    if tune_width < aver:
        logger.warning(f'{tune_width=} is smaller than averaging='
                       + f'{aver}. tune_width set to {aver}.')
        tune_width = aver
    data = deque(maxlen=tune_width)

    logger.info('Hold "q" button to stop power measurements')
    
    logger.debug('Initializing plt')
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1,2)
    #axis for signal from osc
    ax_pm = fig.add_subplot(gs[0,0])
    #axis for energy graph
    ax_pa = fig.add_subplot(gs[0,1])
    
    mean = 0*ureg('J')
    logger.debug('Entering measuring loop')
    while True:
        try:
            laser_amp = pm.get_energy_scr()
        except exceptions.OscilloscopeError as err:
            logger.warning(f'{err.value}. Laser energy = {mean}')
            return mean

        logger.debug(f'measured {laser_amp=}')
        if not laser_amp:
            continue
        data.append(laser_amp)
        
        #ndarray for up to last <aver> values
        tmp_data = pint.Quantity.from_list(
            [x for i,x in enumerate(data) if i<aver])
        mean = tmp_data.mean() # type: ignore
        std = tmp_data.std() # type: ignore
        title = (f'Energy={laser_amp}, '
                + f'Mean (last {aver}) = {mean}, '
                + f'Std (last {aver}) = {std}')
        logger.debug(f'plot {title=}')
        
        #plotting data
        ax_pm.clear()
        ax_pa.clear()
        ax_pm.plot(pm.data)
        
        #add markers for data start and stop
        ax_pm.plot(
            pm.start_ind,
            pm.data[pm.start_ind],
            'o',
            alpha=0.4,
            ms=12,
            color='green')
        ax_pm.plot(
            pm.stop_ind,
            pm.data[pm.stop_ind],
            'o',
            alpha=0.4,
            ms=12,
            color='red')
        ax_pa.plot(tmp_data)
        ax_pa.set_title(title)
        ax_pa.set_ylim(bottom=0)
        fig.canvas.draw()
        plt.pause(0.01)
            
        if keyboard.is_pressed('q'):
            break
        time.sleep(measure_delay.to('s').m)

    return mean

def spectrum(
        hardware: Hardware,
        start_wl: pint.Quantity,
        end_wl: pint.Quantity,
        step: pint.Quantity,
        target_energy: pint.Quantity,
        averaging: int
) -> Union[PaData,None]:
    """Measure dependence of PA signal on excitation wavelength.
    
    Measurements start at <start_wl> wavelength and are taken
    every <step> with the last measurement at <end_wl>,
    so that the last step could be smaller than others.
    Measurements are taken at <target_energy>.
    Each measurement will be averaged <averaging> times.
    """

    logger.info('Start measuring spectra!')
    data = PaData(dims=1, params=['Wavelength'])
    #make steps negative if going from long WLs to short
    if start_wl > end_wl:
        step = -step # type: ignore
    #calculate amount of data points
    d_wl = end_wl-start_wl
    if d_wl%step:
        spectral_points = int(d_wl/step) + 2
    else:
        spectral_points = int(d_wl/step) + 1

    #main measurement cycle
    for i in range(spectral_points):
        if abs(step*i) < abs(d_wl):
            current_wl = start_wl + step.m*i
        else:
            current_wl = end_wl

        logger.info(f'Start measuring point {(i+1)}')
        logger.info(f'Current wavelength is {current_wl}.'
                    +'Please set it!')

        if not set_energy(hardware, current_wl, target_energy, bool(i)):
            return
        measurement, proceed = ameasure_point(hardware, averaging, current_wl)
        data.add_measurement(measurement, [current_wl])
        data.save_tmp()
        if not proceed:
            return data
    logger.info('Spectral scanning complete!')
    data.bp_filter()
    return data

def set_energy(
    hardware: Hardware,
    current_wl: pint.Quantity,
    target_energy: pint.Quantity,
    repeated: bool
) -> bool:
    """Set laser energy for measurements.
    
    Set <target_energy> at <current_wl>.
    <repeated> is flag which indicates that this is not the first
    call of set_energy."""

    if hardware['config']['power_control'] == 'Filters':
        logger.info('Please remove all filters and measure '
                    + 'energy at glass reflection.')
        #measure mean energy at glass reflection
        energy = track_power(hardware, 50)
        logger.info(f'Power meter energy = {energy}')

        #find valid filters combinations for current parameters
        max_filter_comb = hardware['config']['energy']['max_filters']
        logger.debug(f'{max_filter_comb=}')
        filters = glass_calculator(
            current_wl,
            energy,
            target_energy,
            max_filter_comb)
        if not len(filters):
            logger.warning(f'No valid filter combination for '
                    + f'{current_wl}')
            cont_ans = inquirer.confirm(
                message='Do you want to continue?').execute()
            if not cont_ans:
                logger.warning('Spectral measurements terminated!')
                return False
            
        reflection = glass_reflection(current_wl)
        if reflection is not None and reflection != 0:
            target_pm_value = target_energy*reflection
            logger.info(f'Target power meter energy is {target_pm_value}!')
            logger.info('Please set it using filters combination '
                        + ' from above. Additional adjustment by '
                        + 'laser software could be required.')
            track_power(hardware, 50)
        else:
            logger.warning('Target power meter energy '
                            + 'cannot be calculated!')
            return False
    
    elif hardware['config']['power_control'] == 'Glan prism' and not repeated:
        target_pm_value = glan_calc_reverse(target_energy)
        logger.info(f'Target power meter energy is {target_pm_value}!') # type: ignore
        logger.info(f'Please set it using Glan prism.')
        track_power(hardware, 50)
    else:
        logger.error('Unknown power control method! '
                        + 'Measurements terminated!')
        return False
    return True
                
def glass_calculator(wavelength: pint.Quantity,
                     current_energy_pm: pint.Quantity,
                     target_energy: pint.Quantity,
                     max_combinations: int,
                     threshold: float = 2.5,
                     results: int = 5,
                     ) -> dict:
    """Calculate filter combination to get required energy on sample.
    
    Return a dict with up to <results> filter combinations, 
    having the closest transmissions,
    which are higher than required but not more than <threshold> folds higher.
    Accept only wavelengthes, which are present in rsc/ColorGlass.txt.
    """

    #dict for storing found valid combinations
    result = {}
    #file with filter's properties
    sub_folder = 'rsc'
    filename = 'ColorGlass.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        data = np.loadtxt(filename,skiprows=1)
        header = open(filename).readline()
    except FileNotFoundError:
        logger.error('File with color glass data not found!')
        return {}
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        return {}
    
    glass_rm_zeros(data)
    glass_calc_od(data)
    filter_titles = header.split('\n')[0].split('\t')[2:]

    try:
        wl_index = np.where(data[1:,0] == wavelength)[0][0] + 1
    except IndexError:
        logger.error('Target WL is missing in color glass data table!')
        return {}
    # calculated laser energy at sample
    laser_energy = current_energy_pm/data[wl_index,1]*100
    if laser_energy == 0:
        logger.error('Laser radiation is not detected!')
        return {}
    #required total transmission of filters
    target_transm = target_energy/laser_energy
    
    logger.info(f'Target sample energy = {target_energy}')
    logger.info(f'Current sample energy = {laser_energy}')
    logger.info(f'Target transmission = {target_transm*100:.1f} %')

    #build filter_dict = {filter_name: OD at wavelength}
    #find index of wavelength
    filters = {}
    for key, value in zip(filter_titles,data[wl_index,2:]):
        filters.update({key:value})

    filter_combinations = glass_combinations(filters, max_combinations)
    result_comb = glass_limit_comb(
        filter_combinations,
        target_transm,
        results,
        threshold
    )
    logger.info('Valid filter combinations:')
    for comb_name, transmission in result_comb.items():
        logger.info(f'{comb_name}: {transmission*100:.1f} %')

    return result_comb

def glass_rm_zeros(data: np.ndarray) -> None:
    """Replaces zeros in filters data by linear fit from nearest values."""

    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            if data[i+1,j+2] == 0:
                if i == 0:
                    if data[i+2,j+2] == 0 or data[i+3,j+2] == 0:
                        logger.warning('missing value for the smallest WL cannot be calculated!')
                    else:
                        data[i+1,j+2] = 2*data[i+2,j+2] - data[i+3,j+2]
                elif i == data.shape[0]-2:
                    if data[i,j+2] == 0 or data[i-1,j+2] == 0:
                        logger.warning('missing value for the smallest WL cannot be calculated!')
                    else:
                        data[i+1,j+2] = 2*data[i,j+2] - data[i-1,j+2]
                else:
                    if data[i,j+2] == 0 or data[i+2,j+2] == 0:
                        logger.warning('adjacent zeros in filter data are not supported!')
                    else:
                        data[i+1,j+2] = (data[i,j+2] + data[i+2,j+2])/2

def glass_calc_od(data: np.ndarray) -> None:
    """Calculate OD using thickness of filters."""

    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            data[i+1,j+2] = data[i+1,j+2]*data[0,j+2]

def glass_combinations(filters: dict, max_comb: int) -> dict:
    """Calc transmission of combinations up to <max_comb> filters."""

    filter_combinations = {}
    for i in range(max_comb):
        for comb in combinations(filters.items(),i+1):
            key = ''
            value = 0
            for k,v in comb:
                key +=k
                value+=v
            filter_combinations.update({key:math.pow(10,-value)})

    return filter_combinations

def glass_limit_comb(
        filter_combs: dict,
        target_transm: float,
        results: int,
        threshold: float) -> dict:
    """Limit combinations of filters.
    
    Up to <results> with transmission higher than <target_transm>,
    but not higher than <target_transm>*<threshold> will be returned."""

    result = {}
    filter_combs = dict(sorted(
        filter_combs.items(),
        key=lambda item: item[1]))
    
    i=0
    for key, value in filter_combs.items():
        if (value-target_transm) > 0 and value/target_transm < threshold:
            result.update({key: value})
            i += 1
            if i == results:
                break

    return result

def glass_reflection(wl: pint.Quantity) -> Union[float,None]:
    """Get reflection (fraction) from glass at given wavelength.
    
    pm_energy/sample_energy = glass_reflection."""

     #file with filter's properties
    sub_folder = 'rsc'
    filename = 'ColorGlass.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        data = np.loadtxt(filename,skiprows=1)
    except FileNotFoundError:
        logger.error('File with color glass data not found!')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        return
    
    try:
        wl_nm = wl.to('nm')
        wl_index = np.where(data[1:,0] == wl_nm.m)[0][0] + 1
    except IndexError:
        logger.warning('Target WL is missing in color glass data table!')
        return
    
    return data[wl_index,1]/100

def glan_calc_reverse(
        target_energy: pint.Quantity,
        fit_order: int=1
    ) -> Union[pint.Quantity,None]:
    """Calculate energy at power meter for given sample energy.
    
    It is assumed that power meter measures laser energy
    reflected from a thin glass.
    Calculation is based on callibration data from 'rsc/GlanCalibr'.
    <fit_order> is a ploynom order used for fitting callibration data.
    """

    sub_folder = 'rsc'
    filename = 'GlanCalibr.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        calibr_data = np.loadtxt(filename, dtype=np.float64)
    except FileNotFoundError:
        logger.error('File with glan callibration not found!')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        return

    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    if fit_order == 1:
        # target_energy = coef[0]*energy + coef[1]
        energy = (target_energy.to('uJ').m - coef[1])/coef[0]*ureg.uJ
    else:
        logger.warning('Reverse Glan calculation for nonlinear fit is not '
                       'realized! Linear fit used instead!')
        energy = (target_energy.to('uJ').m - coef[1])/coef[0]*ureg.uJ
    
    return energy

def glan_calc(
        energy: pint.Quantity,
        fit_order: int=1
    ) -> Union[pint.Quantity,None]:
    """Calculates energy at sample for a given power meter energy"""

    sub_folder = 'rsc'
    filename = 'GlanCalibr.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        calibr_data = np.loadtxt(filename, dtype=np.float64)
    except FileNotFoundError:
        logger.error('File with glan callibration not found!')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        return

    #get coefficients which fit calibration data with fit_order polynom
    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    #init polynom with the coefficients
    fit = np.poly1d(coef)

    #return the value of polynom at energy
    return fit(energy.to('uJ').m)*ureg.uJ

def ameasure_point(
    hardware: Hardware,
    averaging: int,
    current_wl: pint.Quantity
) -> Tuple[Data_point, bool]:
    """Measure single PA data point with averaging.
    
    second value in the returned tuple (bool) is a flag to 
    continue measurements.
    """

    logger.debug('Start measuring PA data point with averaging.')
    counter = 0
    msmnts: List[Data_point]=[]
    while counter < averaging:
        logger.info(f'Signal at {current_wl} should be measured '
                + f'{averaging-counter} more times.')
        action = set_next_measure_action()
        if action == 'Tune power':
            track_power(hardware, 40)
        elif action == 'Measure':
            tmp_measurement = measure_point(hardware, current_wl)
            if verify_measurement(hardware, tmp_measurement):
                msmnts.append(tmp_measurement)
                counter += 1
                if counter == averaging:
                    measurement = aver_measurements(msmnts)
                    logger.debug('Data point successfully measured!')
                    return measurement, True
       
        elif action == 'Stop measurements':
            if confirm_action():
                measurement = aver_measurements(msmnts)
                logger.warning('Spectral measurement terminated')
                return measurement, False
        else:
            logger.warning('Unknown command in Spectral measure menu!')
    
    logger.warning('Unexpectedly passed after main measure sycle!')
    return new_data_point(), True

def measure_point(
        hardware: Hardware,
        wavelength: pint.Quantity
) -> Data_point:
    """Measure single PA data point."""

    logger.debug('Start measuring PA data point.')
    osc = hardware['osc']
    pm = hardware['power_meter'] #type: ignore
    pa_ch_id = int(hardware['config']['pa_sensor']['connected_chan']) - 1
    pm_ch_id = int(hardware['config']['power_meter']['connected_chan']) - 1
    measurement = new_data_point()
    osc.measure()
    dt = 1/osc.sample_rate
    pa_signal = osc.data[pa_ch_id]
    pa_amp = osc.amp[pa_ch_id]
    pa_signal_raw = osc.data_raw[pa_ch_id]
    start_time = 0*ureg.s
    stop_time = dt*(len(pa_signal)-1)
    pm_signal = osc.data[pm_ch_id]
    pm_energy = pm.energy_from_data(pm_signal, dt)

    measurement.update(
        {
            'wavelength': wavelength,
            'dt': dt,
            'pa_signal_raw': pa_signal_raw,
            'start_time': start_time,
            'stop_time': stop_time,
            'pm_signal': pm_signal,
            'pm_energy': pm_energy,
        }
    )

    logger.debug(f'{wavelength=}')
    logger.debug(f'{dt=}')
    logger.debug(f'Raw PA data has {len(pa_signal_raw)} points '
                    + f'with max value={max(pa_signal_raw)}')
    logger.debug(f'{start_time=}')
    logger.debug(f'{stop_time=}')
    logger.debug(f'power meter data has {len(pm_signal)} points '
                    + f'with max value={max(pm_signal)}')
    logger.debug(f'Laser energy at power meter = {pm_energy}')

    if hardware['config']['power_control'] == 'Filters':
        reflection = glass_reflection(wavelength)
        if reflection is not None and reflection !=0:
            sample_energy = pm_energy/reflection
            pa_signal = pa_signal/sample_energy
            pa_amp = pa_amp/sample_energy
        else:
            sample_energy = 0*ureg.uJ
            pa_amp = 0*ureg.uJ
            logger.warning('Sample energy cannot be '
                            +'calculated')
    elif hardware['config']['power_control'] == 'Glan prism':
        sample_energy = glan_calc(pm_energy)
        if sample_energy is not None and sample_energy != 0:
            pa_signal = pa_signal/sample_energy
            pa_amp = pa_amp/sample_energy
        else:
            logger.warning(f'{sample_energy=}')
            pa_amp = 0*ureg.uJ
            sample_energy = 0*ureg.uJ
    else:
        logger.error('Unknown power control method! '
                    + 'Measurements terminated!')
        return measurement
    
    measurement.update(
        {'sample_energy': sample_energy,
         'pa_signal': pa_signal,
         'max_amp': pa_amp}
    )
    logger.debug(f'{sample_energy=}')
    logger.debug(f'PA data has {len(pa_signal)} points '
                    + f'with max value={max(pa_signal)}')
    logger.debug(f'PA amplitude = {pa_amp}')

    return measurement

def aver_measurements(measurements: List[Data_point]) -> Data_point:
    """Calculate average measurement from a given list.
    
    Actually only amplitude values are averaged, in other cases data
    from the last measurement from the <measurements> is used."""

    result = new_data_point()
    total = len(measurements)
    for measurement in measurements:
        result['dt'] = measurement['dt']
        result['pa_signal'] = measurement['pa_signal']
        result['pa_signal_raw'] = measurement['pa_signal_raw']
        result['pm_signal'] = measurement['pm_signal']
        result['start_time'] = measurement['start_time']
        result['stop_time'] = measurement['stop_time']
        result['wavelength'] = measurement['wavelength']
        
        result['pm_energy'] += measurement['pm_energy'] # type: ignore
        result['sample_energy'] += measurement['sample_energy'] # type: ignore
        result['max_amp'] += measurement['max_amp'] # type: ignore

    result['pm_energy'] = result['pm_energy']/total
    result['sample_energy'] = result['sample_energy']/total
    result['max_amp'] = result['max_amp']/total

    logger.info(f'Average power meter energy {result["pm_energy"]}')
    logger.info(f'Average energy at {result["sample_energy"]}')
    logger.info(f'Average PA signal amp {result["max_amp"]}')
    
    return result

def verify_measurement(
        hardware: Hardware,
        measurement: Data_point
) -> bool:
    """Verify a PA measurement."""

    # update state of power meter
    pm = hardware['power_meter'] #type:ignore
    pm_signal = measurement['pm_signal']
    dt = measurement['dt']
    pa_signal = measurement['pa_signal']
    pm.energy_from_data(pm_signal, dt)

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1,2)
    ax_pm = fig.add_subplot(gs[0,0])
    pm_time = [x*dt for x in range(len(pm_signal))]
    ax_pm.plot(pm_time,pm_signal)

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
    pa_time = [x*dt for x in range(len(pa_signal))]
    ax_pa.plot(pa_time,pa_signal)
    plt.show()

    #confirm that the data is OK
    good_data = inquirer.confirm(
        message='Data looks good?').execute()
    logger.debug(f'{good_data=} was choosen')

    return good_data

def set_next_measure_action() -> str:
    """Choose the next action during PA measurement.
    
    Returned values = ['Tune power'|'Measure'|'Stop measurements'].
    """

    # in future this function can have several implementations
    # depending on whether CLI or GUI mode is used
    measure_ans = inquirer.rawlist(
    message='Chose an action:',
    choices=['Tune power','Measure','Stop measurements']
    ).execute()
    logger.debug(f'{measure_ans=} was choosen')
    return measure_ans

def confirm_action(message: str='') -> bool:
    """Confirm execution of an action."""

    if not message:
        message = 'Are you sure?'
    confirm = inquirer.confirm(message=message).execute()
    logger.debug(f'{confirm=} was choosen.')
    return confirm