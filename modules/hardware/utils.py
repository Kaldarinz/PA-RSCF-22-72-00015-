"""
Hardware utility tools.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

import logging
import os
from itertools import combinations
import math

from pint.facets.plain.quantity import PlainQuantity
import numpy as np

from .glob import hardware
from . import ureg, Q_

logger = logging.getLogger(__name__)


def glass_calculator(
        wavelength: PlainQuantity,
        current_energy_pm: PlainQuantity,
        target_energy: PlainQuantity,
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

    logger.debug('Starting calculation of filter combinations...')
    #file with filter's properties
    sub_folder = 'rsc'
    filename = 'ColorGlass.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        data = np.loadtxt(filename,skiprows=1)
        header = open(filename).readline()
    except FileNotFoundError:
        logger.error('File with color glass data not found!')
        logger.debug('...Terminating.')
        return {}
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return {}
    
    _glass_rm_zeros(data)
    glass_calc_od(data)
    filter_titles = header.split('\n')[0].split('\t')[2:]

    try:
        wl_index = np.where(data[1:,0] == wavelength)[0][0] + 1
    except IndexError:
        logger.error('Target WL is missing in color glass data table!')
        logger.debug('...Terminating')
        return {}
    # calculated laser energy at sample
    laser_energy = current_energy_pm/data[wl_index,1]*100
    if laser_energy == 0:
        logger.error('Laser radiation is not detected!')
        logger.debug('...Terminating')
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

    logger.debug('...Finishing. Filter combinations calculated.')
    return result_comb

def _glass_rm_zeros(data: np.ndarray) -> None:
    """Replaces zeros in filters data by linear fit from nearest values."""

    logger.debug('Starting replacement of zeros in filter data...')
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
    logger.debug('...Finishing. Zeros removed from filter data.')

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
        threshold: float
    ) -> dict:
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

def glass_reflection(wl: PlainQuantity) -> float|None:
    """Get reflection (fraction) from glass at given wavelength.
    
    pm_energy/sample_energy = glass_reflection."""

    logger.debug('Starting calculation of glass reflection...')
     #file with filter's properties
    sub_folder = 'rsc'
    filename = 'ColorGlass.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        data = np.loadtxt(filename,skiprows=1)
    except FileNotFoundError:
        logger.error('File with color glass data not found!')
        logger.debug('...Terminating.')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return
    
    try:
        wl_nm = wl.to('nm')
        wl_index = np.where(data[1:,0] == wl_nm.m)[0][0] + 1
    except IndexError:
        logger.warning('Target WL is missing in color glass data table!')
        logger.debug('...Terminating')
        return
    
    logger.debug('...Finishing. Glass reflection calculated')
    return data[wl_index,1]/100

def glan_calc_reverse(
        target_energy: PlainQuantity,
        fit_order: int=1
    ) -> PlainQuantity|None:
    """Calculate energy at power meter for given sample energy.
    
    It is assumed that power meter measures laser energy
    reflected from a thin glass.
    Calculation is based on callibration data from 'rsc/GlanCalibr'.
    <fit_order> is a ploynom order used for fitting callibration data.
    """

    logger.debug('Starting calculation of power meter '
                 + 'energy from sample energy...')
    sub_folder = 'rsc'
    filename = 'GlanCalibr.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        calibr_data = np.loadtxt(filename, dtype=np.float64)
    except FileNotFoundError:
        logger.error('File with glan callibration not found!')
        logger.debug('...Terminating')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return

    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    if fit_order == 1:
        # target_energy = coef[0]*energy + coef[1]
        energy = (target_energy.to('uJ').m - coef[1])/coef[0]*ureg.uJ
    else:
        logger.warning('Reverse Glan calculation for nonlinear fit is not '
                       'realized! Linear fit used instead!')
        energy = (target_energy.to('uJ').m - coef[1])/coef[0]*ureg.uJ
    
    logger.debug('...Finishing. Power meter energy calculated')
    return energy

def glan_calc(
        energy: PlainQuantity,
        fit_order: int=1
    ) -> PlainQuantity|None:
    """
    Calculates energy at sample for a given power meter energy.
    
    Do not interact with hardware.
    """

    logger.debug('Starting sample energy calculation...')
    sub_folder = 'rsc'
    filename = 'GlanCalibr.txt'
    filename = os.path.join(sub_folder,filename)

    try:
        calibr_data = np.loadtxt(filename, dtype=np.float64)
    except FileNotFoundError:
        logger.error('File with glan callibration not found!')
        logger.debug('...Terminating')
        return
    except ValueError as er:
        logger.error(f'Error while loading color glass data!: {str(er)}')
        logger.debug('...Terminating')
        return

    #get coefficients which fit calibration data with fit_order polynom
    coef = np.polyfit(calibr_data[:,0], calibr_data[:,1],fit_order)

    #init polynom with the coefficients
    fit = np.poly1d(coef)

    #return the value of polynom at energy
    sample_en = fit(energy.to('uJ').m)*ureg.uJ
    logger.debug('...Finishing. Sample energy calculated.')
    return sample_en

def calc_sample_en(
        wavelength: PlainQuantity,
        pm_energy: PlainQuantity
    ) -> PlainQuantity|None:
    """Calculate energy at sample."""

    config = hardware.config
    if config['power_control'] == 'Filters':
        reflection = glass_reflection(wavelength)
        if reflection is not None and reflection:
            sample_energy = pm_energy/reflection    
    elif config['power_control'] == 'Glan prism':
        sample_energy = glan_calc(pm_energy)

    if sample_energy:
        return sample_energy
    else:
        logger.warning('Sample energy cannot be calculated')
        return None
