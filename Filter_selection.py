import numpy as np
import math

filename = 'ColorGlass.txt'

def remove_zeros(data):
    """fill zeros in filters data by linear fit from nearest values"""

    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            if data[i+1,j+2] == 0:
                if i == 0:
                    if data[i+2,j+2] == 0 or data[i+3,j+2] == 0:
                        print('missing value for the smallest WL cannot be calculated!')
                        return data
                    else:
                        data[i+1,j+2] = 2*data[i+2,j+2] - data[i+3,j+2]
                elif i == data.shape[0]-2:
                    if data[i,j+2] == 0 or data[i-1,j+2] == 0:
                        print('missing value for the smallest WL cannot be calculated!')
                        return data
                    else:
                        data[i+1,j+2] = 2*data[i,j+2] - data[i-1,j+2]
                else:
                    if data[i,j+2] == 0 or data[i+2,j+2] == 0:
                        print('adjacent zeros in filter data are not supported!')
                        return data
                    else:
                        data[i+1,j+2] = (data[i,j+2] + data[i+2,j+2])/2
    return data

def calc_od(data):
    """calculates OD using thickness of filters"""
    for j in range(data.shape[1]-2):
        for i in range(data.shape[0]-1):
            data[i+1,j+2] = data[i+1,j+2]*data[0,j+2]
    return data

data = np.loadtxt(filename,skiprows=1)
data = remove_zeros(data)
data = calc_od(data)

header = open(filename).readline()
filters = header.split('\n')[0].split('\t')[2:]

print(filters)

filer_number = 2
all_filters = data.shape[1]-2
#combinations = math.factorial(all_filters)/(math.factorial(filer_number)*math.factorial(all_filters-filer_number))

print(f'all filters = {all_filters}')
#print(f'Combination = {combinations}')

def get_combi(dict, data, filters, filter_number=2):
    """Recursive method for calculation of transmission
    of filter combinations"""

    if len(data) > 1:
        combi_name_1 = filters[0]
        val_1 = data[0]
        for i in range(len(data)-1):
            combi_name_2 = filters[i+1]
            val_2 = data[i+1]
            val = math.pow(10,-(val_1 + val_2))
            combi_name = combi_name_1 +' '+ combi_name_2
            dict.update({combi_name:val})
        next_step_data = data[1:].copy()
        next_step_filters = filters[1:].copy()
        get_combi(dict,next_step_data,next_step_filters)
    else: 
        return dict
    return dict

target_energy= 1000
target_trans = target_energy/data[12,1]
print(f'target WL = {data[12,0]}')
print(f'target transmission = {target_trans}')
all_combinations = {}
result = get_combi(all_combinations, data[12,2:],filters)
combination = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
print(len(combination))

i = 0
for key, value in combination.items():
    if i<6:
        print(f'{key}:{value}')
        i+=1


final_combinations = {}
for key, value in combination.items():
    if (value-target_trans) > 0:
        final_combinations.update({key: value})

print(f'len of result = {len(final_combinations)}')
i = 0
for key, value in dict(sorted(final_combinations.items(), key=lambda item: item[1])).items():
    if i<6:
        print(f'{key}:{value}')
        i+=1
