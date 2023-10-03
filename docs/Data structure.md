# Description of data structure of generated *.hdf5 files

## General structure of .hdf5 file

# version 1.0

/ - <root group>
|.attrs
|  |--'created': str - date and time of data measurement
|  |--'data_points': int - amount of stored PA measurements
|  |--'filename': str - full path to the data file
|  |--'measurement_dims': int - dimensionality of the stored measurement
|  |--'parameter_name': NDArray[str] - independent parameter, changed between measured PA signals
|  |--'parameter_u': List[str] - parameters units
|  |--'updated': str - date and time of last data update
|  |--'version': float - version of data structure
|  |--'zoom_post_time': float - end time from the center of the PA data frame for zoom in data view
|  |--'zoom_pre_time': float - start time from the center of the PA data frame for zoom in data view
|  |--'zoom_u': str - units for pre and post zoom time
|
|--'raw_data' - <group>
   |.attrs
   |  |--'max_len': int - maximum amount of points in PA signal
   |  |--'x_var_name': str - name of the X variable in PA signal
   |  |--'x_var_u': str - units of the X variable
   |  |--'y_var_name': str - name of the Y variable in PA signal
   |  |--'y_var_u': str - name of the Y variable
   |
   |--point001 - <dataset>: ndarray[uint8] - measured PA signal
   |  |--.attrs 
   |    |--'a': float
   |    |--'b': float - y = a*x+b, where 'x' - values from 'data', 'y' - values in 'y var units' scale
   |    |--'param_val': ndarray - value of independent parameter
   |    |--'max_amp': float - (y_max - y_min)
   |    |--'max_amp_u': str - units of 'max_amp'
   |    |--'pm_en': float - laser energy measured by power meter in glass reflection
   |    |--'pm_en_u': str - units of pm_en
   |    |--'sample_en': float - laser energy at sample
   |    |--'sample_en_u': str - units of 'sample_en'
   |    |--'x_var_start': float
   |    |--'x_var_step': float
   |    |--'x_var_stop': float
   |
   |--point002 - <dataset>: ndarray[uint8] - measured PA signal
   |  |--.attrs
   |    |--'a': float
   |    |--'b': float - y = a*x+b, where 'x' - values from 'data', 'y' - values in 'y var units' scale
   |    ...
   ...
