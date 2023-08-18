# Description of data structure of generated *.hdf5 files

## General structure

Attributes in file contain all fields from PaData class along with some additional fields.

/ - <root group>
|.attrs
|  |--'version': float - version of data structure
|  |--'measurement_dims': int - dimensionality of the stored measurement
|  |--'parameter_name': List[str] - independent parameter, changed between measured PA signals
|  |--'parameter_u': List[str] - parameters units
|  |--'data_points': int - amount of stored PA measurements
|  |--'created': str - date and time of data measurement
|  |--'updated': str - date and time of last data update
|  |--'filename': str - full path to the data file
|  |--'zoom_pre_time': float - start time from the center of the PA data frame for zoom in data view
|  |--'zoom_post_time': float - end time from the center of the PA data frame for zoom in data view
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
   |    |--'param_val': ndarray - value of independent parameter
   |    |--'a': float
   |    |--'b': float - y = a*x+b, where 'x' - values from 'data', 'y' - values in 'y var units' scale
   |    |--'x_var_step': float
   |    |--'x_var_start': float
   |    |--'x_var_stop': float
   |    |--'pm_en': float - laser energy measured by power meter in glass reflection
   |    |--'pm_en_u': str - units of pm_en
   |    |--'sample_en': float - laser energy at sample
   |    |--'sample_en_u': str - units of 'sample_en'
   |    |--'max_amp': float - (y_max - y_min)
   |    |--'max_amp_u': str - units of 'max_amp'
   |
   |--point002 - <dataset>: ndarray[uint8] - measured PA signal
   |  |--.attrs
   |    |--'parameter value': int - value of independent parameter
   |    ...
   ...
