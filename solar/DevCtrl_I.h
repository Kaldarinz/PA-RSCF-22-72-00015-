const	PANEL_ALL		= 0;
const  	PANEL_GRATING1		= 1;
const	PANEL_GRATING2		= 2;
const	PANEL_GRATING3		= 4;
const	PANEL_GRATING4		= 8;
const	PANEL_SLIT1		= 16;
const	PANEL_SLIT2		= 32;
const	PANEL_SLIT3		= 64;
const	PANEL_SLIT4		= 128;
const	PANEL_TURNSLIT1		= 256;
const	PANEL_TURNSLIT2		= 512;
const	PANEL_TURRET1		= 1024;
const	PANEL_TURRET2		= 2048;
const	PANEL_SHUTTER1		= 4096;
const	PANEL_SHUTTER2		= 8192;
const	PANEL_SHUTTER3		= 16384;
const	PANEL_SHUTTER4		= 32768;
const	PANEL_FILTER1		= 65536;
const	PANEL_FILTER2		= 131072;
const	PANEL_FILTER3		= 262144;
const	PANEL_FILTER4		= 524288;
const	PANEL_MIRROR1		= 1048576;
const	PANEL_MIRROR2		= 2097152;
const	PANEL_MIRROR3		= 4194304;
const	PANEL_MIRROR4		= 8388608;

const	PANEL_MULTIPOS1		= 67108864;
const	PANEL_MULTIPOS2		= 134217728;
const	PANEL_MULTIPOS3		= 268435456;
const	PANEL_MULTIPOS4		= 536870912;

const	PANEL_DIGITPOS1		= 2147483648;
const	PANEL_DIGITPOS2		= 4294967296;
const	PANEL_DIGITPOS3		= 8589934592;
const	PANEL_DIGITPOS4		= 17179869184;
const	PANEL_DIGITPOS5		= 34359738368;
const	PANEL_DIGITPOS6		= 68719476736;
const	PANEL_DIGITPOS7		= 137438953472;
const	PANEL_DIGITPOS8		= 274877906944;

const	PANEL_BISTABLE1		= 549755813888;
const	PANEL_BISTABLE2		= 1099511627776;

const	PANEL_QUERY1		= 8796093022208;
const	PANEL_QUERY2		= 17592186044416;
const	PANEL_QUERY3		= 35184372088832;
const	PANEL_QUERY4		= 70368744177664;

const	PANEL_DIGITPOSEX1	= 140737488355328;
const	PANEL_DIGITPOSEX2	= 281474976710656;
const	PANEL_DIGITPOSEX3	= 562949953421312;
const	PANEL_DIGITPOSEX4	= 1125899906842624;

const	SDT_GRATING		= 1;
const	SDT_TURRET		= 2;
const	SDT_SLIT		= 3;
const	SDT_TURNSLIT		= 4;
const	SDT_SHUTTER		= 5;
const	SDT_MIRROR		= 6;
const	SDT_FILTER		= 7;
const	SDT_DOUBLEFILTER	= 8;
const	SDT_MULTIPOS		= 9;
const	SDT_DIGITPOSEX		= 10;
const	SDT_BISTABLEDEVICE	= 11;
const	SDT_QUERYDEVICE		= 12;
const	SDT_DIGITPOSEX		= 13;

const	ALL_OK			= 0;
const	RECEIVING_DATA		= 1;
const	PREVIOUSPOS_NEED	= 2;

const	DEVICE_NOTFOUND		= 6;

const	ERROR_OVERFULL		= 7; // 'Overfull of input data buffer.'
const	ERROR_SYMBOLFAULT	= 8; // 'Received symbol is fault.'
const	ERROR_OVERTIME		= 9; // 'Operation time is over, or I2C-interface operation error.'
const	ERROR_INTERFACE		= 10; // 'I2C-interface operation error.'
const	ERROR_HARDWARE		= 11; // 'Unit hardware error.' 
const	ERROR_STEPMOTOR		= 12; // 'Drive error.'
const	ERROR_UNKNOWNSYMBOL	= 13; // 'Unknown symbol from RS232-interface'
const	ERROR_UNKNOWNERROR	= 14; // 'Unknown code.'
const	ERROR_READTIMEOUT	= 15; // 'No connect with device.'
const	ERROR_WRITETIMEOUT	= 16; // 'Serial port is failed.'
const	ERROR_RECEIVEDCODE	= 17; // 'Unknown code.'
const	ERROR_DEVICENOTREADY	= 18; // 'Device not ready.'
const	ERROR_NOTCONNECT	= 19; // 'Serial port is failed.'
const	ERROR_INIZIALIZE	= 20; // 'Error initialization.'
const	ERROR_UNKNOWNPORT	= 21; // 'Unknown serial port.'
const	ERROR_CONVERTATION	= 22; // 'Error convertation.'
const	ERROR_INCORRECTPARAM	= 23; // 'Incorrect parameter.'
const	ERROR_UNKNOWNDEVICE	= 24; // 'Unknown device'
const	ERROR_RANGE		= 25; // 'Out of range.'
const	ERROR_RESET		= 26; // 'Error reset position.'
const	ERROR_MEMORY		= 27; // 'Error of memory.'

const	STATE_ON		= 1;
const	STATE_OFF		= -1;

const	SHUTTER_OPEN		= 3;
const	SHUTTER_CLOSE		= 4;

//Function GetFullDevCtrlVersion is used to retrieve the full software version.
//Parameters:
//Directory is a fully-qualified path to the directory where the file “Devctrl.dll” resides. If Directory is nil, the
//routine will try to retrieve the necessary information from the DLL assuming that it is located in the same directory as
//your application’s executable.
//Providing the returned value is True, Major, Minor, Release, and Build are the major and minor version, release,
//and build numbers, respectively.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetFullDevCtrlVersion)(ansichar* Directory, int32_t* Major, Minor, Release, Build);

//Function GetSupportedDevicesCount is used to retrieve the number of different spectral instruments supported by the software.
//Result: Number of different spectral instruments supported by the software.
typedef int32_t _stdcall (*GetSupportedDevicesCount)(void);

//Function InitDeviceEx2 is used to initialise your spectral instrument. This routine opens the specified COM-port,
//communicates with the device, and loads the necessary configuration ( * .cfg) file.
//Parameters:
//hAppWnd is the main window handle.
//ComPort is the COM-port index (starting from 1). If ComPort = 0, then an automatic search will be performed,
//otherwise starting from the specified index.
//FilePath is the path to the configuration file of your device (e.g., if you have a MSDD1000, it must be a fully qualified
//path to the directory where MSDD1000.cfg resides). If FilePath = nil, then the software will try to access the
//configuration file in the directory where the Device Control DLL (Devctrl.dll) resides.
//If IsShowDialog is True, then an additional message dialog will be displayed asking whether to execute the so-called
//“previous position” operation (see SetPreviousPosition) just after the device initialisation.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*InitDeviceEx2)(HWND hAppWnd, uint8_t* ComPort, ansichar* FilePath, bool IsShowDialog);

//Function CheckConnection is used to check connection with your spectral device.
//Result: True if connection is OK, False otherwise.
typedef bool _stdcall (*CheckConnection)(void); 

//function FreeDevice is used to free active device. This routine closes the used COM-port and saves all the infos on
//the device state into the corresponding * .cfg file.
typedef void _stdcall (*FreeDevice)(void);

//Function GetActiveDevice is used to retrieve the index of the type of active spectral instrument.
//Result:
//2 ->	NP2502
//3 ->	MSDD1004
//4 ->	MS3504
//5 ->	MS7504
//7 ->	MS5004
//10 ->	LPSDP_S380
//15 ->	MS3501
//17 ->	MS2004
//18 ->	MS2001
//19 ->	DM160
//20 ->	MS7504
//21 ->	ML100
//22 ->	MS5204
//23 ->	MS5201
//24 ->	MSDD1002
//25 ->	PHOTO_OBJECTIVE
typedef int32_t _stdcall (*GetActiveDevice)(void);

//Function GetConfigInfo is used to retrieve both the name and the serial number of the active device
//from its configuration file ( * .cfg).
//Parameters:
//DeviceName is the device name, providing the returned value is True.
//SN is the device serial number, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetConfigInfo)(ansichar* DeviceName, int32_t* SN);

//Procedure GetDeviceName is used to retrieve the type of a spectral instrument by its index.
//Parameters:
//DeviceIndex is the index of device type. Can be retrieved by using function GetActiveDevice.
//DeviceName contains the name of device type if successful, or an empty string otherwise.
typedef void _stdcall (*GetDeviceName)(int32_t DeviceIndex, ansichar* DeviceName);

//function SetPermissionMessageBox is used to permit or forbid displaying error messages.
//Parameters:
//If IsPermitted is True then error messages will be displayed, otherwise not.
//By default (i.e., before the very first call of this routine), the error messages are displayed.
typedef void _stdcall (*SetPermissionMessageBox)(bool IsPermitted);

//Procedure ShowDeviceWindowEx is used to display the main software window with selected control panel(s).
//Parameters:
//Panel is the control panel ID (valid values are listed in Table above). If Panel = PANEL_ALL then the main window
//will contain panels for all subdevices (drives) installed in your spectral instrument.
typedef void _stdcall (*ShowDeviceWindowEx)(int64_t Panel);

//Procedure WND_MSG_Register is used to register a spesific message and a destination window.
//Parameters:
//hWindow is the handle of the destination window.
//MsgID is the message ID.
typedef void _stdcall (*WND_MSG_Register)(HWND hWindow, Uint32_t MsgID);

//Procedure MoveDeviceEnabled is used to permit or forbid movement of mobile parts of the active spectral instrument.
//Parameters:
//If IsEnabled is True then all mobile parts are allowed to change their state, otherwise not.
//By default (i.e., before the very first call of this routine), all mobile parts are allowed to change their state,
//providing that the device is correctly initialised. After execution of MoveDeviceEnabled(False) the function
//IsOperationFinished will return False until MoveDeviceEnabled(True) is executed.
typedef void _stdcall (*MoveDeviceEnabled)(bool IsEnabled);

//Function IsOperationFinished is used to check whether the current operation (e.g., changing grating, wavelength,
//slit width, etc.) is still in progress or already finished.
//Parameters: None.
//Result: False if current operation is still in progress, True otherwise.
typedef bool _stdcall (*IsOperationFinished)(void);

//Function GetDeviceError is used to check whether any error encountered during the last operation.
//Result: ERROR_INITIALISATION if device is not yet initialised, ALL_OK if everything is OK (no error), or specified
//error code otherwise (see Error codes for details).
typedef uint8_t _stdcall (*GetDeviceError)(void);

//Function SetPreviousPosition is used to restore correct positions of all mobile parts after switching the device on.
//Parameters: None.
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetPreviousPosition)(void);

//Function GetSubDevicesCount is used to retrieve the number of subdevices of specified type installed in your Solar-
//SpectralInstrumentsspectral instrument.
//Parameters:
//SubDeviceType is the subdevice type (valid values range from SDT_GRATING to SDT_DIGITPOSEX, i.e., all con-
//stants with “SDT” prefix in their name among those listed in constants).
//Result: Number of subdevices of the specified type installed in your SolarSpectralInstrumentsspectral device.
typedef int32_t _stdcall (*GetSubDevicesCount)(uint8_t SubDeviceType);

//Function GetRange is used to retrieve the valid range for a specified subdevice.
//Parameters:
//Panel is the subdeviceID (valid values range from PANEL_GRATING1 to PANEL_GRATING4; from PANEL_DIGITPOS1
//to PANEL_DIGITPOS8; and from PANEL_DIGITPOSEX1 to PANEL_DIGITPOSEX4).
//Min is the valid minimum value for the subdevice (e.g., wavelength range [nm] for a grating),
//providing the returned value is True.
//Max is the valid maximum value for the subdevice (e.g., wavelength range [nm] for a grating),
//providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetRange)(int64_t Panel, float* Min, Max);

//Function SetResetEx is used to reset a subdevice or drive in your spectral device.
//Parameters:
//Panel is the subdevice ID (valid values are from PANEL_GRATING1 to PANEL_GRATING4; from PANEL_SLIT1 to
//PANEL_SLIT4; from PANEL_TURNSLIT1 to PANEL_TURNSLIT2; from PANEL_TURRET1 to PANEL_TURRET2;
//fromPANEL_FILTER1toPANEL_FILTER4; fromPANEL_MIRROR1toPANEL_MIRROR4; fromPANEL_MULTIPOS1
//to PANEL_MULTIPOS4; and from PANEL_DIGITPOSEX1 to PANEL_DIGITPOSEX4).
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetResetEx)(int64_t Panel);

//Function GetGratingMechanismCount is used to retrieve the number of grating mechanisms (e.g., turrets) installed
//in your spectral instrument.
//Result: Number of grating mechanisms (≥ 0).
typedef int32_t _stdcall (*GetGratingMechanismCount)(void);

//Function GetGratingMechanismName is used to query the name of specified grating mechanism (e.g., turret).
//Parameters:
//MechanismIndex is the index of the grating mechanism (starting from 1).
//Result: Name of the specified grating mechanism.
typedef ansichar* _stdcall (*GetGratingMechanismName)(int32_t MechanismIndex);

//Function GetGratingsCount is used to query the number of gratings installed in your spectral device.
//Parameters:
//Panel is the turret ID (valid values range from PANEL_TURRET1 to PANEL_TURRET2).
//GratingsCount is the number of gratings installed (registered) in your spectral device.
//Result: True in all cases.
typedef bool _stdcall (*GetGratingsCount)(int32_t Panel, int32_t* GratingsCount);

//Function GetCurGratingParam is used to query the main parameters of a current grating of the spectral channel.
//Parameters:
//Panel is the spectral channel index (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//Lines is the number of lines (grooves, rulings) in [mm −1 ], providing the returned value is True.
//Blaze is the blazing wavelength in [nm], providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetCurGratingParam)(int32_t Panel, int32_t* Lines, int32_t* Blaze);

//Function GetCurGratingParamEx is used to query the main parameters of a specified grating.
//Parameters:
//GratingIndex is the grating index (starting from 1).
//Lines is the number of lines (grooves, rulings) in [mm −1 ], providing the returned value is True.
//Blaze is the blazing wavelength in [nm], providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetCurGratingParamEx)(int32_t GratingIndex, int32_t* Lines, int32_t* Blaze);

//Function GetGratingInformation is used to retrieve some parameters of specified diffraction grating.
//Parameters:
//GratingIndex is the grating index (starting from 1). If GratingIndex is set to -1,
//then the active grating will be analysed.
//Focus is the focal length [m].
//AngularDeviation is the angular deviation ( θ /2) [radian].
//FocalPlaneTilt (obsolete) is the focal plane tilt [radian], providing the returned value is True.
//Result: True if parameter FocalPlaneTilt retrieved from the configuration file successfully, False otherwise.
typedef bool _stdcall (*GetGratingInformation)(int32_t GratingIndex, float* Focus, float* AngularDeviation, float* FocalPlaneTilt);

//Function IsEchelleGrating is used to check whether specified grating is an Echelle grating.
//Parameters:
//GratingIndex is the grating index (starting from 1). If GratingIndex is set to -1, 
//then the active grating will be analysed.
//Result: True if the grating is an Echelle grating, False otherwise.
typedef bool _stdcall (*IsEchelleGrating)(int32_t GratingIndex);

//Function GetDispersion is used to query the current reciprocal dispersion.
//Parameters:
//Panel is the ID of the current grating (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//Dispersion is the value of the current reciprocal dispersion [nm/mm], providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetDispersion)(int32_t Panel, float* Dispersion);

//Function GetWaveLength is used to query the current wavelength.
//Parameters:
//Panel is the ID of the current grating (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//WL is the value of wavelength [nm], providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetWaveLength)(int32_t Panel, float* WL);

//Function SetWaveLength is used to set a new value of the wavelength.
//Parameters:
//Panel is the ID of the current grating (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//WL is the new value of wavelength [nm] to be set.
//If IsReset is True then a reset operation will be executed prior to setting the new wavelength, otherwise the new
//wavelength will be set directly (i.e., without resetting). Can be omitted; the default value is False.
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetWaveLength)(int32_t Panel, float WL, bool IsReset=FALSE);

//Function GetWaveLengthByPixel is used to calculate the value of wavelength at a specified distance from the central pixel.
//Parameters:
//Panel is the ID of the grating, for which calculations is required (valid values range from PANEL_GRATING1 to
//PANEL_GRATING4).
//Shift is the distance [ µ m] from the so-called “central” pixel to the point in which we need to know the value of
//wavelength. Its value must be negative or positive for pixels with indices lower or higher than that of the central pixel,
//respectively.
//CentralWavelength now specifies only central wavelength, if CentralWavelength < 0, then current central WL will be used.
//if GratingIndex > GratingCount, then current grating will be used and lControlPanels specifies turret index (for now it is PANEL_GRATING1 or PANEL_GRATING2 (which is 1 or 2)).
//if GratingIndex < 0, then Panel specifies which grating to use (as it were before, values PANEL_GRATING1..4).
//if 0 < GratingIndex <= GratingCount, then GratingIndex is required grating index.
//Result: True if successful, False otherwise.
bool _stdcall GetWaveLengthByPixel(int32_t Panel, float Shift, float slPixelSize, float* CentralWavelength, int32_t GratingIndex = -1);

//Function QueryData is used to query data from a subdevice (e.g., PMT). More specifiedally, this routine performs
//single measurement.
//Parameters:
//Panel is the subdevice ID (valid values range from PANEL_QUERY1 to PANEL_QUERY4).
//Timeout is the timeout [ms] for this operation.
//Result: True if successful, False otherwise.
//Remark: The measured value can be retrieved by using function ReadBuffer.
//This function can only be used if spectral instrument has ADC board installed
typedef bool _stdcall (*QueryData)(int64_t Panel, int32_t Timeout);

//Function ReadBuffer is used to retrieve the measured data from the software local buffer.
//Parameters:
//Destination is the pointer to the data buffer.
//DestSize is the data size in bytes.
//Result: True if successful, False otherwise.
//Remark: The measurement itself is started by function QueryData.
//This function can only be used if spectral instrument has ADC board installed
typedef bool _stdcall (*ReadBuffer)(void* Destination, uint16_t* DestSize);

//Function GetNumStepsEx is used to calculate the number of points within specified wavelength range using specified
//wavelength step.
//Parameters:
//Panel is the grating ID (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//StartWL is the required starting wavelength [nm].
//FinishWL is the required final wavelength [nm].
//StepWL is the required wavelength step [nm].
//PointCount is the calculated number of points, providing the returned value is True.
//Providing the returned value is True, the required values of StartWL, FinishWL and StepWL will be
//substituted by real (hardware-bound) values!
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetNumStepsEx)(int32_t Panel, float* StartWL, float* FinishWL, float* StepWL, int32_t* PointCount);

//Function GetWLInRange is used to calculate the wavelength at specified scan point with defined
//start wavelength and wavelength step.
//Parameters:
//Panel is the grating ID (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//StartWL is the required starting wavelength [nm].
//StepWL is the required wavelength step [nm].
//Index is the scan step, for which wavelength is queried
//WL is the calculated wavelength, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetWLInRange)(int32_t Panel, float StartWL, float StepWL, int32_t Index, float* WL);

//Function MakeScan is used to scan a specified wavelength range (e.g., with a PMT).
//The scanning will be started from the current wavelength.
//Parameters:
//Panel is the grating ID (valid values range from PANEL_GRATING1 to PANEL_GRATING4).
//FinalWL is the wavelength [nm] where the scan will be finished.
//StepSize is the wavelength step [nm] of the current scan.
//PointCount is the total number of measured values obtained during the current scan, providing the returned value is True.
//Result: True if successful, False otherwise.
//Remark: The measured values can be retrieved by using function GetMeasuredValue.
typedef bool _stdcall (*MakeScan)(int64_t Panel, float FinalWL, StepSize, int32_t* PointCount);

//Function GetMeasuredValue is used to retrieve the next single value measured during a scan (e.g., with a PMT).
//In order to retrieve all measured values after the scan is complete, call this function PointCount times (see
//parameter PointCount in function MakeScan).
//Parameters:
//Value is the next measured value, providing the returned value is True.
//Result: True if successful, False otherwise.
//Remark: The scan itself is started by function MakeScan.
typedef bool _stdcall (*GetMeasuredValue)(int16_t* Value);

//Function GetTurretPos is used to query the current position of a turret (i.e., the index of the active diffraction grating).
//Parameters:
//Panel is the turret ID (valid values range from PANEL_TURRET1 to PANEL_TURRET2).
//Position is the current position of the turret (i.e., the index of the active diffraction grating), 
//providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetTurretPos)(int32_t Panel, int32_t* Position);

//Function GetTurretPosName is used to query the name of specified position of a turret.
//Parameters:
//Panel is the turret ID (valid values range from PANEL_TURRET1 to PANEL_TURRET2).
//Position is the index of the turret position (starting from 1).
//Name is the name of the turret position, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetTurretPosName)(int32_t Panel, int32_t Position, ansichar* Name);

//Function SetTurret is used to set new position of a Turret (i.e., to set new diffraction grating).
//Parameters:
//Panel is the Turret ID (valid values range from PANEL_TURRET1 to PANEL_TURRET2).
//Position is the index of the new position (i.e., index of grating) to be set (starting from 1).
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetTurret)(int32_t Panel, int32_t Position);

//Function GetShutterState is used to query the shutter state.
//Parameters:
//Panel is the shutter ID (valid values range from PANEL_SHUTTER1 to PANEL_SHUTTER4). Can be omitted; the
//default value is PANEL_SHUTTER1.
//Result: SHUTTER_OPEN if the software is in demo mode or device is not yet initialised; ERROR_INCORRECTPARAM if
//Panel is out of valid range; and for normal operation SHUTTER_OPEN/SHUTTER_CLOSE if the shutter is open/closed, respectively.
typedef uint8_t _stdcall (*GetShutterState)(int32_t Panel=PANEL_SHUTTER1);

//Function OpenShutter is used to open a shutter.
//Parameters:
//Panel is the shutter ID (valid values range from PANEL_SHUTTER1 to PANEL_SHUTTER4). Can be omitted; the default value is PANEL_SHUTTER1.
//If NotIsActive is True then routine tries to open the so-called “active” shutter (e.g., feature of MSDD100x device
//line) even if Panel refers to another shutter. Can be omitted; the default value is True.
//Result: True if operation is started, False otherwise.
typedef bool _stdcall (*OpenShutter)(int32_t Panel=PANEL_SHUTTER1, bool NotIsActive=TRUE);

//Function CloseShutter is used to close a shutter.
//Parameters:
//Panel is the shutter ID (valid values range from PANEL_SHUTTER1 to PANEL_SHUTTER4). Can be omitted; the
//default value is PANEL_SHUTTER1.
//If NotIsActive is True then routine tries to close the so-called “active” shutter (e.g., feature of MSDD1000 device
//line) even if Panel refers to another shutter. Can be omitted; the default value is True.
//Result: True if operation is started, False otherwise.
typedef bool _stdcall (*CloseShutter)(int32_t Panel=PANEL_SHUTTER1, bool NotIsActive=TRUE);

//Function SetShutterMode is used to trigger the control mode of a shutter (either “TTL” or “Soft”).
//Parameters:
//Panel is the shutter ID (valid values range from PANEL_SHUTTER1 to PANEL_SHUTTER4).
//If IsTTL is True then the routine will try to set the shutter control mode to “TTL”, otherwise to “Soft”.
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetShutterMode)(int32_t Panel, bool IsTTL);

//Function GetMirrorPos is used to query the current position of a mirror.
//Parameters:
//Panel is the mirror ID (valid values range from PANEL_MIRROR1 to PANEL_MIRROR4).
//Position is the index of the current position of the mirror, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetMirrorPos)(int32_t Panel, int32_t* Position);

//Function GetMirrorPosName is used to query the name of specified position of a mirror.
//Parameters:
//Panel is the mirror ID (valid values range from PANEL_MIRROR1 to PANEL_MIRROR4).
//Position is the index of the specified position of the mirror (starting from 1).
//Name is the name of the position of the mirror, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetMirrorPosName)(int32_t Panel, int32_t Position, ansichar* Name);

//Function SetMirror is used to set new position of a mirror.
//Parameters:
//Panel is the mirror ID (valid values range from PANEL_MIRROR1 to PANEL_MIRROR4).
//Position is the index of the mirror position to be set (starting from 1).
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetMirror)(int32_t Panel, int32_t  Position);

//Function GetSlitWidth is used to query the current width of a slit.
//Parameters:
//Panel is the slit ID (valid values range from PANEL_SLIT1 to PANEL_SLIT4).
//Width is the value of the current slit width, providing the returned value is True.
//Result: True if successful, False otherwise.
//Remark: Depending on the current setting of IsSpectralWidth in SetSlitWidth, 
//the value of Width will be in [µm] or [nm].
typedef bool _stdcall (*GetSlitWidth)(int32_t Panel,  float* Width);

//Function SetSlitWidth is used to set a new value of the slit width.
//Parameters:
//Panel is the slit ID (valid values range from PANEL_SLIT1 to PANEL_SLIT4).
//Width is the new value of the width to be set.
//If IsSpectralWidth is True then the slit width will be set so as to provide a spectral width of Width[nm](according
//to the current value of dispersion), otherwise the slit width will be set equal to Width [µm]. Can be omitted; the default value is False.
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetSlitWidth)(int32_t Panel, float Width, bool IsSpectralWidth=false);

//Function GetFilter is used to query the active filter index (e.g., in a filter wheel).
//Parameters:
//Panel is the filter ID (valid values range from PANEL_FILTER1 to PANEL_FILTER4).
//FilterIndex is the index of the active filter, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetFilter)(int32_t Panel, int32_t* FilterIndex);

//Function GetFilterNum is used to query the number of positions in a filters holder (e.g., in a filter wheel).
//Parameters:
//Panel is the filter ID (valid values range from PANEL_FILTER1 to PANEL_FILTER4).
//PositionCount is the number of positions in the filters holder (filter wheel), providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetFilterNum)(int32_t Panel, int32_t* PositionCount);

//Function GetFilterCaption is used to query the caption of a filter.
//Parameters:
//Panel is the filter ID (valid values range from PANEL_FILTER1 to PANEL_FILTER4).
//FilterIndex is the index of the filter.
//Caption is the caption (name) of the specified filter, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetFilterCaption)(int32_t Panel, int32_t FilterIndex, ansichar* Caption);

//Function SetFilter is used to set a new filter.
//Parameters:
//Panel is the filter ID (valid values range from PANEL_FILTER1 to PANEL_FILTER4).
//FilterIndex is the index of filter to be set (starting from 1).
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetFilter)(int32_t Panel, int32_t FilterIndex);

//Function GetFilterControlState is used to retrieve the control mode of a filter (either “Auto” or “Manual”).
//Parameters:
//Panel is the filter ID (valid values range from PANEL_FILTER1 to PANEL_FILTER4).
//Result: FC_AUTO or FC_MANUAL if the filter is in the automatic or manual control mode, respectively.
typedef uint8_t _stdcall (*GetFilterControlState)(int32_t Panel);

//Procedure SetFilterControl is used to trigger the control mode of a filter (either “Auto” or “Manual”).
//Parameters:
//Panel is the filter ID (valid values range from PANEL_FILTER1 to PANEL_FILTER4).
//FilterControlState determines the filter control mode(valid values are FC_AUTO and FC_MANUAL -- self-explaining).
typedef void _stdcall (*SetFilterControl)(int32_t Panel, uint8_t FilterControlState);

//Function GetMultiPosDev is used to query the current position of a MultiPos.
//Parameters:
//Panel is the MultiPos ID (valid values range from PANEL_MULTIPOS1 to PANEL_MULTIPOS4).
//Position is the current position of the MultiPos, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetMultiPosDev)(int32_t lControlPanels, int32_t* lPosition);

//Function SetMultiPosDev is used to set new position of a MultiPos.
//Parameters:
//Panel is the MultiPos ID (valid values range from PANEL_MULTIPOS1 to PANEL_MULTIPOS4).
//Position is the index of the new position to be set (starting from 1).
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetMultiPosDev)(int32_t Panel, int32_t Position);

//Function SetState is used to switch certain subdevice on or off.
//Parameters:
//Panel is the subdevice ID (valid values range from PANEL_DIGITPOS1 to PANEL_DIGITPOS8 and from
//PANEL_BISTABLE1 to PANEL_BISTABLE2).
//IsDeviceOn must be either STATE_ON or STATE_OFF (see Other constants).
//Result: True if successful, False otherwise.
typedef bool _stdcall (*SetState)(int64_t Panel, int32_t IsDeviceOn);

//Function GetDigitPosValue is used to query the current value of a DigitPos.
//Parameters:
//Panel is the DigitPos ID (valid values range from PANEL_DIGITPOS1 to PANEL_DIGITPOS8).
//Value is the current value of the DigitPos, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetDigitPosValue)(int64_t Panel, float* Value);

//Function SetDigitPosValue is used to set new value of a DigitPos.
//Parameters:
//Panel is the DigitPos ID (valid values range from PANEL_DIGITPOS1 to PANEL_DIGITPOS8).
//Value is the new value to be set.
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetDigitPosValue)(int64_t Panel, float Value);

//Function GetDigitPosExValue is used to query the current position of a DigitPosEx.
//Parameters:
//Panel is the DigitPosEx ID (valid values range from PANEL_DIGITPOSEX1 to PANEL_DIGITPOSEX4).
//Value is the current position of the DigitPosEx, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetDigitPosExValue)(int64_t Panel, float* Value);

//Function SetDigitPosExValue is used to set new value of a DigitPosEx.
//Parameters:
//Panel is the DigitPosEx ID (valid values range from PANEL_DIGITPOSEX1 to PANEL_DIGITPOSEX4).
//Value is the new value to be set.
//If IsReset is True then a reset operation will be executed prior to setting the new value, otherwise the new value will
//be set directly (i.e., without resetting).
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetDigitPosExValue)(int64_t Panel, float Value, bool IsReset);

//Function GetTurnSlit is used to query the current position of a TurnSlit.
//Parameters:
//Panel is the TurnSlit ID (valid values range from PANEL_TURNSLIT1 to PANEL_TURNSLIT2).
//Angle is the current position of the TurnSlit, providing the returned value is True.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetTurnSlit)(int32_t Panel, float* Angle);

//Function SetTurnSlit is used to set new position of a TurnSlit.
//Parameters:
//Panel is the TurnSlit ID (valid values range from PANEL_TURNSLIT1 to PANEL_TURNSLIT2).
//Angle is the new position (angle [angular degree], valid range from 0 to 22°) to be set.
//Result: True if successful, False otherwise.
//Remark: Use function IsOperationFinished to check whether the operation is still in progress or already finished.
//Use function GetDeviceError to check whether any error is encountered.
typedef bool _stdcall (*SetTurnSlit)(int32_t Panel, float Angle);

//Function GetTurnSlitControlMode is used to query the current control mode of a TurnSlit.
//Parameters:
//Panel is the TurnSlit ID (valid values range from PANEL_TURNSLIT1 to PANEL_TURNSLIT2).
//IsAuto is the current control mode of the TurnSlit, providing the returned value is True. If IsAuto = True, then the
//TurnSlit is turned automatically, otherwise manually.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*GetTurnSlitControlMode)(int32_t Panel; bool* IsAuto);

//Function SetTurnSlitControlMode is used to set the control mode of a TurnSlit.
//Parameters:
//Panel is the TurnSlit ID (valid values range from PANEL_TURNSLIT1 to PANEL_TURNSLIT2).
//IsAuto is the new control mode of the TurnSlit to be set. If IsAuto = True,
//then the TurnSlit will be turned automatically, otherwise manually.
//Result: True if successful, False otherwise.
typedef bool _stdcall (*SetTurnSlitControlMode)(int32_t Panel; bool IsAuto);