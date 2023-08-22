// Created on 27.11.2008 by Dr. Andrey Kamarou
// © Solar TII, Ltd., Mensk, BELARUS

Unit DevCtrl_I;

Interface

Uses
  Windows, Controls, Dialogs, SysUtils;

Const
//State of a DigitPos or Bistable device
  STATE_OFF     = -1;
  STATE_ON      =  1;

//Control mode of a filter 
  FC_AUTO       = 0;
  FC_MANUAL     = 1;

//Time info type for TraceMon (debugging feature)  
  INFO_TIME     = 1;
  INFO_DIFFS    = 2;

//State of a shutter
  SHUTTER_OPEN  = 3;
  SHUTTER_CLOSE = 4;
  
  
//Control (drive) panels
  PANEL_ALL         = 0;

  PANEL_GRATING1    = int64(1) shl  0; // 1;
  PANEL_GRATING2    = int64(1) shl  1; // 2;
  PANEL_GRATING3    = int64(1) shl  2; // 4;
  PANEL_GRATING4    = int64(1) shl  3; // 8;

  PANEL_SLIT1       = int64(1) shl  4; // 16;
  PANEL_SLIT2       = int64(1) shl  5; // 32;
  PANEL_SLIT3       = int64(1) shl  6; // 64;
  PANEL_SLIT4       = int64(1) shl  7; // 128;

  PANEL_TURNSLIT1   = int64(1) shl  8; // 256;
  PANEL_TURNSLIT2   = int64(1) shl  9; // 512;

  PANEL_TURRET1     = int64(1) shl 10; // 1024;
  PANEL_TURRET2     = int64(1) shl 11; // 2048;

  PANEL_SHUTTER1    = int64(1) shl 12; // 4096;
  PANEL_SHUTTER2    = int64(1) shl 13; // 8192;
  PANEL_SHUTTER3    = int64(1) shl 14; // 16384;
  PANEL_SHUTTER4    = int64(1) shl 15; // 32768;

  PANEL_FILTER1     = int64(1) shl 16; // 65536;
  PANEL_FILTER2     = int64(1) shl 17; // 131072;
  PANEL_FILTER3     = int64(1) shl 18; // 262144;
  PANEL_FILTER4     = int64(1) shl 19; // 524288;

  PANEL_MIRROR1     = int64(1) shl 20; // 1048576;
  PANEL_MIRROR2     = int64(1) shl 21; // 2097152;
  PANEL_MIRROR3     = int64(1) shl 22; // 4194304;
  PANEL_MIRROR4     = int64(1) shl 23; // 8388608;

  PANEL_MULTIPOS1   = int64(1) shl 26; // 67108864;
  PANEL_MULTIPOS2   = int64(1) shl 27; // 134217728;
  PANEL_MULTIPOS3   = int64(1) shl 28; // 268435456;
  PANEL_MULTIPOS4   = int64(1) shl 29; // 536870912;

  PANEL_DIGITPOS1   = int64(1) shl 31; // 2147483648;
  PANEL_DIGITPOS2   = int64(1) shl 32; // 4294967296;
  PANEL_DIGITPOS3   = int64(1) shl 33; // 8589934592;
  PANEL_DIGITPOS4   = int64(1) shl 34; // 17179869184;
  PANEL_DIGITPOS5   = int64(1) shl 35; // 34359738368;
  PANEL_DIGITPOS6   = int64(1) shl 36; // 68719476736;
  PANEL_DIGITPOS7   = int64(1) shl 37; // 137438953472;
  PANEL_DIGITPOS8   = int64(1) shl 38; // 274877906944;

  PANEL_BISTABLE1   = int64(1) shl 39; // 549755813888;
  PANEL_BISTABLE2   = int64(1) shl 40; // 1099511627776;

  PANEL_QUERY1      = int64(1) shl 43; // 8796093022208;
  PANEL_QUERY2      = int64(1) shl 44; // 17592186044416;
  PANEL_QUERY3      = int64(1) shl 45; // 35184372088832;
  PANEL_QUERY4      = int64(1) shl 46; // 70368744177664;

  PANEL_DIGITPOSEX1 = int64(1) shl 47; // 140737488355328;
  PANEL_DIGITPOSEX2 = int64(1) shl 48; // 281474976710656;
  PANEL_DIGITPOSEX3 = int64(1) shl 49; // 562949953421312;
  PANEL_DIGITPOSEX4 = int64(1) shl 50; // 1125899906842624;


//Configuration form pages  
  CFG_PAGES_ALL        = 0;
  CFG_PAGE_GRATING     = integer(1) shl  0;
  CFG_PAGE_TURRET      = integer(1) shl  1;
  CFG_PAGE_SLIT        = integer(1) shl  2;
  CFG_PAGE_TURNSLIT    = integer(1) shl  3;
  CFG_PAGE_FILTER      = integer(1) shl  4;
  CFG_PAGE_MIRROR      = integer(1) shl  5;
  CFG_PAGE_MULTIPOS1   = integer(1) shl  6;
  CFG_PAGE_MULTIPOS2   = integer(1) shl  7;
  CFG_PAGE_MULTIPOS3   = integer(1) shl  8;
  CFG_PAGE_MULTIPOS4   = integer(1) shl  9;
  CFG_PAGE_DIGITPOS1   = integer(1) shl 10;
  CFG_PAGE_DIGITPOS2   = integer(1) shl 11;
  CFG_PAGE_DIGITPOS3   = integer(1) shl 12;
  CFG_PAGE_DIGITPOS4   = integer(1) shl 13;
  CFG_PAGE_DIGITPOS5   = integer(1) shl 14;
  CFG_PAGE_DIGITPOS6   = integer(1) shl 15;
  CFG_PAGE_DIGITPOS7   = integer(1) shl 16;
  CFG_PAGE_DIGITPOS8   = integer(1) shl 17;
  CFG_PAGE_DIGITPOSEX1 = integer(1) shl 18;
  CFG_PAGE_DIGITPOSEX2 = integer(1) shl 19;
  CFG_PAGE_DIGITPOSEX3 = integer(1) shl 20;
  CFG_PAGE_DIGITPOSEX4 = integer(1) shl 21;


//Types of subdevices (drives)
  SDT_Grating        =  1;
  SDT_Turret         =  2;
  SDT_Slit           =  3;
  SDT_TurnSlit       =  4;
  SDT_Shutter        =  5;
  SDT_Mirror         =  6;
  SDT_Filter         =  7;
  SDT_DoubleFilter   =  8;
  SDT_MultiPos       =  9;
  SDT_DigitPos       = 10;
  SDT_BiStableDevice = 11;
  SDT_QueryDevice    = 12;
  SDT_DigitPosEx     = 13;

//Error codes for SOLAR TII devices
  ALL_OK               =  0;
  RECEIVING_DATA       =  1;
  PREVIOUSPOS_NEED     =  2;

  DEVICE_NOTFOUND      =  6;

  ERROR_OVERFULL       =  7;
  ERROR_SYMBOLFAULT    =  8;
  ERROR_OVERTIME       =  9;
  ERROR_INTERFACE      = 10;
  ERROR_HARDWARE       = 11;
  ERROR_STEPMOTOR      = 12;
  ERROR_UNKNOWNSYMBOL  = 13;
  ERROR_UNKNOWNERROR   = 14;
  ERROR_READTIMEOUT    = 15;
  ERROR_WRITETIMEOUT   = 16;
  ERROR_RECEIVEDCODE   = 17;
  ERROR_DEVICENOTREADY = 18;
  ERROR_NOTCONNECT     = 19;
  ERROR_INITIALISATION = 20;
  ERROR_UNKNOWNPORT    = 21;
  ERROR_CONVERTATION   = 22;
  ERROR_INCORRECTPARAM = 23;
  ERROR_UNKNOWNDEVICE  = 24;
  ERROR_RANGE          = 25;
  ERROR_RESET          = 26;
  ERROR_MEMORY         = 27;

//Error codes for SOLAR JS devices
  ERROR_UNKNOWN           = 70;
  ERROR_OVERFLOW          = 71;
  ERROR_RESET_EXSLIT      = 72;
  ERROR_RESET_EMSLIT      = 73;
  ERROR_RESET_EXFILTER    = 74;
  ERROR_RESET_EMFILTER    = 75;
  ERROR_RESET_EXDIAPHRAGM = 76;
  ERROR_RESET_EXGRATE     = 77;
  ERROR_RESET_EMGRATE     = 78;
  ERROR_ADC_READY         = 79;
  ERROR_LAMP              = 80;


Type
  TDimension = (dmM, dmMM, dmUM {micron}, dmNM, dmA {Angstroem}, dm1CM {reciprocal centimeter}, dmRad, dmGrad {angular degree}, dmStep);
    
Var
//////////////////////////////////////////////////////////////////////
//     Description of variables for dynamic linking (importing)     //
//////////////////////////////////////////////////////////////////////


//All DLL routines are exported according to STDCALL calling convention!
  CheckConnection          : function: boolean; stdcall;
  CheckDevice              : function: boolean; stdcall;
  CloseShutter             : function(Panel: longint = PANEL_SHUTTER1; NotIsActive: boolean = True): boolean; stdcall;

  FreeDevice               : procedure; stdcall;

  GetActiveDevice          : function: integer; stdcall;
  GetCentralWLByNullPixelWL: function(Panel: longint; Length, PixelSize, NullWL: single; var slCentralWL, slRightWL: single; NumPixel: integer = -1): boolean; stdcall;
  GetConfigInfo            : function(DeviceName: PChar; var SN: longint): boolean; stdcall;
  GetCurGratingParam       : function(Panel: integer; var Lines, Blaze: integer): boolean; stdcall;
  GetCurGratingParamEx	   : function(GratingIndex: integer; var Lines, Blaze: integer): boolean; stdcall;
  GetDevCtrlVersion        : function: integer; stdcall;
  GetDeviceError           : function: byte; stdcall;
  GetDeviceInfo            : function(DeviceName: PChar; var SN: DWORD): boolean; stdcall;
  GetDeviceName            : procedure(DeviceIndex: integer; DeviceName: PChar); stdcall;
  GetDevicePurpose         : function: integer; stdcall;
  GetDigitPosExCorrection  : function(Panel: int64; var Value: single): boolean; stdcall;
  GetDigitPosExValue       : function(Panel: int64; var Value: single): boolean; stdcall;
  GetDigitPosValue         : function(Panel: int64; var Value: single): boolean; stdcall;
  GetDispersion            : function(Panel: longint; var Dispersion: single): boolean; stdcall;
  GetFilter                : function(Panel: longint; var FilterIndex: longint): boolean; stdcall;
  GetFilterCaption         : function(Panel: longint; FilterIndex: longint; Caption: PChar): boolean; stdcall;
  GetFilterControlState    : function(Panel: longint): byte; stdcall;
  GetFilterNum             : function(Panel: longint; var PositionCount: longint): boolean; stdcall;
  GetFullDevCtrlVersion    : function(Directory: PChar; var Major, Minor, Release, Build: integer): boolean; stdcall;
  GetGratingInformation    : function(GratingIndex: integer; var Focus, AngularDeviation, FocalPlaneTilt: single): boolean; stdcall;
  GetGratingMechanismCount : function: integer; stdcall;
  GetGratingMechanismName  : function(MechanismIndex: integer) : PChar; stdcall;
  GetGratingsCount  	     : function(Panel: integer; var GratingsCount: integer): boolean; stdcall;
  GetLampState             : function: integer; stdcall;
  GetMeasuredValue         : function(var Value: DWORD): boolean; stdcall;
  GetMinWLStep             : function(Panel: longint; var MinStep: single): boolean; stdcall;
  GetMirrorPos             : function(Panel: longint; var Position: longint): boolean; stdcall;
  GetMirrorPosName         : function(Panel, Position: longint; Name: PChar): boolean; stdcall;
  GetMultiPosDev           : function(Panel: longint; var Position: longint): boolean; stdcall;
  GetNumSteps              : function(Panel: longint; var StartWL, FinishWL, StepWL: single; var PointCount: integer): boolean; stdcall;
  GetNumStepsEx            : function(Panel: longint; var StartWL, FinishWL, StepWL: single; var PointCount: integer): boolean; stdcall;
  GetRange                 : function(Panel: int64; var Min, Max: single): boolean; stdcall;
  GetShutterState          : function(Panel: longint = PANEL_SHUTTER1): byte; stdcall;
  GetSlitWidth             : function(Panel: longint; var Width: single): boolean; stdcall;
  GetSubDevicesCount       : function(SubDeviceType: byte): integer; stdcall;
  GetSupportedDevicesCount : function: integer; stdcall;
  GetTurnSlit              : function(Panel: longint; var Angle: single): boolean; stdcall;
  GetTurnSlitControlMode   : function(Panel: longint; var IsAuto: boolean): boolean; stdcall;
  GetTurretPos             : function(Panel: longint; var Position: longint): boolean; stdcall;
  GetTurretPosName         : function(Panel, Position: longint; Name: PChar): boolean; stdcall;
  GetWaveLength            : function(Panel: longint; var WL: single): boolean; stdcall;
  GetWaveLengthByPixel     : function(Panel: longint; Shift, PixelSize: single; var WL: single; PixelIndex: integer = -1): boolean; stdcall;
  GetWLInRange             : function(Panel: longint; FromWL, StepWL: single; Index: integer; var WL: single): boolean; stdcall;
  GetWLInRangeEx           : function(Panel: longint; FromWL, StepWL: single; Index: integer; var WL: single): boolean; stdcall;
  GetWLPixelByCentralWL    : function(Panel: longint; Length, PixelSize, CentralWL: single; var WL: single; NumPixel: integer = -1): boolean; stdcall;

  InitDevice               : function(hAppWnd: HWND): boolean; stdcall;
  InitDeviceEx             : function(hAppWnd: HWND; var ComPort: byte; FilePath: PChar): boolean; stdcall;
  InitDeviceEx2            : function(hAppWnd: HWND; var ComPort: byte; FilePath: PChar; IsShowDialog: boolean): boolean; stdcall;
  InitInstrument           : function: boolean; stdcall;
  IsEchelleGrating         : function(GratingIndex: integer): boolean; stdcall;
  IsOperationFinished      : function: boolean; stdcall;

  LockPanels               : function(IsLockPanels: boolean): boolean; stdcall;

  MakeScan                 : function(Panel: int64; FinalWL, StepSize: single; var PointCount: longint): boolean; stdcall;
  MoveBySteps              : function(Panel: longint; StepCount: int64): boolean; stdcall;
  MoveDeviceEnabled        : procedure(IsEnabled: boolean); stdcall;

  OpenShutter              : function(Panel: longint = PANEL_SHUTTER1; NotIsActive: boolean = True): boolean; stdcall;

  QueryData                : function(Panel: int64; Timeout: integer): boolean; stdcall;

  ReadBuffer               : function(Destination: Pointer; var DestSize: DWORD): boolean; stdcall;

  SendByteToCOM            : function(Value: byte): boolean; stdcall;
  SetDigitPosExValue       : function(Panel: int64; Value: single; IsReset: boolean): boolean; stdcall;
  SetDigitPosValue         : function(Panel: int64; Value: single): boolean; stdcall;
  SetFilter                : function(Panel: longint; FilterIndex: longint): boolean; stdcall;
  SetFilterControl         : procedure(Panel: longint; FilterControlState: byte);
  SetInfoType              : function(InfoType: byte): boolean; stdcall;
  SetMirror                : function(Panel: longint; Position: longint): boolean; stdcall;
  SetMultiPosDev           : function(Panel: longint; Position: longint): boolean; stdcall;
  SetParentControl         : procedure(ParentControl: TWinControl); stdcall;
  SetPermissionMessageBox  : procedure(IsPermitted: boolean); stdcall;
  SetPreviousPosition      : function: boolean; stdcall;
  SetReset                 : function(Panel: longint): boolean; stdcall;
  SetResetEx               : function(Panel: int64): boolean; stdcall;
  SetResetGrating          : function(Panel: longint): boolean; stdcall;
  SetShutterMode           : function(Panel: longint; IsTTL: boolean): boolean; stdcall;
  SetSlitWidth             : function(Panel: longint; Width: single; IsSpectralWidth: boolean = False): boolean; stdcall;
  SetState                 : function(Panel: int64; IsDeviceOn: integer): boolean; stdcall;
  SetTurnSlit              : function(Panel: longint; Angle: single): boolean; stdcall;
  SetTurnSlitControlMode   : function(Panel: longint; IsAuto: boolean): boolean; stdcall;
  SetTurret                : function(Panel: longint; Position: longint): boolean; stdcall;
  SetWaveLength            : function(Panel: longint; WL: single; IsReset: boolean = False): boolean; stdcall;
  ShowConfig               : function(Panel: integer): boolean; stdcall;
  ShowDeviceWindow         : procedure(Panel: longint); stdcall;
  ShowDeviceWindowEx       : procedure(Panel: int64); stdcall;
  StepByStep               : function(Panel: longint; StepCount: longint): boolean; stdcall;

  WND_MSG_Register         : procedure(hWindow: HWND; MsgID: UINT); stdcall;

  StepCountToWavelength    : function(StepCount: longint; Grating: Byte; var WaveLength: real) : byte; stdcall;
  WavelengthToStepCount    : function(WaveLength: real; Grating: Byte; var StepCount: longint) : byte; stdcall;
  
  CalculateGratingFocus    :function(lControlPanels       : longint;
                                     slLength1            : single; //[mkm] Distance from central pixel to first peak
                                     slWavelength1        : Single; //[nm] Wavelength at distance to first peak
                                     slLength2            : single; //[mkm] Distance from central pixel to 2nd peak
                                     slWavelength2        : Single  //[nm] Wavelength at distance to 2nd peak
                                    ) : boolean; stdcall;


function LoadDevCtrlDLL(Path: string; IsCheckAllRoutines: boolean = True): THandle;


Implementation


function LoadDevCtrlDLL(Path: string; IsCheckAllRoutines: boolean = True): THandle;
var
  h: THandle;
  IsOK: Boolean;
  str: string;

  function Import(Name: PChar): Pointer; //nested routine
  begin
    //Attention! According to the description of "GetProcAddress" routine in Win32 SDK, parameter Name here is case-sensitive!
    //The spelling and case of the function name pointed to by Name must be identical to that in the EXPORTS statement of the source DLL file!
    //The correct spelling of the routines exported by a DLL can be checked with "Anywhere PE Viewer" (freeware), for example.
    Result:= GetProcAddress(h, Name);
    if IsCheckAllRoutines
    then if Result = nil
         then begin
                IsOK:= False;
                str:= str + #13#10 + Name;
              end;  
  end;
  
begin
  IsOK:= False;
  str:= '';

  Result:= LoadLibrary(PChar(Path));

  if Result <> 0
  then begin
         IsOK:= True;
         h:= Result;

         @CheckConnection           := Import('CheckConnection');
         @CheckDevice               := Import('CheckDevice');
         @CloseShutter              := Import('CloseShutter');

         @FreeDevice                := Import('FreeDevice');

         @GetActiveDevice           := Import('GetActiveDevice');
         @GetCentralWLByNullPixelWL := Import('GetCentralWLByNullPixelWL');
         @GetConfigInfo             := Import('GetConfigInfo');
         @GetCurGratingParam        := Import('GetCurGratingParam');
         @GetCurGratingParamEx      := Import('GetCurGratingParamEx');
         @GetDevCtrlVersion         := Import('GetDevCtrlVersion');
         @GetDeviceError            := Import('GetDeviceError');
         @GetDeviceInfo             := Import('GetDeviceInfo');
         @GetDeviceName             := Import('GetDeviceName');
         @GetDevicePurpose          := Import('GetDevicePurpose');
         @GetDigitPosExCorrection   := Import('GetDigitPosExCorrection');
         @GetDigitPosExValue        := Import('GetDigitPosExValue');
         @GetDigitPosValue          := Import('GetDigitPosValue');
         @GetDispersion             := Import('GetDispersion');
         @GetFilter                 := Import('GetFilter');
         @GetFilterCaption          := Import('GetFilterCaption');
         @GetFilterControlState     := Import('GetFilterControlState');
         @GetFilterNum              := Import('GetFilterNum');
         @GetFullDevCtrlVersion     := Import('GetFullDevCtrlVersion');
         @GetGratingInformation     := Import('GetGratingInformation');
         @GetGratingMechanismCount  := Import('GetGratingMechanismCount');
         @GetGratingMechanismName   := Import('GetGratingMechanismName');
         @GetGratingsCount          := Import('GetGratingsCount');
         @GetLampState              := Import('GetLampState');
         @GetMeasuredValue          := Import('GetMeasuredValue');
         @GetMinWLStep              := Import('GetMinWLStep');
         @GetMirrorPos              := Import('GetMirrorPos');
         @GetMirrorPosName          := Import('GetMirrorPosName');
         @GetMultiPosDev            := Import('GetMultiPosDev');
         @GetNumSteps               := Import('GetNumSteps');
         @GetNumStepsEx             := Import('GetNumStepsEx');
         @GetRange                  := Import('GetRange');
         @GetShutterState           := Import('GetShutterState');
         @GetSlitWidth              := Import('GetSlitWidth');
         @GetSubDevicesCount        := Import('GetSubDevicesCount');
         @GetSupportedDevicesCount  := Import('GetSupportedDevicesCount');
         @GetTurnSlit               := Import('GetTurnSlit');
         @GetTurnSlitControlMode    := Import('GetTurnSlitControlMode');
         @GetTurretPos              := Import('GetTurretPos');
         @GetTurretPosName          := Import('GetTurretPosName');
         @GetWaveLength             := Import('GetWaveLength');
         @GetWaveLengthByPixel      := Import('GetWaveLengthByPixel');
         @GetWLInRange              := Import('GetWLInRange');
         @GetWLInRangeEx            := Import('GetWLInRangeEx');
         @GetWLPixelByCentralWL     := Import('GetWLPixelByCentralWL');

         @InitDevice                := Import('InitDevice');
         @InitDeviceEx              := Import('InitDeviceEx');
         @InitDeviceEx2             := Import('InitDeviceEx2');
         @InitInstrument            := Import('InitInstrument');
         @IsEchelleGrating          := Import('IsEchelleGrating');
         @IsOperationFinished       := Import('IsOperationFinished');

         @LockPanels                := Import('LockPanels');

         @MakeScan                  := Import('MakeScan');
         @MoveBySteps               := Import('MoveBySteps');
         @MoveDeviceEnabled         := Import('MoveDeviceEnabled');

         @OpenShutter               := Import('OpenShutter');

         @QueryData                 := Import('QueryData');

         @ReadBuffer                := Import('ReadBuffer');

         @SendByteToCOM             := Import('SendByteToCOM');
         @SetDigitPosExValue        := Import('SetDigitPosExValue');
         @SetDigitPosValue          := Import('SetDigitPosValue');
         @SetFilter                 := Import('SetFilter');
         @SetFilterControl          := Import('SetFilterControl');
         @SetInfoType               := Import('SetInfoType');
         @SetMirror                 := Import('SetMirror');
         @SetMultiPosDev            := Import('SetMultiPosDev');
         @SetParentControl          := Import('SetParentControl');
         @SetPermissionMessageBox   := Import('SetPermissionMessageBox');
         @SetPreviousPosition       := Import('SetPreviousPosition');
         @SetReset                  := Import('SetReset');
         @SetResetEx                := Import('SetResetEx');
         @SetResetGrating           := Import('SetResetGrating');
         @SetShutterMode            := Import('SetShutterMode');
         @SetSlitWidth              := Import('SetSlitWidth');
         @SetState                  := Import('SetState');
         @SetTurnSlit               := Import('SetTurnSlit');
         @SetTurnSlitControlMode    := Import('SetTurnSlitControlMode');
         @SetTurret                 := Import('SetTurret');
         @SetWaveLength             := Import('SetWaveLength');
         @ShowConfig                := Import('ShowConfig');
         @ShowDeviceWindow          := Import('ShowDeviceWindow');
         @ShowDeviceWindowEx        := Import('ShowDeviceWindowEx');
         @StepByStep                := Import('StepByStep');

         @WND_MSG_Register          := Import('WND_MSG_Register');

         @StepCountToWavelength     := Import('StepCountToWavelengthEx');
         @WavelengthToStepCount     := Import('WavelengthToStepCountEx');
         @CalculateGratingFocus     := Import('CalculateGratingFocusEx');

         if not IsOK
         then MessageDlg('Failed to locate some functions in '+
                         ExtractFileName(Path)+'!'+#13#10+
                         'These are as follows:'+
                         str, mtWarning, [mbOK], 0);
       end
  else begin
         Result:= INVALID_HANDLE_VALUE;
         MessageDlg('Failed to load the Device Control Library (Devctrl.dll)!', mtWarning, [mbOK], 0);
       end;
end;

END.
