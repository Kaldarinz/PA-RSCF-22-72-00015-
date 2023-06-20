
import ctypes
import sys
import pathlib
from ctypes import c_char_p, wintypes
import win32gui

class Handle():

    def __init__(self) -> None:
        self.hwnd = 0

    def enum_handler(self, hwnd, lParam):
        if win32gui.GetWindowText(hwnd) == 'MS5204i_15098':
            print(f'HWND = {hwnd}')
            self.hwnd = hwnd
    
    def get_handle(self):
        win32gui.EnumWindows(self.enum_handler, None)

    

if __name__ == "__main__":
    libname = str(pathlib.Path().absolute())
    libname += '\solar'
    print("libname: ", libname)

    # Load the shared library into c types.
    if sys.platform.startswith("win"):
        DevCtrl = ctypes.WinDLL(libname+"\DevCtrl.dll")
    else:
        print('Program terminated!')

    DevCtrl.InitDeviceEx2.argtypes = [wintypes.HWND, ctypes.c_byte, ctypes.c_char_p, ctypes.c_bool]
    DevCtrl.InitDeviceEx2.restype = ctypes.c_bool
    path_to_cfg = ctypes.c_char_p(bytes(libname, 'utf-8'))
    handle = Handle()
    handle.get_handle()
    initDev = DevCtrl.InitDeviceEx2(handle.hwnd, 0, path_to_cfg, True)
    print(f'Init = {initDev}')

    DevCtrl.GetDevCtrlVersion.restype = ctypes.c_int
    print(f'Init = {DevCtrl.GetDevCtrlVersion()}')

   
