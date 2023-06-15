#!/usr/bin/env python
""" Simple examples of calling C functions through ctypes module. """
import ctypes
import sys
import pathlib
from ctypes import wintypes
from traceback import print_tb

if __name__ == "__main__":
    libname = str(pathlib.Path().absolute())
    print("libname: ", libname)

    # Load the shared library into c types.
    if sys.platform.startswith("win"):
        DevCtrl = ctypes.WinDLL(libname+"\DevCtrl.dll")
    else:
        print('Program terminated!')

    # You need tell ctypes that the function returns a float

    DevCtrl.GetSupportedDevicesCount.restype = ctypes.c_int
    answer = DevCtrl.GetSupportedDevicesCount()
    print(f"Answer = {answer}")

    DevCtrl.InitInstrument.restype = ctypes.c_bool
    print(f'Init = {DevCtrl.InitInstrument()}')

    DevCtrl.InitDeviceEx2.argtypes = [wintypes.HWND, ctypes.c_byte, ctypes.c_char_p, ctypes.c_bool]
    DevCtrl.InitDeviceEx2.restype = ctypes.c_bool
    path_to_cfg = b'C:\Users\antonpopov\python\solar\MS5204_15098.cfg\n'
    initDev = DevCtrl.InitDeviceEx2(459460, 0, path_to_cfg, True)
    print(f'Init = {initDev}')

    DevCtrl.CheckConnection.restype = ctypes.c_bool
    print(f'Init = {DevCtrl.CheckConnection()}')

   
