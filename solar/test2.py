
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

if __name__ == '__main__':

    handle = Handle()
    handle.get_handle()
    print(handle.hwnd)