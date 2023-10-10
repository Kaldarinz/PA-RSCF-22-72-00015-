"""
Latest developing version
"""

from __future__ import annotations

import os, os.path
import sys
import logging, logging.config
from datetime import datetime
from typing import Optional

from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
from PyQt5 import QtCore
import pint
import yaml
from pylablib.devices.Thorlabs import KinesisMotor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QWidget, QPlainTextEdit
)
from PyQt5.uic import loadUi

from modules import ureg, Q_
import modules.validators as vd
from modules.pa_data import PaData
import modules.pa_logic as pa_logic
from modules.exceptions import StageError, HardwareError
import modules.data_classes as dc
from modules.utils import confirm_action
from gui.PA_main_window_ui import Ui_MainWindow

def init_logs(window: Window) -> logging.Logger:
    """Initiate logging"""

    with open('rsc/log_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        log_config = config['official']
        max_files = int(config['additional']['maxfiles'])
    filesnames = [f for f in os.listdir('./logs') if f.endswith('log')]
    filesnames.sort(key = lambda x: os.path.getctime('./logs/' + x))
    while len(filesnames) > max_files:
        old_file = filesnames.pop(0)
        os.remove('./logs/' + old_file)

    # adding current data to name of log files
    for i in (log_config["handlers"].keys()):
        # Check if filename in handler.
        # Console handlers might be present in config file
        if 'filename' in log_config['handlers'][i]:
            log_filename = log_config["handlers"][i]["filename"]
            base, extension = os.path.splitext(log_filename)
            today = datetime.today().strftime("_%Y_%m_%d-%H-%M-%S")
            log_filename = f"{base}{today}{extension}"
            log_config["handlers"][i]["filename"] = log_filename

    #adding textbox handler. I cannot handle it with file config.
    logTextBox = window.LogTextEdit
    logTextBox.setFormatter()
    logging.config.dictConfig(log_config)
    logTextBox.setFormatter(
        logging.Formatter(
            '%(levelname)s - %(message)s'))
    logging.getLogger('root').addHandler(logTextBox)
    logging.getLogger('root').setLevel(logging.INFO)
    logging.getLogger('modules').addHandler(logTextBox)
    logging.getLogger('modules').setLevel(logging.INFO)

    return logging.getLogger()

def init_hardware() -> None:
    """CLI for hardware init"""

    try:
        pa_logic.init_hardware()
    except HardwareError:
        logger.error('Hardware initialization failed')

class Window(QMainWindow, Ui_MainWindow):
    """Application MainWindow."""

    def __init__(self, parent:QMainWindow = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()
        self.LogTextEdit = QTextEditLogger(self.LogTextEdit)

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        self.action_Init.triggered.connect(init_hardware)

class QTextEditLogger(logging.Handler, QPlainTextEdit):
    appendLogText = QtCore.pyqtSignal(str)

    def __init__(self, parent: QPlainTextEdit):
        super().__init__()
        QPlainTextEdit.__init__(self)
        self.widget = parent
        self.widget.setReadOnly(True)
        self.appendLogText.connect(self.widget.appendPlainText)

    def emit(self, record):
        msg = self.format(record)
        self.appendLogText.emit(msg)

def dumb_handler(logger:QTextEditLogger) ->QTextEditLogger:
    return(logger)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    win = Window()
    logger = init_logs(win)
    logger.info('Starting application')

    ureg = pint.UnitRegistry(auto_reduce_dimensions=True) # type: ignore
    ureg.default_format = '~P'
    logger.debug('Pint activated (measured values with units)')
    ureg.setup_matplotlib(True)
    logger.debug('MatplotLib handlers for pint activated')
    pa_logic.load_config()
    logger.debug('Initializing PaData class for storing data')
    data = PaData()
    win.show()
    sys.exit(app.exec())