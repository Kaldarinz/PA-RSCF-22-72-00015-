"""
Latest developing GUI version

Main module.

Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from __future__ import annotations

import os, os.path
import sys
import time
import logging, logging.config
from datetime import datetime
from typing import Callable, cast, Literal

import yaml
from pint.facets.plain.quantity import PlainQuantity

import PySide6
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from PySide6.QtCore import (
    Qt,
    QThread,
    QThreadPool,
    Slot,
    Signal,
    QTimer
)
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QApplication,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout
)
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
    QPixmap
)
import numpy as np
from modules import ureg, Q_
from modules.hardware import utils as hutils
from modules.pa_data import (
    PaData
)

import modules.pa_logic as pa_logic
from modules.data_classes import (
    Worker,
    EnergyMeasurement,
    MapData,
    StagesStatus,
    Position
)
from modules.constants import (
    MSMNT_MODES,
    SCAN_MODES
)
from modules.gui.designer.PA_main_window_ui import Ui_MainWindow
from modules.gui.designer.designer_widgets import (
    DataViewer,
    LoggerWidget,
    MeasureWidget,
    PointMeasureWidget,
    CurveMeasureWidget,
    MapMeasureWidget,
    PowerMeterMonitor,
    MotorView
)

def init_logs() -> tuple[logging.Logger, QLogHandler]:
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

    #adding wdiget handler. I cannot handle it with file config.
    q_log_handler = QLogHandler()
    q_log_handler.thread.start()
    logging.config.dictConfig(log_config)
    logging.getLogger('root').addHandler(q_log_handler)
    logging.getLogger('modules').addHandler(q_log_handler)

    return logging.getLogger('pa_cli'), q_log_handler

class Window(QMainWindow,Ui_MainWindow,):
    """Application MainWindow."""

    energy_measured = Signal(EnergyMeasurement)
    "Emitted when PM data is read."
    motors_status_updated = Signal(StagesStatus)
    "Emitted, when new stages status is read."
    position_updated = Signal(Position)
    "Emitted, when new current position is measured"

    def __init__(
            self,
            parent: QMainWindow | None=None,
        ) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
        #Long-running threads
        self.q_logger = q_log_handler

        #Threadpool
        self.pool = QThreadPool()

        # Workers
        self.astep_worker: Worker | None = None

        # Timers
        self.timer_motor = QTimer()
        self.timer_motor.setInterval(100)
        self.timer_pm = QTimer()
        self.timer_pm.setInterval(250)

        # Icons
        self.pixmap_c = QPixmap(
            u":/icons/qt_resources/plug-connect.png"
        )
        "Connected state icon."
        self.pixmap_dc = QPixmap(
            u":/icons/qt_resources/plug-disconnect-prohibition.png"
        )
        "Disconnected state icon."

        # Hardware connection state
        self.init_state: bool = False
        "Hardware connection flag."

        self.set_widgets()
        self.connectSignalsSlots()
        self.set_shortcuts()

    def set_widgets(self):
        """Add and configure custom widgets."""

        ###############################################################
        ### Data viewer ###
        ###############################################################
        self.data_viewer = DataViewer()
        self.sw_central.addWidget(self.data_viewer)

        ###############################################################
        ### Log viewer ###
        ###############################################################
        self.d_log = LoggerWidget()
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.d_log)

        ###############################################################
        ### Motors control ###
        ###############################################################
        self.d_motors = MotorView(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.d_motors
        )
        self.d_motors.hide()

        ###############################################################
        ### 0D measurements ###
        ###############################################################
        self.p_point = PointMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_point)
        # Init power meter monitor
        self.init_pm_monitor(self.p_point.pm_monitor)

        ###############################################################
        ### 1D measurements ###
        ###############################################################
        self.p_curve = CurveMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_curve)
        # Init power meter monitor
        self.init_pm_monitor(self.p_curve.pm_monitor)

        ###############################################################
        ### 2D measurements ###
        ###############################################################
        self.p_map = MapMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_map)

        ###############################################################
        ### General controls ###
        ###############################################################
        # Scan mode selector
        cb = QComboBox()
        cb.addItems(MSMNT_MODES)
        cb.setEditable(False)
        cb.setCurrentIndex(self.sw_central.currentIndex())
        self.tb_mode.addWidget(cb)
        self.cb_mode_select = cb
        self.upd_mode(self.cb_mode_select.currentText())

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        ###############################################################
        ### logs ###
        ###############################################################
        self.q_logger.thread.log.connect(
            self.d_log.te_log.appendPlainText
        )
        self.action_Logs.toggled.connect(self.d_log.setVisible)
        self.d_log.visibilityChanged.connect(self.action_Logs.setChecked)

        ###############################################################
        ### actions ###
        ###############################################################
        # Hardware initialization
        self.action_Init.triggered.connect(self.init_hardware)
        # Date change event
        self.data_viewer.data_changed.connect(self.set_save_btns)
        # Data view
        self.action_Data.toggled.connect(self.activate_data_viwer)
        self.action_Open.triggered.connect(self.open_file)
        # Close file
        self.action_Close.triggered.connect(self.close_file)
        # Save file
        self.action_Save.triggered.connect(
            lambda x: self.save_file(False)
        )
        # Save as file
        self.actionSave_As.triggered.connect(
            lambda x: self.save_file(True)
        )
        # Export data
        self.actionExport_Data.triggered.connect(self.export_data)
        # Mode selection
        self.cb_mode_select.currentTextChanged.connect(self.upd_mode)
        self.action_measure_PA.toggled.connect(
            self.activate_measure_widget
        )
        # About
        self.actionAbout.triggered.connect(self.about)

        ###############################################################
        ### 0D measurements ###
        ###############################################################
        # Measure point
        self.p_point.measure_point.connect(
            lambda wl: self.measure_point(self.p_point, wl)
        )
        # Point measured
        self.p_point.point_measured.connect(
            self.data_viewer.add_point_to_data
        )

        ###############################################################
        ### 1D measurements ###
        ###############################################################
        # Measure point
        self.p_curve.measure_point.connect(
            lambda wl: self.measure_point(self.p_curve, wl))
        # Curve finished
        self.p_curve.curve_obtained.connect(
            self.data_viewer.add_curve_to_data
        )

        ###############################################################
        ### 2D measurements ###
        ###############################################################
        # Scan started
        self.p_map.scan_started.connect(self.measure_map)
        # Scan finished
        self.p_map.scan_obtained.connect(
            self.data_viewer.add_map_to_data
        )
        # Calculate auto step
        self.p_map.calc_astepClicked.connect(self.calc_astep)

        # Set scan speed
        self.p_map.sb_speed.quantSet.connect(
            self.set_scan_speed
        )
        # Update current position
        self.position_updated.connect(self.p_map.set_cur_pos)
        # Move to click
        self.p_map.plot_scan.pos_selected.connect(
            lambda pos: self.start_worker(
                pa_logic.move_to,
                new_pos = pos)
        )

        ###############################################################
        ### Motor dock widget ###
        ###############################################################
        # Show/hide widget
        self.action_position.toggled.connect(self.d_motors.setVisible)
        self.d_motors.visibilityChanged.connect(
            self.action_position.setChecked
        )
        # Buttons
        # Home
        self.d_motors.btn_home.clicked.connect(
            lambda: self.start_worker(pa_logic.home)
        )
        # Move to
        self.d_motors.pos_selected.connect(
            lambda pos: self.start_worker(
                pa_logic.move_to,
                new_pos = pos)
        )
        # Break al stages
        self.d_motors.btn_stop.clicked.connect(
            lambda: self.start_worker(pa_logic.break_all_stages)
        )
        # Jog
        # X direction
        self._conn_jog_btn(
            self.d_motors.btn_x_left_move, 'x', '-'
        )
        self._conn_jog_btn(
            self.d_motors.btn_x_right_move, 'x', '+'
        )
        # Y direction
        self._conn_jog_btn(
            self.d_motors.btn_y_left_move, 'y', '-'
        )
        self._conn_jog_btn(
            self.d_motors.btn_y_right_move, 'y', '+'
        )
        # Z direction
        self._conn_jog_btn(
            self.d_motors.btn_z_left_move, 'z', '-'
        )
        self._conn_jog_btn(
            self.d_motors.btn_z_right_move, 'z', '+'
        )
        ### Update status and position
        self.position_updated.connect(self.d_motors.set_position)
        self.motors_status_updated.connect(self.d_motors.upd_status)

    def set_shortcuts(self) -> None:

        # Delete measurements
        del_sc = QShortcut(
            QKeySequence.StandardKey.Delete,
            self.data_viewer.lv_content,
            context = Qt.ShortcutContext.WidgetShortcut
        )
        del_sc.activated.connect(self.data_viewer.del_msmsnt)

    @Slot(bool)
    def set_save_btns(self, data_exist: bool) -> None:
        """Set state of `Save` and `Save as` buttons."""

        self.actionSave_As.setEnabled(data_exist)
        # Set enable for save only if data was loaded from file
        data_title = self.data_viewer.lbl_data_title.text()
        data_title = data_title.split('*')[0]
        if data_title.endswith('hdf5') and data_exist:
            self.action_Save.setEnabled(data_exist)

    @Slot(PlainQuantity)
    def set_scan_speed(self, speed: PlainQuantity) -> None:
        """
        Set scan stage.
        
        Implement emulation if hardware is disconnected.
        """

        # Emulation
        if not self.init_state:
            self.p_map.sb_cur_speed.quantity = speed
            return
        # Real request otherwise
        set_vel_worker = Worker(
            pa_logic.set_stages_speed,
            speed = speed
        )
        set_vel_worker.signals.result.connect(
            self.p_map.sb_cur_speed.set_quantity
        )
        self.pool.start(set_vel_worker)

    @Slot(QProgressDialog)
    def calc_astep(self, pb: QProgressDialog) -> None:
        """
        Calculate auto step.
        
        This slot handles communication with hardware for autostep
        calculation.\n
        Implement emulation if hardware is disconnected.
        """
        if self.astep_worker is not None:
            self.astep_worker.signals.disconnect_all()
            self.astep_worker = None

        # Start emulation if hardware is not init
        if not self.init_state:
            self.astep_worker = Worker(
                func=pa_logic.en_meas_fast_cont_emul,
                max_count = pb.maximum()
            )
        else:
            self.astep_worker = Worker(
                func = pa_logic.meas_cont,
                data = SCAN_MODES[self.p_map.cb_mode.currentText()],
                max_count = pb.maximum()
            )
        # Progress signal increase progress bar
        self.astep_worker.signals.progess.connect(
            lambda x: pb.setValue(pb.value() + 1)
        )
        # Set results
        self.astep_worker.signals.result.connect(
            self.p_map.set_astep
        )
        # Resume normal osc operation after procedure finished
        if self.init_state:
            self.astep_worker.signals.finished.connect(
                self.start_worker(pa_logic.osc_run_normal)
            )
        # if progress bar was cancelled or finished, stop measurements
        pb.canceled.connect(
            lambda: self.stop_worker(self.astep_worker)
        )
        self.pool.start(self.astep_worker)

    def _conn_jog_btn(
            self,
            btn: QPushButton,
            axes: Literal['x', 'y', 'z'],
            direction: Literal['-', '+']
    ) -> None:
        """Connect signals for jog button."""

        btn.pressed.connect(
            lambda: self.start_worker(
                pa_logic.stage_jog,
                axes = axes,
                direction = direction
                )
        )
        btn.released.connect(
            lambda: self.start_worker(pa_logic.stage_stop, axes = axes)
        )

    @Slot()
    def open_file(self) -> None:
        """Load data file."""

        logger.debug('Start loading data...')
        filename = self.get_filename()
        if filename:
            self.close_file()
            data = PaData()
            data.load(filename)
            logger.info(f'Data with {data.attrs.measurements_count} '
                        + 'PA measurements loaded!')
            basename = os.path.basename(filename)
            self.data_viewer.lbl_data_title.setText(basename)
            self.data_viewer.data = data

    def get_filename(self) -> str:
        """Launch a dialog to open a file."""

        path = os.path.dirname(__file__)
        initial_dir = os.path.join(path, 'measuring results')
        filename = QFileDialog.getOpenFileName(
            self,
            caption='Choose a file',
            dir=initial_dir,
            filter='Hierarchical Data Format (*.hdf5)'
        )[0]
        return filename

    @Slot(bool)
    def save_file(self, new_file: bool) -> None:
        """Save data to file."""

        path = os.path.dirname(__file__)
        initial_dir = os.path.join(path, 'measuring results')
        if not new_file:
            file_title = self.data_viewer.lbl_data_title.text()
            file_title = file_title.split('*')[0]
            filename = os.path.join(initial_dir, file_title)
        else:
            filename = QFileDialog.getSaveFileName(
                self,
                caption='Set filename',
                dir=initial_dir,
                filter='Hierarchical Data Format (*.hdf5)'
            )[0]
        if self.data_viewer.data is not None:
            self.data_viewer.data.save(filename)
            # remove marker of changed data from data title
            basename = os.path.basename(filename)
            self.data_viewer.lbl_data_title.setText(basename)

    @Slot()
    def export_data(self) -> None:
        """Export data to txt."""

        self.export_diag = QWidget()
        self.export_diag.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.export_diag.setWindowTitle('Choose data to export')
        layout = QVBoxLayout()
        self.export_diag.setLayout(layout)
        p = Parameter.create(
            name = 'Choose data to export',
            type = 'group',
            children = self.data_viewer.data.get_content() # type: ignore
        )
        config = Parameter.create(
            name = 'Config',
            type = 'group',
            children = [
                {'name': 'delimeter', 'type': 'str', 'value': ';'},
                {'name': 'format', 'type': 'str', 'value': r'%10.6e'}
            ]
        )
        p.insertChild(0, config)
        # Collapse all
        def activate(param):
            for ch in param:
                if isinstance(ch, GroupParameter):
                    ch.setOpts(expanded=False)
                    activate(ch)
        activate(p)

        # Save btn
        def _export(_):
            path = os.path.dirname(__file__)
            initial_dir = os.path.join(path, 'measuring results')
            filename = QFileDialog.getSaveFileName(
                self,
                caption='Set filename',
                dir=initial_dir,
                filter='Text file (*.txt)'
            )[0]
            self.data_viewer.data.export_data(filename, p.getValues()) # type: ignore
        btn = Parameter.create(
            name = 'Export Data',
            type = 'action'
        )
        btn.sigActivated.connect(_export)

        p.insertChild(0, btn)
        t = ParameterTree()
        t.setParameters(p, showTop = False)
        layout.addWidget(t)
        self.export_diag.show()

    @Slot()
    def close_file(self) -> None:
        """Close current file."""

        if self.data_viewer.data is not None and self.data_viewer.data.changed:
            if self.confirm_action('Data was changed.'
                                + ' Do you really want to close the file?'):
                self._file_close_proc()
        else:
            self._file_close_proc()

    def _file_close_proc(self) -> None:
        """Actual file close."""

        self.data_viewer.data = None

    def activate_data_viwer(self, activated: bool) -> None:
        """Toogle data viewer."""

        if activated:
            self.sw_central.setCurrentWidget(self.data_viewer)
            self.action_measure_PA.setChecked(False)
        else:
            self.upd_mode(self.cb_mode_select.currentText())

    @Slot()
    def activate_measure_widget(self, activated: bool) -> None:
        """Toggle measre widget."""

        if activated:
            self.upd_mode(self.cb_mode_select.currentText())
        else:
            self.action_Data.setChecked(True)

    @Slot(MapData)
    def measure_map(self, scan: MapData) -> None:
        """
        Start 2D scanning.
        
        Implement emulation if hardware is disconnected.
        """

        if self.init_state:
            scan_worker = Worker(pa_logic.scan_2d, scan = scan)
        # Emulate scan if hardware is disconnected
        else:
            scan_worker = Worker(
                pa_logic.scan_2d_emul,
                scan = scan,
                speed = self.p_map.sb_cur_speed.quantity
        )
        self.p_map.scan_worker = scan_worker
        scan_worker.signals.progess.connect(self.p_map.plot_scan.upd_scan)
        scan_worker.signals.finished.connect(self.p_map.scan_finished)
        self.pool.start(scan_worker)

    @Slot(PlainQuantity)
    def measure_point(
            self,
            caller: MeasureWidget,
            wl: PlainQuantity) -> None:
        """
        Measure single PA point.
        
        Implement emulation if hardware is disconnected.
        """
        
        if self.init_state:
            worker = Worker(pa_logic.measure_point, wl)
        else:
            worker = Worker(pa_logic.measure_point_emul, wl)
        worker.signals.result.connect(
            caller.verify_msmnt
        )
        self.pool.start(worker)

    def upd_mode(self, value: str) -> None:
        """Change sw_main for current measuring mode."""

        if value == 'Single point':
            self.sw_central.setCurrentWidget(self.p_point)
        elif value == 'Curve':
            self.sw_central.setCurrentWidget(self.p_curve)
        elif value == 'Map':
            self.sw_central.setCurrentWidget(self.p_map)
        else:
            logger.warning(f'Unsupported mode selected: {value}')

        self.action_measure_PA.setChecked(True)
        self.action_Data.setChecked(False)

    def closeEvent(self, event) -> None:
        """Exis routines."""

        # Check if some data is not saved
        if self.data_viewer.lbl_data_title.text().endswith('*'):
            if not self.confirm_action(
                title = 'Data not saved',
                message = ('You have not saved data. '
                           + 'Do you really want to exit?')
            ):
                event.ignore()
                return
        elif not self.confirm_action(
            title = 'Confirm Close',
            message = 'Do you really want to exit?'
        ):
            event.ignore()
            return

        logger.info('Stopping application')
        # Close hardware actors
        pa_logic.close_hardware()
        self.pool.clear()

    def init_hardware(self) -> None:
        """
        Hardware initiation.
        
        Additionally launch energy measurement worker
        if oscilloscope was initiated.
        """
        worker = Worker(pa_logic.init_hardware)
        self.pool.start(worker)
        worker.signals.result.connect(self.post_init)

    def post_init(self, state: bool) -> None:
        """Routines to be done after initialization of hardware."""

        # Update hardware connection flag
        self.init_state = state
        if not state:
            self.action_Init.setIcon(self.pixmap_dc)
            return
        self.action_Init.setIcon(self.pixmap_c)
        # Start energy measurement cycle.
        if pa_logic.osc_open():
            if not self.timer_pm.isActive():
                self.timer_pm.timeout.connect(self.upd_pm)
                self.timer_pm.start()
                logger.info('Energy measurement cycle started!')
        else:
            logger.info('Failed to start energy mesurement cycle!')

        # Start motor position update timer
        if pa_logic.stages_open():
            if not self.timer_motor.isActive():
                self.timer_motor.timeout.connect(self.upd_motor)
                self.timer_motor.start()
            logger.info('Motors position and status cycle started!')
            # update motor speed
            self.set_scan_speed(self.p_map.sb_speed.quantity)
        else:
            logger.info('Failed to start motors measureemnts cycle!')

    def upd_pm(self) -> None:
        """Update power meter information."""

        pm_worker = Worker(pa_logic.en_meas_fast)
        pm_worker.signals.result.connect(
            self.energy_measured.emit
        )
        self.pool.start(pm_worker)

    def upd_motor(self) -> None:
        """Update information on motors widget."""

        # Update position information
        self.start_cb_worker(
            self.position_updated.emit,
            pa_logic.stages_position
        )

        # Update status information
        self.start_cb_worker(
            self.motors_status_updated.emit,
            pa_logic.stages_status
        )

    def init_pm_monitor(
            self,
            monitor: PowerMeterMonitor
        ) -> None:
        """Initialize power meter monitor."""

        self.energy_measured.connect(monitor.add_msmnt)

    def start_cb_worker(
            self,
            callback: Callable,
            func: Callable,
            *args,
            **kwargs
    ) -> None:
        """
        Execute `func` in a separate thread, return value to `callback`.

        `callback` function is called when `func` emit `results` signal.
        Value returned by `results` is sent as `callback` arguments.
        """

        worker = Worker(func, *args, **kwargs)
        worker.signals.result.connect(callback)
        self.pool.start(worker)

    def start_worker(
            self,
            func: Callable,
            *args,
            **kwargs
    ) -> None:
        """
        Execute ``func`` in a separate thread.

        This usually used for sending simple hardware commands.
        """

        worker = Worker(func, *args, **kwargs)
        self.pool.start(worker)

    def stop_worker(self, worker: Worker | None) -> None:
        """Stop and delete worker."""


        if worker is not None:
            worker.kwargs['flags']['is_running'] = False
            # delete worker
            for key, value in self.__dict__.items():
                if value is worker:
                    self.__dict__[key] = None

    def pause_worker(
            self,
            is_paused: bool,
            worker: Worker | None
        ) -> None:
        """Pause or resume worker."""

        if worker is not None:
            worker.kwargs['flags']['pause'] = is_paused

    def confirm_action(self,
            title: str='Confirm',
            message: str='Are you sure?'
        ) -> bool:
        """Dialog to confirm an action."""

        answer = QMessageBox.question(
            self,
            title,
            message
        )
        if answer == QMessageBox.StandardButton.Yes:
            result = True
        else:
            result = False
        logger.debug(f'{message} = {result}')
        return result

    def about(self) -> None:
        """About popup window."""

        info = QMessageBox()
        info.setWindowTitle('About programm')
        info.setText(
            """
            Programm for photoacoustic measurements using experimental
            setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

            Author: Anton Popov
            contact: a.popov.fizte@gmail.com
            
            Created with financial support from Russian Scince Foundation.
            Grant # 22-72-00015

            2024
            """
        )
        info.exec()

class LoggerThread(QThread):
    """Thread for logging."""

    log = Signal(str)
    data = []

    @Slot()
    def run(self) -> None:
        self.waiting_for_data = True
        while True:
            while self.waiting_for_data:
                time.sleep(0.01)
            while len(self.data):
                self.log.emit(self.data.pop(0))
            self.waiting_for_data = True

    def get_data(self, log_entry: str) -> None:
        """Store a log entry for displaying."""
        self.data.append(log_entry)
        self.waiting_for_data = False

class QLogHandler(logging.Handler):
    """Log handler for GUI."""

    def __init__(self):
        super().__init__()
        self.thread = LoggerThread()
        self.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                '%I:%M:%S'
            )
        )
        self.setLevel(logging.INFO)

    def emit(self, record):
        msg = self.format(record)
        self.thread.get_data(msg)

if __name__ == '__main__':

    pg.setConfigOptions(antialias=True, background='w', foreground='k')
    app = QApplication(sys.argv)
    logger, q_log_handler = init_logs()
    win = Window()
    
    logger.info('Starting application')
    pa_logic.load_config()
    win.showMaximized()
    sys.exit(app.exec())