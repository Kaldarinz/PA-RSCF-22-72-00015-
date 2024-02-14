"""
Latest developing GUI version
"""

from __future__ import annotations

import os, os.path
import sys
import time
import logging, logging.config
from datetime import datetime
from typing import Iterable, Callable, cast, Literal
from dataclasses import field, fields
from functools import partial

import pint
import yaml
from pint.facets.plain.quantity import PlainQuantity
from matplotlib.backend_bases import PickEvent
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import PySide6
import pyqtgraph as pg
from PySide6.QtCore import (
    Qt,
    QObject,
    QThread,
    QThreadPool,
    Slot,
    Signal,
    QModelIndex,
    QTimer
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMessageBox,
    QLineEdit,
    QPushButton,
    QLayout,
    QComboBox,
    QTreeWidgetItem,
    QFileDialog,
    QMessageBox,
    QProgressDialog
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
    PaData,
    MeasuredPoint
)

import modules.pa_logic as pa_logic
from modules.utils import (
    upd_plot,
    form_quant
)   
from modules.data_classes import (
    Worker,
    Measurement,
    EnergyMeasurement,
    Position,
    MapData
)
from modules.constants import (
    POINT_SIGNALS,
    MSMNTS_SIGNALS
)
from modules.gui.widgets import (
    MplCanvas,
    MplNavCanvas,
    QuantSpinBox,
)
from modules.gui.designer.PA_main_window_ui import Ui_MainWindow
from modules.gui.designer.designer_widgets import (
    PAVerifierDialog,
    DataViewer,
    LoggerWidget,
    PointMeasureWidget,
    CurveMeasureWidget,
    MapMeasureWidget,
    PowerMeterMonitor,
    CurveView,
    PointView,
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
    "Signal, emitted when PM data is obtained."

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

        # Timers
        self.timer_motor = QTimer()
        self.timer_motor.setInterval(100)
        self.timer_pm = QTimer()
        self.timer_pm.setInterval(200)

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

        # Initiate widgets
        self.set_widgets()

        #Connect custom signals and slots
        self.connectSignalsSlots()

    def set_widgets(self):
        """Add and configure custom widgets."""

        #Measurement verifier dialog
        self.pa_verifier = PAVerifierDialog()

        ### Data viewer ###
        self.data_viewer = DataViewer()
        self.sw_central.addWidget(self.data_viewer)

        ### Log viewer ###
        self.d_log = LoggerWidget()
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.d_log)

        ### Motors control ###
        self.d_motors = MotorView(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.d_motors
        )
        self.d_motors.set_range(
            (Q_(0, 'mm'), Q_(25, 'mm'))
        )
        self.d_motors.hide()

        ### 0D measurements ###
        self.p_point = PointMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_point)
        # Initial value of SetPoint
        self.upd_sp(
            self.p_point,
            self.p_point.sb_sample_sp
        )
        # Init power meter monitor
        self.init_pm_monitor(self.p_point.pm_monitor)

        ### 1D measurements ###
        self.p_curve = CurveMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_curve)
        # Initial value of SetPoint
        self.upd_sp(
            self.p_curve,
            self.p_curve.sb_sample_sp
        )
        # Init power meter monitor
        self.init_pm_monitor(self.p_curve.pm_monitor)

        ### 2D measurements ###
        self.p_map = MapMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_map)

        # Scan mode selector
        cb = QComboBox()
        cb.addItems(
            ('Single point', 'Curve', 'Map')
        )
        cb.setEditable(False)
        cb.setCurrentIndex(self.sw_central.currentIndex())
        self.upd_mode(cb.currentText())
        self.tb_mode.addWidget(cb)
        self.cb_mode_select = cb
        self.upd_mode(self.cb_mode_select.currentText())

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        ### logs ###
        self.q_logger.thread.log.connect(self.d_log.te_log.appendPlainText)

        ### Various actions ###
        # Hardware initialization
        self.action_Init.triggered.connect(self.init_hardware)
        # Date change event
        self.data_viewer.data_changed.connect(self.set_save_btns)
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

        ### Shortcuts ###
        # Delete measurements
        del_sc = QShortcut(
            QKeySequence.StandardKey.Delete,
            self.data_viewer.lv_content,
            context = Qt.ShortcutContext.WidgetShortcut
        )
        del_sc.activated.connect(self.del_msmsnt)

        ### Mode selection ###
        self.cb_mode_select.currentTextChanged.connect(self.upd_mode)
        self.action_measure_PA.toggled.connect(self.activate_measure_widget)

        ### 0D measurements ###
        # Measure button
        self.p_point.btn_measure.clicked.connect(
            lambda x: self.measure(self.p_point)
        )
        # Sample SetPoint changed 
        self.p_point.sb_sample_sp.valueChanged.connect(
            lambda x: self.upd_sp(
                self.p_point,
                self.p_point.sb_sample_sp
            )
        )
        # PM SetPoint changed
        self.p_point.sb_pm_sp.valueChanged.connect(
            lambda x: self.upd_sp(
                self.p_point,
                self.p_point.sb_pm_sp
            )
        )
        # Current PM energy value
        self.p_point.pm_monitor.le_cur_en.textChanged.connect(
            self.p_point.le_pm_en.setText
        )
        # Current Sample energy value
        self.p_point.pm_monitor.le_cur_en.textChanged.connect(
            lambda val: self.p_point.le_sample_en.setText(
                str(hutils.glan_calc(ureg(val)))
            )
        )

        ### 1D measurements ###
        # Button Set Parameter
        self.p_curve.btn_run.clicked.connect(self.run_curve)
        # Button Measure
        self.p_curve.btn_measure.clicked.connect(
            lambda x: self.measure(self.p_curve)
        )
        # Button restart
        self.p_curve.btn_restart.clicked.connect(
            lambda x: self.restart_measurement(self.p_curve)
        )
        # Button stop
        self.p_curve.btn_stop.clicked.connect(
            lambda x: self.stop_measurement(self.p_curve)
        )
        # Sample energy SetPoint 
        self.p_curve.sb_sample_sp.valueChanged.connect(
            lambda x: self.upd_sp(
                self.p_curve,
                self.p_curve.sb_sample_sp
            )
        )
        # Power meter energy SetPoint
        self.p_curve.sb_pm_sp.valueChanged.connect(
            lambda x: self.upd_sp(
                self.p_curve,
                self.p_curve.sb_pm_sp
            )
        )
        # Current PM energy value
        self.p_curve.pm_monitor.le_cur_en.textChanged.connect(
            self.p_curve.le_pm_en.setText
        )
        # Current Sample energy value
        self.p_curve.pm_monitor.le_cur_en.textChanged.connect(
            lambda val: self.p_curve.le_sample_en.setText(
                str(hutils.glan_calc(ureg(val)))
            )
        )

        ### 2D measurements ###
        # Scan
        self.p_map.scan_started.connect(self.measure_map)
        # Scan finished
        self.p_map.scan_obtained.connect(self.save_map)
        # Calculate auto step
        self.p_map.calc_astepClicked.connect(self.calc_astep)

        # Set scan speed
        self.p_map.sb_speed.quantSet.connect(
            self.set_scan_speed
        )

        ### Data view###
        self.action_Data.toggled.connect(self.activate_data_viwer)
        self.action_Open.triggered.connect(self.open_file)
        
        # Measurement title change
        self.data_viewer.content_model.dataChanged.connect(
            self._msmnt_changed
        )

        ### Logs ###
        self.action_Logs.toggled.connect(self.d_log.setVisible)
        self.d_log.visibilityChanged.connect(self.action_Logs.setChecked)

        ### Motor dock widget ###
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
        self.d_motors.btn_go.clicked.connect(
            lambda: self.start_worker(
                pa_logic.move_to,
                new_pos = self.d_motors.current_pos())
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

    @Slot(bool)
    def set_save_btns(self, data_exist: bool) -> None:
        """Set state of `Save` and `Save as` buttons."""

        self.actionSave_As.setEnabled(data_exist)
        # Set enable for save only if data was loaded from file
        data_title = self.data_viewer.lbl_data_title.text()
        data_title = data_title.split('*')[0]
        if data_title.endswith('hdf5') and data_exist:
            self.action_Save.setEnabled(data_exist)

    @Slot(MapData)
    def measure_map(self, scan: MapData) -> None:
        """
        Start 2D scanning.
        
        Implement emulation if hardware is disconnected.
        """

        logger.info('measure_map slot called.')
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

    @Slot(MapData)
    def save_map(self, scan: MapData) -> None:
        """Save scanned data."""

        # create new data structure if necessary
        if self.data is None:
            self.data = PaData()
        self.data.add_map(scan)
        logger.debug('Map data saved.')
        self.data_changed.emit()

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

    @Slot(QProgressDialog)
    def calc_astep(self, pb: QProgressDialog) -> None:
        """
        Calculate auto step.
        
        This slot handles communication with hardware for autostep
        calculation.\n
        Implement emulation if hardware is disconnected.
        """

        # Start emulation if hardware is not init
        if not self.init_state:
            self.astep_worker = Worker(
                func=pa_logic.en_meas_fast_cont_emul,
                max_count = pb.maximum()
            )
        else:
            self.astep_worker = Worker(
                func = pa_logic.meas_cont,
                data = 'en_fast',
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
            self.data = data

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
                caption='Choose a file',
                dir=initial_dir,
                filter='Hierarchical Data Format (*.hdf5)'
            )[0]
        if self.data is not None:
            if self.data.attrs.version < 1.2:
                self.data.attrs.version = PaData.VERSION
            self.data.save(filename)
            # remove marker of changed data from data title
            data_title = self.data_viewer.lbl_data_title.text()
            data_title = data_title.split('*')[0]
            self.data_viewer.lbl_data_title.setText(data_title)

    @Slot()
    def close_file(self) -> None:
        """Close current file."""

        if self.data is not None and self.data.changed:
            if self.confirm_action('Data was changed.'
                                + ' Do you really want to close the file?'):
                self._file_close_proc()
        else:
            self._file_close_proc()

    def _file_close_proc(self) -> None:
        """Actual file close."""

        self.data_viewer.data = None

    @Slot(QModelIndex, QModelIndex, list)
    def _msmnt_changed(
            self,
            top_left: QModelIndex,
            bot_right: QModelIndex,
            roles: list
        ) -> None:
        """Change measurement title."""

        # Get old measurement title
        title = ''
        for title, val in self.data.measurements.items(): # type: ignore
            if val == self.data_viewer.s_measurement:
                break
        if not title:
            logger.warning('Error in measurement title change.')
            return
        # If new title is invalid, set old title back
        if not top_left.data():
            self.data_viewer.content_model.setData(
                top_left,
                title,
                role = Qt.ItemDataRole.EditRole
            )
            return
        
        # Update data
        msmnts = self.data.measurements # type: ignore
        msmnts[top_left.data()] = msmnts.pop(title)

    @Slot()
    def del_msmsnt(self) -> None:
        """Delete currently selected measurement."""

        # Get currently selected index
        val = self.data_viewer.lv_content.selectedIndexes()[0]
        # Remove measurement from data
        self.data.measurements.pop(val.data()) # type: ignore
        # Trigger update of widgets
        self.data_changed.emit()
        logger.info(f'Measurement: {val.data()} removed.')

    def set_map_view(self) -> None:
        """Set central part of data_viewer for map view."""

        if (self.data is None
            or self.data_viewer.s_measurement is None):
            logger.warning(
                'Map view cannot be updated. Data is missing.'
            )
            return 
            
        view = self.data_viewer.p_2d
        self.data_viewer.sw_view.setCurrentWidget(view)
        view.plot_map.set_data(self.data.maps[self.data_viewer.s_msmnt_title])       

    def set_cb_detail_view(
            self,
            parent: PointView | CurveView
        ) -> None:
        """
        Set combobox with signals for detail view.
        
        Implies that combobox is ``cb_detail_select``.
        ``parent`` should have attribute with this name.
        """

        # Disconnect old slots
        try:
            parent.cb_detail_select.currentTextChanged.disconnect()
        except:
            logger.debug('No slots were connected to cb_detail_select.')

        # Clear old values
        parent.cb_detail_select.clear()

        # Set new values
        parent.cb_detail_select.addItems(
            [key for key in POINT_SIGNALS.keys()]
        )

        # If necessary, set dtype_point attribute
        if self.data_viewer.s_point_dtype:
            for key,val in POINT_SIGNALS.items():
                if val == self.data_viewer.s_point_dtype:
                    parent.cb_detail_select.setCurrentText(key)
                    break
        else:
            self.data_viewer.s_point_dtype = POINT_SIGNALS[
                parent.cb_detail_select.currentText()
            ]

        # Connect slot again
        parent.cb_detail_select.currentTextChanged.connect(
            self.upd_point_info
        )

    def set_data_description_view(self, data: PaData) -> None:
        """Set data description tv_info."""

        tv = self.data_viewer.tv_info
        if self.data_viewer.s_measurement is None:
            logger.error('Describption cannot be build. Measurement is not set.')
            return
        measurement = self.data_viewer.s_measurement

        # file attributes
        # clear if existed
        if getattr(tv, 'fmd', None) is not None:
            index = tv.indexOfTopLevelItem(tv.fmd)
            tv.takeTopLevelItem(index)
        tv.fmd = QTreeWidgetItem(tv, ['File attributes'])
        tv.fmd.setExpanded(True)
        for field in fields(data.attrs):
            QTreeWidgetItem(
                tv.fmd,
                [field.name, str(getattr(data.attrs, field.name))]
            )

        # measurement attributes
        # clear if existed
        if getattr(tv, 'mmd', None) is not None:
            index = tv.indexOfTopLevelItem(tv.mmd)
            tv.takeTopLevelItem(index)
        tv.mmd = QTreeWidgetItem(tv, ['Measurement attributes'])
        tv.mmd.setExpanded(True)
        msmnt_title = ''
        for key, val in data.measurements.items():
            if val == measurement:
                msmnt_title = key
                break
        QTreeWidgetItem(tv.mmd, ['Title', msmnt_title])
        for field in fields(measurement.attrs):
            QTreeWidgetItem(
                tv.mmd,
                [field.name, str(getattr(
                    measurement.attrs,
                    field.name
                    ))]
            )
        
        # point attributes
        self.upd_point_info()

    def upd_point_info(self, new_text: str | None=None) -> None:
        """Update plot and text information about current datapoint."""

        if new_text is not None:
            self.data_viewer.s_point_dtype = POINT_SIGNALS[new_text]

        if (self.data is None
            or self.data_viewer.s_measurement is None):
            logger.warning(
                'Point info cannot be updated. Some data is missing.'
            )
            return

        if self.data_viewer.s_measurement.attrs.measurement_dims == 1:
            # update plot
            active_dview = self.data_viewer.sw_view.currentWidget()
            active_dview = cast(CurveView, active_dview)
            upd_plot(
                active_dview.plot_detail, # type: ignore
                *self.data.point_data_plot(
                    msmnt = self.data_viewer.s_measurement,
                    index = self.data_viewer.s_point_ind,
                    dtype = self.data_viewer.s_point_dtype
                )
            )

        # clear existing description
        tv = self.data_viewer.tv_info 
        if tv.pmd is not None:
            index = tv.indexOfTopLevelItem(tv.pmd)
            tv.takeTopLevelItem(index)
        # set new description
        tv.pmd = QTreeWidgetItem(tv, ['Point attributes'])
        tv.pmd.setExpanded(True)
        dp_title = PaData._build_name(self.data_viewer.s_point_ind + 1)
        dp = self.data_viewer.s_measurement.data[dp_title] #type: ignore
        QTreeWidgetItem(tv.pmd, ['Title', dp_title])
        # PointMetaData
        for field in fields(dp.attrs):
            val = getattr(dp.attrs, field.name)
            if type(val) == list:
                val = [str(x) for x in val]
            QTreeWidgetItem(tv.pmd, [field.name, str(val)])
        dtype = getattr(dp, self.data_viewer.s_point_dtype)
        for field in fields(dtype):
            if field.name in ['data', 'data_raw']:
                continue
            QTreeWidgetItem(
                tv.pmd,
                [field.name, str(getattr(dtype, field.name))]
            )

    def pick_event(
            self,
            widget:MplCanvas,
            event: PickEvent,
            parent: CurveView | None=None,
            measurement: Measurement | None=None):
        """Callback method for processing data picking on plot."""

        if self.data_viewer.s_measurement is None:
            logger.warning(
                'pick event cannot be processed. Measurement is missing.'
            )
            return

        # Update datamarker on plot
        if widget._marker_ref is None:
            logger.warning('Marker is not set.')
            return
        widget._marker_ref.set_data(
            [widget.xdata[event.ind[0]]], # type: ignore
            [widget.ydata[event.ind[0]]] # type: ignore
        )
        widget.draw()
        # Update index of currently selected data
        self.data_viewer.s_point_ind = event.ind[0] # type: ignore
        dp_name = PaData._build_name(event.ind[0] + 1) # type: ignore
        self.data_viewer.s_datapoint = self.data_viewer.s_measurement.data[dp_name]
        # update point widgets
        self.upd_point_info()

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

    def _set_layout_enabled(
            self,
            layout: QLayout,
            enabled: bool
        ) -> None:
        """Set enabled for all QWidgets in a layout."""

        for i in range(layout.count()):
            child = layout.itemAt(i)
            if child.layout() is not None:
                self._set_layout_enabled(child.layout(), enabled)
            else:
                child.widget().setEnabled(enabled) # type: ignore

    def run_curve(self) -> None:
        """
        Start 1D measurement.
        
        Launch energy adjustment.\n
        Set ``p_curve.step``, ``p_curve.spectral_points`` attributes.
        Also set progress bar and realted widgets.
        """

        # Activate curve measure button
        self._set_layout_enabled(
            self.p_curve.lo_measure_ctrls,
            True
        )
        # But keep restart deactivated
        self.p_curve.btn_restart.setEnabled(False)

        # Deactivate paramter controls
        self._set_layout_enabled(
            self.p_curve.lo_parameter,
            False
        )

        page = self.p_curve
        
        # calculate amount of paramter values
        param_start = page.sb_from.quantity
        param_end = page.sb_to.quantity
        step = page.sb_step.quantity
        delta = abs(param_end-param_start)
        if delta%step:
            max_steps = int(delta/step) + 2
        else:
            max_steps = int(delta/step) + 1
        page.max_steps = max_steps

        # set paramter name attribute
        page.parameter = [page.cb_mode.currentText()]

        # create array with all parameter values
        # and set it as attribute along with step
        spectral_points = []
        if param_start < param_end:
            page.step = step
            for i in range(max_steps-1):
                spectral_points.append(param_start+i*step)
            spectral_points.append(param_end)
        else:
            page.step = -step
            for i in range(max_steps-1):
                spectral_points.append(param_end+i*step)
            spectral_points.append(param_start)
        page.param_points = Q_.from_list(spectral_points)
        page.current_point = 0

    def measure(
            self,
            mode_widget: CurveMeasureWidget | PointMeasureWidget
        ) -> None:
        """Measure a point for 1D PA measurement."""

        # block measure button
        mode_widget.btn_measure.setEnabled(False)

        # Launch measurement
        wl = mode_widget.sb_cur_param.quantity
        worker = Worker(pa_logic.measure_point, wl)
        worker.signals.result.connect(
            lambda data: self.verify_measurement(
                mode_widget,
                data
            )
        )
        self.pool.start(worker)

    def verify_measurement(
            self,
            parent: CurveMeasureWidget | PointMeasureWidget,
            data: MeasuredPoint | None
        ) -> None:
        """
        Verify a measurement.
        
        Attributes
        ----------
        ``parent`` is widget, which stores current point index.
        """

        # Check if data was obtained
        if data is None:
            info_dial = QMessageBox()
            info_dial.setWindowTitle('Bad data')
            info_dial.setText('Error during data reading!')
            info_dial.exec()
            parent.btn_measure.setEnabled(True)
            return None

        plot_pa = self.pa_verifier.plot_pa
        plot_pm = self.pa_verifier.plot_pm

        # Power meter plot
        ydata=data.pm_signal
        xdata=Q_(
            np.arange(len(data.pm_signal))*data.dt_pm.m, # type: ignore
            data.dt_pm.u
        )
        upd_plot(
            base_widget = plot_pm,
            ydata = ydata,
            xdata = xdata
        )

        # PA signal plot
        start = data.start_time
        stop = data.stop_time.to(start.u)
        upd_plot(
            base_widget = plot_pa,
            ydata = data.pa_signal,
            xdata = Q_(
                np.linspace(
                    start.m,
                    stop.m,
                    len(data.pa_signal), # type: ignore
                    endpoint=True), # type: ignore
                start.u
            ) # type: ignore
        )

        # pa_verifier.exec return 1 if data is ok and 0 otherwise
        if self.pa_verifier.exec():
            # create new data structure if necessary
            if self.data is None:
                self.data = PaData()
            # Create new measurement if it is the first measured point
            # in curve measurement
            if isinstance(parent, CurveMeasureWidget):
                if parent.current_point == 0:
                    msmnt_title, _ = self.data.add_measurement(
                        dims = 1,
                        params = parent.parameter
                    ) # type: ignore
                    # Activate restart button
                    parent.btn_restart.setEnabled(True)
                # otherwise take title of the last measurement
                else:
                    msmnt_title = PaData._build_name(
                        self.data.attrs.measurements_count,
                        'measurement'
                )
            # for point data new measurement is created each time
            else:
                msmnt_title, _ = self.data.add_measurement(
                    dims = 0,
                    params = []
                ) # type: ignore
            # add datapoint to the measurement
            msmnt = self.data.measurements[msmnt_title]
            cur_param = []
            if isinstance(parent, CurveMeasureWidget):
                cur_param.append(parent.sb_cur_param.quantity)
            self.data.add_point(
                measurement = msmnt,
                data = data,
                param_val = cur_param.copy()
            )
            
            if isinstance(parent, CurveMeasureWidget):
                # update current point attribute, which will trigger 
                # update of realted widgets on parent
                parent.current_point += 1
                # Enable measure button if not last point
                if parent.current_point < parent.max_steps:
                    parent.btn_measure.setEnabled(True)
                # Enable parameter control if it was the last point
                else:
                    self._set_layout_enabled(
                        parent.lo_parameter,
                        True
                    )
                # update measurement plot
                upd_plot(
                    parent.plot_measurement,
                    *self.data.param_data_plot(msmnt = msmnt)
                )
            else:
                # Enable measure button for Point measuremnt
                parent.btn_measure.setEnabled(True) 
            # Update data_viewer
            self.data_changed.emit()
        # Enable measure button if measured point is bad
        else:
            parent.btn_measure.setEnabled(True)

    def restart_measurement(
            self,
            msmnt_page: CurveMeasureWidget
        ) -> None:
        """Restart a measurement."""

        msmnt_page.current_point = 0

    def stop_measurement(
            self,
            msmnt_page: CurveMeasureWidget
        ) -> None:
        """
        Stop a measurement.
        
        Stop the measurement, by activating paramter controls
        and deactivating measurement controls.
        """

        msmnt_page.current_point = 0
        self._set_layout_enabled(
            msmnt_page.lo_parameter,
            True
        )
        self._set_layout_enabled(
            msmnt_page.lo_measure_ctrls,
            False
        )

    def upd_sp(
            self,
            parent: PointMeasureWidget | CurveMeasureWidget,
            caller: QuantSpinBox) -> None:
        """Update Set Point data."""

        # Update PM SetPoint if Sample SetPoint was set
        if caller == parent.sb_sample_sp:
            new_sp = hutils.glan_calc_reverse(caller.quantity)
            if new_sp is None:
                logger.error('New set point cannot be calculated')
                return
            self.set_spinbox_silent(parent.sb_pm_sp, new_sp)
        # Update Sample SetPoint if PM SetPoint was set   
        elif caller == parent.sb_pm_sp:
            new_sp = hutils.glan_calc(caller.quantity)
            if new_sp is None:
                logger.error('New set point cannot be calculated')
                return
            self.set_spinbox_silent(parent.sb_sample_sp, new_sp)

        # Update SetPoint on plot
        parent.pm_monitor.plot_right.sp = parent.sb_pm_sp.quantity
                
    def set_spinbox_silent(
            self,
            sb: QuantSpinBox,
            value: PlainQuantity
        ) -> None:
        """Set ``value`` to ``sb`` without firing events."""

        sb.blockSignals(True)
        sb.quantity = value.to(sb.quantity.u)
        sb.blockSignals(False)

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

        # Request information only if dock with motors is visible
        if not self.d_motors.isVisible():
            return
        
        # Update position information
        self.start_cb_worker(
            self.d_motors.set_position,
            pa_logic.stages_position
        )

        # Update status information
        self.start_cb_worker(
            self.d_motors.upd_status,
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

    def get_layout(self, widget: QWidget) -> QLayout|None:
        """Return parent layout for a widget."""

        parent = widget.parent()
        child_layouts = self._unpack_layout(parent.children())
        for layout in child_layouts:
            if layout.indexOf(widget) > -1:
                return layout

    def _unpack_layout(self, childs: list[QObject]) -> list[QLayout]:
        """Recurrent function to find all layouts in a list of widgets."""

        result = []
        for item in childs:
            if isinstance(item, QLayout):
                result.append(item)
                result.extend(self._unpack_layout(item.children()))
        return result

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

    ureg = pint.UnitRegistry(auto_reduce_dimensions=True) # type: ignore
    ureg.default_format = '~P'
    logger.debug('Pint activated (measured values with units)')
    ureg.setup_matplotlib(True)
    logger.debug('MatplotLib handlers for pint activated')
    pa_logic.load_config()
    win.showMaximized()
    sys.exit(app.exec())