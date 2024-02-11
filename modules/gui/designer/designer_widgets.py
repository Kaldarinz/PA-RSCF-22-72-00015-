"""
Widgets, built in Qt Designer, and used as a relatively big parts of
the progamm.
"""
import logging
from collections import deque
from datetime import datetime
import math
from time import sleep, time

from PySide6.QtCore import (
    Signal,
    Qt,
    QStringListModel,
    Slot,
    QTimer
)
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QDockWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QProgressDialog
)
from PySide6.QtGui import (
    QPixmap
)
import numpy as np
from pint.facets.plain.quantity import PlainQuantity
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from . import (
    verify_measure_ui,
    data_viewer_ui,
    log_dock_widget_ui,
    point_measure_widget_ui,
    curve_measure_widget_ui,
    map_measure_widget_ui,
    pm_monitor_ui,
    curve_data_view_ui,
    point_data_view_ui,
    motor_control_ui
)

from ... import Q_
from ...pa_data import Measurement
from ...data_classes import (
    DataPoint,
    Position,
    QuantRect,
    StagesStatus,
    EnergyMeasurement,
    MapData,
    Worker,
    WorkerFlags
)
from ...constants import (
    DetailedSignals
)
from ...utils import (
    upd_plot,
    form_quant,
    btn_set_silent
)

from ..widgets import (
    MplCanvas,
    QuantSpinBox
)

logger = logging.getLogger(__name__)

class PAVerifierDialog(QDialog, verify_measure_ui.Ui_Dialog):
    """Dialog window for verification of a PA measurement."""

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

class DataViewer(QWidget, data_viewer_ui.Ui_Form):
    """Data viewer widget."""

    data_changed = Signal(bool)

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.p_1d = CurveView(self)
        self.p_0d = PointView(self)
        self.p_empty = QWidget()
        layout = QVBoxLayout()
        lbl = QLabel('No data to show')
        layout.addWidget(lbl)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.p_empty.setLayout(layout)
        self.sw_view.addWidget(self.p_1d)
        self.sw_view.addWidget(self.p_0d)
        self.sw_view.addWidget(self.p_empty)
        self.sw_view.setCurrentWidget(self.p_empty)
        self.measurement: Measurement|None = None
        "Currently displayed data."
        self.datapoint: DataPoint|None = None
        "Selected datapoint."
        self.data_index: int = 0
        "Index of the selceted datapoint."
        self.dtype_point: str = ''
        "Data type of the currently displayed datapoint."

        # Set model for file content
        self.content_model = QStringListModel()
        self.lv_content.setModel(self.content_model)

        # This is bullshit, but I cannot find other solution.
        # With default stretch factor central widget is too narrow 
        # for no reason.
        self.splitter.setStretchFactor(0,1)
        self.splitter.setStretchFactor(1,90)
        self.splitter.setStretchFactor(2,1)

class LoggerWidget(QDockWidget, log_dock_widget_ui.Ui_d_log):
    """Logger dock widget."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class CurveMeasureWidget(QWidget,curve_measure_widget_ui.Ui_Form):
    """1D PhotoAcoustic measurements widget."""

    cur_p_changed = Signal()
    "Signal change of current data point."

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.pm_monitor = PowerMeterMonitor(self)
        self.lo_measure.replaceWidget(
            0,
            self.pm_monitor
        )
        self.parameter: list[str]
        "List of parameter names."
        self.max_steps: int = 1
        "Amount of parameter points in measurement."
        self.step: PlainQuantity
        "Parameter step value."
        self.param_points: PlainQuantity
        "Array with all param values."
        self._current_point: int = 0

        # Auto update widgets when current_point is changed
        self.cur_p_changed.connect(self.upd_widgets)

    @property
    def current_point(self) -> int:
        """
        Index of current parameter value.
        
        Automatically updates related widgets."""

        return self._current_point
    
    @current_point.setter
    def current_point(self, val: int) -> None:

        self._current_point = val
        self.cur_p_changed.emit()

    def upd_widgets(self) -> None:
        """Update widgets, related to current scan point.
        
        Updates progress bar, its label and current param."""

        # progress bar
        pb = int(self.current_point/self.max_steps*100)
        self.pb.setValue(pb)

        # progress bar label
        text = f'{self.current_point}/{self.max_steps}'
        self.lbl_pb.setText(text)

        
        if self.current_point < self.max_steps:
            # current param value
            new_param = self.param_points[self.current_point] # type: ignore
            self.sb_cur_param.quantity = new_param
        else:
            # disable measure if last point
            self.btn_measure.setEnabled(False)

class PointMeasureWidget(QWidget,point_measure_widget_ui.Ui_Form):
    """0D PhotoAcoustic measurements widget."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.pm_monitor = PowerMeterMonitor(self)
        self.lo_measure.replaceWidget(
            self.placeholder_pm_monitor,
            self.pm_monitor
        )

class MapMeasureWidget(QWidget,map_measure_widget_ui.Ui_map_measure):
    """2D PhotoAcoustic measurements widget."""

    calc_astepClicked = Signal(QProgressDialog)
    astep_calculated = Signal(bool)
    "Emitted, when auto step is calculated. Always return `True`."
    scan_started = Signal(MapData)
    scan_worker: Worker | None = None

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # Zoom state of plot
        self._zoomed_out = False
        # max scan range
        self.XMAX = Q_(25, 'mm')
        self.YMAX = Q_(25, 'mm')

        # Set Axis titles
        self._new_scan_plane()

        # Set default range
        self.set_zoomed_out(True)

        # Set default selection
        self.plot_scan.set_def_selarea()
        # We should manually call this slot, since it is not yet connected
        self._sel_changed(self.plot_scan.selected_area)

        # Set enable state for axis controls
        self._new_dir()

        # Set default vals
        self.sb_cur_speed.quantity = Q_(0, 'm/s')
        self.sb_tpp.quantity = Q_(0, 's')

        # Scan timer
        self.scan_timer: QTimer | None = None
        self._scan_started: datetime | None = None

        self.connect_signals_slots()

    def connect_signals_slots(self):
        """Connect signals and slots."""

        # Scan size X changed
        self.sb_sizeX.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(width = val)
        )
        # Scan size Y changed
        self.sb_sizeY.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(height = val)
        )
        # Scan center X changed
        self.sb_centerX.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(
                left = val - self.sb_sizeX.quantity/2)
        )
        # Scan center Y changed
        self.sb_centerY.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(
                bottom = val - self.sb_sizeY.quantity/2)
        )
        # Scan points changed
        self.sb_pointsX.quantSet.connect(lambda val: self._points_set())
        self.sb_pointsY.quantSet.connect(lambda val: self._points_set())
        # Selected area changed from plot
        self.plot_scan.selection_changed.connect(self._sel_changed)
        # Toogle selected area visibility
        self.btn_plot_area_sel.toggled.connect(
            self.plot_scan.set_selarea_vis
        )
        # Set maximum plot size visible
        self.btn_plot_full_area.pressed.connect(lambda: self.set_zoomed_out(True))
        # Expand scan to full plot area
        self.btn_plot_fit.pressed.connect(lambda: self.set_zoomed_out(False))
        # Clear plot
        self.btn_plot_clear.clicked.connect(self.clear_data)
        # Scan plane changed
        self.cb_scanplane.currentTextChanged.connect(
            lambda _: self._new_scan_plane()
        )
        # Scan direction changed
        self.cb_scandir.currentTextChanged.connect( lambda _: self._new_dir())
        # Auto step calculation
        self.btn_astep_calc.clicked.connect(lambda _: self.calc_astep())
        # Current speed changed
        self.sb_cur_speed.valueChanged.connect(lambda _: self._set_est_dur())
        # Scan
        self.btn_start.toggled.connect(self.scan)
        # Stop scan
        self.btn_stop.clicked.connect(self.stop_scan)
        
    def scan(self, state: bool) -> None:
        """Launch scanning by emitting ``scan_started``."""

        if state:
            # Check if scan step was already calculated
            if self.sb_tpp.quantity.m == 0:
                self.calc_astep()
                self.astep_calculated.connect(self.scan)
                return
            else:
                try:
                    self.astep_calculated.disconnect(self.scan)
                except:
                    pass
            # Set buttons state
            self.btn_start.setEnabled(False)
            self.btn_plot_clear.setEnabled(False)
            # Create MapData object
            scan = MapData(
                center = self.center,
                width = self.sb_sizeX.quantity,
                height = self.sb_sizeY.quantity,
                hpoints = int(self.sb_pointsX.value()),
                vpoints = int(self.sb_pointsY.value()),
                scan_plane = self.cb_scanplane.currentText(), # type: ignore
                scan_dir = self.cb_scandir.currentText(),
                wavelength = self.sb_wl.quantity
            )
            # Load scan data to plot
            self.plot_scan.set_data(data=scan)
            self.set_zoomed_out(False)
            # Initiate scan timer
            if self.scan_timer is None:    
                self.scan_timer = QTimer(self)
                self.scan_timer.timeout.connect(self._upd_scan_time)
            self._scan_started = datetime.now()
            self.scan_timer.start(100)
            # Emit signal to start actual scanning from main window
            self.scan_started.emit(scan)

    @Slot()
    def stop_scan(self) -> None:
        """Stop current scan."""

        if self.scan_worker is None:
            return
        # Stop actual measurement
        self.scan_worker.flags['is_running'] = False
        
    @Slot()
    def scan_finished(self) -> None:
        """Scan finished procedures."""

        # Change state of buttons
        self.btn_start.setChecked(False)
        self.btn_start.setEnabled(True)
        self.btn_plot_clear.setEnabled(True)
        # Stop scan timer
        if self.scan_timer is None:
            logger.warning('Scan timer is missing.')
        else:
            self.scan_timer.stop()
        if self.plot_scan.data is not None:
            lines = len(self.plot_scan.data.data)
            dur = self.le_dur.text()
            logger.info(f'{lines} lines scanned. Scan duration {dur}')

        logger.info('Scanning finished.')

    @Slot()
    def clear_data(self) -> None:
        """Clear scanned data."""

        self.plot_scan.clear_plot()
        self.le_dur.setText('')
        self.set_zoomed_out(True)

    def calc_astep(self, count: int=30) -> QProgressDialog:
        """Launch auto step calculation."""

        # Create progress bar dialog
        pb = QProgressDialog(self)
        #pb.setAutoReset(False)
        pb.setAutoClose(True)
        pb.setLabelText('Calculating auto step size...')
        pb.setRange(0, count)
        # Set appear immediatelly
        pb.setMinimumDuration(0)
        pb.setValue(0)
        # Emit signal to launch worker for measuring.
        self.calc_astepClicked.emit(pb)
        pb.show()
        return pb

    @Slot()
    def set_astep(self, data: list[EnergyMeasurement]) -> None:
        """
        Set auto step and points from data list.
        
        ``data`` should contain results of performance test of signal
        acquisition system with at least 4 EnergyMeasurements, but
        for better precision it is better to provide 30-50 EnergyMeasurements.

        The algorithm will calculate average dt between signal measurements
        and use it to calculate step size and amount of points.
        Calculated dt is set to sb_tpp spinbox.
        """

        if len(data) < 4:
            logger.warning(
                'Auto step cannot be calculated. Too few data.'
            )
            return

        # Calculate duration of all measurements
        delta = data[-1].datetime - data[0].datetime
        # Multiplication by 2 is current implementation detail.
        # Currently signal is measured from only one channel during
        # the test. But actuall PA measurements will require both 
        # channels, which are similar in all params hence multiplication
        # by 2.
        dt = Q_(delta.total_seconds()*2, 's')
        # Force update of the current stage speed
        self.sb_speed.quantSet.emit(self.sb_speed.quantity)
        # Get current speed
        speed = self.sb_cur_speed.quantity
        # Average time per 1 step
        time_per_p = dt/(len(data) - 1)
        step = speed * time_per_p
        # Line size depends on orientation of fast scan direction
        if self.cb_scandir.currentText().startswith('H'):
            line_size = self.sb_sizeX.quantity
        else:
            line_size = self.sb_sizeY.quantity

        points = int(line_size/step) + 1

        ### Set calculated values
        # dt
        self.sb_tpp.quantity = time_per_p
        # Step
        x_units = self.sb_stepX.quantity.u
        self.sb_stepX.quantity = step.to(x_units)
        y_units = self.sb_stepY.quantity.u
        self.sb_stepY.quantity = step.to(y_units)
        # Points
        if self.cb_scandir.currentText().startswith('H'):
            self.sb_pointsX.setValue(points)
            self.sb_pointsY.setValue(
               int(self.sb_sizeY.quantity/step) + 1 
            )
        else:
            self.sb_pointsX.setValue(
               int(self.sb_sizeX.quantity/step) + 1 
            )
            self.sb_pointsY.setValue(points)
        # Set estimated scan duration
        self._set_est_dur()
        # Emit signal
        self.astep_calculated.emit(True)
    
    def set_zoomed_out(self, state: bool) -> None:
        """Set whether maximum scan area is visible."""
        if state and self._zoomed_out:
            return
        if not state and not self._zoomed_out:
            return
        
        self._zoomed_out = state
        self.plot_scan.abs_coords = state
        if state:
            self.plot_scan.set_scanrange(self.XMAX, self.YMAX)
            self.plot_scan.set_abs_scan_pos(True)
        else:
            self.plot_scan.set_scanrange(
                self.plot_scan.data.width,
                self.plot_scan.data.height
            )
            self.plot_scan.set_abs_scan_pos(False)

    def _set_est_dur(self) -> None:
        """Set estimated duration of scan."""

        if self.sb_tpp.quantity.m == 0:
            return
        # Get current speed
        speed = self.sb_cur_speed.quantity
        # Line size depends on orientation of fast scan direction
        if self.cb_scandir.currentText().startswith('H'):
            line_size = self.sb_sizeX.quantity
            spoints = self.sb_pointsY.value()
            est_dur = (line_size*spoints + self.sb_sizeY.quantity)/speed
        else:
            line_size = self.sb_sizeY.quantity
            spoints = self.sb_pointsX.value()
            est_dur = (line_size*spoints + self.sb_sizeX.quantity)/speed
        tot_s = est_dur.to('s').m
        hours = int(tot_s // 3600)
        mins = int((tot_s % 3600) // 60)
        secs = (tot_s % 3600) % 60
        strdur = f'{hours:02d}:{mins:02d}:{secs:04.1f}'
        self.le_estdur.setText(strdur)

    @Slot()
    def _sel_changed(self, new_sel: QuantRect | None) -> None:
        """This slot is always called when selection is changed."""
        # Uncheck selection button if selection is not visible
        if new_sel is None:
            btn_set_silent(self.btn_plot_area_sel, False)
            return
        # Otherwise update spin boxes
        x, y, width, height = new_sel
        cx = x + width/2
        cy = y + height/2
        xstep = width/(self.sb_pointsX.value() - 1)
        ystep = height/(self.sb_pointsY.value() - 1)
        size_changed = False
        if self.sb_centerX.quantity != cx:
            self.sb_centerX.quantity = cx # type: ignore
        if self.sb_centerY.quantity != cy:
            self.sb_centerY.quantity = cy # type: ignore
        if self.sb_sizeX.quantity != width:
            self.sb_sizeX.quantity = width
            size_changed = True
        if self.sb_sizeY.quantity != height:
            self.sb_sizeY.quantity = height
            size_changed = True
        if self.sb_stepX != xstep:
            self.sb_stepX.quantity = xstep
        if self.sb_stepY != ystep:
            self.sb_stepY.quantity = ystep
        if size_changed:
            self._set_est_dur()
        
    @Slot()
    def _points_set(self) -> None:
        """Update scan step, when amount of points is changed."""

        xstep = self.sb_sizeX.quantity/(self.sb_pointsX.value() - 1)
        ystep = self.sb_sizeY.quantity/(self.sb_pointsY.value() - 1)
        if self.sb_stepX.quantity != xstep:
            self.sb_stepX.quantity = xstep
        if self.sb_stepY.quantity != ystep:
            self.sb_stepY.quantity = ystep
        # Update estimated scan duration
        self._set_est_dur()

    @Slot()
    def _new_scan_plane(self) -> None:
        """Slot, called when new scan plane is selected."""

        sel_plane = self.cb_scanplane.currentText()
        self.plot_scan.hlabel = sel_plane[0]
        self.plot_scan.vlabel = sel_plane[1]
        # Labels for scan area
        self.lbl_areaX.setText(sel_plane[0])
        self.lbl_areaY.setText(sel_plane[1])

    @Slot()
    def _new_dir(self) -> None:
        """Callback for new scan direction."""
        
        # Update est scan duration
        self._set_est_dur()
        # Check if auto step calculation is enabled
        if not self.chb_astep.isChecked():
            return
        # Set enable state for axis controls
        if self.cb_scandir.currentText()[0] == 'H':
            self.sb_pointsX.setEnabled(False)
            self.sb_pointsY.setEnabled(True)
        else:
            self.sb_pointsX.setEnabled(True)
            self.sb_pointsY.setEnabled(False)

    @Slot()
    def _upd_scan_time(self) -> None:
        """Update scan duration."""

        if self._scan_started is not None:
            dt = datetime.now() - self._scan_started
            s, ms = str(dt).split('.')
            self.le_dur.setText(f'{s}.{ms[:1]}')
        else:
            logger.warning('Scan timer cannot be updated. Start time is None.')

    @property
    def center(self) -> Position:
        """Absolute position of scan center."""

        # Get titles of horizontal and vertical axes
        haxis = self.cb_scanplane.currentText()[0]
        vaxis = self.cb_scanplane.currentText()[1]
        # Get values of horizontal and vertical coordinates
        hcoord = self.sb_centerX.quantity
        vcoord = self.sb_centerY.quantity
        # Covert relative coordinate to absolute
        if not self.zoomed_out:
            if (shift:= self.plot_scan._rel_to_abs(quant = True)) is not None:
                hcoord += shift[0]
                vcoord += shift[1]
        # Generate position
        center = Position.from_tuples([
            (haxis, hcoord), (vaxis, vcoord)
        ])
        return center

    @property
    def zoomed_out(self) -> bool:
        """Whether maximum scan area is visible."""
        return self._zoomed_out
    @zoomed_out.setter
    def zoomed_out(self, state: bool) -> None:
        logger.warning('Zoomed out attribute is read only!')

class PowerMeterMonitor(QWidget,pm_monitor_ui.Ui_Form):
    """Power meter monitor.
    
    Contains two plot areas:\n
    ``plot_left`` and ``plot_right``.
    """

    def __init__(
            self,
            parent: QWidget | None=None,
            tune_width: int=50,
            aver: int=10  
        ) -> None:
        """
        Initialization of Power meter monitor.

        ``tune_width`` - size of energy adjustment plot (right).\n
        ``aver`` - amount of measurements used to calc mean and std.
        """
        super().__init__(parent)
        self.setupUi(self)
        self.tune_width = tune_width
        self.aver = aver
        self.connect_signals_slots()

    def connect_signals_slots(self):
        """Connect signals and slots."""

        # Start button
        self.btn_start.toggled.connect(
            lambda enable: self.btn_pause.setChecked(not enable)
        )
        # Pause button
        self.btn_pause.toggled.connect(
            lambda enable: self.btn_start.setChecked(not enable)
        )

        ### Stop button
        # Reset buttons
        self.btn_stop.clicked.connect(
            lambda: btn_set_silent(self.btn_start, False)
        )
        self.btn_stop.clicked.connect(
            lambda: btn_set_silent(self.btn_pause, False)
        )
        # Clear plots
        self.btn_stop.clicked.connect(
            lambda: upd_plot(base_widget=self.plot_left, ydata=[])
        )
        self.btn_stop.clicked.connect(
            lambda: upd_plot(base_widget=self.plot_right, ydata=[])
        )

    def add_msmnt(self, measurement: EnergyMeasurement) -> None:
        """"Add energy measurement."""

        # Update information only if Start btn is checked 
        # and widget is visible
        logger.debug(f'add_msmnt called for {self.objectName()}')
        if (not self.btn_start.isChecked()) or (not self.isVisible()):
            return
        # Check if measurement is not empty
        if measurement.energy.m is np.nan:
            return
        # Skip duplicate read
        if np.array_equal(
            measurement.signal.m,
            self.plot_left.ydata
        ):
            return
        # Plot signal
        upd_plot(
            base_widget = self.plot_left,
            ydata = measurement.signal,
            marker = [measurement.istart, measurement.istop]
        )

        # Plot tune graph
        ytune = self.plot_right.ydata
        # Check if some data already present
        if ytune.ndim == 0:
            upd_plot(
                base_widget = self.plot_right,
                ydata = PlainQuantity.from_list([measurement.energy])
            )
        else:
            energy = measurement.energy.to(self.plot_right.ylabel)
            qu = deque(ytune, maxlen = self.tune_width)
            qu.append(energy.m)
            ytune = np.array(qu)
            upd_plot(
                base_widget = self.plot_right,
                ydata = ytune
            )
        
        # Update line edits with energy info
        units = self.plot_right.ylabel
        ytune = self.plot_right.ydata
        self.le_cur_en.setText(form_quant(measurement.energy))
        aver = Q_(ytune.mean(), units)
        self.le_aver_en.setText(form_quant(aver))
        std = Q_(ytune.std(), units)
        self.le_std_en.setText(form_quant(std))

class CurveView(QWidget,curve_data_view_ui.Ui_Form):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
class PointView(QWidget, point_data_view_ui.Ui_Form):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class MotorView(QDockWidget, motor_control_ui.Ui_DockWidget):
    """Mechanical positioning widget."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.x: PlainQuantity = Q_(0, 'mm')
        self.y: PlainQuantity = Q_(0, 'mm')
        self.z: PlainQuantity = Q_(0, 'mm')

        # Set fixed scales for all plots.
        self.plot_xy.fixed_scales = True
        self.plot_xz.fixed_scales = True
        self.plot_yz.fixed_scales = True

        self.fmt = 'o'
        "Format string for plotting."

        self.pixmap_c = QPixmap(
            u":/icons/qt_resources/plug-connect.png"
        )
        "Connected state icon."
        self.pixmap_dc = QPixmap(
            u":/icons/qt_resources/plug-disconnect-prohibition.png"
        )
        "Disconnected state icon."

        # Set default range
        self.set_range(
            (Q_(0, 'mm'), Q_(25, 'mm'))
        )

        # Set initial position of plots
        upd_plot(
            self.plot_xy,
            ydata = [self.y.m],
            xdata = [self.x.m],
            fmt = self.fmt
        )
        upd_plot(
            self.plot_xz,
            ydata = [self.z.m],
            xdata = [self.x.m],
            fmt = self.fmt
        )
        upd_plot(
            self.plot_yz,
            ydata = [self.z.m],
            xdata = [self.y.m],
            fmt = self.fmt
        )

    @Slot(Position)
    def set_position(self, position: Position) -> None:
        """Update current coordinate."""

        if position is None:
            logger.warning('Invalid coordinate in set position.')
        # Update current position line edits and attributes.
        if position.x is not None and position.x != self.x:
            self.x = position.x.to('mm')
            self.le_x_cur_pos.setText(f'{self.x:.3f~P}')
            self.sb_x_new_pos.setValue(self.x.m)
        if position.y is not None and position.y != self.y:
            self.y = position.y.to('mm')
            self.le_y_cur_pos.setText(f'{self.y:.3f~P}')
            self.sb_y_new_pos.setValue(self.y.m)
        if position.z is not None and position.z != self.z:
            self.z = position.z.to('mm')
            self.le_z_cur_pos.setText(f'{self.z:.3f~P}')
            self.sb_z_new_pos.setValue(self.z.m)
        
        # Update plots if necessary
        self._upd_plot(self.x.m, self.y.m, self.plot_xy)
        self._upd_plot(self.x.m, self.z.m, self.plot_xz)
        self._upd_plot(self.z.m, self.y.m, self.plot_yz)

    def set_range(
            self,
            range: tuple[PlainQuantity, PlainQuantity]
        ) -> None:
        """Set range of valid coordinates."""

        min_val = range[0].to('mm').m
        max_val = range[1].to('mm').m
        range_mm = [min_val, max_val]
        # Set range for plots
        self.plot_xy.axes.set_xlim(range_mm) # type: ignore
        self.plot_xy.axes.set_ylim(range_mm) # type: ignore
        self.plot_xz.axes.set_xlim(range_mm) # type: ignore
        self.plot_xz.axes.set_ylim(range_mm) # type: ignore
        self.plot_yz.axes.set_xlim(range_mm) # type: ignore
        self.plot_yz.axes.set_ylim(range_mm) # type: ignore

        # Set plot titles
        self.plot_xy.xlabel = 'X, [mm]'
        self.plot_xy.ylabel = 'Y, [mm]'
        self.plot_xz.xlabel = 'X, [mm]'
        self.plot_xz.ylabel = 'Z, [mm]'
        self.plot_yz.xlabel = 'Z, [mm]'
        self.plot_yz.ylabel = 'Y, [mm]'
        # Set range for spin boxes
        for sb in iter([
            self.sb_x_new_pos,
            self.sb_y_new_pos,
            self.sb_z_new_pos
        ]):
            sb.setMinimum(min_val)
            sb.setMaximum(max_val)

    def upd_status(
            self,
            status: StagesStatus
        ) -> None:
        """Update status of motors."""

        self._upd_axes_status(
            status.x_open,
            status.x_status,
            self.icon_x_status,
            self.lbl_x_status
        )
        self._upd_axes_status(
            status.y_open,
            status.y_status,
            self.icon_y_status,
            self.lbl_y_status
        )
        self._upd_axes_status(
            status.z_open,
            status.z_status,
            self.icon_z_status,
            self.lbl_z_status
        )

    def _upd_plot(
            self,
            x: float,
            y: float,
            plot: MplCanvas
        ) -> None:
        """Update plot coordinate."""

        if x != plot.xdata[0] or y != plot.ydata[0]:
            upd_plot(
            base_widget = plot,
            ydata = [y],
            xdata = [x],
            fmt = self.fmt
        )

    def _upd_axes_status(
            self,
            is_opened: bool | None,
            status: list[str],
            icon: QLabel,
            lbl: QLabel
        ) -> None:
        """Update status widgets for an axes."""

        if is_opened is None or len(status) == 0:
            return
        if is_opened:
            icon.setPixmap(self.pixmap_c)
            lbl.setText(','.join(status))
        else:
            icon.setPixmap(self.pixmap_dc)
            lbl.setText('Disconnected')

    def current_pos(self) -> Position:
        """Get position to send stages."""

        pos = Position()
        pos.x = Q_(self.sb_x_new_pos.value(), 'mm')
        pos.y = Q_(self.sb_y_new_pos.value(), 'mm')
        pos.z = Q_(self.sb_z_new_pos.value(), 'mm')
        return pos