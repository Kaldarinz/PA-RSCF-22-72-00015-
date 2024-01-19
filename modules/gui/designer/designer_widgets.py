"""
Classes with initaited widgets, built in Qt Designer.
"""
import logging
from collections import deque

from PySide6.QtCore import (
    Signal,
    Qt,
    QStringListModel,
    Slot
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
    StagesStatus,
    EnergyMeasurement,
    MapData
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
    scan_started = Signal(MapData)

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # Set Axis titles
        self._new_scan_plane()

        # Set default range
        self.plot_scan.set_scanrange(Q_(25, 'mm'), Q_(25, 'mm'))

        # Set default selection
        self.plot_scan.set_selarea(Q_(11, 'mm'), Q_(11, 'mm'), Q_(4, 'mm'), Q_(4, 'mm'))

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
        # Selected area changed from plot
        self.plot_scan.selection_changed.connect(self._sel_changed)
        # Scan plane changed
        self.cb_scanplane.currentTextChanged.connect(lambda _: self._new_scan_plane())
        # Auto step calculation
        self.btn_astep_calc.clicked.connect(
            lambda state: self.calc_astep())
        # Scan
        self.btn_start.clicked.connect(self.scan)

        # Test
        self.btn_tst.clicked.connect(self.scan)

    def scan(self) -> None:
        """Launch scanning by emitting ``scan_started``."""

        # Create MapData object
        scan = MapData(
            center = self.center,
            width = self.sb_sizeX.quantity,
            height = self.sb_sizeY.quantity,
            hpoints = self.sb_pointsX.value(),
            vpoints = self.sb_pointsY.value(),
            scan_plane = self.cb_scanplane.currentText(),
            scan_dir = self.cb_scandir.currentText(),
            wavelength = self.sb_wl.quantity
        )
        # Load scan data to plot
        self.plot_scan.set_data(scan)
        # Emit signal to start actual scanning from main window
        self.scan_started.emit(scan)

    def calc_astep(self, count: int=30) -> None:
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
        self.calc_astepClicked.emit(pb)
        pb.show()

    @Slot()
    def set_astep(self, data: list[EnergyMeasurement]) -> None:
        """
        Set auto step and points from data list.
        
        ``data`` should contain results of performance test of signal
        acquisition system with at least 4 EnergyMeasurements, but
        for better precision it is better to provide 30-50 EnergyMeasurements.

        The algorithm will calculate average dt between signal measurements
        and use it to calculate step size and amount of points.
        """

        if len(data) < 4:
            logger.warning(
                'Auto step cannot be calculated. Too few data.'
            )
            return

        # Calculate duration of all measurements
        delta = data[-1].datetime - data[0].datetime
        dt = Q_(delta.seconds + delta.microseconds/1_000_000, 's')
        speed = self.sb_speed.quantity
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
            est_dur = 1.2*(line_size + 2*self.sb_sizeY.quantity)/speed
        else:
            self.sb_pointsX.setValue(
               int(self.sb_sizeX.quantity/step) + 1 
            )
            self.sb_pointsY.setValue(points)
            est_dur = 1.2*(line_size + 2*self.sb_sizeX.quantity)/speed
        # Est time
        self.le_estdur.setText(str(est_dur))

    @Slot()
    def _sel_changed(self, new_sel: tuple[PlainQuantity]) -> None:

        x, y, width, height = new_sel
        cx = x + width/2
        cy = y + height/2
        if self.sb_centerX.quantity != cx:
            self.sb_centerX.quantity = cx
        if self.sb_centerY.quantity != cy:
            self.sb_centerY.quantity = cy
        if self.sb_sizeX.quantity != width:
            self.sb_sizeX.quantity = width
        if self.sb_sizeY.quantity != height:
            self.sb_sizeY.quantity = height

    @Slot()
    def _new_scan_plane(self) -> None:
        """Slot, called when new scan plane is selected."""

        sel_plane = self.cb_scanplane.currentText()
        self.plot_scan.hlabel = sel_plane[0]
        self.plot_scan.vlabel = sel_plane[1]

    @property
    def center(self) -> Position:
        """Position of scan center."""

        # Get titles of horizontal and vertical axes
        haxis = self.cb_scanplane.currentText()[0]
        vaxis = self.cb_scanplane.currentText()[1]
        # Get values of horizontal and vertical coordinates
        hcoord = self.sb_centerX.quantity
        vcoord = self.sb_centerY.quantity
        # Generate position
        center = Position.from_tuples([
            (haxis, hcoord), (vaxis, vcoord)
        ])
        return center

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