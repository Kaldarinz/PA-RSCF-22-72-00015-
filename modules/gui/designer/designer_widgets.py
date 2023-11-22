"""
Classes with initaited widgets, built in Qt Designer.
"""
import logging

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
    QLabel
)
from PySide6.QtGui import (
    QPixmap
)
from pint.facets.plain.quantity import PlainQuantity
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from . import (
    verify_measure_ui,
    data_viewer_ui,
    log_dock_widget_ui,
    pm_dock_widget_ui,
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
    DetailedSignals,
    Coordinate
)
from ...utils import (
    upd_plot
)
from ...pa_logic import (
    stages_status,
    stages_position
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

    def __init__(self, parent: QWidget | None = None) -> None:
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class PowerMeterWidget(QDockWidget, pm_dock_widget_ui.Ui_d_pm):
    """Power meter dock widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class CurveMeasureWidget(QWidget,curve_measure_widget_ui.Ui_Form):
    """1D PhotoAcoustic measurements widget."""

    cur_p_changed = Signal()
    "Signal change of current data point."

    def __init__(self, parent: QWidget | None = None) -> None:
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.pm_monitor = PowerMeterMonitor(self)
        self.lo_measure.replaceWidget(
            self.placeholder_pm_monitor,
            self.pm_monitor
        )

class MapMeasureWidget(QWidget,map_measure_widget_ui.Ui_Form):
    """2D PhotoAcoustic measurements widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class PowerMeterMonitor(QWidget,pm_monitor_ui.Ui_Form):
    """Power meter monitor.
    
    Contains two plot areas:\n
    ``plot_left`` and ``plot_right``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class CurveView(QWidget,curve_data_view_ui.Ui_Form):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
class PointView(QWidget, point_data_view_ui.Ui_Form):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class MotorView(QDockWidget, motor_control_ui.Ui_DockWidget):
    """Mechanical positioning widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
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

    @Slot(Coordinate)
    def set_position(self, position: Coordinate) -> None:
        """Update current coordinate."""

        if position is None:
            logger.warning('Invalid coordinate in set position.')
        # Update current position line edits and attributes.
        if position.x is not None:
            self.x = position.x.to('mm')
            self.le_x_cur_pos.setText(str(self.x))
        if position.y is not None:
            self.y = position.y.to('mm')
            self.le_y_cur_pos.setText(str(self.y))
        if position.z is not None:
            self.z = position.z.to('mm')
            self.le_z_cur_pos.setText(str(self.z))

        # Update plot positions
        upd_plot(
            base_widget = self.plot_xy,
            ydata = [self.y.m],
            xdata = [self.x.m],
            fmt = self.fmt
        )
        upd_plot(
            base_widget = self.plot_xz,
            ydata = [self.z.to('mm').m],
            xdata = [self.x.to('mm').m],
            fmt = self.fmt
        )
        upd_plot(
            base_widget = self.plot_yz,
            ydata = [self.y.to('mm').m],
            xdata = [self.z.to('mm').m],
            fmt = self.fmt
        )

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
            status_lst: list[tuple[bool, str]]
        ) -> None:
        """Update status of motors."""

        # Length of list with status can vary, but it assumed to satuses
        # for X, Y, Z axes if they are present.
        for status, icon, lbl in zip(
            status_lst,
            [
                self.icon_x_status,
                self.icon_y_status,
                self.icon_z_status
            ],
             [
                self.lbl_x_status,
                self.lbl_y_status,
                self.lbl_z_status
             ]
        ):
            # Set icons and status text
            if status[0]:
                icon.setPixmap(self.pixmap_c)
                lbl.setText(status[1])
            else:
                icon.setPixmap(self.pixmap_dc)
                lbl.setText('Disconnected')
