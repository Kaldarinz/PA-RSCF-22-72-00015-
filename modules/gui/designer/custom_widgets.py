"""
Classes with initaited widgets, built in Qt Designer.
"""
from PySide6.QtCore import (
    Signal,
    Qt
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
    point_data_view_ui
)

from ...pa_data import Measurement
from ...data_classes import (
    DataPoint,
    DetailedSignals
)


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
        """Update widgets, related to corrent scan point.
        
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
    
