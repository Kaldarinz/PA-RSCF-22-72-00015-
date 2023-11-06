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
    power_meter_monitor_ui,
    curve_data_view_ui,
    point_data_view_ui
)

from ...pa_data import Measurement
from ...data_classes import DataPoint


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
        self.dtype_point: str|None = None
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.pm_monitor = PowerMeterMonitor(self)
        self.lo_measure.replaceWidget(
            self.placeholder_pm_monitor,
            self.pm_monitor
        )
        self.spectral_points: int
        self.step: PlainQuantity
        # self.lo_run.replaceWidget(
        #     self.btn_run,
        #     QPushButton('WTF')
        # )

class PointMeasureWidget(QWidget,point_measure_widget_ui.Ui_Form):
    """0D PhotoAcoustic measurements widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class MapMeasureWidget(QWidget,map_measure_widget_ui.Ui_Form):
    """2D PhotoAcoustic measurements widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class PowerMeterMonitor(QWidget,power_meter_monitor_ui.Ui_pm_monitor):
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
        self.nav_curve = NavigationToolbar2QT(self.plot_curve, self)
        self.lo_curve.addWidget(self.nav_curve)
        self.nav_detail = NavigationToolbar2QT(self.plot_detail, self)
        self.lo_detail.addWidget(self.nav_detail)
        

class PointView(QWidget, point_data_view_ui.Ui_Form):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
    
