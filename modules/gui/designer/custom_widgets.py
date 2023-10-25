"""
Classes with initaited widgets, built in Qt Designer.
"""

from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QDockWidget
)

from . import (
    verify_measure_ui,
    data_viewer_ui,
    log_dock_widget_ui,
    pm_dock_widget_ui
)


class PAVerifierDialog(QDialog, verify_measure_ui.Ui_Dialog):
    """Dialog window for verification of a PA measurement."""

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

class DataViewer(QWidget, data_viewer_ui.Ui_Form):
    """Data viewer widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

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

