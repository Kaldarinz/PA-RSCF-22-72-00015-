# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'power_meter_monitor.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QSizePolicy, QSplitter, QVBoxLayout,
    QWidget)

from ..widgets import MplCanvas

class Ui_pm_monitor(object):
    def setupUi(self, pm_monitor):
        if not pm_monitor.objectName():
            pm_monitor.setObjectName(u"pm_monitor")
        pm_monitor.resize(400, 300)
        self.verticalLayout = QVBoxLayout(pm_monitor)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(pm_monitor)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.plot_left = MplCanvas(self.splitter)
        self.plot_left.setObjectName(u"plot_left")
        self.splitter.addWidget(self.plot_left)
        self.plot_right = MplCanvas(self.splitter)
        self.plot_right.setObjectName(u"plot_right")
        self.splitter.addWidget(self.plot_right)

        self.verticalLayout.addWidget(self.splitter)


        self.retranslateUi(pm_monitor)

        QMetaObject.connectSlotsByName(pm_monitor)
    # setupUi

    def retranslateUi(self, pm_monitor):
        pm_monitor.setWindowTitle(QCoreApplication.translate("pm_monitor", u"Form", None))
    # retranslateUi

