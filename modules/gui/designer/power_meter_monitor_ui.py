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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget)

from ..widgets import MplCanvas

class Ui_pm_monitor(object):
    def setupUi(self, pm_monitor):
        if not pm_monitor.objectName():
            pm_monitor.setObjectName(u"pm_monitor")
        pm_monitor.resize(882, 594)
        self.horizontalLayout = QHBoxLayout(pm_monitor)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, -1, 0, -1)
        self.splitter = QSplitter(pm_monitor)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_signal = QLabel(self.layoutWidget)
        self.lbl_signal.setObjectName(u"lbl_signal")
        self.lbl_signal.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_signal)

        self.plot_left = MplCanvas(self.layoutWidget)
        self.plot_left.setObjectName(u"plot_left")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_left.sizePolicy().hasHeightForWidth())
        self.plot_left.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.plot_left)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lbl_tune = QLabel(self.layoutWidget1)
        self.lbl_tune.setObjectName(u"lbl_tune")
        self.lbl_tune.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.lbl_tune)

        self.plot_right = MplCanvas(self.layoutWidget1)
        self.plot_right.setObjectName(u"plot_right")
        sizePolicy.setHeightForWidth(self.plot_right.sizePolicy().hasHeightForWidth())
        self.plot_right.setSizePolicy(sizePolicy)

        self.verticalLayout_2.addWidget(self.plot_right)

        self.splitter.addWidget(self.layoutWidget1)

        self.horizontalLayout.addWidget(self.splitter)


        self.retranslateUi(pm_monitor)

        QMetaObject.connectSlotsByName(pm_monitor)
    # setupUi

    def retranslateUi(self, pm_monitor):
        pm_monitor.setWindowTitle(QCoreApplication.translate("pm_monitor", u"Form", None))
        self.lbl_signal.setText(QCoreApplication.translate("pm_monitor", u"Power Meter Signal Shape", None))
        self.lbl_tune.setText(QCoreApplication.translate("pm_monitor", u"Power Meter Energy Tune", None))
    # retranslateUi

