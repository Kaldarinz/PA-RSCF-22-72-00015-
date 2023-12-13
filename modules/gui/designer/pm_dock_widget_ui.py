# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pm_dock_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QDockWidget, QSizePolicy, QWidget)

from .designer_widgets import PowerMeterMonitor

class Ui_d_pm(object):
    def setupUi(self, d_pm):
        if not d_pm.objectName():
            d_pm.setObjectName(u"d_pm")
        d_pm.resize(893, 304)
        d_pm.setFloating(True)
        self.pm = PowerMeterMonitor()
        self.pm.setObjectName(u"pm")
        d_pm.setWidget(self.pm)

        self.retranslateUi(d_pm)

        QMetaObject.connectSlotsByName(d_pm)
    # setupUi

    def retranslateUi(self, d_pm):
        d_pm.setWindowTitle(QCoreApplication.translate("d_pm", u"Power Meter", None))
    # retranslateUi

