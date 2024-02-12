# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'map_data_view.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QSizePolicy, QWidget)

from ..widgets import PgMap

class Ui_Map_view(object):
    def setupUi(self, Map_view):
        if not Map_view.objectName():
            Map_view.setObjectName(u"Map_view")
        Map_view.resize(782, 694)
        self.horizontalLayout = QHBoxLayout(Map_view)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.plot_map = PgMap(Map_view)
        self.plot_map.setObjectName(u"plot_map")

        self.horizontalLayout.addWidget(self.plot_map)


        self.retranslateUi(Map_view)

        QMetaObject.connectSlotsByName(Map_view)
    # setupUi

    def retranslateUi(self, Map_view):
        Map_view.setWindowTitle(QCoreApplication.translate("Map_view", u"Form", None))
    # retranslateUi

