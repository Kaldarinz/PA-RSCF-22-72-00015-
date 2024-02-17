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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QSizePolicy,
    QSpacerItem, QSplitter, QVBoxLayout, QWidget)

from ..widgets import (MplNavCanvas, PgMap)

class Ui_Map_view(object):
    def setupUi(self, Map_view):
        if not Map_view.objectName():
            Map_view.setObjectName(u"Map_view")
        Map_view.resize(940, 739)
        self.horizontalLayout_3 = QHBoxLayout(Map_view)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.splitter = QSplitter(Map_view)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_2 = QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.cb_curve_select = QComboBox(self.widget)
        self.cb_curve_select.setObjectName(u"cb_curve_select")

        self.horizontalLayout_2.addWidget(self.cb_curve_select)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.plot_map = PgMap(self.widget)
        self.plot_map.setObjectName(u"plot_map")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_map.sizePolicy().hasHeightForWidth())
        self.plot_map.setSizePolicy(sizePolicy)
        self.plot_map.setMinimumSize(QSize(600, 0))

        self.verticalLayout_2.addWidget(self.plot_map)

        self.splitter.addWidget(self.widget)
        self.widget1 = QWidget(self.splitter)
        self.widget1.setObjectName(u"widget1")
        self.verticalLayout = QVBoxLayout(self.widget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cb_detail_select = QComboBox(self.widget1)
        self.cb_detail_select.setObjectName(u"cb_detail_select")

        self.horizontalLayout.addWidget(self.cb_detail_select)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.plot_detail = MplNavCanvas(self.widget1)
        self.plot_detail.setObjectName(u"plot_detail")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plot_detail.sizePolicy().hasHeightForWidth())
        self.plot_detail.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.plot_detail)

        self.splitter.addWidget(self.widget1)

        self.horizontalLayout_3.addWidget(self.splitter)


        self.retranslateUi(Map_view)

        QMetaObject.connectSlotsByName(Map_view)
    # setupUi

    def retranslateUi(self, Map_view):
        Map_view.setWindowTitle(QCoreApplication.translate("Map_view", u"Form", None))
    # retranslateUi

