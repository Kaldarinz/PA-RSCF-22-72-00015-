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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QSizePolicy, QSpacerItem, QSplitter, QVBoxLayout,
    QWidget)

from ..widgets import (PgMap, PgPlot)

class Ui_Map_view(object):
    def setupUi(self, Map_view):
        if not Map_view.objectName():
            Map_view.setObjectName(u"Map_view")
        Map_view.resize(1060, 739)
        self.horizontalLayout_3 = QHBoxLayout(Map_view)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.splitter = QSplitter(Map_view)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lbl_curve_select = QLabel(self.layoutWidget)
        self.lbl_curve_select.setObjectName(u"lbl_curve_select")

        self.horizontalLayout_2.addWidget(self.lbl_curve_select)

        self.cb_curve_select = QComboBox(self.layoutWidget)
        self.cb_curve_select.setObjectName(u"cb_curve_select")

        self.horizontalLayout_2.addWidget(self.cb_curve_select)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.plot_map = PgMap(self.layoutWidget)
        self.plot_map.setObjectName(u"plot_map")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_map.sizePolicy().hasHeightForWidth())
        self.plot_map.setSizePolicy(sizePolicy)
        self.plot_map.setMinimumSize(QSize(600, 0))

        self.verticalLayout_2.addWidget(self.plot_map)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lbl_detail_select = QLabel(self.layoutWidget1)
        self.lbl_detail_select.setObjectName(u"lbl_detail_select")

        self.horizontalLayout.addWidget(self.lbl_detail_select)

        self.cb_detail_select = QComboBox(self.layoutWidget1)
        self.cb_detail_select.setObjectName(u"cb_detail_select")

        self.horizontalLayout.addWidget(self.cb_detail_select)

        self.lbl_sample = QLabel(self.layoutWidget1)
        self.lbl_sample.setObjectName(u"lbl_sample")

        self.horizontalLayout.addWidget(self.lbl_sample)

        self.cb_sample = QComboBox(self.layoutWidget1)
        self.cb_sample.setObjectName(u"cb_sample")

        self.horizontalLayout.addWidget(self.cb_sample)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.plot_detail = PgPlot(self.layoutWidget1)
        self.plot_detail.setObjectName(u"plot_detail")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plot_detail.sizePolicy().hasHeightForWidth())
        self.plot_detail.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.plot_detail)

        self.splitter.addWidget(self.layoutWidget1)

        self.horizontalLayout_3.addWidget(self.splitter)


        self.retranslateUi(Map_view)

        QMetaObject.connectSlotsByName(Map_view)
    # setupUi

    def retranslateUi(self, Map_view):
        Map_view.setWindowTitle(QCoreApplication.translate("Map_view", u"Form", None))
        self.lbl_curve_select.setText(QCoreApplication.translate("Map_view", u"Map Signal", None))
        self.lbl_detail_select.setText(QCoreApplication.translate("Map_view", u"Point Signal", None))
        self.lbl_sample.setText(QCoreApplication.translate("Map_view", u"Sample", None))
    # retranslateUi

