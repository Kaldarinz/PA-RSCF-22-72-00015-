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
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QVBoxLayout, QWidget)

from ..compound_widgets import PointView
from ..widgets import PgMap

class Ui_Map_view(object):
    def setupUi(self, Map_view):
        if not Map_view.objectName():
            Map_view.setObjectName(u"Map_view")
        Map_view.resize(1060, 739)
        self.verticalLayout = QVBoxLayout(Map_view)
        self.verticalLayout.setObjectName(u"verticalLayout")
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

        self.btn_tst = QPushButton(self.layoutWidget)
        self.btn_tst.setObjectName(u"btn_tst")

        self.horizontalLayout_2.addWidget(self.btn_tst)

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
        self.pv = PointView(self.splitter)
        self.pv.setObjectName(u"pv")
        sizePolicy.setHeightForWidth(self.pv.sizePolicy().hasHeightForWidth())
        self.pv.setSizePolicy(sizePolicy)
        self.splitter.addWidget(self.pv)

        self.verticalLayout.addWidget(self.splitter)


        self.retranslateUi(Map_view)

        QMetaObject.connectSlotsByName(Map_view)
    # setupUi

    def retranslateUi(self, Map_view):
        Map_view.setWindowTitle(QCoreApplication.translate("Map_view", u"Form", None))
        self.lbl_curve_select.setText(QCoreApplication.translate("Map_view", u"Map Signal", None))
        self.btn_tst.setText(QCoreApplication.translate("Map_view", u"test", None))
    # retranslateUi

