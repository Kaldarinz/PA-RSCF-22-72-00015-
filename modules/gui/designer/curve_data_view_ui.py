# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'curve_data_view.ui'
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
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

from ..compound_widgets import PointView
from ..widgets import PgPlot

class Ui_Curve_view(object):
    def setupUi(self, Curve_view):
        if not Curve_view.objectName():
            Curve_view.setObjectName(u"Curve_view")
        Curve_view.resize(1367, 796)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Curve_view.sizePolicy().hasHeightForWidth())
        Curve_view.setSizePolicy(sizePolicy)
        self.horizontalLayout_2 = QHBoxLayout(Curve_view)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lo_curve = QVBoxLayout()
        self.lo_curve.setObjectName(u"lo_curve")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lbl_curve_select = QLabel(Curve_view)
        self.lbl_curve_select.setObjectName(u"lbl_curve_select")

        self.horizontalLayout.addWidget(self.lbl_curve_select)

        self.cb_curve_select = QComboBox(Curve_view)
        self.cb_curve_select.setObjectName(u"cb_curve_select")
        self.cb_curve_select.setMinimumSize(QSize(140, 0))

        self.horizontalLayout.addWidget(self.cb_curve_select)

        self.horizontalSpacer = QSpacerItem(373, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.lo_curve.addLayout(self.horizontalLayout)

        self.plot_curve = PgPlot(Curve_view)
        self.plot_curve.setObjectName(u"plot_curve")
        sizePolicy.setHeightForWidth(self.plot_curve.sizePolicy().hasHeightForWidth())
        self.plot_curve.setSizePolicy(sizePolicy)

        self.lo_curve.addWidget(self.plot_curve)


        self.horizontalLayout_2.addLayout(self.lo_curve)

        self.pv = PointView(Curve_view)
        self.pv.setObjectName(u"pv")
        sizePolicy.setHeightForWidth(self.pv.sizePolicy().hasHeightForWidth())
        self.pv.setSizePolicy(sizePolicy)

        self.horizontalLayout_2.addWidget(self.pv)


        self.retranslateUi(Curve_view)

        QMetaObject.connectSlotsByName(Curve_view)
    # setupUi

    def retranslateUi(self, Curve_view):
        Curve_view.setWindowTitle(QCoreApplication.translate("Curve_view", u"Form", None))
        self.lbl_curve_select.setText(QCoreApplication.translate("Curve_view", u"Curve Signal", None))
    # retranslateUi

