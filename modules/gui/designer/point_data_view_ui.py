# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'point_data_view.ui'
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

from ..widgets import MplNavCanvas

class Ui_Point_view(object):
    def setupUi(self, Point_view):
        if not Point_view.objectName():
            Point_view.setObjectName(u"Point_view")
        Point_view.resize(649, 585)
        self.verticalLayout = QVBoxLayout(Point_view)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lbl_detail_select = QLabel(Point_view)
        self.lbl_detail_select.setObjectName(u"lbl_detail_select")

        self.horizontalLayout.addWidget(self.lbl_detail_select)

        self.cb_detail_select = QComboBox(Point_view)
        self.cb_detail_select.setObjectName(u"cb_detail_select")
        self.cb_detail_select.setMinimumSize(QSize(140, 0))

        self.horizontalLayout.addWidget(self.cb_detail_select)

        self.lbl_sample = QLabel(Point_view)
        self.lbl_sample.setObjectName(u"lbl_sample")

        self.horizontalLayout.addWidget(self.lbl_sample)

        self.cb_sample = QComboBox(Point_view)
        self.cb_sample.setObjectName(u"cb_sample")

        self.horizontalLayout.addWidget(self.cb_sample)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.plot_detail = MplNavCanvas(Point_view)
        self.plot_detail.setObjectName(u"plot_detail")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_detail.sizePolicy().hasHeightForWidth())
        self.plot_detail.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.plot_detail)


        self.retranslateUi(Point_view)

        QMetaObject.connectSlotsByName(Point_view)
    # setupUi

    def retranslateUi(self, Point_view):
        Point_view.setWindowTitle(QCoreApplication.translate("Point_view", u"Form", None))
        self.lbl_detail_select.setText(QCoreApplication.translate("Point_view", u"Signal", None))
        self.lbl_sample.setText(QCoreApplication.translate("Point_view", u"Sample", None))
    # retranslateUi

