# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pm_monitor.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QVBoxLayout, QWidget)

from ..widgets import PgPlot

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(995, 623)
        self.verticalLayout_3 = QVBoxLayout(Form)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 6, 0, 6)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer)

        self.lbl_cur_en = QLabel(Form)
        self.lbl_cur_en.setObjectName(u"lbl_cur_en")

        self.horizontalLayout_9.addWidget(self.lbl_cur_en)

        self.le_cur_en = QLineEdit(Form)
        self.le_cur_en.setObjectName(u"le_cur_en")
        self.le_cur_en.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_cur_en.sizePolicy().hasHeightForWidth())
        self.le_cur_en.setSizePolicy(sizePolicy)
        self.le_cur_en.setMinimumSize(QSize(70, 20))
        self.le_cur_en.setMaximumSize(QSize(50, 16777215))
        self.le_cur_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_cur_en)

        self.lbl_aver_en = QLabel(Form)
        self.lbl_aver_en.setObjectName(u"lbl_aver_en")

        self.horizontalLayout_9.addWidget(self.lbl_aver_en)

        self.le_aver_en = QLineEdit(Form)
        self.le_aver_en.setObjectName(u"le_aver_en")
        self.le_aver_en.setEnabled(True)
        sizePolicy.setHeightForWidth(self.le_aver_en.sizePolicy().hasHeightForWidth())
        self.le_aver_en.setSizePolicy(sizePolicy)
        self.le_aver_en.setMinimumSize(QSize(70, 20))
        self.le_aver_en.setMaximumSize(QSize(50, 16777215))
        self.le_aver_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_aver_en)

        self.lbl_std_en = QLabel(Form)
        self.lbl_std_en.setObjectName(u"lbl_std_en")

        self.horizontalLayout_9.addWidget(self.lbl_std_en)

        self.le_std_en = QLineEdit(Form)
        self.le_std_en.setObjectName(u"le_std_en")
        self.le_std_en.setEnabled(True)
        sizePolicy.setHeightForWidth(self.le_std_en.sizePolicy().hasHeightForWidth())
        self.le_std_en.setSizePolicy(sizePolicy)
        self.le_std_en.setMinimumSize(QSize(70, 20))
        self.le_std_en.setMaximumSize(QSize(50, 16777215))
        self.le_std_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_std_en)

        self.btn_start = QPushButton(Form)
        self.btn_start.setObjectName(u"btn_start")
        font = QFont()
        font.setBold(True)
        self.btn_start.setFont(font)
        self.btn_start.setCheckable(True)

        self.horizontalLayout_9.addWidget(self.btn_start)

        self.btn_pause = QPushButton(Form)
        self.btn_pause.setObjectName(u"btn_pause")
        self.btn_pause.setFont(font)
        self.btn_pause.setCheckable(True)

        self.horizontalLayout_9.addWidget(self.btn_pause)

        self.btn_stop = QPushButton(Form)
        self.btn_stop.setObjectName(u"btn_stop")
        font1 = QFont()
        font1.setBold(True)
        font1.setUnderline(False)
        font1.setStrikeOut(False)
        self.btn_stop.setFont(font1)

        self.horizontalLayout_9.addWidget(self.btn_stop)


        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy1)
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lbl_signal = QLabel(self.layoutWidget)
        self.lbl_signal.setObjectName(u"lbl_signal")
        self.lbl_signal.setFont(font)
        self.lbl_signal.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.lbl_signal)

        self.plot_left = PgPlot(self.layoutWidget)
        self.plot_left.setObjectName(u"plot_left")
        sizePolicy1.setHeightForWidth(self.plot_left.sizePolicy().hasHeightForWidth())
        self.plot_left.setSizePolicy(sizePolicy1)

        self.verticalLayout_2.addWidget(self.plot_left)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_tune = QLabel(self.layoutWidget1)
        self.lbl_tune.setObjectName(u"lbl_tune")
        self.lbl_tune.setFont(font)
        self.lbl_tune.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_tune)

        self.plot_right = PgPlot(self.layoutWidget1)
        self.plot_right.setObjectName(u"plot_right")
        sizePolicy1.setHeightForWidth(self.plot_right.sizePolicy().hasHeightForWidth())
        self.plot_right.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.plot_right)

        self.splitter.addWidget(self.layoutWidget1)

        self.verticalLayout_3.addWidget(self.splitter)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lbl_cur_en.setText(QCoreApplication.translate("Form", u"Current Energy:", None))
        self.lbl_aver_en.setText(QCoreApplication.translate("Form", u"Average Energy:", None))
        self.lbl_std_en.setText(QCoreApplication.translate("Form", u"std:", None))
        self.btn_start.setText(QCoreApplication.translate("Form", u"START", None))
        self.btn_pause.setText(QCoreApplication.translate("Form", u"PAUSE", None))
        self.btn_stop.setText(QCoreApplication.translate("Form", u"STOP", None))
        self.lbl_signal.setText(QCoreApplication.translate("Form", u"Power Meter Signal Shape", None))
        self.lbl_tune.setText(QCoreApplication.translate("Form", u"Power Meter Energy Tune", None))
    # retranslateUi

