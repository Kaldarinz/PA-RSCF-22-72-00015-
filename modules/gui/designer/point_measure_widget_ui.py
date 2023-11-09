# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'point_measure_widget.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QFrame, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QVBoxLayout, QWidget)

from ..widgets import QuantSpinBox
from . import qt_resources_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1073, 732)
        self.verticalLayout_2 = QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.lbl_cur_param = QLabel(Form)
        self.lbl_cur_param.setObjectName(u"lbl_cur_param")
        font = QFont()
        font.setBold(True)
        self.lbl_cur_param.setFont(font)

        self.horizontalLayout_12.addWidget(self.lbl_cur_param)

        self.sb_cur_param = QuantSpinBox(Form)
        self.sb_cur_param.setObjectName(u"sb_cur_param")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sb_cur_param.sizePolicy().hasHeightForWidth())
        self.sb_cur_param.setSizePolicy(sizePolicy)
        self.sb_cur_param.setMinimumSize(QSize(71, 0))
        self.sb_cur_param.setMaximumSize(QSize(71, 16777215))
        self.sb_cur_param.setBaseSize(QSize(0, 0))
        self.sb_cur_param.setReadOnly(False)
        self.sb_cur_param.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_cur_param.setDecimals(0)

        self.horizontalLayout_12.addWidget(self.sb_cur_param)


        self.horizontalLayout.addLayout(self.horizontalLayout_12)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.lo_measure = QHBoxLayout()
        self.lo_measure.setObjectName(u"lo_measure")
        self.w_measure = QWidget(Form)
        self.w_measure.setObjectName(u"w_measure")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.w_measure.sizePolicy().hasHeightForWidth())
        self.w_measure.setSizePolicy(sizePolicy1)
        self.verticalLayout = QVBoxLayout(self.w_measure)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.btn_measure = QPushButton(self.w_measure)
        self.btn_measure.setObjectName(u"btn_measure")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.btn_measure.sizePolicy().hasHeightForWidth())
        self.btn_measure.setSizePolicy(sizePolicy2)
        self.btn_measure.setFont(font)
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control-stop.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_measure.setIcon(icon)
        self.btn_measure.setCheckable(False)

        self.verticalLayout.addWidget(self.btn_measure)

        self.line_8 = QFrame(self.w_measure)
        self.line_8.setObjectName(u"line_8")
        self.line_8.setFrameShape(QFrame.HLine)
        self.line_8.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_8)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.lbl_sample_en = QLabel(self.w_measure)
        self.lbl_sample_en.setObjectName(u"lbl_sample_en")

        self.horizontalLayout_4.addWidget(self.lbl_sample_en)

        self.sb_sample_en = QuantSpinBox(self.w_measure)
        self.sb_sample_en.setObjectName(u"sb_sample_en")
        sizePolicy.setHeightForWidth(self.sb_sample_en.sizePolicy().hasHeightForWidth())
        self.sb_sample_en.setSizePolicy(sizePolicy)
        self.sb_sample_en.setMinimumSize(QSize(71, 0))
        self.sb_sample_en.setMaximumSize(QSize(71, 16777215))
        self.sb_sample_en.setReadOnly(True)
        self.sb_sample_en.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_sample_en.setDecimals(1)
        self.sb_sample_en.setMinimum(0.000000000000000)
        self.sb_sample_en.setMaximum(2000.000000000000000)
        self.sb_sample_en.setSingleStep(50.000000000000000)
        self.sb_sample_en.setValue(0.000000000000000)

        self.horizontalLayout_4.addWidget(self.sb_sample_en)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.lbl_sample_sp = QLabel(self.w_measure)
        self.lbl_sample_sp.setObjectName(u"lbl_sample_sp")

        self.horizontalLayout_7.addWidget(self.lbl_sample_sp)

        self.sb_sample_sp = QuantSpinBox(self.w_measure)
        self.sb_sample_sp.setObjectName(u"sb_sample_sp")
        sizePolicy.setHeightForWidth(self.sb_sample_sp.sizePolicy().hasHeightForWidth())
        self.sb_sample_sp.setSizePolicy(sizePolicy)
        self.sb_sample_sp.setMinimumSize(QSize(71, 0))
        self.sb_sample_sp.setMaximumSize(QSize(71, 16777215))
        self.sb_sample_sp.setDecimals(1)
        self.sb_sample_sp.setMinimum(0.000000000000000)
        self.sb_sample_sp.setMaximum(5000.000000000000000)
        self.sb_sample_sp.setSingleStep(50.000000000000000)
        self.sb_sample_sp.setValue(500.000000000000000)

        self.horizontalLayout_7.addWidget(self.sb_sample_sp)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.line_6 = QFrame(self.w_measure)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShape(QFrame.HLine)
        self.line_6.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_6)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.lbl_pm_en = QLabel(self.w_measure)
        self.lbl_pm_en.setObjectName(u"lbl_pm_en")

        self.horizontalLayout_3.addWidget(self.lbl_pm_en)

        self.sb_pm_en = QuantSpinBox(self.w_measure)
        self.sb_pm_en.setObjectName(u"sb_pm_en")
        sizePolicy.setHeightForWidth(self.sb_pm_en.sizePolicy().hasHeightForWidth())
        self.sb_pm_en.setSizePolicy(sizePolicy)
        self.sb_pm_en.setMinimumSize(QSize(71, 0))
        self.sb_pm_en.setMaximumSize(QSize(71, 16777215))
        self.sb_pm_en.setReadOnly(True)
        self.sb_pm_en.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_pm_en.setDecimals(1)
        self.sb_pm_en.setMinimum(0.000000000000000)
        self.sb_pm_en.setMaximum(2000.000000000000000)
        self.sb_pm_en.setValue(0.000000000000000)

        self.horizontalLayout_3.addWidget(self.sb_pm_en)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.lbl_pm_sp = QLabel(self.w_measure)
        self.lbl_pm_sp.setObjectName(u"lbl_pm_sp")

        self.horizontalLayout_6.addWidget(self.lbl_pm_sp)

        self.sb_pm_sp = QuantSpinBox(self.w_measure)
        self.sb_pm_sp.setObjectName(u"sb_pm_sp")
        sizePolicy.setHeightForWidth(self.sb_pm_sp.sizePolicy().hasHeightForWidth())
        self.sb_pm_sp.setSizePolicy(sizePolicy)
        self.sb_pm_sp.setMinimumSize(QSize(71, 0))
        self.sb_pm_sp.setMaximumSize(QSize(71, 16777215))
        self.sb_pm_sp.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.sb_pm_sp.setDecimals(1)
        self.sb_pm_sp.setMinimum(0.000000000000000)
        self.sb_pm_sp.setMaximum(600.000000000000000)
        self.sb_pm_sp.setSingleStep(1.000000000000000)
        self.sb_pm_sp.setValue(500.000000000000000)

        self.horizontalLayout_6.addWidget(self.sb_pm_sp)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.line_5 = QFrame(self.w_measure)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.HLine)
        self.line_5.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_5)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.lbl_aver = QLabel(self.w_measure)
        self.lbl_aver.setObjectName(u"lbl_aver")

        self.horizontalLayout_5.addWidget(self.lbl_aver)

        self.sb_aver = QSpinBox(self.w_measure)
        self.sb_aver.setObjectName(u"sb_aver")
        self.sb_aver.setEnabled(False)
        sizePolicy.setHeightForWidth(self.sb_aver.sizePolicy().hasHeightForWidth())
        self.sb_aver.setSizePolicy(sizePolicy)
        self.sb_aver.setMinimumSize(QSize(71, 0))
        self.sb_aver.setMaximumSize(QSize(43, 16777215))
        self.sb_aver.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_aver.setMinimum(1)
        self.sb_aver.setMaximum(10)
        self.sb_aver.setSingleStep(1)
        self.sb_aver.setValue(1)

        self.horizontalLayout_5.addWidget(self.sb_aver)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.verticalSpacer = QSpacerItem(17, 22, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.lo_measure.addWidget(self.w_measure)

        self.placeholder_pm_monitor = QWidget(Form)
        self.placeholder_pm_monitor.setObjectName(u"placeholder_pm_monitor")

        self.lo_measure.addWidget(self.placeholder_pm_monitor)


        self.verticalLayout_2.addLayout(self.lo_measure)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lbl_cur_param.setText(QCoreApplication.translate("Form", u"Set Wavelength", None))
#if QT_CONFIG(statustip)
        self.btn_measure.setStatusTip(QCoreApplication.translate("Form", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_measure.setText(QCoreApplication.translate("Form", u"MEASURE", None))
        self.lbl_sample_en.setText(QCoreApplication.translate("Form", u"Sample Energy", None))
        self.sb_sample_en.setSuffix(QCoreApplication.translate("Form", u" uJ", None))
        self.lbl_sample_sp.setText(QCoreApplication.translate("Form", u"SetPoint", None))
        self.sb_sample_sp.setPrefix("")
        self.sb_sample_sp.setSuffix(QCoreApplication.translate("Form", u" uJ", None))
        self.lbl_pm_en.setText(QCoreApplication.translate("Form", u"Power Meter Energy", None))
        self.sb_pm_en.setSuffix(QCoreApplication.translate("Form", u" uJ", None))
        self.lbl_pm_sp.setText(QCoreApplication.translate("Form", u"SetPoint", None))
        self.sb_pm_sp.setSuffix(QCoreApplication.translate("Form", u" uJ", None))
        self.lbl_aver.setText(QCoreApplication.translate("Form", u"Averaging", None))
        self.sb_aver.setSuffix("")
        self.sb_aver.setPrefix("")
    # retranslateUi

