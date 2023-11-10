# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'curve_measure_widget.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QSplitter, QVBoxLayout, QWidget)

from ..widgets import (MplCanvas, QuantSpinBox)
from . import qt_resources_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1073, 732)
        self.verticalLayout_2 = QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btn_run = QPushButton(Form)
        self.btn_run.setObjectName(u"btn_run")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_run.sizePolicy().hasHeightForWidth())
        self.btn_run.setSizePolicy(sizePolicy)
        font = QFont()
        font.setBold(True)
        self.btn_run.setFont(font)
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_run.setIcon(icon)
        self.btn_run.setCheckable(True)

        self.horizontalLayout.addWidget(self.btn_run)

        self.line_3 = QFrame(Form)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_3)

        self.lbl_mode = QLabel(Form)
        self.lbl_mode.setObjectName(u"lbl_mode")

        self.horizontalLayout.addWidget(self.lbl_mode)

        self.cb_mode = QComboBox(Form)
        self.cb_mode.addItem("")
        self.cb_mode.addItem("")
        self.cb_mode.addItem("")
        self.cb_mode.setObjectName(u"cb_mode")

        self.horizontalLayout.addWidget(self.cb_mode)

        self.line_2 = QFrame(Form)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_2)

        self.lbl_from = QLabel(Form)
        self.lbl_from.setObjectName(u"lbl_from")

        self.horizontalLayout.addWidget(self.lbl_from)

        self.sb_from = QuantSpinBox(Form)
        self.sb_from.setObjectName(u"sb_from")
        self.sb_from.setDecimals(0)
        self.sb_from.setMinimum(500.000000000000000)
        self.sb_from.setMaximum(1000.000000000000000)
        self.sb_from.setSingleStep(10.000000000000000)
        self.sb_from.setValue(740.000000000000000)

        self.horizontalLayout.addWidget(self.sb_from)

        self.lbl_to = QLabel(Form)
        self.lbl_to.setObjectName(u"lbl_to")

        self.horizontalLayout.addWidget(self.lbl_to)

        self.sb_to = QuantSpinBox(Form)
        self.sb_to.setObjectName(u"sb_to")
        self.sb_to.setDecimals(0)
        self.sb_to.setMinimum(500.000000000000000)
        self.sb_to.setMaximum(1000.000000000000000)
        self.sb_to.setSingleStep(10.000000000000000)
        self.sb_to.setValue(750.000000000000000)

        self.horizontalLayout.addWidget(self.sb_to)

        self.line = QFrame(Form)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line)

        self.lbl_step = QLabel(Form)
        self.lbl_step.setObjectName(u"lbl_step")

        self.horizontalLayout.addWidget(self.lbl_step)

        self.sb_step = QuantSpinBox(Form)
        self.sb_step.setObjectName(u"sb_step")
        self.sb_step.setDecimals(0)
        self.sb_step.setMinimum(1.000000000000000)
        self.sb_step.setValue(10.000000000000000)

        self.horizontalLayout.addWidget(self.sb_step)

        self.line_7 = QFrame(Form)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.VLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_7)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.lo_measure = QHBoxLayout(self.layoutWidget)
        self.lo_measure.setObjectName(u"lo_measure")
        self.lo_measure.setContentsMargins(0, 0, 0, 0)
        self.w_measure = QWidget(self.layoutWidget)
        self.w_measure.setObjectName(u"w_measure")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.w_measure.sizePolicy().hasHeightForWidth())
        self.w_measure.setSizePolicy(sizePolicy1)
        self.verticalLayout = QVBoxLayout(self.w_measure)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btn_measure = QPushButton(self.w_measure)
        self.btn_measure.setObjectName(u"btn_measure")
        sizePolicy.setHeightForWidth(self.btn_measure.sizePolicy().hasHeightForWidth())
        self.btn_measure.setSizePolicy(sizePolicy)
        self.btn_measure.setFont(font)
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/control-stop.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_measure.setIcon(icon1)
        self.btn_measure.setCheckable(False)

        self.horizontalLayout_2.addWidget(self.btn_measure)

        self.btn_restart = QPushButton(self.w_measure)
        self.btn_restart.setObjectName(u"btn_restart")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.btn_restart.sizePolicy().hasHeightForWidth())
        self.btn_restart.setSizePolicy(sizePolicy2)
        self.btn_restart.setMinimumSize(QSize(30, 0))
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/arrow-circle-225-left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_restart.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.btn_restart)

        self.btn_stop = QPushButton(self.w_measure)
        self.btn_stop.setObjectName(u"btn_stop")
        sizePolicy2.setHeightForWidth(self.btn_stop.sizePolicy().hasHeightForWidth())
        self.btn_stop.setSizePolicy(sizePolicy2)
        self.btn_stop.setMinimumSize(QSize(30, 0))
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon3)

        self.horizontalLayout_2.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.lbl_measured_p = QLabel(self.w_measure)
        self.lbl_measured_p.setObjectName(u"lbl_measured_p")

        self.horizontalLayout_11.addWidget(self.lbl_measured_p)

        self.pb = QProgressBar(self.w_measure)
        self.pb.setObjectName(u"pb")
        sizePolicy.setHeightForWidth(self.pb.sizePolicy().hasHeightForWidth())
        self.pb.setSizePolicy(sizePolicy)
        self.pb.setMinimumSize(QSize(10, 0))
        self.pb.setValue(0)
        self.pb.setTextVisible(False)

        self.horizontalLayout_11.addWidget(self.pb)

        self.lbl_pb = QLabel(self.w_measure)
        self.lbl_pb.setObjectName(u"lbl_pb")
        self.lbl_pb.setLineWidth(0)
        self.lbl_pb.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.horizontalLayout_11.addWidget(self.lbl_pb)


        self.verticalLayout.addLayout(self.horizontalLayout_11)

        self.line_4 = QFrame(self.w_measure)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.lbl_cur_param = QLabel(self.w_measure)
        self.lbl_cur_param.setObjectName(u"lbl_cur_param")
        self.lbl_cur_param.setFont(font)

        self.horizontalLayout_12.addWidget(self.lbl_cur_param)

        self.sb_cur_param = QuantSpinBox(self.w_measure)
        self.sb_cur_param.setObjectName(u"sb_cur_param")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.sb_cur_param.sizePolicy().hasHeightForWidth())
        self.sb_cur_param.setSizePolicy(sizePolicy3)
        self.sb_cur_param.setMinimumSize(QSize(71, 0))
        self.sb_cur_param.setMaximumSize(QSize(71, 16777215))
        self.sb_cur_param.setBaseSize(QSize(0, 0))
        self.sb_cur_param.setReadOnly(True)
        self.sb_cur_param.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_cur_param.setDecimals(0)

        self.horizontalLayout_12.addWidget(self.sb_cur_param)


        self.verticalLayout.addLayout(self.horizontalLayout_12)

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

        self.le_sample_en = QLineEdit(self.w_measure)
        self.le_sample_en.setObjectName(u"le_sample_en")
        sizePolicy2.setHeightForWidth(self.le_sample_en.sizePolicy().hasHeightForWidth())
        self.le_sample_en.setSizePolicy(sizePolicy2)
        self.le_sample_en.setMinimumSize(QSize(71, 0))
        self.le_sample_en.setMaximumSize(QSize(71, 16777215))

        self.horizontalLayout_4.addWidget(self.le_sample_en)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.lbl_sample_sp = QLabel(self.w_measure)
        self.lbl_sample_sp.setObjectName(u"lbl_sample_sp")

        self.horizontalLayout_7.addWidget(self.lbl_sample_sp)

        self.sb_sample_sp = QuantSpinBox(self.w_measure)
        self.sb_sample_sp.setObjectName(u"sb_sample_sp")
        sizePolicy3.setHeightForWidth(self.sb_sample_sp.sizePolicy().hasHeightForWidth())
        self.sb_sample_sp.setSizePolicy(sizePolicy3)
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

        self.le_pm_en = QLineEdit(self.w_measure)
        self.le_pm_en.setObjectName(u"le_pm_en")
        sizePolicy3.setHeightForWidth(self.le_pm_en.sizePolicy().hasHeightForWidth())
        self.le_pm_en.setSizePolicy(sizePolicy3)
        self.le_pm_en.setMaximumSize(QSize(71, 16777215))

        self.horizontalLayout_3.addWidget(self.le_pm_en)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.lbl_pm_sp = QLabel(self.w_measure)
        self.lbl_pm_sp.setObjectName(u"lbl_pm_sp")

        self.horizontalLayout_6.addWidget(self.lbl_pm_sp)

        self.sb_pm_sp = QuantSpinBox(self.w_measure)
        self.sb_pm_sp.setObjectName(u"sb_pm_sp")
        sizePolicy3.setHeightForWidth(self.sb_pm_sp.sizePolicy().hasHeightForWidth())
        self.sb_pm_sp.setSizePolicy(sizePolicy3)
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
        sizePolicy3.setHeightForWidth(self.sb_aver.sizePolicy().hasHeightForWidth())
        self.sb_aver.setSizePolicy(sizePolicy3)
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

        self.placeholder_pm_monitor = QWidget(self.layoutWidget)
        self.placeholder_pm_monitor.setObjectName(u"placeholder_pm_monitor")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.placeholder_pm_monitor.sizePolicy().hasHeightForWidth())
        self.placeholder_pm_monitor.setSizePolicy(sizePolicy4)

        self.lo_measure.addWidget(self.placeholder_pm_monitor)

        self.splitter.addWidget(self.layoutWidget)
        self.plot_measurement = MplCanvas(self.splitter)
        self.plot_measurement.setObjectName(u"plot_measurement")
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.plot_measurement.sizePolicy().hasHeightForWidth())
        self.plot_measurement.setSizePolicy(sizePolicy5)
        self.plot_measurement.setMinimumSize(QSize(0, 0))
        self.splitter.addWidget(self.plot_measurement)

        self.verticalLayout_2.addWidget(self.splitter)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
#if QT_CONFIG(statustip)
        self.btn_run.setStatusTip(QCoreApplication.translate("Form", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_run.setText(QCoreApplication.translate("Form", u"RUN", None))
#if QT_CONFIG(statustip)
        self.lbl_mode.setStatusTip(QCoreApplication.translate("Form", u"Choose an independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_mode.setText(QCoreApplication.translate("Form", u"Mode", None))
        self.cb_mode.setItemText(0, QCoreApplication.translate("Form", u"Wavelength", None))
        self.cb_mode.setItemText(1, QCoreApplication.translate("Form", u"Energy", None))
        self.cb_mode.setItemText(2, QCoreApplication.translate("Form", u"Concentration", None))

#if QT_CONFIG(statustip)
        self.cb_mode.setStatusTip(QCoreApplication.translate("Form", u"Choose an independent variable", None))
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(statustip)
        self.lbl_from.setStatusTip(QCoreApplication.translate("Form", u"Start value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_from.setText(QCoreApplication.translate("Form", u"From", None))
#if QT_CONFIG(statustip)
        self.sb_from.setStatusTip(QCoreApplication.translate("Form", u"Start value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_from.setSuffix(QCoreApplication.translate("Form", u" nm", None))
#if QT_CONFIG(statustip)
        self.lbl_to.setStatusTip(QCoreApplication.translate("Form", u"End value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_to.setText(QCoreApplication.translate("Form", u"To", None))
#if QT_CONFIG(statustip)
        self.sb_to.setStatusTip(QCoreApplication.translate("Form", u"End value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_to.setSuffix(QCoreApplication.translate("Form", u" nm", None))
#if QT_CONFIG(statustip)
        self.lbl_step.setStatusTip(QCoreApplication.translate("Form", u"Step value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_step.setText(QCoreApplication.translate("Form", u"Step", None))
#if QT_CONFIG(statustip)
        self.sb_step.setStatusTip(QCoreApplication.translate("Form", u"Step value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_step.setSuffix(QCoreApplication.translate("Form", u" nm", None))
#if QT_CONFIG(statustip)
        self.btn_measure.setStatusTip(QCoreApplication.translate("Form", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_measure.setText(QCoreApplication.translate("Form", u"MEASURE", None))
        self.btn_restart.setText("")
        self.btn_stop.setText("")
        self.lbl_measured_p.setText(QCoreApplication.translate("Form", u"Measured points", None))
        self.lbl_pb.setText("")
        self.lbl_cur_param.setText(QCoreApplication.translate("Form", u"Set Wavelength", None))
        self.lbl_sample_en.setText(QCoreApplication.translate("Form", u"Sample Energy", None))
        self.lbl_sample_sp.setText(QCoreApplication.translate("Form", u"SetPoint", None))
        self.sb_sample_sp.setPrefix("")
        self.sb_sample_sp.setSuffix(QCoreApplication.translate("Form", u" uJ", None))
        self.lbl_pm_en.setText(QCoreApplication.translate("Form", u"Power Meter Energy", None))
        self.lbl_pm_sp.setText(QCoreApplication.translate("Form", u"SetPoint", None))
        self.sb_pm_sp.setSuffix(QCoreApplication.translate("Form", u" uJ", None))
        self.lbl_aver.setText(QCoreApplication.translate("Form", u"Averaging", None))
        self.sb_aver.setSuffix("")
        self.sb_aver.setPrefix("")
    # retranslateUi

