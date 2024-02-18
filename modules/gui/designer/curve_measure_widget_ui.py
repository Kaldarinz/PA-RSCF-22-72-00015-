# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'curve_measure_widget.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..widgets import (MplCanvas, QuantSpinBox)
from . import qt_resources_rc

class Ui_Curve_measure_widget(object):
    def setupUi(self, Curve_measure_widget):
        if not Curve_measure_widget.objectName():
            Curve_measure_widget.setObjectName(u"Curve_measure_widget")
        Curve_measure_widget.resize(1117, 842)
        self.horizontalLayout_2 = QHBoxLayout(Curve_measure_widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.lbl_measure_settings = QLabel(Curve_measure_widget)
        self.lbl_measure_settings.setObjectName(u"lbl_measure_settings")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_measure_settings.sizePolicy().hasHeightForWidth())
        self.lbl_measure_settings.setSizePolicy(sizePolicy)
        font = QFont()
        font.setBold(True)
        self.lbl_measure_settings.setFont(font)
        self.lbl_measure_settings.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.lbl_measure_settings)

        self.lo_parameter = QVBoxLayout()
        self.lo_parameter.setObjectName(u"lo_parameter")
        self.grid_lo = QGridLayout()
        self.grid_lo.setObjectName(u"grid_lo")
        self.lbl_mode = QLabel(Curve_measure_widget)
        self.lbl_mode.setObjectName(u"lbl_mode")
        sizePolicy.setHeightForWidth(self.lbl_mode.sizePolicy().hasHeightForWidth())
        self.lbl_mode.setSizePolicy(sizePolicy)

        self.grid_lo.addWidget(self.lbl_mode, 0, 0, 1, 1)

        self.cb_mode = QComboBox(Curve_measure_widget)
        self.cb_mode.addItem("")
        self.cb_mode.addItem("")
        self.cb_mode.addItem("")
        self.cb_mode.setObjectName(u"cb_mode")
        sizePolicy1 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.cb_mode.sizePolicy().hasHeightForWidth())
        self.cb_mode.setSizePolicy(sizePolicy1)
        self.cb_mode.setMaximumSize(QSize(16777215, 16777215))

        self.grid_lo.addWidget(self.cb_mode, 0, 1, 1, 1)

        self.lbl_from = QLabel(Curve_measure_widget)
        self.lbl_from.setObjectName(u"lbl_from")
        sizePolicy.setHeightForWidth(self.lbl_from.sizePolicy().hasHeightForWidth())
        self.lbl_from.setSizePolicy(sizePolicy)

        self.grid_lo.addWidget(self.lbl_from, 1, 0, 1, 1)

        self.sb_from = QuantSpinBox(Curve_measure_widget)
        self.sb_from.setObjectName(u"sb_from")
        self.sb_from.setDecimals(0)
        self.sb_from.setMinimum(0.000000000000000)
        self.sb_from.setMaximum(10000.000000000000000)
        self.sb_from.setSingleStep(10.000000000000000)
        self.sb_from.setValue(740.000000000000000)

        self.grid_lo.addWidget(self.sb_from, 1, 1, 1, 1)

        self.lbl_to = QLabel(Curve_measure_widget)
        self.lbl_to.setObjectName(u"lbl_to")
        sizePolicy.setHeightForWidth(self.lbl_to.sizePolicy().hasHeightForWidth())
        self.lbl_to.setSizePolicy(sizePolicy)

        self.grid_lo.addWidget(self.lbl_to, 2, 0, 1, 1)

        self.sb_to = QuantSpinBox(Curve_measure_widget)
        self.sb_to.setObjectName(u"sb_to")
        self.sb_to.setDecimals(0)
        self.sb_to.setMinimum(0.000000000000000)
        self.sb_to.setMaximum(10000.000000000000000)
        self.sb_to.setSingleStep(10.000000000000000)
        self.sb_to.setValue(750.000000000000000)

        self.grid_lo.addWidget(self.sb_to, 2, 1, 1, 1)

        self.lbl_step = QLabel(Curve_measure_widget)
        self.lbl_step.setObjectName(u"lbl_step")
        sizePolicy.setHeightForWidth(self.lbl_step.sizePolicy().hasHeightForWidth())
        self.lbl_step.setSizePolicy(sizePolicy)

        self.grid_lo.addWidget(self.lbl_step, 3, 0, 1, 1)

        self.sb_step = QuantSpinBox(Curve_measure_widget)
        self.sb_step.setObjectName(u"sb_step")
        self.sb_step.setDecimals(0)
        self.sb_step.setMinimum(0.000000000000000)
        self.sb_step.setMaximum(1000.000000000000000)
        self.sb_step.setValue(10.000000000000000)

        self.grid_lo.addWidget(self.sb_step, 3, 1, 1, 1)


        self.lo_parameter.addLayout(self.grid_lo)

        self.btn_run = QPushButton(Curve_measure_widget)
        self.btn_run.setObjectName(u"btn_run")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.btn_run.sizePolicy().hasHeightForWidth())
        self.btn_run.setSizePolicy(sizePolicy2)
        self.btn_run.setFont(font)
        self.btn_run.setCheckable(False)

        self.lo_parameter.addWidget(self.btn_run)


        self.verticalLayout_2.addLayout(self.lo_parameter)

        self.line = QFrame(Curve_measure_widget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_2.addWidget(self.line)

        self.w_measure = QWidget(Curve_measure_widget)
        self.w_measure.setObjectName(u"w_measure")
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.w_measure.sizePolicy().hasHeightForWidth())
        self.w_measure.setSizePolicy(sizePolicy3)
        self.verticalLayout = QVBoxLayout(self.w_measure)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lo_measure_ctrls = QHBoxLayout()
        self.lo_measure_ctrls.setObjectName(u"lo_measure_ctrls")
        self.btn_measure = QPushButton(self.w_measure)
        self.btn_measure.setObjectName(u"btn_measure")
        self.btn_measure.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.btn_measure.sizePolicy().hasHeightForWidth())
        self.btn_measure.setSizePolicy(sizePolicy2)
        self.btn_measure.setFont(font)
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control-stop.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_measure.setIcon(icon)
        self.btn_measure.setCheckable(False)

        self.lo_measure_ctrls.addWidget(self.btn_measure)

        self.btn_restart = QPushButton(self.w_measure)
        self.btn_restart.setObjectName(u"btn_restart")
        self.btn_restart.setEnabled(False)
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.btn_restart.sizePolicy().hasHeightForWidth())
        self.btn_restart.setSizePolicy(sizePolicy4)
        self.btn_restart.setMinimumSize(QSize(30, 0))
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/arrow-circle-225-left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_restart.setIcon(icon1)

        self.lo_measure_ctrls.addWidget(self.btn_restart)

        self.btn_stop = QPushButton(self.w_measure)
        self.btn_stop.setObjectName(u"btn_stop")
        self.btn_stop.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.btn_stop.sizePolicy().hasHeightForWidth())
        self.btn_stop.setSizePolicy(sizePolicy4)
        self.btn_stop.setMinimumSize(QSize(30, 0))
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon2)

        self.lo_measure_ctrls.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.lo_measure_ctrls)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.lbl_measured_p = QLabel(self.w_measure)
        self.lbl_measured_p.setObjectName(u"lbl_measured_p")
        sizePolicy3.setHeightForWidth(self.lbl_measured_p.sizePolicy().hasHeightForWidth())
        self.lbl_measured_p.setSizePolicy(sizePolicy3)

        self.horizontalLayout_11.addWidget(self.lbl_measured_p)

        self.pb = QProgressBar(self.w_measure)
        self.pb.setObjectName(u"pb")
        sizePolicy2.setHeightForWidth(self.pb.sizePolicy().hasHeightForWidth())
        self.pb.setSizePolicy(sizePolicy2)
        self.pb.setMinimumSize(QSize(10, 0))
        self.pb.setValue(0)
        self.pb.setTextVisible(False)

        self.horizontalLayout_11.addWidget(self.pb)

        self.lbl_pb = QLabel(self.w_measure)
        self.lbl_pb.setObjectName(u"lbl_pb")
        sizePolicy3.setHeightForWidth(self.lbl_pb.sizePolicy().hasHeightForWidth())
        self.lbl_pb.setSizePolicy(sizePolicy3)
        self.lbl_pb.setLineWidth(0)
        self.lbl_pb.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.horizontalLayout_11.addWidget(self.lbl_pb)


        self.verticalLayout.addLayout(self.horizontalLayout_11)

        self.line_4 = QFrame(self.w_measure)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lbl_cur_param = QLabel(self.w_measure)
        self.lbl_cur_param.setObjectName(u"lbl_cur_param")
        self.lbl_cur_param.setFont(font)

        self.horizontalLayout.addWidget(self.lbl_cur_param)

        self.sb_cur_param = QuantSpinBox(self.w_measure)
        self.sb_cur_param.setObjectName(u"sb_cur_param")
        self.sb_cur_param.setMaximumSize(QSize(94, 16777215))
        self.sb_cur_param.setReadOnly(True)
        self.sb_cur_param.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_cur_param.setMaximum(10000.000000000000000)

        self.horizontalLayout.addWidget(self.sb_cur_param)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.lbl_cur_wl = QLabel(self.w_measure)
        self.lbl_cur_wl.setObjectName(u"lbl_cur_wl")
        sizePolicy.setHeightForWidth(self.lbl_cur_wl.sizePolicy().hasHeightForWidth())
        self.lbl_cur_wl.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setBold(False)
        self.lbl_cur_wl.setFont(font1)

        self.horizontalLayout_12.addWidget(self.lbl_cur_wl)

        self.sb_cur_wl = QuantSpinBox(self.w_measure)
        self.sb_cur_wl.setObjectName(u"sb_cur_wl")
        sizePolicy5 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.sb_cur_wl.sizePolicy().hasHeightForWidth())
        self.sb_cur_wl.setSizePolicy(sizePolicy5)
        self.sb_cur_wl.setMinimumSize(QSize(0, 0))
        self.sb_cur_wl.setMaximumSize(QSize(94, 16777215))
        self.sb_cur_wl.setBaseSize(QSize(0, 0))
        self.sb_cur_wl.setReadOnly(False)
        self.sb_cur_wl.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.sb_cur_wl.setDecimals(0)
        self.sb_cur_wl.setMaximum(10000.000000000000000)
        self.sb_cur_wl.setValue(740.000000000000000)

        self.horizontalLayout_12.addWidget(self.sb_cur_wl)


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
        sizePolicy.setHeightForWidth(self.lbl_sample_en.sizePolicy().hasHeightForWidth())
        self.lbl_sample_en.setSizePolicy(sizePolicy)

        self.horizontalLayout_4.addWidget(self.lbl_sample_en)

        self.le_sample_en = QLineEdit(self.w_measure)
        self.le_sample_en.setObjectName(u"le_sample_en")
        sizePolicy2.setHeightForWidth(self.le_sample_en.sizePolicy().hasHeightForWidth())
        self.le_sample_en.setSizePolicy(sizePolicy2)
        self.le_sample_en.setMinimumSize(QSize(0, 0))
        self.le_sample_en.setMaximumSize(QSize(94, 16777215))

        self.horizontalLayout_4.addWidget(self.le_sample_en)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.lbl_sample_sp = QLabel(self.w_measure)
        self.lbl_sample_sp.setObjectName(u"lbl_sample_sp")
        sizePolicy.setHeightForWidth(self.lbl_sample_sp.sizePolicy().hasHeightForWidth())
        self.lbl_sample_sp.setSizePolicy(sizePolicy)

        self.horizontalLayout_7.addWidget(self.lbl_sample_sp)

        self.sb_sample_sp = QuantSpinBox(self.w_measure)
        self.sb_sample_sp.setObjectName(u"sb_sample_sp")
        sizePolicy5.setHeightForWidth(self.sb_sample_sp.sizePolicy().hasHeightForWidth())
        self.sb_sample_sp.setSizePolicy(sizePolicy5)
        self.sb_sample_sp.setMinimumSize(QSize(0, 0))
        self.sb_sample_sp.setMaximumSize(QSize(94, 16777215))
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
        sizePolicy.setHeightForWidth(self.lbl_pm_en.sizePolicy().hasHeightForWidth())
        self.lbl_pm_en.setSizePolicy(sizePolicy)

        self.horizontalLayout_3.addWidget(self.lbl_pm_en)

        self.le_pm_en = QLineEdit(self.w_measure)
        self.le_pm_en.setObjectName(u"le_pm_en")
        sizePolicy1.setHeightForWidth(self.le_pm_en.sizePolicy().hasHeightForWidth())
        self.le_pm_en.setSizePolicy(sizePolicy1)
        self.le_pm_en.setMaximumSize(QSize(94, 16777215))

        self.horizontalLayout_3.addWidget(self.le_pm_en)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.lbl_pm_sp = QLabel(self.w_measure)
        self.lbl_pm_sp.setObjectName(u"lbl_pm_sp")
        sizePolicy.setHeightForWidth(self.lbl_pm_sp.sizePolicy().hasHeightForWidth())
        self.lbl_pm_sp.setSizePolicy(sizePolicy)

        self.horizontalLayout_6.addWidget(self.lbl_pm_sp)

        self.sb_pm_sp = QuantSpinBox(self.w_measure)
        self.sb_pm_sp.setObjectName(u"sb_pm_sp")
        sizePolicy5.setHeightForWidth(self.sb_pm_sp.sizePolicy().hasHeightForWidth())
        self.sb_pm_sp.setSizePolicy(sizePolicy5)
        self.sb_pm_sp.setMinimumSize(QSize(0, 0))
        self.sb_pm_sp.setMaximumSize(QSize(94, 16777215))
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
        self.lbl_aver.setEnabled(False)
        sizePolicy.setHeightForWidth(self.lbl_aver.sizePolicy().hasHeightForWidth())
        self.lbl_aver.setSizePolicy(sizePolicy)

        self.horizontalLayout_5.addWidget(self.lbl_aver)

        self.sb_aver = QSpinBox(self.w_measure)
        self.sb_aver.setObjectName(u"sb_aver")
        self.sb_aver.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.sb_aver.sizePolicy().hasHeightForWidth())
        self.sb_aver.setSizePolicy(sizePolicy2)
        self.sb_aver.setMinimumSize(QSize(0, 0))
        self.sb_aver.setMaximumSize(QSize(94, 16777215))
        self.sb_aver.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_aver.setMinimum(1)
        self.sb_aver.setMaximum(10)
        self.sb_aver.setSingleStep(1)
        self.sb_aver.setValue(1)

        self.horizontalLayout_5.addWidget(self.sb_aver)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.line_3 = QFrame(self.w_measure)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.verticalSpacer = QSpacerItem(17, 22, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.verticalLayout_2.addWidget(self.w_measure)


        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.line_2 = QFrame(Curve_measure_widget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.line_2)

        self.lo_measure = QSplitter(Curve_measure_widget)
        self.lo_measure.setObjectName(u"lo_measure")
        sizePolicy6 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.lo_measure.sizePolicy().hasHeightForWidth())
        self.lo_measure.setSizePolicy(sizePolicy6)
        self.lo_measure.setOrientation(Qt.Vertical)
        self.placeholder_pm_monitor = QWidget(self.lo_measure)
        self.placeholder_pm_monitor.setObjectName(u"placeholder_pm_monitor")
        sizePolicy7 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.placeholder_pm_monitor.sizePolicy().hasHeightForWidth())
        self.placeholder_pm_monitor.setSizePolicy(sizePolicy7)
        self.lo_measure.addWidget(self.placeholder_pm_monitor)
        self.plot_measurement = MplCanvas(self.lo_measure)
        self.plot_measurement.setObjectName(u"plot_measurement")
        sizePolicy6.setHeightForWidth(self.plot_measurement.sizePolicy().hasHeightForWidth())
        self.plot_measurement.setSizePolicy(sizePolicy6)
        self.plot_measurement.setMinimumSize(QSize(0, 0))
        self.lo_measure.addWidget(self.plot_measurement)

        self.horizontalLayout_2.addWidget(self.lo_measure)


        self.retranslateUi(Curve_measure_widget)

        QMetaObject.connectSlotsByName(Curve_measure_widget)
    # setupUi

    def retranslateUi(self, Curve_measure_widget):
        Curve_measure_widget.setWindowTitle(QCoreApplication.translate("Curve_measure_widget", u"Form", None))
        self.lbl_measure_settings.setText(QCoreApplication.translate("Curve_measure_widget", u"Set Parameter Settings", None))
#if QT_CONFIG(statustip)
        self.lbl_mode.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Choose an independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_mode.setText(QCoreApplication.translate("Curve_measure_widget", u"Parameter", None))
        self.cb_mode.setItemText(0, QCoreApplication.translate("Curve_measure_widget", u"Wavelength", None))
        self.cb_mode.setItemText(1, QCoreApplication.translate("Curve_measure_widget", u"Energy", None))
        self.cb_mode.setItemText(2, QCoreApplication.translate("Curve_measure_widget", u"Concentration", None))

#if QT_CONFIG(statustip)
        self.cb_mode.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Choose an independent variable", None))
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(statustip)
        self.lbl_from.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Start value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_from.setText(QCoreApplication.translate("Curve_measure_widget", u"From", None))
#if QT_CONFIG(statustip)
        self.sb_from.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Start value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_from.setSuffix(QCoreApplication.translate("Curve_measure_widget", u" nm", None))
#if QT_CONFIG(statustip)
        self.lbl_to.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"End value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_to.setText(QCoreApplication.translate("Curve_measure_widget", u"To", None))
#if QT_CONFIG(statustip)
        self.sb_to.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"End value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_to.setSuffix(QCoreApplication.translate("Curve_measure_widget", u" nm", None))
#if QT_CONFIG(statustip)
        self.lbl_step.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Step value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_step.setText(QCoreApplication.translate("Curve_measure_widget", u"Step", None))
#if QT_CONFIG(statustip)
        self.sb_step.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Step value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_step.setSuffix(QCoreApplication.translate("Curve_measure_widget", u" nm", None))
#if QT_CONFIG(statustip)
        self.btn_run.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_run.setText(QCoreApplication.translate("Curve_measure_widget", u"Set Parameter", None))
#if QT_CONFIG(statustip)
        self.btn_measure.setStatusTip(QCoreApplication.translate("Curve_measure_widget", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_measure.setText(QCoreApplication.translate("Curve_measure_widget", u"MEASURE", None))
        self.btn_restart.setText("")
        self.btn_stop.setText("")
        self.lbl_measured_p.setText(QCoreApplication.translate("Curve_measure_widget", u"Measured points", None))
        self.lbl_pb.setText(QCoreApplication.translate("Curve_measure_widget", u"0/0", None))
        self.lbl_cur_param.setText(QCoreApplication.translate("Curve_measure_widget", u"Set Parameter", None))
        self.lbl_cur_wl.setText(QCoreApplication.translate("Curve_measure_widget", u"Wavelength", None))
        self.sb_cur_wl.setSuffix(QCoreApplication.translate("Curve_measure_widget", u" nm", None))
        self.lbl_sample_en.setText(QCoreApplication.translate("Curve_measure_widget", u"Sample Energy", None))
        self.lbl_sample_sp.setText(QCoreApplication.translate("Curve_measure_widget", u"SetPoint", None))
        self.sb_sample_sp.setPrefix("")
        self.sb_sample_sp.setSuffix(QCoreApplication.translate("Curve_measure_widget", u" uJ", None))
        self.lbl_pm_en.setText(QCoreApplication.translate("Curve_measure_widget", u"Power Meter Energy", None))
        self.lbl_pm_sp.setText(QCoreApplication.translate("Curve_measure_widget", u"SetPoint", None))
        self.sb_pm_sp.setSuffix(QCoreApplication.translate("Curve_measure_widget", u" uJ", None))
        self.lbl_aver.setText(QCoreApplication.translate("Curve_measure_widget", u"Averaging", None))
        self.sb_aver.setSuffix("")
        self.sb_aver.setPrefix("")
    # retranslateUi

