# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'map_measure_widget.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLayout, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

from ..widgets import (PgMap, QuantSpinBox)
from . import qt_resources_rc

class Ui_map_measure(object):
    def setupUi(self, map_measure):
        if not map_measure.objectName():
            map_measure.setObjectName(u"map_measure")
        map_measure.resize(1269, 907)
        self.horizontalLayout_6 = QHBoxLayout(map_measure)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.lbl_scanarea = QLabel(map_measure)
        self.lbl_scanarea.setObjectName(u"lbl_scanarea")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_scanarea.sizePolicy().hasHeightForWidth())
        self.lbl_scanarea.setSizePolicy(sizePolicy)
        font = QFont()
        font.setBold(True)
        self.lbl_scanarea.setFont(font)
        self.lbl_scanarea.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_scanarea)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.sb_sizeX = QuantSpinBox(map_measure)
        self.sb_sizeX.setObjectName(u"sb_sizeX")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.sb_sizeX.sizePolicy().hasHeightForWidth())
        self.sb_sizeX.setSizePolicy(sizePolicy1)
        self.sb_sizeX.setMinimumSize(QSize(0, 0))
        self.sb_sizeX.setDecimals(2)
        self.sb_sizeX.setMaximum(25.000000000000000)
        self.sb_sizeX.setValue(4.000000000000000)

        self.gridLayout.addWidget(self.sb_sizeX, 1, 2, 1, 1)

        self.lbl_points = QLabel(map_measure)
        self.lbl_points.setObjectName(u"lbl_points")

        self.gridLayout.addWidget(self.lbl_points, 0, 3, 1, 1)

        self.lbl_areaY = QLabel(map_measure)
        self.lbl_areaY.setObjectName(u"lbl_areaY")
        self.lbl_areaY.setFont(font)

        self.gridLayout.addWidget(self.lbl_areaY, 2, 0, 1, 1)

        self.lbl_step = QLabel(map_measure)
        self.lbl_step.setObjectName(u"lbl_step")

        self.gridLayout.addWidget(self.lbl_step, 0, 4, 1, 1)

        self.lbl_size = QLabel(map_measure)
        self.lbl_size.setObjectName(u"lbl_size")

        self.gridLayout.addWidget(self.lbl_size, 0, 2, 1, 1)

        self.sb_stepX = QuantSpinBox(map_measure)
        self.sb_stepX.setObjectName(u"sb_stepX")
        sizePolicy1.setHeightForWidth(self.sb_stepX.sizePolicy().hasHeightForWidth())
        self.sb_stepX.setSizePolicy(sizePolicy1)
        self.sb_stepX.setMinimumSize(QSize(0, 0))
        self.sb_stepX.setReadOnly(True)
        self.sb_stepX.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_stepX.setDecimals(2)
        self.sb_stepX.setMaximum(999.990000000000009)

        self.gridLayout.addWidget(self.sb_stepX, 1, 4, 1, 1)

        self.sb_sizeY = QuantSpinBox(map_measure)
        self.sb_sizeY.setObjectName(u"sb_sizeY")
        sizePolicy1.setHeightForWidth(self.sb_sizeY.sizePolicy().hasHeightForWidth())
        self.sb_sizeY.setSizePolicy(sizePolicy1)
        self.sb_sizeY.setMinimumSize(QSize(0, 0))
        self.sb_sizeY.setDecimals(2)
        self.sb_sizeY.setMaximum(25.000000000000000)
        self.sb_sizeY.setValue(4.000000000000000)

        self.gridLayout.addWidget(self.sb_sizeY, 2, 2, 1, 1)

        self.sb_stepY = QuantSpinBox(map_measure)
        self.sb_stepY.setObjectName(u"sb_stepY")
        sizePolicy1.setHeightForWidth(self.sb_stepY.sizePolicy().hasHeightForWidth())
        self.sb_stepY.setSizePolicy(sizePolicy1)
        self.sb_stepY.setMinimumSize(QSize(0, 0))
        self.sb_stepY.setReadOnly(True)
        self.sb_stepY.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_stepY.setDecimals(2)
        self.sb_stepY.setMaximum(999.990000000000009)

        self.gridLayout.addWidget(self.sb_stepY, 2, 4, 1, 1)

        self.sb_centerX = QuantSpinBox(map_measure)
        self.sb_centerX.setObjectName(u"sb_centerX")
        sizePolicy1.setHeightForWidth(self.sb_centerX.sizePolicy().hasHeightForWidth())
        self.sb_centerX.setSizePolicy(sizePolicy1)
        self.sb_centerX.setMinimumSize(QSize(0, 0))
        self.sb_centerX.setDecimals(2)
        self.sb_centerX.setMaximum(25.000000000000000)
        self.sb_centerX.setValue(12.500000000000000)

        self.gridLayout.addWidget(self.sb_centerX, 1, 1, 1, 1)

        self.sb_centerY = QuantSpinBox(map_measure)
        self.sb_centerY.setObjectName(u"sb_centerY")
        sizePolicy1.setHeightForWidth(self.sb_centerY.sizePolicy().hasHeightForWidth())
        self.sb_centerY.setSizePolicy(sizePolicy1)
        self.sb_centerY.setMinimumSize(QSize(0, 0))
        self.sb_centerY.setDecimals(2)
        self.sb_centerY.setMaximum(25.000000000000000)
        self.sb_centerY.setValue(12.500000000000000)

        self.gridLayout.addWidget(self.sb_centerY, 2, 1, 1, 1)

        self.lbl_center = QLabel(map_measure)
        self.lbl_center.setObjectName(u"lbl_center")

        self.gridLayout.addWidget(self.lbl_center, 0, 1, 1, 1)

        self.lbl_areaX = QLabel(map_measure)
        self.lbl_areaX.setObjectName(u"lbl_areaX")
        self.lbl_areaX.setFont(font)

        self.gridLayout.addWidget(self.lbl_areaX, 1, 0, 1, 1)

        self.sb_pointsX = QuantSpinBox(map_measure)
        self.sb_pointsX.setObjectName(u"sb_pointsX")
        sizePolicy1.setHeightForWidth(self.sb_pointsX.sizePolicy().hasHeightForWidth())
        self.sb_pointsX.setSizePolicy(sizePolicy1)
        self.sb_pointsX.setMinimumSize(QSize(68, 0))
        self.sb_pointsX.setDecimals(0)
        self.sb_pointsX.setMinimum(2.000000000000000)
        self.sb_pointsX.setMaximum(1024.000000000000000)
        self.sb_pointsX.setValue(10.000000000000000)

        self.gridLayout.addWidget(self.sb_pointsX, 1, 3, 1, 1)

        self.sb_pointsY = QuantSpinBox(map_measure)
        self.sb_pointsY.setObjectName(u"sb_pointsY")
        sizePolicy1.setHeightForWidth(self.sb_pointsY.sizePolicy().hasHeightForWidth())
        self.sb_pointsY.setSizePolicy(sizePolicy1)
        self.sb_pointsY.setMinimumSize(QSize(0, 0))
        self.sb_pointsY.setDecimals(0)
        self.sb_pointsY.setMinimum(2.000000000000000)
        self.sb_pointsY.setMaximum(1024.000000000000000)
        self.sb_pointsY.setValue(10.000000000000000)

        self.gridLayout.addWidget(self.sb_pointsY, 2, 3, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_2 = QLabel(map_measure)
        self.label_2.setObjectName(u"label_2")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)

        self.horizontalLayout_4.addWidget(self.label_2)

        self.btn_plot_full_area = QPushButton(map_measure)
        self.btn_plot_full_area.setObjectName(u"btn_plot_full_area")
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.btn_plot_full_area.sizePolicy().hasHeightForWidth())
        self.btn_plot_full_area.setSizePolicy(sizePolicy3)
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/map.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_plot_full_area.setIcon(icon)

        self.horizontalLayout_4.addWidget(self.btn_plot_full_area)

        self.btn_plot_clear = QPushButton(map_measure)
        self.btn_plot_clear.setObjectName(u"btn_plot_clear")
        sizePolicy3.setHeightForWidth(self.btn_plot_clear.sizePolicy().hasHeightForWidth())
        self.btn_plot_clear.setSizePolicy(sizePolicy3)
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/map--minus.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_plot_clear.setIcon(icon1)

        self.horizontalLayout_4.addWidget(self.btn_plot_clear)

        self.btn_plot_fit = QPushButton(map_measure)
        self.btn_plot_fit.setObjectName(u"btn_plot_fit")
        sizePolicy3.setHeightForWidth(self.btn_plot_fit.sizePolicy().hasHeightForWidth())
        self.btn_plot_fit.setSizePolicy(sizePolicy3)
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/map-resize.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_plot_fit.setIcon(icon2)

        self.horizontalLayout_4.addWidget(self.btn_plot_fit)

        self.line_4 = QFrame(map_measure)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.VLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_4.addWidget(self.line_4)

        self.btn_plot_area_sel = QPushButton(map_measure)
        self.btn_plot_area_sel.setObjectName(u"btn_plot_area_sel")
        sizePolicy3.setHeightForWidth(self.btn_plot_area_sel.sizePolicy().hasHeightForWidth())
        self.btn_plot_area_sel.setSizePolicy(sizePolicy3)
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/selection-resize.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_plot_area_sel.setIcon(icon3)
        self.btn_plot_area_sel.setCheckable(True)
        self.btn_plot_area_sel.setChecked(True)
        self.btn_plot_area_sel.setFlat(False)

        self.horizontalLayout_4.addWidget(self.btn_plot_area_sel)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.line = QFrame(map_measure)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, -1, -1, -1)
        self.btn_start = QPushButton(map_measure)
        self.btn_start.setObjectName(u"btn_start")
        self.btn_start.setFont(font)
        icon4 = QIcon()
        icon4.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_start.setIcon(icon4)
        self.btn_start.setCheckable(True)

        self.horizontalLayout.addWidget(self.btn_start)

        self.btn_pause = QPushButton(map_measure)
        self.btn_pause.setObjectName(u"btn_pause")
        icon5 = QIcon()
        icon5.addFile(u":/icons/qt_resources/control-pause.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_pause.setIcon(icon5)
        self.btn_pause.setCheckable(True)

        self.horizontalLayout.addWidget(self.btn_pause)

        self.btn_restart = QPushButton(map_measure)
        self.btn_restart.setObjectName(u"btn_restart")
        icon6 = QIcon()
        icon6.addFile(u":/icons/qt_resources/arrow-circle-225-left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_restart.setIcon(icon6)

        self.horizontalLayout.addWidget(self.btn_restart)

        self.btn_stop = QPushButton(map_measure)
        self.btn_stop.setObjectName(u"btn_stop")
        icon7 = QIcon()
        icon7.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon7)

        self.horizontalLayout.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lbl_estdur = QLabel(map_measure)
        self.lbl_estdur.setObjectName(u"lbl_estdur")

        self.horizontalLayout_2.addWidget(self.lbl_estdur)

        self.le_estdur = QLineEdit(map_measure)
        self.le_estdur.setObjectName(u"le_estdur")
        self.le_estdur.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.le_estdur.sizePolicy().hasHeightForWidth())
        self.le_estdur.setSizePolicy(sizePolicy1)
        self.le_estdur.setMaximumSize(QSize(100, 16777215))
        self.le_estdur.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.le_estdur)

        self.lbl_dur = QLabel(map_measure)
        self.lbl_dur.setObjectName(u"lbl_dur")

        self.horizontalLayout_2.addWidget(self.lbl_dur)

        self.le_dur = QLineEdit(map_measure)
        self.le_dur.setObjectName(u"le_dur")
        self.le_dur.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.le_dur.sizePolicy().hasHeightForWidth())
        self.le_dur.setSizePolicy(sizePolicy1)
        self.le_dur.setMaximumSize(QSize(100, 16777215))
        self.le_dur.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.le_dur)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.line_3 = QFrame(map_measure)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.lbl_params = QLabel(map_measure)
        self.lbl_params.setObjectName(u"lbl_params")
        sizePolicy.setHeightForWidth(self.lbl_params.sizePolicy().hasHeightForWidth())
        self.lbl_params.setSizePolicy(sizePolicy)
        self.lbl_params.setFont(font)
        self.lbl_params.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_params)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.lbl_speed = QLabel(map_measure)
        self.lbl_speed.setObjectName(u"lbl_speed")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.lbl_speed)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.sb_speed = QuantSpinBox(map_measure)
        self.sb_speed.setObjectName(u"sb_speed")
        sizePolicy3.setHeightForWidth(self.sb_speed.sizePolicy().hasHeightForWidth())
        self.sb_speed.setSizePolicy(sizePolicy3)
        self.sb_speed.setMinimumSize(QSize(95, 0))
        self.sb_speed.setDecimals(2)
        self.sb_speed.setMaximum(40.000000000000000)
        self.sb_speed.setValue(2.000000000000000)

        self.horizontalLayout_5.addWidget(self.sb_speed)

        self.label = QLabel(map_measure)
        self.label.setObjectName(u"label")
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        self.label.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_5.addWidget(self.label)

        self.sb_cur_speed = QuantSpinBox(map_measure)
        self.sb_cur_speed.setObjectName(u"sb_cur_speed")
        sizePolicy1.setHeightForWidth(self.sb_cur_speed.sizePolicy().hasHeightForWidth())
        self.sb_cur_speed.setSizePolicy(sizePolicy1)
        self.sb_cur_speed.setMinimumSize(QSize(50, 0))
        self.sb_cur_speed.setFrame(True)
        self.sb_cur_speed.setReadOnly(True)
        self.sb_cur_speed.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_cur_speed.setMaximum(999.990000000000009)
        self.sb_cur_speed.setValue(2.000000000000000)

        self.horizontalLayout_5.addWidget(self.sb_cur_speed)


        self.formLayout.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout_5)

        self.lbl_astep = QLabel(map_measure)
        self.lbl_astep.setObjectName(u"lbl_astep")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.lbl_astep)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.chb_astep = QCheckBox(map_measure)
        self.chb_astep.setObjectName(u"chb_astep")
        sizePolicy3.setHeightForWidth(self.chb_astep.sizePolicy().hasHeightForWidth())
        self.chb_astep.setSizePolicy(sizePolicy3)
        self.chb_astep.setCheckable(True)
        self.chb_astep.setChecked(True)

        self.horizontalLayout_3.addWidget(self.chb_astep)

        self.btn_astep_calc = QPushButton(map_measure)
        self.btn_astep_calc.setObjectName(u"btn_astep_calc")
        sizePolicy3.setHeightForWidth(self.btn_astep_calc.sizePolicy().hasHeightForWidth())
        self.btn_astep_calc.setSizePolicy(sizePolicy3)
        self.btn_astep_calc.setMinimumSize(QSize(50, 0))

        self.horizontalLayout_3.addWidget(self.btn_astep_calc)

        self.lbl_tpp = QLabel(map_measure)
        self.lbl_tpp.setObjectName(u"lbl_tpp")
        sizePolicy2.setHeightForWidth(self.lbl_tpp.sizePolicy().hasHeightForWidth())
        self.lbl_tpp.setSizePolicy(sizePolicy2)
        self.lbl_tpp.setMinimumSize(QSize(53, 0))

        self.horizontalLayout_3.addWidget(self.lbl_tpp)

        self.sb_tpp = QuantSpinBox(map_measure)
        self.sb_tpp.setObjectName(u"sb_tpp")
        self.sb_tpp.setMinimumSize(QSize(50, 0))
        self.sb_tpp.setReadOnly(True)
        self.sb_tpp.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_tpp.setDecimals(2)
        self.sb_tpp.setMaximum(999.990000000000009)

        self.horizontalLayout_3.addWidget(self.sb_tpp)


        self.formLayout.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_3)

        self.lbl_scandir = QLabel(map_measure)
        self.lbl_scandir.setObjectName(u"lbl_scandir")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.lbl_scandir)

        self.cb_scandir = QComboBox(map_measure)
        icon8 = QIcon()
        icon8.addFile(u":/icons/qt_resources/SCAN_HTL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon8, "")
        icon9 = QIcon()
        icon9.addFile(u":/icons/qt_resources/SCAN_HTR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon9, "")
        icon10 = QIcon()
        icon10.addFile(u":/icons/qt_resources/SCAN_HBR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon10, "")
        icon11 = QIcon()
        icon11.addFile(u":/icons/qt_resources/SCAN_HBL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon11, "")
        icon12 = QIcon()
        icon12.addFile(u":/icons/qt_resources/SCAN_VBR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon12, "")
        icon13 = QIcon()
        icon13.addFile(u":/icons/qt_resources/SCAN_VBL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon13, "")
        icon14 = QIcon()
        icon14.addFile(u":/icons/qt_resources/SCAN_VTR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon14, "")
        icon15 = QIcon()
        icon15.addFile(u":/icons/qt_resources/SCAN_VTL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon15, "")
        self.cb_scandir.setObjectName(u"cb_scandir")
        self.cb_scandir.setFrame(True)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.cb_scandir)

        self.lbl_sig = QLabel(map_measure)
        self.lbl_sig.setObjectName(u"lbl_sig")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.lbl_sig)

        self.cb_sig = QComboBox(map_measure)
        self.cb_sig.addItem("")
        self.cb_sig.setObjectName(u"cb_sig")
        self.cb_sig.setFrame(True)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.cb_sig)

        self.lbl_scanplane = QLabel(map_measure)
        self.lbl_scanplane.setObjectName(u"lbl_scanplane")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.lbl_scanplane)

        self.cb_scanplane = QComboBox(map_measure)
        self.cb_scanplane.addItem("")
        self.cb_scanplane.addItem("")
        self.cb_scanplane.addItem("")
        self.cb_scanplane.setObjectName(u"cb_scanplane")
        self.cb_scanplane.setFrame(True)

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.cb_scanplane)

        self.lbl_wl = QLabel(map_measure)
        self.lbl_wl.setObjectName(u"lbl_wl")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.lbl_wl)

        self.sb_wl = QuantSpinBox(map_measure)
        self.sb_wl.setObjectName(u"sb_wl")
        self.sb_wl.setDecimals(0)
        self.sb_wl.setMinimum(200.000000000000000)
        self.sb_wl.setMaximum(2000.000000000000000)
        self.sb_wl.setSingleStep(10.000000000000000)
        self.sb_wl.setValue(700.000000000000000)

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.sb_wl)


        self.verticalLayout.addLayout(self.formLayout)

        self.line_2 = QFrame(map_measure)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.btn_tst = QPushButton(map_measure)
        self.btn_tst.setObjectName(u"btn_tst")

        self.verticalLayout.addWidget(self.btn_tst)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_6.addLayout(self.verticalLayout)

        self.plot_scan = PgMap(map_measure)
        self.plot_scan.setObjectName(u"plot_scan")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.plot_scan.sizePolicy().hasHeightForWidth())
        self.plot_scan.setSizePolicy(sizePolicy4)

        self.horizontalLayout_6.addWidget(self.plot_scan)

        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(1, 9)
        QWidget.setTabOrder(self.sb_centerX, self.sb_centerY)
        QWidget.setTabOrder(self.sb_centerY, self.sb_sizeX)
        QWidget.setTabOrder(self.sb_sizeX, self.sb_sizeY)
        QWidget.setTabOrder(self.sb_sizeY, self.sb_pointsX)
        QWidget.setTabOrder(self.sb_pointsX, self.sb_pointsY)
        QWidget.setTabOrder(self.sb_pointsY, self.sb_stepX)
        QWidget.setTabOrder(self.sb_stepX, self.sb_stepY)
        QWidget.setTabOrder(self.sb_stepY, self.btn_start)
        QWidget.setTabOrder(self.btn_start, self.btn_pause)
        QWidget.setTabOrder(self.btn_pause, self.btn_restart)
        QWidget.setTabOrder(self.btn_restart, self.btn_stop)
        QWidget.setTabOrder(self.btn_stop, self.le_estdur)
        QWidget.setTabOrder(self.le_estdur, self.le_dur)
        QWidget.setTabOrder(self.le_dur, self.chb_astep)
        QWidget.setTabOrder(self.chb_astep, self.cb_scandir)
        QWidget.setTabOrder(self.cb_scandir, self.cb_sig)
        QWidget.setTabOrder(self.cb_sig, self.cb_scanplane)

        self.retranslateUi(map_measure)

        QMetaObject.connectSlotsByName(map_measure)
    # setupUi

    def retranslateUi(self, map_measure):
        map_measure.setWindowTitle(QCoreApplication.translate("map_measure", u"Form", None))
        self.lbl_scanarea.setText(QCoreApplication.translate("map_measure", u"Scan Area", None))
        self.sb_sizeX.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.lbl_points.setText(QCoreApplication.translate("map_measure", u"Points", None))
        self.lbl_areaY.setText(QCoreApplication.translate("map_measure", u"Y", None))
        self.lbl_step.setText(QCoreApplication.translate("map_measure", u"Step", None))
        self.lbl_size.setText(QCoreApplication.translate("map_measure", u"Size", None))
        self.sb_stepX.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.sb_sizeY.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.sb_stepY.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.sb_centerX.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.sb_centerY.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.lbl_center.setText(QCoreApplication.translate("map_measure", u"Center", None))
        self.lbl_areaX.setText(QCoreApplication.translate("map_measure", u"X", None))
        self.label_2.setText(QCoreApplication.translate("map_measure", u"Visible plot area", None))
#if QT_CONFIG(tooltip)
        self.btn_plot_full_area.setToolTip(QCoreApplication.translate("map_measure", u"Set maximum plot size", None))
#endif // QT_CONFIG(tooltip)
        self.btn_plot_full_area.setText("")
#if QT_CONFIG(tooltip)
        self.btn_plot_clear.setToolTip(QCoreApplication.translate("map_measure", u"Clear scanned data from plot", None))
#endif // QT_CONFIG(tooltip)
        self.btn_plot_clear.setText("")
#if QT_CONFIG(tooltip)
        self.btn_plot_fit.setToolTip(QCoreApplication.translate("map_measure", u"Fit data to full plot", None))
#endif // QT_CONFIG(tooltip)
        self.btn_plot_fit.setText("")
#if QT_CONFIG(tooltip)
        self.btn_plot_area_sel.setToolTip(QCoreApplication.translate("map_measure", u"Select scan area", None))
#endif // QT_CONFIG(tooltip)
        self.btn_plot_area_sel.setText("")
        self.btn_start.setText(QCoreApplication.translate("map_measure", u"Scan", None))
        self.btn_pause.setText("")
        self.btn_restart.setText("")
        self.btn_stop.setText("")
        self.lbl_estdur.setText(QCoreApplication.translate("map_measure", u"Est time", None))
        self.lbl_dur.setText(QCoreApplication.translate("map_measure", u"Scan time", None))
        self.lbl_params.setText(QCoreApplication.translate("map_measure", u"Parameters", None))
        self.lbl_speed.setText(QCoreApplication.translate("map_measure", u"Scan speed", None))
#if QT_CONFIG(tooltip)
        self.sb_speed.setToolTip(QCoreApplication.translate("map_measure", u"Set scan speed. Equal for all axes.", None))
#endif // QT_CONFIG(tooltip)
        self.sb_speed.setSuffix(QCoreApplication.translate("map_measure", u" mm/s", None))
        self.label.setText(QCoreApplication.translate("map_measure", u"Cur speed", None))
#if QT_CONFIG(tooltip)
        self.sb_cur_speed.setToolTip(QCoreApplication.translate("map_measure", u"Currently set scan speed. Equal for all axes.", None))
#endif // QT_CONFIG(tooltip)
        self.sb_cur_speed.setSuffix(QCoreApplication.translate("map_measure", u" m/s", None))
        self.lbl_astep.setText(QCoreApplication.translate("map_measure", u"Auto step", None))
        self.chb_astep.setText("")
#if QT_CONFIG(tooltip)
        self.btn_astep_calc.setToolTip(QCoreApplication.translate("map_measure", u"Measure average time between ADC measureemnts.", None))
#endif // QT_CONFIG(tooltip)
        self.btn_astep_calc.setText(QCoreApplication.translate("map_measure", u"Calculate", None))
        self.lbl_tpp.setText(QCoreApplication.translate("map_measure", u"Point", None))
#if QT_CONFIG(tooltip)
        self.sb_tpp.setToolTip(QCoreApplication.translate("map_measure", u"Average time between scan points. Limited by ADC speed.", None))
#endif // QT_CONFIG(tooltip)
        self.sb_tpp.setSuffix(QCoreApplication.translate("map_measure", u" s", None))
        self.lbl_scandir.setText(QCoreApplication.translate("map_measure", u"Direction", None))
        self.cb_scandir.setItemText(0, QCoreApplication.translate("map_measure", u"HLT", None))
        self.cb_scandir.setItemText(1, QCoreApplication.translate("map_measure", u"HRT", None))
        self.cb_scandir.setItemText(2, QCoreApplication.translate("map_measure", u"HRB", None))
        self.cb_scandir.setItemText(3, QCoreApplication.translate("map_measure", u"HLB", None))
        self.cb_scandir.setItemText(4, QCoreApplication.translate("map_measure", u"VRB", None))
        self.cb_scandir.setItemText(5, QCoreApplication.translate("map_measure", u"VLB", None))
        self.cb_scandir.setItemText(6, QCoreApplication.translate("map_measure", u"VRT", None))
        self.cb_scandir.setItemText(7, QCoreApplication.translate("map_measure", u"VLT", None))

        self.lbl_sig.setText(QCoreApplication.translate("map_measure", u"Signal", None))
        self.cb_sig.setItemText(0, QCoreApplication.translate("map_measure", u"Amplitude", None))

        self.lbl_scanplane.setText(QCoreApplication.translate("map_measure", u"Plane", None))
        self.cb_scanplane.setItemText(0, QCoreApplication.translate("map_measure", u"XY", None))
        self.cb_scanplane.setItemText(1, QCoreApplication.translate("map_measure", u"XZ", None))
        self.cb_scanplane.setItemText(2, QCoreApplication.translate("map_measure", u"YZ", None))

        self.lbl_wl.setText(QCoreApplication.translate("map_measure", u"Wavelength", None))
        self.sb_wl.setSuffix(QCoreApplication.translate("map_measure", u" nm", None))
        self.btn_tst.setText(QCoreApplication.translate("map_measure", u"Test", None))
    # retranslateUi

