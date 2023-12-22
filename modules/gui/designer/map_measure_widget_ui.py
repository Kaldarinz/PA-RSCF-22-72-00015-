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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QSpinBox, QVBoxLayout, QWidget)

from ..widgets import (MplMap, QuantSpinBox)
from . import qt_resources_rc

class Ui_map_measure(object):
    def setupUi(self, map_measure):
        if not map_measure.objectName():
            map_measure.setObjectName(u"map_measure")
        map_measure.resize(924, 639)
        self.horizontalLayout_3 = QHBoxLayout(map_measure)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(6, -1, -1, -1)
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
        self.gridLayout.setContentsMargins(0, -1, -1, -1)
        self.rb_lockstep = QRadioButton(map_measure)
        self.rb_lockstep.setObjectName(u"rb_lockstep")

        self.gridLayout.addWidget(self.rb_lockstep, 3, 3, 1, 1)

        self.lbl_size = QLabel(map_measure)
        self.lbl_size.setObjectName(u"lbl_size")

        self.gridLayout.addWidget(self.lbl_size, 0, 1, 1, 1)

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

        self.gridLayout.addWidget(self.sb_sizeX, 1, 1, 1, 1)

        self.lbl_lock = QLabel(map_measure)
        self.lbl_lock.setObjectName(u"lbl_lock")

        self.gridLayout.addWidget(self.lbl_lock, 3, 0, 1, 1)

        self.lbl_points = QLabel(map_measure)
        self.lbl_points.setObjectName(u"lbl_points")

        self.gridLayout.addWidget(self.lbl_points, 0, 2, 1, 1)

        self.lbl_step = QLabel(map_measure)
        self.lbl_step.setObjectName(u"lbl_step")

        self.gridLayout.addWidget(self.lbl_step, 0, 3, 1, 1)

        self.lbl_areaX = QLabel(map_measure)
        self.lbl_areaX.setObjectName(u"lbl_areaX")
        self.lbl_areaX.setFont(font)

        self.gridLayout.addWidget(self.lbl_areaX, 1, 0, 1, 1)

        self.sb_sizeY = QuantSpinBox(map_measure)
        self.sb_sizeY.setObjectName(u"sb_sizeY")
        sizePolicy1.setHeightForWidth(self.sb_sizeY.sizePolicy().hasHeightForWidth())
        self.sb_sizeY.setSizePolicy(sizePolicy1)
        self.sb_sizeY.setMinimumSize(QSize(0, 0))
        self.sb_sizeY.setDecimals(2)
        self.sb_sizeY.setMaximum(25.000000000000000)
        self.sb_sizeY.setValue(4.000000000000000)

        self.gridLayout.addWidget(self.sb_sizeY, 2, 1, 1, 1)

        self.sb_pointsY = QSpinBox(map_measure)
        self.sb_pointsY.setObjectName(u"sb_pointsY")
        sizePolicy1.setHeightForWidth(self.sb_pointsY.sizePolicy().hasHeightForWidth())
        self.sb_pointsY.setSizePolicy(sizePolicy1)
        self.sb_pointsY.setMinimumSize(QSize(0, 0))

        self.gridLayout.addWidget(self.sb_pointsY, 2, 2, 1, 1)

        self.rb_lockpoints = QRadioButton(map_measure)
        self.rb_lockpoints.setObjectName(u"rb_lockpoints")

        self.gridLayout.addWidget(self.rb_lockpoints, 3, 2, 1, 1)

        self.sb_stepX = QuantSpinBox(map_measure)
        self.sb_stepX.setObjectName(u"sb_stepX")
        sizePolicy1.setHeightForWidth(self.sb_stepX.sizePolicy().hasHeightForWidth())
        self.sb_stepX.setSizePolicy(sizePolicy1)
        self.sb_stepX.setMinimumSize(QSize(0, 0))
        self.sb_stepX.setDecimals(2)

        self.gridLayout.addWidget(self.sb_stepX, 1, 3, 1, 1)

        self.sb_pointsX = QSpinBox(map_measure)
        self.sb_pointsX.setObjectName(u"sb_pointsX")
        sizePolicy1.setHeightForWidth(self.sb_pointsX.sizePolicy().hasHeightForWidth())
        self.sb_pointsX.setSizePolicy(sizePolicy1)
        self.sb_pointsX.setMinimumSize(QSize(68, 0))

        self.gridLayout.addWidget(self.sb_pointsX, 1, 2, 1, 1)

        self.lbl_areaY = QLabel(map_measure)
        self.lbl_areaY.setObjectName(u"lbl_areaY")
        self.lbl_areaY.setFont(font)

        self.gridLayout.addWidget(self.lbl_areaY, 2, 0, 1, 1)

        self.sb_stepY = QuantSpinBox(map_measure)
        self.sb_stepY.setObjectName(u"sb_stepY")
        sizePolicy1.setHeightForWidth(self.sb_stepY.sizePolicy().hasHeightForWidth())
        self.sb_stepY.setSizePolicy(sizePolicy1)
        self.sb_stepY.setMinimumSize(QSize(0, 0))
        self.sb_stepY.setDecimals(2)

        self.gridLayout.addWidget(self.sb_stepY, 2, 3, 1, 1)

        self.rb_locksize = QRadioButton(map_measure)
        self.rb_locksize.setObjectName(u"rb_locksize")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.rb_locksize.sizePolicy().hasHeightForWidth())
        self.rb_locksize.setSizePolicy(sizePolicy2)

        self.gridLayout.addWidget(self.rb_locksize, 3, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

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
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_start.setIcon(icon)

        self.horizontalLayout.addWidget(self.btn_start)

        self.btn_pause = QPushButton(map_measure)
        self.btn_pause.setObjectName(u"btn_pause")
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/control-pause.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_pause.setIcon(icon1)

        self.horizontalLayout.addWidget(self.btn_pause)

        self.btn_restart = QPushButton(map_measure)
        self.btn_restart.setObjectName(u"btn_restart")
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/arrow-circle-225-left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_restart.setIcon(icon2)

        self.horizontalLayout.addWidget(self.btn_restart)

        self.btn_stop = QPushButton(map_measure)
        self.btn_stop.setObjectName(u"btn_stop")
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon3)

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
        self.le_estdur.setMaximumSize(QSize(81, 16777215))
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
        self.le_dur.setMaximumSize(QSize(81, 16777215))
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
        self.formLayout.setContentsMargins(0, -1, -1, -1)
        self.lbl_speed = QLabel(map_measure)
        self.lbl_speed.setObjectName(u"lbl_speed")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.lbl_speed)

        self.sb_speed = QuantSpinBox(map_measure)
        self.sb_speed.setObjectName(u"sb_speed")
        sizePolicy1.setHeightForWidth(self.sb_speed.sizePolicy().hasHeightForWidth())
        self.sb_speed.setSizePolicy(sizePolicy1)
        self.sb_speed.setMinimumSize(QSize(0, 0))
        self.sb_speed.setDecimals(2)
        self.sb_speed.setValue(2.000000000000000)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.sb_speed)

        self.lbl_astep = QLabel(map_measure)
        self.lbl_astep.setObjectName(u"lbl_astep")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.lbl_astep)

        self.chb_astep = QCheckBox(map_measure)
        self.chb_astep.setObjectName(u"chb_astep")
        self.chb_astep.setChecked(True)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.chb_astep)

        self.lbl_scandir = QLabel(map_measure)
        self.lbl_scandir.setObjectName(u"lbl_scandir")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.lbl_scandir)

        self.cb_scandir = QComboBox(map_measure)
        icon4 = QIcon()
        icon4.addFile(u":/icons/qt_resources/SCAN_HTL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon4, "")
        icon5 = QIcon()
        icon5.addFile(u":/icons/qt_resources/SCAN_HTR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon5, "")
        icon6 = QIcon()
        icon6.addFile(u":/icons/qt_resources/SCAN_HBR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon6, "")
        icon7 = QIcon()
        icon7.addFile(u":/icons/qt_resources/SCAN_HBL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon7, "")
        icon8 = QIcon()
        icon8.addFile(u":/icons/qt_resources/SCAN_VBR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon8, "")
        icon9 = QIcon()
        icon9.addFile(u":/icons/qt_resources/SCAN_VBL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon9, "")
        icon10 = QIcon()
        icon10.addFile(u":/icons/qt_resources/SCAN_VTR.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon10, "")
        icon11 = QIcon()
        icon11.addFile(u":/icons/qt_resources/SCAN_VTL.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cb_scandir.addItem(icon11, "")
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


        self.verticalLayout.addLayout(self.formLayout)

        self.line_2 = QFrame(map_measure)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.plot_scan = MplMap(map_measure)
        self.plot_scan.setObjectName(u"plot_scan")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.plot_scan.sizePolicy().hasHeightForWidth())
        self.plot_scan.setSizePolicy(sizePolicy3)

        self.horizontalLayout_3.addWidget(self.plot_scan)


        self.retranslateUi(map_measure)

        QMetaObject.connectSlotsByName(map_measure)
    # setupUi

    def retranslateUi(self, map_measure):
        map_measure.setWindowTitle(QCoreApplication.translate("map_measure", u"Form", None))
        self.lbl_scanarea.setText(QCoreApplication.translate("map_measure", u"Scan Area", None))
        self.rb_lockstep.setText("")
        self.lbl_size.setText(QCoreApplication.translate("map_measure", u"Size", None))
        self.sb_sizeX.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.lbl_lock.setText(QCoreApplication.translate("map_measure", u"Lock", None))
        self.lbl_points.setText(QCoreApplication.translate("map_measure", u"Points", None))
        self.lbl_step.setText(QCoreApplication.translate("map_measure", u"Step", None))
        self.lbl_areaX.setText(QCoreApplication.translate("map_measure", u"X", None))
        self.sb_sizeY.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.rb_lockpoints.setText("")
        self.sb_stepX.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.lbl_areaY.setText(QCoreApplication.translate("map_measure", u"Y", None))
        self.sb_stepY.setSuffix(QCoreApplication.translate("map_measure", u" mm", None))
        self.rb_locksize.setText("")
        self.btn_start.setText(QCoreApplication.translate("map_measure", u"Scan", None))
        self.btn_pause.setText("")
        self.btn_restart.setText("")
        self.btn_stop.setText("")
        self.lbl_estdur.setText(QCoreApplication.translate("map_measure", u"Est time", None))
        self.lbl_dur.setText(QCoreApplication.translate("map_measure", u"Time", None))
        self.lbl_params.setText(QCoreApplication.translate("map_measure", u"Parameters", None))
        self.lbl_speed.setText(QCoreApplication.translate("map_measure", u"Speed", None))
        self.sb_speed.setSuffix(QCoreApplication.translate("map_measure", u" mm/s", None))
        self.lbl_astep.setText(QCoreApplication.translate("map_measure", u"Auto step", None))
        self.chb_astep.setText("")
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

    # retranslateUi

