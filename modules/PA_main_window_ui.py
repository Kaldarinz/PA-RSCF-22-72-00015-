# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PA_main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QDockWidget,
    QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMenu, QMenuBar, QPlainTextEdit,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QStackedWidget, QStatusBar, QToolBar, QVBoxLayout,
    QWidget)

from .widgets import MplCanvas
from . import qt_resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(873, 857)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QSize(16777215, 16777215))
        MainWindow.setDockNestingEnabled(True)
        self.action_Open = QAction(MainWindow)
        self.action_Open.setObjectName(u"action_Open")
        self.action_New = QAction(MainWindow)
        self.action_New.setObjectName(u"action_New")
        self.actionO_pen_recent = QAction(MainWindow)
        self.actionO_pen_recent.setObjectName(u"actionO_pen_recent")
        self.action_Save = QAction(MainWindow)
        self.action_Save.setObjectName(u"action_Save")
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.action_Close = QAction(MainWindow)
        self.action_Close.setObjectName(u"action_Close")
        self.action_Exit = QAction(MainWindow)
        self.action_Exit.setObjectName(u"action_Exit")
        self.action_Init = QAction(MainWindow)
        self.action_Init.setObjectName(u"action_Init")
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/battery-plug.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_Init.setIcon(icon)
        self.action_measure_PA = QAction(MainWindow)
        self.action_measure_PA.setObjectName(u"action_measure_PA")
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/spectrum.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_measure_PA.setIcon(icon1)
        self.action_Power_Meter = QAction(MainWindow)
        self.action_Power_Meter.setObjectName(u"action_Power_Meter")
        self.action_Power_Meter.setCheckable(True)
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/application-monitor.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_Power_Meter.setIcon(icon2)
        self.action_Logs = QAction(MainWindow)
        self.action_Logs.setObjectName(u"action_Logs")
        self.action_Logs.setCheckable(True)
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/sticky-note-text.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_Logs.setIcon(icon3)
        self.action_Data = QAction(MainWindow)
        self.action_Data.setObjectName(u"action_Data")
        self.action_Data.setCheckable(True)
        icon4 = QIcon()
        icon4.addFile(u":/icons/qt_resources/television-image.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_Data.setIcon(icon4)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy1)
        self.dock_pm = QDockWidget(self.centralwidget)
        self.dock_pm.setObjectName(u"dock_pm")
        self.dock_pm.setEnabled(True)
        self.dock_pm.setGeometry(QRect(10, 740, 666, 250))
        self.dock_pm.setMinimumSize(QSize(626, 250))
        self.dock_pm.setAcceptDrops(True)
        self.dock_pm.setFloating(True)
        self.w_pm = QWidget()
        self.w_pm.setObjectName(u"w_pm")
        self.verticalLayout_2 = QVBoxLayout(self.w_pm)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 6, 0, 0)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.lbl_cur_en = QLabel(self.w_pm)
        self.lbl_cur_en.setObjectName(u"lbl_cur_en")

        self.horizontalLayout_9.addWidget(self.lbl_cur_en)

        self.le_cur_en = QLineEdit(self.w_pm)
        self.le_cur_en.setObjectName(u"le_cur_en")
        self.le_cur_en.setEnabled(True)
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.le_cur_en.sizePolicy().hasHeightForWidth())
        self.le_cur_en.setSizePolicy(sizePolicy2)
        self.le_cur_en.setMinimumSize(QSize(70, 20))
        self.le_cur_en.setMaximumSize(QSize(50, 16777215))
        self.le_cur_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_cur_en)

        self.lbl_aver_en = QLabel(self.w_pm)
        self.lbl_aver_en.setObjectName(u"lbl_aver_en")

        self.horizontalLayout_9.addWidget(self.lbl_aver_en)

        self.le_aver_en = QLineEdit(self.w_pm)
        self.le_aver_en.setObjectName(u"le_aver_en")
        self.le_aver_en.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.le_aver_en.sizePolicy().hasHeightForWidth())
        self.le_aver_en.setSizePolicy(sizePolicy2)
        self.le_aver_en.setMinimumSize(QSize(70, 20))
        self.le_aver_en.setMaximumSize(QSize(50, 16777215))
        self.le_aver_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_aver_en)

        self.lbl_std_en = QLabel(self.w_pm)
        self.lbl_std_en.setObjectName(u"lbl_std_en")

        self.horizontalLayout_9.addWidget(self.lbl_std_en)

        self.le_std_en = QLineEdit(self.w_pm)
        self.le_std_en.setObjectName(u"le_std_en")
        self.le_std_en.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.le_std_en.sizePolicy().hasHeightForWidth())
        self.le_std_en.setSizePolicy(sizePolicy2)
        self.le_std_en.setMinimumSize(QSize(70, 20))
        self.le_std_en.setMaximumSize(QSize(50, 16777215))
        self.le_std_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_std_en)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer)

        self.btn_pm_start = QPushButton(self.w_pm)
        self.btn_pm_start.setObjectName(u"btn_pm_start")
        font = QFont()
        font.setBold(True)
        self.btn_pm_start.setFont(font)

        self.horizontalLayout_9.addWidget(self.btn_pm_start)

        self.btn_pm_stop = QPushButton(self.w_pm)
        self.btn_pm_stop.setObjectName(u"btn_pm_stop")
        font1 = QFont()
        font1.setBold(True)
        font1.setUnderline(False)
        font1.setStrikeOut(False)
        self.btn_pm_stop.setFont(font1)

        self.horizontalLayout_9.addWidget(self.btn_pm_stop)


        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.plot_pm_left = MplCanvas(self.w_pm)
        self.plot_pm_left.setObjectName(u"plot_pm_left")
        sizePolicy1.setHeightForWidth(self.plot_pm_left.sizePolicy().hasHeightForWidth())
        self.plot_pm_left.setSizePolicy(sizePolicy1)

        self.horizontalLayout_10.addWidget(self.plot_pm_left)

        self.plot_pm_right = MplCanvas(self.w_pm)
        self.plot_pm_right.setObjectName(u"plot_pm_right")
        sizePolicy1.setHeightForWidth(self.plot_pm_right.sizePolicy().hasHeightForWidth())
        self.plot_pm_right.setSizePolicy(sizePolicy1)

        self.horizontalLayout_10.addWidget(self.plot_pm_right)


        self.verticalLayout_2.addLayout(self.horizontalLayout_10)

        self.dock_pm.setWidget(self.w_pm)
        self.dock_log = QDockWidget(self.centralwidget)
        self.dock_log.setObjectName(u"dock_log")
        self.dock_log.setGeometry(QRect(20, 530, 256, 214))
        self.dock_log.setFloating(True)
        self.dock_log.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dockWidgetContents_7 = QWidget()
        self.dockWidgetContents_7.setObjectName(u"dockWidgetContents_7")
        self.horizontalLayout = QHBoxLayout(self.dockWidgetContents_7)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.te_log = QPlainTextEdit(self.dockWidgetContents_7)
        self.te_log.setObjectName(u"te_log")
        self.te_log.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.te_log.sizePolicy().hasHeightForWidth())
        self.te_log.setSizePolicy(sizePolicy1)
        self.te_log.setMinimumSize(QSize(0, 0))
        self.te_log.setMaximumSize(QSize(16777215, 16777215))
        self.te_log.setReadOnly(True)

        self.horizontalLayout.addWidget(self.te_log)

        self.dock_log.setWidget(self.dockWidgetContents_7)
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setGeometry(QRect(20, 30, 583, 466))
        sizePolicy1.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy1)
        self.stackedWidget.setMaximumSize(QSize(16777215, 16777215))
        self.p_point = QWidget()
        self.p_point.setObjectName(u"p_point")
        sizePolicy1.setHeightForWidth(self.p_point.sizePolicy().hasHeightForWidth())
        self.p_point.setSizePolicy(sizePolicy1)
        self.btn_point_run = QPushButton(self.p_point)
        self.btn_point_run.setObjectName(u"btn_point_run")
        self.btn_point_run.setGeometry(QRect(760, 190, 75, 24))
        self.btn_point_run.setFont(font)
        icon5 = QIcon()
        icon5.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_point_run.setIcon(icon5)
        self.btn_point_run.setCheckable(True)
        self.stackedWidget.addWidget(self.p_point)
        self.p_curve = QWidget()
        self.p_curve.setObjectName(u"p_curve")
        sizePolicy1.setHeightForWidth(self.p_curve.sizePolicy().hasHeightForWidth())
        self.p_curve.setSizePolicy(sizePolicy1)
        self.verticalLayout_3 = QVBoxLayout(self.p_curve)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btn_curve_run = QPushButton(self.p_curve)
        self.btn_curve_run.setObjectName(u"btn_curve_run")
        self.btn_curve_run.setFont(font)
        self.btn_curve_run.setIcon(icon5)
        self.btn_curve_run.setCheckable(True)

        self.horizontalLayout_2.addWidget(self.btn_curve_run)

        self.line_3 = QFrame(self.p_curve)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.line_3)

        self.lbl_curve_mode = QLabel(self.p_curve)
        self.lbl_curve_mode.setObjectName(u"lbl_curve_mode")

        self.horizontalLayout_2.addWidget(self.lbl_curve_mode)

        self.cb_curve_mode = QComboBox(self.p_curve)
        self.cb_curve_mode.addItem("")
        self.cb_curve_mode.addItem("")
        self.cb_curve_mode.addItem("")
        self.cb_curve_mode.setObjectName(u"cb_curve_mode")

        self.horizontalLayout_2.addWidget(self.cb_curve_mode)

        self.line_2 = QFrame(self.p_curve)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.line_2)

        self.lbl_curve_from = QLabel(self.p_curve)
        self.lbl_curve_from.setObjectName(u"lbl_curve_from")

        self.horizontalLayout_2.addWidget(self.lbl_curve_from)

        self.sb_curve_from = QSpinBox(self.p_curve)
        self.sb_curve_from.setObjectName(u"sb_curve_from")
        self.sb_curve_from.setMinimum(600)
        self.sb_curve_from.setMaximum(1000)
        self.sb_curve_from.setSingleStep(10)
        self.sb_curve_from.setStepType(QAbstractSpinBox.DefaultStepType)
        self.sb_curve_from.setValue(740)

        self.horizontalLayout_2.addWidget(self.sb_curve_from)

        self.lbl_curve_to = QLabel(self.p_curve)
        self.lbl_curve_to.setObjectName(u"lbl_curve_to")

        self.horizontalLayout_2.addWidget(self.lbl_curve_to)

        self.spinBox_2 = QSpinBox(self.p_curve)
        self.spinBox_2.setObjectName(u"spinBox_2")
        self.spinBox_2.setMinimum(600)
        self.spinBox_2.setMaximum(1000)
        self.spinBox_2.setSingleStep(10)
        self.spinBox_2.setValue(750)

        self.horizontalLayout_2.addWidget(self.spinBox_2)

        self.line = QFrame(self.p_curve)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.line)

        self.lbl_curve_step = QLabel(self.p_curve)
        self.lbl_curve_step.setObjectName(u"lbl_curve_step")

        self.horizontalLayout_2.addWidget(self.lbl_curve_step)

        self.spinBox_3 = QSpinBox(self.p_curve)
        self.spinBox_3.setObjectName(u"spinBox_3")
        self.spinBox_3.setMinimum(1)
        self.spinBox_3.setMaximum(500)
        self.spinBox_3.setValue(10)

        self.horizontalLayout_2.addWidget(self.spinBox_3)

        self.line_7 = QFrame(self.p_curve)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.VLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.line_7)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.btn_curve_measure = QPushButton(self.p_curve)
        self.btn_curve_measure.setObjectName(u"btn_curve_measure")
        self.btn_curve_measure.setFont(font)
        icon6 = QIcon()
        icon6.addFile(u":/icons/qt_resources/control-stop.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_curve_measure.setIcon(icon6)
        self.btn_curve_measure.setCheckable(True)

        self.verticalLayout.addWidget(self.btn_curve_measure)

        self.line_4 = QFrame(self.p_curve)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.lbl_curve_energy = QLabel(self.p_curve)
        self.lbl_curve_energy.setObjectName(u"lbl_curve_energy")
        font2 = QFont()
        font2.setPointSize(8)
        font2.setBold(True)
        self.lbl_curve_energy.setFont(font2)

        self.verticalLayout.addWidget(self.lbl_curve_energy)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.lbl_curve_sample = QLabel(self.p_curve)
        self.lbl_curve_sample.setObjectName(u"lbl_curve_sample")

        self.horizontalLayout_4.addWidget(self.lbl_curve_sample)

        self.sb_curve_sample_energy = QSpinBox(self.p_curve)
        self.sb_curve_sample_energy.setObjectName(u"sb_curve_sample_energy")
        self.sb_curve_sample_energy.setMaximumSize(QSize(43, 16777215))
        self.sb_curve_sample_energy.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_curve_sample_energy.setMinimum(10)
        self.sb_curve_sample_energy.setMaximum(1500)
        self.sb_curve_sample_energy.setSingleStep(100)
        self.sb_curve_sample_energy.setValue(500)

        self.horizontalLayout_4.addWidget(self.sb_curve_sample_energy)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.sb_curve_sample_sp = QSpinBox(self.p_curve)
        self.sb_curve_sample_sp.setObjectName(u"sb_curve_sample_sp")
        self.sb_curve_sample_sp.setMinimum(10)
        self.sb_curve_sample_sp.setMaximum(1500)
        self.sb_curve_sample_sp.setSingleStep(100)
        self.sb_curve_sample_sp.setValue(500)

        self.verticalLayout.addWidget(self.sb_curve_sample_sp)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.lbl_curve_pm = QLabel(self.p_curve)
        self.lbl_curve_pm.setObjectName(u"lbl_curve_pm")

        self.horizontalLayout_3.addWidget(self.lbl_curve_pm)

        self.sb_curve_pm_energy = QSpinBox(self.p_curve)
        self.sb_curve_pm_energy.setObjectName(u"sb_curve_pm_energy")
        self.sb_curve_pm_energy.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sb_curve_pm_energy.setMinimum(10)
        self.sb_curve_pm_energy.setMaximum(1500)
        self.sb_curve_pm_energy.setSingleStep(100)
        self.sb_curve_pm_energy.setValue(500)

        self.horizontalLayout_3.addWidget(self.sb_curve_pm_energy)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.sb_curve_pm_sp = QSpinBox(self.p_curve)
        self.sb_curve_pm_sp.setObjectName(u"sb_curve_pm_sp")
        self.sb_curve_pm_sp.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.sb_curve_pm_sp.setMinimum(10)
        self.sb_curve_pm_sp.setMaximum(1500)
        self.sb_curve_pm_sp.setSingleStep(100)
        self.sb_curve_pm_sp.setValue(500)

        self.verticalLayout.addWidget(self.sb_curve_pm_sp)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_5.addLayout(self.verticalLayout)

        self.plot_curve_signal = MplCanvas(self.p_curve)
        self.plot_curve_signal.setObjectName(u"plot_curve_signal")
        sizePolicy1.setHeightForWidth(self.plot_curve_signal.sizePolicy().hasHeightForWidth())
        self.plot_curve_signal.setSizePolicy(sizePolicy1)
        self.plot_curve_signal.setMinimumSize(QSize(100, 50))

        self.horizontalLayout_5.addWidget(self.plot_curve_signal)

        self.plot_curve_adjust = MplCanvas(self.p_curve)
        self.plot_curve_adjust.setObjectName(u"plot_curve_adjust")
        sizePolicy1.setHeightForWidth(self.plot_curve_adjust.sizePolicy().hasHeightForWidth())
        self.plot_curve_adjust.setSizePolicy(sizePolicy1)
        self.plot_curve_adjust.setMinimumSize(QSize(100, 50))

        self.horizontalLayout_5.addWidget(self.plot_curve_adjust)


        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        self.plot_curve_pa = MplCanvas(self.p_curve)
        self.plot_curve_pa.setObjectName(u"plot_curve_pa")
        sizePolicy1.setHeightForWidth(self.plot_curve_pa.sizePolicy().hasHeightForWidth())
        self.plot_curve_pa.setSizePolicy(sizePolicy1)
        self.plot_curve_pa.setMinimumSize(QSize(300, 200))

        self.verticalLayout_3.addWidget(self.plot_curve_pa)

        self.stackedWidget.addWidget(self.p_curve)
        self.p_map = QWidget()
        self.p_map.setObjectName(u"p_map")
        self.btn_map_run = QPushButton(self.p_map)
        self.btn_map_run.setObjectName(u"btn_map_run")
        self.btn_map_run.setGeometry(QRect(110, 80, 75, 24))
        self.btn_map_run.setFont(font)
        self.btn_map_run.setIcon(icon5)
        self.btn_map_run.setCheckable(True)
        self.stackedWidget.addWidget(self.p_map)
        MainWindow.setCentralWidget(self.centralwidget)
        self.dock_log.raise_()
        self.dock_pm.raise_()
        self.stackedWidget.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 873, 21))
        self.menu_Data = QMenu(self.menubar)
        self.menu_Data.setObjectName(u"menu_Data")
        self.menu_Hardware = QMenu(self.menubar)
        self.menu_Hardware.setObjectName(u"menu_Hardware")
        self.menu_Measurements = QMenu(self.menubar)
        self.menu_Measurements.setObjectName(u"menu_Measurements")
        self.menu_Tools = QMenu(self.menubar)
        self.menu_Tools.setObjectName(u"menu_Tools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.tb_main = QToolBar(MainWindow)
        self.tb_main.setObjectName(u"tb_main")
        self.tb_main.setAllowedAreas(Qt.TopToolBarArea)
        self.tb_main.setFloatable(False)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.tb_main)
        self.tb_mode = QToolBar(MainWindow)
        self.tb_mode.setObjectName(u"tb_mode")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.tb_mode)
        self.tb_tools = QToolBar(MainWindow)
        self.tb_tools.setObjectName(u"tb_tools")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.tb_tools)

        self.menubar.addAction(self.menu_Data.menuAction())
        self.menubar.addAction(self.menu_Hardware.menuAction())
        self.menubar.addAction(self.menu_Measurements.menuAction())
        self.menubar.addAction(self.menu_Tools.menuAction())
        self.menu_Data.addAction(self.action_New)
        self.menu_Data.addAction(self.action_Open)
        self.menu_Data.addAction(self.actionO_pen_recent)
        self.menu_Data.addSeparator()
        self.menu_Data.addAction(self.action_Save)
        self.menu_Data.addAction(self.actionSave_As)
        self.menu_Data.addSeparator()
        self.menu_Data.addAction(self.action_Close)
        self.menu_Data.addSeparator()
        self.menu_Data.addAction(self.action_Exit)
        self.menu_Hardware.addAction(self.action_Init)
        self.menu_Hardware.addSeparator()
        self.menu_Measurements.addAction(self.action_measure_PA)
        self.menu_Measurements.addSeparator()
        self.menu_Measurements.addAction(self.action_Power_Meter)
        self.menu_Measurements.addSeparator()
        self.menu_Tools.addAction(self.action_Logs)
        self.menu_Tools.addSeparator()
        self.tb_main.addAction(self.action_Init)
        self.tb_main.addSeparator()
        self.tb_main.addAction(self.action_Data)
        self.tb_main.addSeparator()
        self.tb_main.addAction(self.action_measure_PA)
        self.tb_main.addSeparator()
        self.tb_tools.addAction(self.action_Logs)
        self.tb_tools.addAction(self.action_Power_Meter)

        self.retranslateUi(MainWindow)
        self.dock_log.visibilityChanged.connect(self.action_Logs.setChecked)
        self.action_Logs.toggled.connect(self.dock_log.setVisible)
        self.dock_pm.visibilityChanged.connect(self.action_Power_Meter.setChecked)
        self.action_Power_Meter.toggled.connect(self.dock_pm.setVisible)

        self.stackedWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Photoacoustics", None))
#if QT_CONFIG(statustip)
        MainWindow.setStatusTip(QCoreApplication.translate("MainWindow", u"Set PhotoAcoustic measuring mode", None))
#endif // QT_CONFIG(statustip)
        self.action_Open.setText(QCoreApplication.translate("MainWindow", u"&Open", None))
        self.action_New.setText(QCoreApplication.translate("MainWindow", u"&New", None))
        self.actionO_pen_recent.setText(QCoreApplication.translate("MainWindow", u"O&pen Recent", None))
        self.action_Save.setText(QCoreApplication.translate("MainWindow", u"&Save", None))
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Save &As", None))
        self.action_Close.setText(QCoreApplication.translate("MainWindow", u"&Close", None))
        self.action_Exit.setText(QCoreApplication.translate("MainWindow", u"&Exit", None))
        self.action_Init.setText(QCoreApplication.translate("MainWindow", u"&Init All", None))
#if QT_CONFIG(statustip)
        self.action_Init.setStatusTip(QCoreApplication.translate("MainWindow", u"Initialize hardware", None))
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        self.action_Init.setWhatsThis(QCoreApplication.translate("MainWindow", u"Start initialization procedure to start working with the system", None))
#endif // QT_CONFIG(whatsthis)
        self.action_measure_PA.setText(QCoreApplication.translate("MainWindow", u"Measure PA Signal", None))
        self.action_Power_Meter.setText(QCoreApplication.translate("MainWindow", u"&Power Meter", None))
#if QT_CONFIG(statustip)
        self.action_Power_Meter.setStatusTip(QCoreApplication.translate("MainWindow", u"Start laser energy measurement", None))
#endif // QT_CONFIG(statustip)
        self.action_Logs.setText(QCoreApplication.translate("MainWindow", u"&Logs", None))
#if QT_CONFIG(statustip)
        self.action_Logs.setStatusTip(QCoreApplication.translate("MainWindow", u"Open logs", None))
#endif // QT_CONFIG(statustip)
        self.action_Data.setText(QCoreApplication.translate("MainWindow", u"Data", None))
#if QT_CONFIG(tooltip)
        self.action_Data.setToolTip(QCoreApplication.translate("MainWindow", u"Open data viewer", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.action_Data.setStatusTip(QCoreApplication.translate("MainWindow", u"Data viewer", None))
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(accessibility)
        self.dock_pm.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
        self.dock_pm.setWindowTitle(QCoreApplication.translate("MainWindow", u"Power Meter", None))
        self.lbl_cur_en.setText(QCoreApplication.translate("MainWindow", u"Current Energy:", None))
        self.lbl_aver_en.setText(QCoreApplication.translate("MainWindow", u"Average Energy:", None))
        self.lbl_std_en.setText(QCoreApplication.translate("MainWindow", u"std Energy:", None))
        self.btn_pm_start.setText(QCoreApplication.translate("MainWindow", u"START", None))
        self.btn_pm_stop.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
        self.dock_log.setWindowTitle(QCoreApplication.translate("MainWindow", u"Logs", None))
#if QT_CONFIG(statustip)
        self.btn_point_run.setStatusTip(QCoreApplication.translate("MainWindow", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_point_run.setText(QCoreApplication.translate("MainWindow", u"RUN", None))
#if QT_CONFIG(statustip)
        self.btn_curve_run.setStatusTip(QCoreApplication.translate("MainWindow", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_curve_run.setText(QCoreApplication.translate("MainWindow", u"RUN", None))
#if QT_CONFIG(statustip)
        self.lbl_curve_mode.setStatusTip(QCoreApplication.translate("MainWindow", u"Choose an independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_curve_mode.setText(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.cb_curve_mode.setItemText(0, QCoreApplication.translate("MainWindow", u"Wavelength", None))
        self.cb_curve_mode.setItemText(1, QCoreApplication.translate("MainWindow", u"Energy", None))
        self.cb_curve_mode.setItemText(2, QCoreApplication.translate("MainWindow", u"Concentration", None))

#if QT_CONFIG(statustip)
        self.cb_curve_mode.setStatusTip(QCoreApplication.translate("MainWindow", u"Choose an independent variable", None))
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(statustip)
        self.lbl_curve_from.setStatusTip(QCoreApplication.translate("MainWindow", u"Start value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_curve_from.setText(QCoreApplication.translate("MainWindow", u"From", None))
#if QT_CONFIG(statustip)
        self.sb_curve_from.setStatusTip(QCoreApplication.translate("MainWindow", u"Start value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.sb_curve_from.setSuffix(QCoreApplication.translate("MainWindow", u" nm", None))
#if QT_CONFIG(statustip)
        self.lbl_curve_to.setStatusTip(QCoreApplication.translate("MainWindow", u"End value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_curve_to.setText(QCoreApplication.translate("MainWindow", u"To", None))
#if QT_CONFIG(statustip)
        self.spinBox_2.setStatusTip(QCoreApplication.translate("MainWindow", u"End value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.spinBox_2.setSuffix(QCoreApplication.translate("MainWindow", u" nm", None))
#if QT_CONFIG(statustip)
        self.lbl_curve_step.setStatusTip(QCoreApplication.translate("MainWindow", u"Step value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.lbl_curve_step.setText(QCoreApplication.translate("MainWindow", u"Step", None))
#if QT_CONFIG(statustip)
        self.spinBox_3.setStatusTip(QCoreApplication.translate("MainWindow", u"Step value of the independent variable", None))
#endif // QT_CONFIG(statustip)
        self.spinBox_3.setSuffix(QCoreApplication.translate("MainWindow", u" nm", None))
#if QT_CONFIG(statustip)
        self.btn_curve_measure.setStatusTip(QCoreApplication.translate("MainWindow", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_curve_measure.setText(QCoreApplication.translate("MainWindow", u"MEASURE", None))
        self.lbl_curve_energy.setText(QCoreApplication.translate("MainWindow", u"Energy", None))
        self.lbl_curve_sample.setText(QCoreApplication.translate("MainWindow", u"Sample", None))
        self.sb_curve_sample_energy.setSuffix(QCoreApplication.translate("MainWindow", u" uJ", None))
        self.sb_curve_sample_energy.setPrefix("")
        self.sb_curve_sample_sp.setSuffix(QCoreApplication.translate("MainWindow", u" uJ", None))
        self.sb_curve_sample_sp.setPrefix(QCoreApplication.translate("MainWindow", u"SetPoint: ", None))
        self.lbl_curve_pm.setText(QCoreApplication.translate("MainWindow", u"Power Meter", None))
        self.sb_curve_pm_energy.setSuffix(QCoreApplication.translate("MainWindow", u" uJ", None))
        self.sb_curve_pm_energy.setPrefix("")
        self.sb_curve_pm_sp.setSuffix(QCoreApplication.translate("MainWindow", u" uJ", None))
        self.sb_curve_pm_sp.setPrefix(QCoreApplication.translate("MainWindow", u"SetPoint: ", None))
#if QT_CONFIG(statustip)
        self.btn_map_run.setStatusTip(QCoreApplication.translate("MainWindow", u"Start PhotoAcoustic measurement", None))
#endif // QT_CONFIG(statustip)
        self.btn_map_run.setText(QCoreApplication.translate("MainWindow", u"RUN", None))
        self.menu_Data.setTitle(QCoreApplication.translate("MainWindow", u"&Data", None))
        self.menu_Hardware.setTitle(QCoreApplication.translate("MainWindow", u"&Hardware", None))
        self.menu_Measurements.setTitle(QCoreApplication.translate("MainWindow", u"&Measurements", None))
        self.menu_Tools.setTitle(QCoreApplication.translate("MainWindow", u"&Tools", None))
        self.tb_main.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main", None))
#if QT_CONFIG(statustip)
        self.tb_main.setStatusTip(QCoreApplication.translate("MainWindow", u"Basic operations", None))
#endif // QT_CONFIG(statustip)
        self.tb_mode.setWindowTitle(QCoreApplication.translate("MainWindow", u"Measuring mode", None))
#if QT_CONFIG(statustip)
        self.tb_mode.setStatusTip(QCoreApplication.translate("MainWindow", u"Set PhotoAcoustic measuring mode", None))
#endif // QT_CONFIG(statustip)
        self.tb_tools.setWindowTitle(QCoreApplication.translate("MainWindow", u"Tools", None))
#if QT_CONFIG(statustip)
        self.tb_tools.setStatusTip(QCoreApplication.translate("MainWindow", u"Tools", None))
#endif // QT_CONFIG(statustip)
    # retranslateUi

