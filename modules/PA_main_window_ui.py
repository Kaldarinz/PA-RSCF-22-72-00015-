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
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMenu, QMenuBar, QPlainTextEdit, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QStackedWidget,
    QStatusBar, QToolBar, QVBoxLayout, QWidget)

from .widgets import MplCanvas
from . import qt_resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1577, 1089)
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
        self.dock_pm.setGeometry(QRect(9, 669, 666, 250))
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
        self.dock_log.setGeometry(QRect(9, 339, 256, 214))
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
        self.stackedWidget.setGeometry(QRect(9, 9, 596, 108))
        sizePolicy1.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy1)
        self.stackedWidget.setMaximumSize(QSize(16777215, 16777215))
        self.page = QWidget()
        self.page.setObjectName(u"page")
        sizePolicy1.setHeightForWidth(self.page.sizePolicy().hasHeightForWidth())
        self.page.setSizePolicy(sizePolicy1)
        self.stackedWidget.addWidget(self.page)
        self.p_curve = QWidget()
        self.p_curve.setObjectName(u"p_curve")
        sizePolicy1.setHeightForWidth(self.p_curve.sizePolicy().hasHeightForWidth())
        self.p_curve.setSizePolicy(sizePolicy1)
        self.horizontalLayout_3 = QHBoxLayout(self.p_curve)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btn_curve_run = QPushButton(self.p_curve)
        self.btn_curve_run.setObjectName(u"btn_curve_run")
        self.btn_curve_run.setFont(font)
        icon4 = QIcon()
        icon4.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_curve_run.setIcon(icon4)
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

        self.spinBox = QSpinBox(self.p_curve)
        self.spinBox.setObjectName(u"spinBox")

        self.horizontalLayout_2.addWidget(self.spinBox)

        self.lbl_curve_to = QLabel(self.p_curve)
        self.lbl_curve_to.setObjectName(u"lbl_curve_to")

        self.horizontalLayout_2.addWidget(self.lbl_curve_to)

        self.spinBox_2 = QSpinBox(self.p_curve)
        self.spinBox_2.setObjectName(u"spinBox_2")

        self.horizontalLayout_2.addWidget(self.spinBox_2)

        self.lbl_curve_range_u = QLabel(self.p_curve)
        self.lbl_curve_range_u.setObjectName(u"lbl_curve_range_u")
        self.lbl_curve_range_u.setMinimumSize(QSize(40, 0))

        self.horizontalLayout_2.addWidget(self.lbl_curve_range_u)

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

        self.horizontalLayout_2.addWidget(self.spinBox_3)

        self.lbl_curve_step_u = QLabel(self.p_curve)
        self.lbl_curve_step_u.setObjectName(u"lbl_curve_step_u")
        self.lbl_curve_step_u.setMinimumSize(QSize(40, 0))

        self.horizontalLayout_2.addWidget(self.lbl_curve_step_u)

        self.line_7 = QFrame(self.p_curve)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.VLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.line_7)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.plot_pa_1d = MplCanvas(self.p_curve)
        self.plot_pa_1d.setObjectName(u"plot_pa_1d")
        sizePolicy1.setHeightForWidth(self.plot_pa_1d.sizePolicy().hasHeightForWidth())
        self.plot_pa_1d.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.plot_pa_1d)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.stackedWidget.addWidget(self.p_curve)
        MainWindow.setCentralWidget(self.centralwidget)
        self.dock_log.raise_()
        self.dock_pm.raise_()
        self.stackedWidget.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1577, 21))
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
        MainWindow.addToolBar(Qt.TopToolBarArea, self.tb_main)
        self.tb_mode = QToolBar(MainWindow)
        self.tb_mode.setObjectName(u"tb_mode")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.tb_mode)
        MainWindow.insertToolBarBreak(self.tb_mode)

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
        self.tb_main.addAction(self.action_Power_Meter)
        self.tb_main.addSeparator()
        self.tb_main.addAction(self.action_measure_PA)
        self.tb_main.addSeparator()
        self.tb_main.addAction(self.action_Logs)

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
        self.action_Open.setText(QCoreApplication.translate("MainWindow", u"&Open", None))
        self.action_New.setText(QCoreApplication.translate("MainWindow", u"&New", None))
        self.actionO_pen_recent.setText(QCoreApplication.translate("MainWindow", u"O&pen Recent", None))
        self.action_Save.setText(QCoreApplication.translate("MainWindow", u"&Save", None))
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Save &As", None))
        self.action_Close.setText(QCoreApplication.translate("MainWindow", u"&Close", None))
        self.action_Exit.setText(QCoreApplication.translate("MainWindow", u"&Exit", None))
        self.action_Init.setText(QCoreApplication.translate("MainWindow", u"&Init All", None))
        self.action_measure_PA.setText(QCoreApplication.translate("MainWindow", u"Measure PA Signal", None))
        self.action_Power_Meter.setText(QCoreApplication.translate("MainWindow", u"&Power Meter", None))
        self.action_Logs.setText(QCoreApplication.translate("MainWindow", u"&Logs", None))
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
        self.btn_curve_run.setText(QCoreApplication.translate("MainWindow", u"RUN", None))
        self.lbl_curve_mode.setText(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.cb_curve_mode.setItemText(0, QCoreApplication.translate("MainWindow", u"Wavelength", None))
        self.cb_curve_mode.setItemText(1, QCoreApplication.translate("MainWindow", u"Energy", None))
        self.cb_curve_mode.setItemText(2, QCoreApplication.translate("MainWindow", u"Concentration", None))

        self.lbl_curve_from.setText(QCoreApplication.translate("MainWindow", u"From", None))
        self.lbl_curve_to.setText(QCoreApplication.translate("MainWindow", u"To", None))
        self.lbl_curve_range_u.setText(QCoreApplication.translate("MainWindow", u"Units", None))
        self.lbl_curve_step.setText(QCoreApplication.translate("MainWindow", u"Step", None))
        self.lbl_curve_step_u.setText(QCoreApplication.translate("MainWindow", u"Units", None))
        self.menu_Data.setTitle(QCoreApplication.translate("MainWindow", u"&Data", None))
        self.menu_Hardware.setTitle(QCoreApplication.translate("MainWindow", u"&Hardware", None))
        self.menu_Measurements.setTitle(QCoreApplication.translate("MainWindow", u"&Measurements", None))
        self.menu_Tools.setTitle(QCoreApplication.translate("MainWindow", u"&Tools", None))
        self.tb_main.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
        self.tb_mode.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar_2", None))
    # retranslateUi

