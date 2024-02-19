# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PA_main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QMenu,
    QMenuBar, QSizePolicy, QStackedWidget, QStatusBar,
    QToolBar, QWidget)
from . import qt_resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1033, 570)
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
        self.action_Save.setEnabled(False)
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.actionSave_As.setEnabled(False)
        self.action_Close = QAction(MainWindow)
        self.action_Close.setObjectName(u"action_Close")
        self.action_Exit = QAction(MainWindow)
        self.action_Exit.setObjectName(u"action_Exit")
        self.action_Init = QAction(MainWindow)
        self.action_Init.setObjectName(u"action_Init")
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/plug-disconnect-prohibition.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_Init.setIcon(icon)
        self.action_measure_PA = QAction(MainWindow)
        self.action_measure_PA.setObjectName(u"action_measure_PA")
        self.action_measure_PA.setCheckable(True)
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
        self.action_position = QAction(MainWindow)
        self.action_position.setObjectName(u"action_position")
        self.action_position.setCheckable(True)
        icon5 = QIcon()
        icon5.addFile(u":/icons/qt_resources/joystick.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_position.setIcon(icon5)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy1)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.sw_central = QStackedWidget(self.centralwidget)
        self.sw_central.setObjectName(u"sw_central")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.sw_central.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.sw_central.addWidget(self.page_2)

        self.horizontalLayout.addWidget(self.sw_central)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1033, 22))
        self.menu_Data = QMenu(self.menubar)
        self.menu_Data.setObjectName(u"menu_Data")
        self.menu_Hardware = QMenu(self.menubar)
        self.menu_Hardware.setObjectName(u"menu_Hardware")
        self.menu_Measurements = QMenu(self.menubar)
        self.menu_Measurements.setObjectName(u"menu_Measurements")
        self.menu_Tools = QMenu(self.menubar)
        self.menu_Tools.setObjectName(u"menu_Tools")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
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
        self.menubar.addAction(self.menuHelp.menuAction())
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
        self.menu_Hardware.addAction(self.action_position)
        self.menu_Measurements.addAction(self.action_measure_PA)
        self.menu_Measurements.addSeparator()
        self.menu_Measurements.addAction(self.action_Power_Meter)
        self.menu_Measurements.addSeparator()
        self.menu_Tools.addAction(self.action_Logs)
        self.menu_Tools.addSeparator()
        self.menuHelp.addAction(self.actionAbout)
        self.tb_main.addAction(self.action_Init)
        self.tb_main.addSeparator()
        self.tb_main.addAction(self.action_Data)
        self.tb_main.addSeparator()
        self.tb_main.addAction(self.action_measure_PA)
        self.tb_main.addSeparator()
        self.tb_tools.addAction(self.action_Logs)
        self.tb_tools.addAction(self.action_Power_Meter)
        self.tb_tools.addAction(self.action_position)

        self.retranslateUi(MainWindow)

        self.sw_central.setCurrentIndex(0)


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
        self.action_position.setText(QCoreApplication.translate("MainWindow", u"Positioning", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.menu_Data.setTitle(QCoreApplication.translate("MainWindow", u"&Data", None))
        self.menu_Hardware.setTitle(QCoreApplication.translate("MainWindow", u"&Hardware", None))
        self.menu_Measurements.setTitle(QCoreApplication.translate("MainWindow", u"&Measurements", None))
        self.menu_Tools.setTitle(QCoreApplication.translate("MainWindow", u"&Tools", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
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

