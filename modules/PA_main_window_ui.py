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
from PySide6.QtWidgets import (QApplication, QDockWidget, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QPlainTextEdit, QPushButton, QSizePolicy, QSpacerItem,
    QStatusBar, QToolBar, QVBoxLayout, QWidget)

from .widgets import MplCanvas
from . import qt_resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1067, 871)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QSize(16777215, 16777215))
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
        self.action_Point_measurement = QAction(MainWindow)
        self.action_Point_measurement.setObjectName(u"action_Point_measurement")
        self.action_Power_Meter = QAction(MainWindow)
        self.action_Power_Meter.setObjectName(u"action_Power_Meter")
        self.action_Power_Meter.setCheckable(True)
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/application-monitor.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_Power_Meter.setIcon(icon1)
        self.action_Logs = QAction(MainWindow)
        self.action_Logs.setObjectName(u"action_Logs")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.LogTextEdit = QPlainTextEdit(self.centralwidget)
        self.LogTextEdit.setObjectName(u"LogTextEdit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.LogTextEdit.sizePolicy().hasHeightForWidth())
        self.LogTextEdit.setSizePolicy(sizePolicy1)
        self.LogTextEdit.setMinimumSize(QSize(0, 0))
        self.LogTextEdit.setMaximumSize(QSize(16777215, 200))

        self.verticalLayout.addWidget(self.LogTextEdit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1067, 21))
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
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.dock_pm = QDockWidget(MainWindow)
        self.dock_pm.setObjectName(u"dock_pm")
        self.dock_pm.setEnabled(True)
        self.dock_pm.setMinimumSize(QSize(584, 250))
        self.dock_pm.setAcceptDrops(True)
        self.dock_pm.setFloating(True)
        self.w_pm = QWidget()
        self.w_pm.setObjectName(u"w_pm")
        self.verticalLayout_2 = QVBoxLayout(self.w_pm)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.lbl_cur_en = QLabel(self.w_pm)
        self.lbl_cur_en.setObjectName(u"lbl_cur_en")

        self.horizontalLayout_9.addWidget(self.lbl_cur_en)

        self.le_cur_en = QLineEdit(self.w_pm)
        self.le_cur_en.setObjectName(u"le_cur_en")
        self.le_cur_en.setEnabled(False)
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.le_cur_en.sizePolicy().hasHeightForWidth())
        self.le_cur_en.setSizePolicy(sizePolicy2)
        self.le_cur_en.setMinimumSize(QSize(50, 20))
        self.le_cur_en.setMaximumSize(QSize(50, 16777215))
        self.le_cur_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_cur_en)

        self.lbl_aver_en = QLabel(self.w_pm)
        self.lbl_aver_en.setObjectName(u"lbl_aver_en")

        self.horizontalLayout_9.addWidget(self.lbl_aver_en)

        self.le_aver_en = QLineEdit(self.w_pm)
        self.le_aver_en.setObjectName(u"le_aver_en")
        self.le_aver_en.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.le_aver_en.sizePolicy().hasHeightForWidth())
        self.le_aver_en.setSizePolicy(sizePolicy2)
        self.le_aver_en.setMinimumSize(QSize(50, 20))
        self.le_aver_en.setMaximumSize(QSize(50, 16777215))
        self.le_aver_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_aver_en)

        self.lbl_std_en = QLabel(self.w_pm)
        self.lbl_std_en.setObjectName(u"lbl_std_en")

        self.horizontalLayout_9.addWidget(self.lbl_std_en)

        self.le_std_en = QLineEdit(self.w_pm)
        self.le_std_en.setObjectName(u"le_std_en")
        self.le_std_en.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.le_std_en.sizePolicy().hasHeightForWidth())
        self.le_std_en.setSizePolicy(sizePolicy2)
        self.le_std_en.setMinimumSize(QSize(50, 20))
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
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.plot_pm_left.sizePolicy().hasHeightForWidth())
        self.plot_pm_left.setSizePolicy(sizePolicy3)

        self.horizontalLayout_10.addWidget(self.plot_pm_left)

        self.plot_pm_right = MplCanvas(self.w_pm)
        self.plot_pm_right.setObjectName(u"plot_pm_right")
        sizePolicy3.setHeightForWidth(self.plot_pm_right.sizePolicy().hasHeightForWidth())
        self.plot_pm_right.setSizePolicy(sizePolicy3)

        self.horizontalLayout_10.addWidget(self.plot_pm_right)


        self.verticalLayout_2.addLayout(self.horizontalLayout_10)

        self.dock_pm.setWidget(self.w_pm)
        MainWindow.addDockWidget(Qt.RightDockWidgetArea, self.dock_pm)

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
        self.menu_Measurements.addAction(self.action_Point_measurement)
        self.menu_Measurements.addSeparator()
        self.menu_Measurements.addAction(self.action_Power_Meter)
        self.menu_Tools.addAction(self.action_Logs)
        self.menu_Tools.addSeparator()
        self.toolBar.addAction(self.action_Init)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_Power_Meter)

        self.retranslateUi(MainWindow)
        self.action_Power_Meter.toggled.connect(self.dock_pm.setVisible)
        self.dock_pm.visibilityChanged.connect(self.action_Power_Meter.setChecked)

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
        self.action_Point_measurement.setText(QCoreApplication.translate("MainWindow", u"&Point Measurement", None))
        self.action_Power_Meter.setText(QCoreApplication.translate("MainWindow", u"&Power Meter", None))
        self.action_Logs.setText(QCoreApplication.translate("MainWindow", u"&Logs", None))
        self.menu_Data.setTitle(QCoreApplication.translate("MainWindow", u"&Data", None))
        self.menu_Hardware.setTitle(QCoreApplication.translate("MainWindow", u"&Hardware", None))
        self.menu_Measurements.setTitle(QCoreApplication.translate("MainWindow", u"&Measurements", None))
        self.menu_Tools.setTitle(QCoreApplication.translate("MainWindow", u"&Tools", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
#if QT_CONFIG(accessibility)
        self.dock_pm.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
        self.dock_pm.setWindowTitle(QCoreApplication.translate("MainWindow", u"Power Meter", None))
        self.lbl_cur_en.setText(QCoreApplication.translate("MainWindow", u"Current Energy:", None))
        self.lbl_aver_en.setText(QCoreApplication.translate("MainWindow", u"Average Energy:", None))
        self.lbl_std_en.setText(QCoreApplication.translate("MainWindow", u"std Energy:", None))
        self.btn_pm_start.setText(QCoreApplication.translate("MainWindow", u"START", None))
        self.btn_pm_stop.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
    # retranslateUi

