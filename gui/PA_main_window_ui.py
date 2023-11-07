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
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenu, QMenuBar,
    QPlainTextEdit, QSizePolicy, QSpacerItem, QStatusBar,
    QToolBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(977, 487)
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
        self.action_Point_measurement = QAction(MainWindow)
        self.action_Point_measurement.setObjectName(u"action_Point_measurement")
        self.action_Power_Meter = QAction(MainWindow)
        self.action_Power_Meter.setObjectName(u"action_Power_Meter")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalSpacer = QSpacerItem(20, 204, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.LogTextEdit = QPlainTextEdit(self.centralwidget)
        self.LogTextEdit.setObjectName(u"LogTextEdit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.menubar.setGeometry(QRect(0, 0, 977, 21))
        self.menu_Data = QMenu(self.menubar)
        self.menu_Data.setObjectName(u"menu_Data")
        self.menu_Hardware = QMenu(self.menubar)
        self.menu_Hardware.setObjectName(u"menu_Hardware")
        self.menu_Measurements = QMenu(self.menubar)
        self.menu_Measurements.setObjectName(u"menu_Measurements")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.menubar.addAction(self.menu_Data.menuAction())
        self.menubar.addAction(self.menu_Hardware.menuAction())
        self.menubar.addAction(self.menu_Measurements.menuAction())
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
        self.toolBar.addAction(self.action_Init)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
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
        self.menu_Data.setTitle(QCoreApplication.translate("MainWindow", u"&Data", None))
        self.menu_Hardware.setTitle(QCoreApplication.translate("MainWindow", u"&Hardware", None))
        self.menu_Measurements.setTitle(QCoreApplication.translate("MainWindow", u"&Measurements", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

