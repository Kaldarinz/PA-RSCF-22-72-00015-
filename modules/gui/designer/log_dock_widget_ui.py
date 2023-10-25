# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'log_dock_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QDockWidget, QHBoxLayout, QPlainTextEdit,
    QSizePolicy, QWidget)

class Ui_d_log(object):
    def setupUi(self, d_log):
        if not d_log.objectName():
            d_log.setObjectName(u"d_log")
        d_log.resize(642, 499)
        d_log.setFloating(True)
        d_log.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.w_log = QWidget()
        self.w_log.setObjectName(u"w_log")
        self.horizontalLayout = QHBoxLayout(self.w_log)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.te_log = QPlainTextEdit(self.w_log)
        self.te_log.setObjectName(u"te_log")
        self.te_log.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.te_log.sizePolicy().hasHeightForWidth())
        self.te_log.setSizePolicy(sizePolicy)
        self.te_log.setMinimumSize(QSize(0, 0))
        self.te_log.setMaximumSize(QSize(16777215, 16777215))
        self.te_log.setReadOnly(True)

        self.horizontalLayout.addWidget(self.te_log)

        d_log.setWidget(self.w_log)

        self.retranslateUi(d_log)

        QMetaObject.connectSlotsByName(d_log)
    # setupUi

    def retranslateUi(self, d_log):
        d_log.setWindowTitle(QCoreApplication.translate("d_log", u"Logs", None))
    # retranslateUi

