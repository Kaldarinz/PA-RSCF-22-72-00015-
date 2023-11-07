# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pm_dock_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QDockWidget, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

from ..widgets import MplCanvas

class Ui_d_pm(object):
    def setupUi(self, d_pm):
        if not d_pm.objectName():
            d_pm.setObjectName(u"d_pm")
        d_pm.resize(893, 304)
        d_pm.setFloating(True)
        self.w_pm = QWidget()
        self.w_pm.setObjectName(u"w_pm")
        self.verticalLayout = QVBoxLayout(self.w_pm)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.lbl_cur_en = QLabel(self.w_pm)
        self.lbl_cur_en.setObjectName(u"lbl_cur_en")

        self.horizontalLayout_9.addWidget(self.lbl_cur_en)

        self.le_cur_en = QLineEdit(self.w_pm)
        self.le_cur_en.setObjectName(u"le_cur_en")
        self.le_cur_en.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_cur_en.sizePolicy().hasHeightForWidth())
        self.le_cur_en.setSizePolicy(sizePolicy)
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
        sizePolicy.setHeightForWidth(self.le_aver_en.sizePolicy().hasHeightForWidth())
        self.le_aver_en.setSizePolicy(sizePolicy)
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
        sizePolicy.setHeightForWidth(self.le_std_en.sizePolicy().hasHeightForWidth())
        self.le_std_en.setSizePolicy(sizePolicy)
        self.le_std_en.setMinimumSize(QSize(70, 20))
        self.le_std_en.setMaximumSize(QSize(50, 16777215))
        self.le_std_en.setReadOnly(True)

        self.horizontalLayout_9.addWidget(self.le_std_en)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer)

        self.btn_start = QPushButton(self.w_pm)
        self.btn_start.setObjectName(u"btn_start")
        font = QFont()
        font.setBold(True)
        self.btn_start.setFont(font)

        self.horizontalLayout_9.addWidget(self.btn_start)

        self.btn_pause = QPushButton(self.w_pm)
        self.btn_pause.setObjectName(u"btn_pause")
        self.btn_pause.setFont(font)
        self.btn_pause.setCheckable(True)

        self.horizontalLayout_9.addWidget(self.btn_pause)

        self.btn_stop = QPushButton(self.w_pm)
        self.btn_stop.setObjectName(u"btn_stop")
        font1 = QFont()
        font1.setBold(True)
        font1.setUnderline(False)
        font1.setStrikeOut(False)
        self.btn_stop.setFont(font1)

        self.horizontalLayout_9.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.plot_left = MplCanvas(self.w_pm)
        self.plot_left.setObjectName(u"plot_left")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plot_left.sizePolicy().hasHeightForWidth())
        self.plot_left.setSizePolicy(sizePolicy1)

        self.horizontalLayout_10.addWidget(self.plot_left)

        self.plot_right = MplCanvas(self.w_pm)
        self.plot_right.setObjectName(u"plot_right")
        sizePolicy1.setHeightForWidth(self.plot_right.sizePolicy().hasHeightForWidth())
        self.plot_right.setSizePolicy(sizePolicy1)

        self.horizontalLayout_10.addWidget(self.plot_right)


        self.verticalLayout.addLayout(self.horizontalLayout_10)

        d_pm.setWidget(self.w_pm)

        self.retranslateUi(d_pm)

        QMetaObject.connectSlotsByName(d_pm)
    # setupUi

    def retranslateUi(self, d_pm):
        d_pm.setWindowTitle(QCoreApplication.translate("d_pm", u"Power Meter", None))
        self.lbl_cur_en.setText(QCoreApplication.translate("d_pm", u"Current Energy:", None))
        self.lbl_aver_en.setText(QCoreApplication.translate("d_pm", u"Average Energy:", None))
        self.lbl_std_en.setText(QCoreApplication.translate("d_pm", u"std Energy:", None))
        self.btn_start.setText(QCoreApplication.translate("d_pm", u"START", None))
        self.btn_pause.setText(QCoreApplication.translate("d_pm", u"PAUSE", None))
        self.btn_stop.setText(QCoreApplication.translate("d_pm", u"STOP", None))
    # retranslateUi

