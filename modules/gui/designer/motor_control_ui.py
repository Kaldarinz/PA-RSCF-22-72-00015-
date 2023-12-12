# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'motor_control.ui'
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
from PySide6.QtWidgets import (QApplication, QDockWidget, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)

from ..widgets import MplCanvas
from . import qt_resources_rc
from . import qt_resources_rc

class Ui_DockWidget(object):
    def setupUi(self, DockWidget):
        if not DockWidget.objectName():
            DockWidget.setObjectName(u"DockWidget")
        DockWidget.resize(922, 749)
        DockWidget.setFloating(True)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(6, -1, -1, -1)
        self.lbl_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_cur_pos.setObjectName(u"lbl_cur_pos")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_cur_pos.sizePolicy().hasHeightForWidth())
        self.lbl_cur_pos.setSizePolicy(sizePolicy)
        font = QFont()
        font.setBold(True)
        self.lbl_cur_pos.setFont(font)
        self.lbl_cur_pos.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.lbl_cur_pos)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(6, -1, -1, 6)
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.lbl_x_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_x_cur_pos.setObjectName(u"lbl_x_cur_pos")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lbl_x_cur_pos.sizePolicy().hasHeightForWidth())
        self.lbl_x_cur_pos.setSizePolicy(sizePolicy1)
        self.lbl_x_cur_pos.setFont(font)

        self.gridLayout_2.addWidget(self.lbl_x_cur_pos, 0, 0, 1, 1)

        self.icon_x_status = QLabel(self.dockWidgetContents)
        self.icon_x_status.setObjectName(u"icon_x_status")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.icon_x_status.sizePolicy().hasHeightForWidth())
        self.icon_x_status.setSizePolicy(sizePolicy2)
        self.icon_x_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-disconnect-prohibition.png"))

        self.gridLayout_2.addWidget(self.icon_x_status, 0, 1, 1, 1)

        self.le_x_cur_pos = QLineEdit(self.dockWidgetContents)
        self.le_x_cur_pos.setObjectName(u"le_x_cur_pos")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.le_x_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_x_cur_pos.setSizePolicy(sizePolicy3)
        self.le_x_cur_pos.setMaximumSize(QSize(16777215, 16777215))
        self.le_x_cur_pos.setReadOnly(True)

        self.gridLayout_2.addWidget(self.le_x_cur_pos, 0, 2, 1, 1)

        self.lbl_x_status = QLabel(self.dockWidgetContents)
        self.lbl_x_status.setObjectName(u"lbl_x_status")
        self.lbl_x_status.setMinimumSize(QSize(120, 0))

        self.gridLayout_2.addWidget(self.lbl_x_status, 0, 3, 1, 1)

        self.lbl_y_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_y_cur_pos.setObjectName(u"lbl_y_cur_pos")
        self.lbl_y_cur_pos.setFont(font)

        self.gridLayout_2.addWidget(self.lbl_y_cur_pos, 1, 0, 1, 1)

        self.icon_y_status = QLabel(self.dockWidgetContents)
        self.icon_y_status.setObjectName(u"icon_y_status")
        sizePolicy.setHeightForWidth(self.icon_y_status.sizePolicy().hasHeightForWidth())
        self.icon_y_status.setSizePolicy(sizePolicy)
        self.icon_y_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-disconnect-prohibition.png"))

        self.gridLayout_2.addWidget(self.icon_y_status, 1, 1, 1, 1)

        self.le_y_cur_pos = QLineEdit(self.dockWidgetContents)
        self.le_y_cur_pos.setObjectName(u"le_y_cur_pos")
        sizePolicy4 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.le_y_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_y_cur_pos.setSizePolicy(sizePolicy4)
        self.le_y_cur_pos.setMaximumSize(QSize(16777215, 16777215))
        self.le_y_cur_pos.setReadOnly(True)

        self.gridLayout_2.addWidget(self.le_y_cur_pos, 1, 2, 1, 1)

        self.lbl_y_status = QLabel(self.dockWidgetContents)
        self.lbl_y_status.setObjectName(u"lbl_y_status")

        self.gridLayout_2.addWidget(self.lbl_y_status, 1, 3, 1, 1)

        self.lbl_z_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_z_cur_pos.setObjectName(u"lbl_z_cur_pos")
        self.lbl_z_cur_pos.setFont(font)

        self.gridLayout_2.addWidget(self.lbl_z_cur_pos, 2, 0, 1, 1)

        self.icon_z_status = QLabel(self.dockWidgetContents)
        self.icon_z_status.setObjectName(u"icon_z_status")
        sizePolicy.setHeightForWidth(self.icon_z_status.sizePolicy().hasHeightForWidth())
        self.icon_z_status.setSizePolicy(sizePolicy)
        self.icon_z_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-disconnect-prohibition.png"))

        self.gridLayout_2.addWidget(self.icon_z_status, 2, 1, 1, 1)

        self.le_z_cur_pos = QLineEdit(self.dockWidgetContents)
        self.le_z_cur_pos.setObjectName(u"le_z_cur_pos")
        sizePolicy4.setHeightForWidth(self.le_z_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_z_cur_pos.setSizePolicy(sizePolicy4)
        self.le_z_cur_pos.setMaximumSize(QSize(16777215, 16777215))
        self.le_z_cur_pos.setReadOnly(True)

        self.gridLayout_2.addWidget(self.le_z_cur_pos, 2, 2, 1, 1)

        self.lbl_z_status = QLabel(self.dockWidgetContents)
        self.lbl_z_status.setObjectName(u"lbl_z_status")

        self.gridLayout_2.addWidget(self.lbl_z_status, 2, 3, 1, 1)


        self.horizontalLayout_4.addLayout(self.gridLayout_2)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.line = QFrame(self.dockWidgetContents)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_2.addWidget(self.line)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(6, 6, -1, -1)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, 6, 6, -1)
        self.lbl_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_new_pos.setObjectName(u"lbl_new_pos")
        sizePolicy.setHeightForWidth(self.lbl_new_pos.sizePolicy().hasHeightForWidth())
        self.lbl_new_pos.setSizePolicy(sizePolicy)
        self.lbl_new_pos.setFont(font)
        self.lbl_new_pos.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_new_pos)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setSizeConstraint(QLayout.SetMinimumSize)
        self.formLayout_2.setHorizontalSpacing(6)
        self.lbl_x_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_x_new_pos.setObjectName(u"lbl_x_new_pos")
        self.lbl_x_new_pos.setFont(font)

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.lbl_x_new_pos)

        self.sb_x_new_pos = QDoubleSpinBox(self.dockWidgetContents)
        self.sb_x_new_pos.setObjectName(u"sb_x_new_pos")
        sizePolicy4.setHeightForWidth(self.sb_x_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos.setSizePolicy(sizePolicy4)
        self.sb_x_new_pos.setMinimumSize(QSize(74, 0))
        self.sb_x_new_pos.setDecimals(2)

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.sb_x_new_pos)

        self.lbl_y_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_y_new_pos.setObjectName(u"lbl_y_new_pos")
        self.lbl_y_new_pos.setFont(font)

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.lbl_y_new_pos)

        self.sb_y_new_pos = QDoubleSpinBox(self.dockWidgetContents)
        self.sb_y_new_pos.setObjectName(u"sb_y_new_pos")
        sizePolicy4.setHeightForWidth(self.sb_y_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_y_new_pos.setSizePolicy(sizePolicy4)
        self.sb_y_new_pos.setDecimals(2)

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.sb_y_new_pos)

        self.lbl_z_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_z_new_pos.setObjectName(u"lbl_z_new_pos")
        self.lbl_z_new_pos.setFont(font)

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.lbl_z_new_pos)

        self.sb_z_new_pos = QDoubleSpinBox(self.dockWidgetContents)
        self.sb_z_new_pos.setObjectName(u"sb_z_new_pos")
        sizePolicy4.setHeightForWidth(self.sb_z_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_z_new_pos.setSizePolicy(sizePolicy4)
        self.sb_z_new_pos.setDecimals(2)

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.sb_z_new_pos)


        self.verticalLayout.addLayout(self.formLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btn_home = QPushButton(self.dockWidgetContents)
        self.btn_home.setObjectName(u"btn_home")
        self.btn_home.setMinimumSize(QSize(64, 0))
        self.btn_home.setMaximumSize(QSize(60, 16777215))
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/home--arrow.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_home.setIcon(icon)

        self.horizontalLayout.addWidget(self.btn_home)

        self.btn_go = QPushButton(self.dockWidgetContents)
        self.btn_go.setObjectName(u"btn_go")
        sizePolicy4.setHeightForWidth(self.btn_go.sizePolicy().hasHeightForWidth())
        self.btn_go.setSizePolicy(sizePolicy4)
        self.btn_go.setMinimumSize(QSize(0, 0))
        self.btn_go.setMaximumSize(QSize(16777215, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_go.setIcon(icon1)

        self.horizontalLayout.addWidget(self.btn_go)

        self.btn_stop = QPushButton(self.dockWidgetContents)
        self.btn_stop.setObjectName(u"btn_stop")
        sizePolicy4.setHeightForWidth(self.btn_stop.sizePolicy().hasHeightForWidth())
        self.btn_stop.setSizePolicy(sizePolicy4)
        self.btn_stop.setMinimumSize(QSize(0, 0))
        self.btn_stop.setMaximumSize(QSize(16777215, 16777215))
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon2)

        self.horizontalLayout.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.line_2 = QFrame(self.dockWidgetContents)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.lbl_move = QLabel(self.dockWidgetContents)
        self.lbl_move.setObjectName(u"lbl_move")
        sizePolicy.setHeightForWidth(self.lbl_move.sizePolicy().hasHeightForWidth())
        self.lbl_move.setSizePolicy(sizePolicy)
        self.lbl_move.setFont(font)
        self.lbl_move.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_move)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.lbl_x_move = QLabel(self.dockWidgetContents)
        self.lbl_x_move.setObjectName(u"lbl_x_move")
        sizePolicy.setHeightForWidth(self.lbl_x_move.sizePolicy().hasHeightForWidth())
        self.lbl_x_move.setSizePolicy(sizePolicy)
        self.lbl_x_move.setMaximumSize(QSize(10, 16777215))
        self.lbl_x_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_x_move, 0, 0, 1, 1)

        self.btn_x_left_move = QPushButton(self.dockWidgetContents)
        self.btn_x_left_move.setObjectName(u"btn_x_left_move")
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/control-double-180.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_x_left_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_x_left_move, 0, 1, 1, 1)

        self.btn_x_right_move = QPushButton(self.dockWidgetContents)
        self.btn_x_right_move.setObjectName(u"btn_x_right_move")
        icon4 = QIcon()
        icon4.addFile(u":/icons/qt_resources/control-double.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_x_right_move.setIcon(icon4)

        self.gridLayout.addWidget(self.btn_x_right_move, 0, 2, 1, 1)

        self.lbl_y_move = QLabel(self.dockWidgetContents)
        self.lbl_y_move.setObjectName(u"lbl_y_move")
        sizePolicy.setHeightForWidth(self.lbl_y_move.sizePolicy().hasHeightForWidth())
        self.lbl_y_move.setSizePolicy(sizePolicy)
        self.lbl_y_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_y_move, 1, 0, 1, 1)

        self.btn_y_left_move = QPushButton(self.dockWidgetContents)
        self.btn_y_left_move.setObjectName(u"btn_y_left_move")
        self.btn_y_left_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_y_left_move, 1, 1, 1, 1)

        self.btn_y_right_move = QPushButton(self.dockWidgetContents)
        self.btn_y_right_move.setObjectName(u"btn_y_right_move")
        self.btn_y_right_move.setIcon(icon4)

        self.gridLayout.addWidget(self.btn_y_right_move, 1, 2, 1, 1)

        self.lbl_z_move = QLabel(self.dockWidgetContents)
        self.lbl_z_move.setObjectName(u"lbl_z_move")
        sizePolicy.setHeightForWidth(self.lbl_z_move.sizePolicy().hasHeightForWidth())
        self.lbl_z_move.setSizePolicy(sizePolicy)
        self.lbl_z_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_z_move, 2, 0, 1, 1)

        self.btn_z_left_move = QPushButton(self.dockWidgetContents)
        self.btn_z_left_move.setObjectName(u"btn_z_left_move")
        self.btn_z_left_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_z_left_move, 2, 1, 1, 1)

        self.btn_z_right_move = QPushButton(self.dockWidgetContents)
        self.btn_z_right_move.setObjectName(u"btn_z_right_move")
        self.btn_z_right_move.setIcon(icon4)

        self.gridLayout.addWidget(self.btn_z_right_move, 2, 2, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.line_3 = QFrame(self.dockWidgetContents)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.le_test = QLineEdit(self.dockWidgetContents)
        self.le_test.setObjectName(u"le_test")
        sizePolicy4.setHeightForWidth(self.le_test.sizePolicy().hasHeightForWidth())
        self.le_test.setSizePolicy(sizePolicy4)

        self.verticalLayout.addWidget(self.le_test)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.line_4 = QFrame(self.dockWidgetContents)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.VLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_3.addWidget(self.line_4)

        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.plot_xz = MplCanvas(self.dockWidgetContents)
        self.plot_xz.setObjectName(u"plot_xz")
        sizePolicy5 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.plot_xz.sizePolicy().hasHeightForWidth())
        self.plot_xz.setSizePolicy(sizePolicy5)

        self.gridLayout_4.addWidget(self.plot_xz, 0, 0, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.plot_xy = MplCanvas(self.dockWidgetContents)
        self.plot_xy.setObjectName(u"plot_xy")
        sizePolicy5.setHeightForWidth(self.plot_xy.sizePolicy().hasHeightForWidth())
        self.plot_xy.setSizePolicy(sizePolicy5)

        self.gridLayout_3.addWidget(self.plot_xy, 0, 0, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_3, 1, 0, 1, 1)

        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.plot_yz = MplCanvas(self.dockWidgetContents)
        self.plot_yz.setObjectName(u"plot_yz")
        sizePolicy5.setHeightForWidth(self.plot_yz.sizePolicy().hasHeightForWidth())
        self.plot_yz.setSizePolicy(sizePolicy5)

        self.gridLayout_5.addWidget(self.plot_yz, 0, 0, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_5, 1, 1, 1, 1)


        self.horizontalLayout_3.addLayout(self.gridLayout_6)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        DockWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(DockWidget)

        QMetaObject.connectSlotsByName(DockWidget)
    # setupUi

    def retranslateUi(self, DockWidget):
        DockWidget.setWindowTitle(QCoreApplication.translate("DockWidget", u"Motor Mover", None))
        self.lbl_cur_pos.setText(QCoreApplication.translate("DockWidget", u"Position & Status", None))
        self.lbl_x_cur_pos.setText(QCoreApplication.translate("DockWidget", u"X", None))
        self.icon_x_status.setText("")
        self.le_x_cur_pos.setText(QCoreApplication.translate("DockWidget", u"0", None))
        self.lbl_x_status.setText(QCoreApplication.translate("DockWidget", u"Not Set", None))
        self.lbl_y_cur_pos.setText(QCoreApplication.translate("DockWidget", u"Y", None))
        self.icon_y_status.setText("")
        self.le_y_cur_pos.setText(QCoreApplication.translate("DockWidget", u"0", None))
        self.lbl_y_status.setText(QCoreApplication.translate("DockWidget", u"Not Set", None))
        self.lbl_z_cur_pos.setText(QCoreApplication.translate("DockWidget", u"Z", None))
        self.icon_z_status.setText("")
        self.le_z_cur_pos.setText(QCoreApplication.translate("DockWidget", u"0", None))
        self.lbl_z_status.setText(QCoreApplication.translate("DockWidget", u"Not Set", None))
        self.lbl_new_pos.setText(QCoreApplication.translate("DockWidget", u"New Position", None))
        self.lbl_x_new_pos.setText(QCoreApplication.translate("DockWidget", u"X", None))
        self.sb_x_new_pos.setSuffix(QCoreApplication.translate("DockWidget", u" mm", None))
        self.lbl_y_new_pos.setText(QCoreApplication.translate("DockWidget", u"Y", None))
        self.sb_y_new_pos.setSuffix(QCoreApplication.translate("DockWidget", u" mm", None))
        self.lbl_z_new_pos.setText(QCoreApplication.translate("DockWidget", u"Z", None))
        self.sb_z_new_pos.setSuffix(QCoreApplication.translate("DockWidget", u" mm", None))
        self.btn_home.setText(QCoreApplication.translate("DockWidget", u"Home", None))
        self.btn_go.setText(QCoreApplication.translate("DockWidget", u"Go to", None))
        self.btn_stop.setText("")
        self.lbl_move.setText(QCoreApplication.translate("DockWidget", u"Move", None))
        self.lbl_x_move.setText(QCoreApplication.translate("DockWidget", u"X", None))
        self.btn_x_left_move.setText("")
        self.btn_x_right_move.setText("")
        self.lbl_y_move.setText(QCoreApplication.translate("DockWidget", u"Y", None))
        self.btn_y_left_move.setText("")
        self.btn_y_right_move.setText("")
        self.lbl_z_move.setText(QCoreApplication.translate("DockWidget", u"Z", None))
        self.btn_z_left_move.setText("")
        self.btn_z_right_move.setText("")
    # retranslateUi

