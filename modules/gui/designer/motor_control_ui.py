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
    QSlider, QSpacerItem, QVBoxLayout, QWidget)

from ..widgets import MplCanvas
from . import qt_resources_rc
from . import qt_resources_rc

class Ui_DockWidget(object):
    def setupUi(self, DockWidget):
        if not DockWidget.objectName():
            DockWidget.setObjectName(u"DockWidget")
        DockWidget.resize(654, 518)
        DockWidget.setFloating(True)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.horizontalLayout = QHBoxLayout(self.dockWidgetContents)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
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

        self.verticalLayout.addWidget(self.lbl_cur_pos)

        self.line_5 = QFrame(self.dockWidgetContents)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.VLine)
        self.line_5.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_5)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.lbl_x_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_x_cur_pos.setObjectName(u"lbl_x_cur_pos")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lbl_x_cur_pos.sizePolicy().hasHeightForWidth())
        self.lbl_x_cur_pos.setSizePolicy(sizePolicy1)
        self.lbl_x_cur_pos.setFont(font)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.lbl_x_cur_pos)

        self.le_x_cur_pos = QLineEdit(self.dockWidgetContents)
        self.le_x_cur_pos.setObjectName(u"le_x_cur_pos")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.le_x_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_x_cur_pos.setSizePolicy(sizePolicy2)
        self.le_x_cur_pos.setMaximumSize(QSize(16777215, 16777215))
        self.le_x_cur_pos.setReadOnly(True)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.le_x_cur_pos)

        self.lbl_y_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_y_cur_pos.setObjectName(u"lbl_y_cur_pos")
        self.lbl_y_cur_pos.setFont(font)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.lbl_y_cur_pos)

        self.le_y_cur_pos = QLineEdit(self.dockWidgetContents)
        self.le_y_cur_pos.setObjectName(u"le_y_cur_pos")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.le_y_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_y_cur_pos.setSizePolicy(sizePolicy3)
        self.le_y_cur_pos.setMaximumSize(QSize(16777215, 16777215))
        self.le_y_cur_pos.setReadOnly(True)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.le_y_cur_pos)

        self.lbl_z_cur_pos = QLabel(self.dockWidgetContents)
        self.lbl_z_cur_pos.setObjectName(u"lbl_z_cur_pos")
        self.lbl_z_cur_pos.setFont(font)

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.lbl_z_cur_pos)

        self.le_z_cur_pos = QLineEdit(self.dockWidgetContents)
        self.le_z_cur_pos.setObjectName(u"le_z_cur_pos")
        sizePolicy3.setHeightForWidth(self.le_z_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_z_cur_pos.setSizePolicy(sizePolicy3)
        self.le_z_cur_pos.setMaximumSize(QSize(16777215, 16777215))
        self.le_z_cur_pos.setReadOnly(True)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.le_z_cur_pos)


        self.verticalLayout.addLayout(self.formLayout)

        self.line = QFrame(self.dockWidgetContents)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

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
        self.lbl_x_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_x_new_pos.setObjectName(u"lbl_x_new_pos")
        self.lbl_x_new_pos.setFont(font)

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.lbl_x_new_pos)

        self.sb_x_new_pos = QDoubleSpinBox(self.dockWidgetContents)
        self.sb_x_new_pos.setObjectName(u"sb_x_new_pos")
        sizePolicy3.setHeightForWidth(self.sb_x_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos.setSizePolicy(sizePolicy3)
        self.sb_x_new_pos.setMinimumSize(QSize(74, 0))
        self.sb_x_new_pos.setDecimals(2)

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.sb_x_new_pos)

        self.lbl_y_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_y_new_pos.setObjectName(u"lbl_y_new_pos")
        self.lbl_y_new_pos.setFont(font)

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.lbl_y_new_pos)

        self.sb_y_new_pos = QDoubleSpinBox(self.dockWidgetContents)
        self.sb_y_new_pos.setObjectName(u"sb_y_new_pos")
        sizePolicy3.setHeightForWidth(self.sb_y_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_y_new_pos.setSizePolicy(sizePolicy3)
        self.sb_y_new_pos.setDecimals(2)

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.sb_y_new_pos)

        self.lbl_z_new_pos = QLabel(self.dockWidgetContents)
        self.lbl_z_new_pos.setObjectName(u"lbl_z_new_pos")
        self.lbl_z_new_pos.setFont(font)

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.lbl_z_new_pos)

        self.sb_z_new_pos = QDoubleSpinBox(self.dockWidgetContents)
        self.sb_z_new_pos.setObjectName(u"sb_z_new_pos")
        sizePolicy3.setHeightForWidth(self.sb_z_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_z_new_pos.setSizePolicy(sizePolicy3)
        self.sb_z_new_pos.setDecimals(2)

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.sb_z_new_pos)


        self.verticalLayout.addLayout(self.formLayout_2)

        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setSizeConstraint(QLayout.SetMinimumSize)
        self.formLayout_3.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.btn_go = QPushButton(self.dockWidgetContents)
        self.btn_go.setObjectName(u"btn_go")
        sizePolicy3.setHeightForWidth(self.btn_go.sizePolicy().hasHeightForWidth())
        self.btn_go.setSizePolicy(sizePolicy3)
        self.btn_go.setMinimumSize(QSize(50, 0))
        self.btn_go.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_go.setIcon(icon)

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.btn_go)

        self.btn_stop = QPushButton(self.dockWidgetContents)
        self.btn_stop.setObjectName(u"btn_stop")
        sizePolicy3.setHeightForWidth(self.btn_stop.sizePolicy().hasHeightForWidth())
        self.btn_stop.setSizePolicy(sizePolicy3)
        self.btn_stop.setMinimumSize(QSize(50, 0))
        self.btn_stop.setMaximumSize(QSize(16777215, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon1)

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.btn_stop)


        self.verticalLayout.addLayout(self.formLayout_3)

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
        self.lbl_x_move.setMaximumSize(QSize(6, 16777215))
        self.lbl_x_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_x_move, 0, 0, 1, 1)

        self.btn_x_left_move = QPushButton(self.dockWidgetContents)
        self.btn_x_left_move.setObjectName(u"btn_x_left_move")
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/control-double-180.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_x_left_move.setIcon(icon2)

        self.gridLayout.addWidget(self.btn_x_left_move, 0, 1, 1, 1)

        self.btn_x_right_move = QPushButton(self.dockWidgetContents)
        self.btn_x_right_move.setObjectName(u"btn_x_right_move")
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/control-double.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_x_right_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_x_right_move, 0, 2, 1, 1)

        self.lbl_y_move = QLabel(self.dockWidgetContents)
        self.lbl_y_move.setObjectName(u"lbl_y_move")
        sizePolicy.setHeightForWidth(self.lbl_y_move.sizePolicy().hasHeightForWidth())
        self.lbl_y_move.setSizePolicy(sizePolicy)
        self.lbl_y_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_y_move, 1, 0, 1, 1)

        self.btn_y_left_move = QPushButton(self.dockWidgetContents)
        self.btn_y_left_move.setObjectName(u"btn_y_left_move")
        self.btn_y_left_move.setIcon(icon2)

        self.gridLayout.addWidget(self.btn_y_left_move, 1, 1, 1, 1)

        self.btn_y_right_move = QPushButton(self.dockWidgetContents)
        self.btn_y_right_move.setObjectName(u"btn_y_right_move")
        self.btn_y_right_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_y_right_move, 1, 2, 1, 1)

        self.lbl_z_move = QLabel(self.dockWidgetContents)
        self.lbl_z_move.setObjectName(u"lbl_z_move")
        sizePolicy.setHeightForWidth(self.lbl_z_move.sizePolicy().hasHeightForWidth())
        self.lbl_z_move.setSizePolicy(sizePolicy)
        self.lbl_z_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_z_move, 2, 0, 1, 1)

        self.btn_z_left_move = QPushButton(self.dockWidgetContents)
        self.btn_z_left_move.setObjectName(u"btn_z_left_move")
        self.btn_z_left_move.setIcon(icon2)

        self.gridLayout.addWidget(self.btn_z_left_move, 2, 1, 1, 1)

        self.btn_z_right_move = QPushButton(self.dockWidgetContents)
        self.btn_z_right_move.setObjectName(u"btn_z_right_move")
        self.btn_z_right_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_z_right_move, 2, 2, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.line_3 = QFrame(self.dockWidgetContents)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.lbl_status = QLabel(self.dockWidgetContents)
        self.lbl_status.setObjectName(u"lbl_status")
        sizePolicy.setHeightForWidth(self.lbl_status.sizePolicy().hasHeightForWidth())
        self.lbl_status.setSizePolicy(sizePolicy)
        self.lbl_status.setFont(font)
        self.lbl_status.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_status)

        self.gridLayout_8 = QGridLayout()
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setSizeConstraint(QLayout.SetMaximumSize)
        self.lbl_y_title_status = QLabel(self.dockWidgetContents)
        self.lbl_y_title_status.setObjectName(u"lbl_y_title_status")
        sizePolicy1.setHeightForWidth(self.lbl_y_title_status.sizePolicy().hasHeightForWidth())
        self.lbl_y_title_status.setSizePolicy(sizePolicy1)
        self.lbl_y_title_status.setFont(font)

        self.gridLayout_8.addWidget(self.lbl_y_title_status, 1, 0, 1, 1)

        self.lbl_z_title_status = QLabel(self.dockWidgetContents)
        self.lbl_z_title_status.setObjectName(u"lbl_z_title_status")
        sizePolicy1.setHeightForWidth(self.lbl_z_title_status.sizePolicy().hasHeightForWidth())
        self.lbl_z_title_status.setSizePolicy(sizePolicy1)
        self.lbl_z_title_status.setFont(font)

        self.gridLayout_8.addWidget(self.lbl_z_title_status, 2, 0, 1, 1)

        self.icon_x_status = QLabel(self.dockWidgetContents)
        self.icon_x_status.setObjectName(u"icon_x_status")
        sizePolicy4 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.icon_x_status.sizePolicy().hasHeightForWidth())
        self.icon_x_status.setSizePolicy(sizePolicy4)
        self.icon_x_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-disconnect-prohibition.png"))

        self.gridLayout_8.addWidget(self.icon_x_status, 0, 1, 1, 1)

        self.icon_z_status = QLabel(self.dockWidgetContents)
        self.icon_z_status.setObjectName(u"icon_z_status")
        sizePolicy.setHeightForWidth(self.icon_z_status.sizePolicy().hasHeightForWidth())
        self.icon_z_status.setSizePolicy(sizePolicy)
        self.icon_z_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-disconnect-prohibition.png"))

        self.gridLayout_8.addWidget(self.icon_z_status, 2, 1, 1, 1)

        self.icon_y_status = QLabel(self.dockWidgetContents)
        self.icon_y_status.setObjectName(u"icon_y_status")
        sizePolicy.setHeightForWidth(self.icon_y_status.sizePolicy().hasHeightForWidth())
        self.icon_y_status.setSizePolicy(sizePolicy)
        self.icon_y_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-disconnect-prohibition.png"))

        self.gridLayout_8.addWidget(self.icon_y_status, 1, 1, 1, 1)

        self.lbl_x_title_satus = QLabel(self.dockWidgetContents)
        self.lbl_x_title_satus.setObjectName(u"lbl_x_title_satus")
        sizePolicy4.setHeightForWidth(self.lbl_x_title_satus.sizePolicy().hasHeightForWidth())
        self.lbl_x_title_satus.setSizePolicy(sizePolicy4)
        self.lbl_x_title_satus.setFont(font)

        self.gridLayout_8.addWidget(self.lbl_x_title_satus, 0, 0, 1, 1)

        self.lbl_x_status = QLabel(self.dockWidgetContents)
        self.lbl_x_status.setObjectName(u"lbl_x_status")
        self.lbl_x_status.setMinimumSize(QSize(120, 0))

        self.gridLayout_8.addWidget(self.lbl_x_status, 0, 2, 1, 1)

        self.lbl_y_status = QLabel(self.dockWidgetContents)
        self.lbl_y_status.setObjectName(u"lbl_y_status")

        self.gridLayout_8.addWidget(self.lbl_y_status, 1, 2, 1, 1)

        self.lbl_z_status = QLabel(self.dockWidgetContents)
        self.lbl_z_status.setObjectName(u"lbl_z_status")

        self.gridLayout_8.addWidget(self.lbl_z_status, 2, 2, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_8)

        self.line_4 = QFrame(self.dockWidgetContents)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.slide_xz_z = QSlider(self.dockWidgetContents)
        self.slide_xz_z.setObjectName(u"slide_xz_z")
        self.slide_xz_z.setOrientation(Qt.Vertical)

        self.gridLayout_4.addWidget(self.slide_xz_z, 0, 0, 1, 1)

        self.plot_xz = MplCanvas(self.dockWidgetContents)
        self.plot_xz.setObjectName(u"plot_xz")
        sizePolicy5 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.plot_xz.sizePolicy().hasHeightForWidth())
        self.plot_xz.setSizePolicy(sizePolicy5)

        self.gridLayout_4.addWidget(self.plot_xz, 0, 1, 1, 1)

        self.slide_xz_x = QSlider(self.dockWidgetContents)
        self.slide_xz_x.setObjectName(u"slide_xz_x")
        self.slide_xz_x.setMaximum(250)
        self.slide_xz_x.setPageStep(10)
        self.slide_xz_x.setOrientation(Qt.Horizontal)
        self.slide_xz_x.setTickPosition(QSlider.TicksBelow)
        self.slide_xz_x.setTickInterval(10)

        self.gridLayout_4.addWidget(self.slide_xz_x, 1, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.slide_xy_y = QSlider(self.dockWidgetContents)
        self.slide_xy_y.setObjectName(u"slide_xy_y")
        self.slide_xy_y.setOrientation(Qt.Vertical)

        self.gridLayout_3.addWidget(self.slide_xy_y, 0, 0, 1, 1)

        self.plot_xy = MplCanvas(self.dockWidgetContents)
        self.plot_xy.setObjectName(u"plot_xy")
        sizePolicy5.setHeightForWidth(self.plot_xy.sizePolicy().hasHeightForWidth())
        self.plot_xy.setSizePolicy(sizePolicy5)

        self.gridLayout_3.addWidget(self.plot_xy, 0, 1, 1, 1)

        self.slide_xy_x = QSlider(self.dockWidgetContents)
        self.slide_xy_x.setObjectName(u"slide_xy_x")
        self.slide_xy_x.setOrientation(Qt.Horizontal)

        self.gridLayout_3.addWidget(self.slide_xy_x, 1, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_3, 1, 0, 1, 1)

        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.slide_yz_y = QSlider(self.dockWidgetContents)
        self.slide_yz_y.setObjectName(u"slide_yz_y")
        self.slide_yz_y.setOrientation(Qt.Vertical)

        self.gridLayout_5.addWidget(self.slide_yz_y, 0, 0, 1, 1)

        self.plot_yz = MplCanvas(self.dockWidgetContents)
        self.plot_yz.setObjectName(u"plot_yz")
        sizePolicy5.setHeightForWidth(self.plot_yz.sizePolicy().hasHeightForWidth())
        self.plot_yz.setSizePolicy(sizePolicy5)

        self.gridLayout_5.addWidget(self.plot_yz, 0, 1, 1, 1)

        self.slide_yz_z = QSlider(self.dockWidgetContents)
        self.slide_yz_z.setObjectName(u"slide_yz_z")
        self.slide_yz_z.setOrientation(Qt.Horizontal)

        self.gridLayout_5.addWidget(self.slide_yz_z, 1, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_5, 1, 1, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout_6)

        DockWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(DockWidget)

        QMetaObject.connectSlotsByName(DockWidget)
    # setupUi

    def retranslateUi(self, DockWidget):
        DockWidget.setWindowTitle(QCoreApplication.translate("DockWidget", u"DockWidget", None))
        self.lbl_cur_pos.setText(QCoreApplication.translate("DockWidget", u"Current Position", None))
        self.lbl_x_cur_pos.setText(QCoreApplication.translate("DockWidget", u"X", None))
        self.lbl_y_cur_pos.setText(QCoreApplication.translate("DockWidget", u"Y", None))
        self.lbl_z_cur_pos.setText(QCoreApplication.translate("DockWidget", u"Z", None))
        self.lbl_new_pos.setText(QCoreApplication.translate("DockWidget", u"New Position", None))
        self.lbl_x_new_pos.setText(QCoreApplication.translate("DockWidget", u"X", None))
        self.sb_x_new_pos.setSuffix(QCoreApplication.translate("DockWidget", u" mm", None))
        self.lbl_y_new_pos.setText(QCoreApplication.translate("DockWidget", u"Y", None))
        self.sb_y_new_pos.setSuffix(QCoreApplication.translate("DockWidget", u" mm", None))
        self.lbl_z_new_pos.setText(QCoreApplication.translate("DockWidget", u"Z", None))
        self.sb_z_new_pos.setSuffix(QCoreApplication.translate("DockWidget", u" mm", None))
        self.btn_go.setText(QCoreApplication.translate("DockWidget", u"Go", None))
        self.btn_stop.setText(QCoreApplication.translate("DockWidget", u"STOP", None))
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
        self.lbl_status.setText(QCoreApplication.translate("DockWidget", u"Status", None))
        self.lbl_y_title_status.setText(QCoreApplication.translate("DockWidget", u"Y", None))
        self.lbl_z_title_status.setText(QCoreApplication.translate("DockWidget", u"Z", None))
        self.icon_x_status.setText("")
        self.icon_z_status.setText("")
        self.icon_y_status.setText("")
        self.lbl_x_title_satus.setText(QCoreApplication.translate("DockWidget", u"X", None))
        self.lbl_x_status.setText(QCoreApplication.translate("DockWidget", u"Not Set", None))
        self.lbl_y_status.setText(QCoreApplication.translate("DockWidget", u"Not Set", None))
        self.lbl_z_status.setText(QCoreApplication.translate("DockWidget", u"Not Set", None))
    # retranslateUi

