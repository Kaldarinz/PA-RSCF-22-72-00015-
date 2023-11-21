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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QVBoxLayout, QWidget)

from ..widgets.py import MplCanvas
from . import qt_resources_rc
from . import qt_resources_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(982, 693)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.lbl_cur_pos = QLabel(Form)
        self.lbl_cur_pos.setObjectName(u"lbl_cur_pos")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_cur_pos.sizePolicy().hasHeightForWidth())
        self.lbl_cur_pos.setSizePolicy(sizePolicy)
        font = QFont()
        font.setBold(True)
        self.lbl_cur_pos.setFont(font)

        self.verticalLayout.addWidget(self.lbl_cur_pos)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.lbl_x_cur_pos = QLabel(Form)
        self.lbl_x_cur_pos.setObjectName(u"lbl_x_cur_pos")
        self.lbl_x_cur_pos.setFont(font)

        self.gridLayout_2.addWidget(self.lbl_x_cur_pos, 0, 0, 1, 1)

        self.le_x_cur_pos = QLineEdit(Form)
        self.le_x_cur_pos.setObjectName(u"le_x_cur_pos")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.le_x_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_x_cur_pos.setSizePolicy(sizePolicy1)
        self.le_x_cur_pos.setMaximumSize(QSize(74, 16777215))
        self.le_x_cur_pos.setReadOnly(True)

        self.gridLayout_2.addWidget(self.le_x_cur_pos, 0, 1, 1, 1)

        self.lbl_y_cur_pos = QLabel(Form)
        self.lbl_y_cur_pos.setObjectName(u"lbl_y_cur_pos")
        self.lbl_y_cur_pos.setFont(font)

        self.gridLayout_2.addWidget(self.lbl_y_cur_pos, 1, 0, 1, 1)

        self.le_y_cur_pos = QLineEdit(Form)
        self.le_y_cur_pos.setObjectName(u"le_y_cur_pos")
        sizePolicy1.setHeightForWidth(self.le_y_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_y_cur_pos.setSizePolicy(sizePolicy1)
        self.le_y_cur_pos.setMaximumSize(QSize(74, 16777215))
        self.le_y_cur_pos.setReadOnly(True)

        self.gridLayout_2.addWidget(self.le_y_cur_pos, 1, 1, 1, 1)

        self.lbl_z_cur_pos = QLabel(Form)
        self.lbl_z_cur_pos.setObjectName(u"lbl_z_cur_pos")
        self.lbl_z_cur_pos.setFont(font)

        self.gridLayout_2.addWidget(self.lbl_z_cur_pos, 2, 0, 1, 1)

        self.le_z_cur_pos = QLineEdit(Form)
        self.le_z_cur_pos.setObjectName(u"le_z_cur_pos")
        sizePolicy1.setHeightForWidth(self.le_z_cur_pos.sizePolicy().hasHeightForWidth())
        self.le_z_cur_pos.setSizePolicy(sizePolicy1)
        self.le_z_cur_pos.setMaximumSize(QSize(74, 16777215))
        self.le_z_cur_pos.setReadOnly(True)

        self.gridLayout_2.addWidget(self.le_z_cur_pos, 2, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_2)

        self.line = QFrame(Form)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.lbl_new_pos = QLabel(Form)
        self.lbl_new_pos.setObjectName(u"lbl_new_pos")
        sizePolicy.setHeightForWidth(self.lbl_new_pos.sizePolicy().hasHeightForWidth())
        self.lbl_new_pos.setSizePolicy(sizePolicy)
        self.lbl_new_pos.setFont(font)

        self.verticalLayout.addWidget(self.lbl_new_pos)

        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setSizeConstraint(QLayout.SetMinimumSize)
        self.lbl_x_new_pos = QLabel(Form)
        self.lbl_x_new_pos.setObjectName(u"lbl_x_new_pos")
        self.lbl_x_new_pos.setFont(font)

        self.gridLayout_7.addWidget(self.lbl_x_new_pos, 0, 0, 1, 1)

        self.doubleSpinBox = QDoubleSpinBox(Form)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        sizePolicy1.setHeightForWidth(self.doubleSpinBox.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox.setSizePolicy(sizePolicy1)
        self.doubleSpinBox.setMinimumSize(QSize(74, 0))

        self.gridLayout_7.addWidget(self.doubleSpinBox, 0, 1, 1, 1)

        self.lbl_y_new_pos = QLabel(Form)
        self.lbl_y_new_pos.setObjectName(u"lbl_y_new_pos")
        self.lbl_y_new_pos.setFont(font)

        self.gridLayout_7.addWidget(self.lbl_y_new_pos, 1, 0, 1, 1)

        self.doubleSpinBox_2 = QDoubleSpinBox(Form)
        self.doubleSpinBox_2.setObjectName(u"doubleSpinBox_2")
        sizePolicy1.setHeightForWidth(self.doubleSpinBox_2.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_2.setSizePolicy(sizePolicy1)

        self.gridLayout_7.addWidget(self.doubleSpinBox_2, 1, 1, 1, 1)

        self.lbl_z_new_pos = QLabel(Form)
        self.lbl_z_new_pos.setObjectName(u"lbl_z_new_pos")
        self.lbl_z_new_pos.setFont(font)

        self.gridLayout_7.addWidget(self.lbl_z_new_pos, 2, 0, 1, 1)

        self.doubleSpinBox_3 = QDoubleSpinBox(Form)
        self.doubleSpinBox_3.setObjectName(u"doubleSpinBox_3")
        sizePolicy1.setHeightForWidth(self.doubleSpinBox_3.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_3.setSizePolicy(sizePolicy1)

        self.gridLayout_7.addWidget(self.doubleSpinBox_3, 2, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_7)

        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setSizeConstraint(QLayout.SetMinimumSize)
        self.formLayout_3.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.btn_go = QPushButton(Form)
        self.btn_go.setObjectName(u"btn_go")
        sizePolicy1.setHeightForWidth(self.btn_go.sizePolicy().hasHeightForWidth())
        self.btn_go.setSizePolicy(sizePolicy1)
        self.btn_go.setMinimumSize(QSize(50, 0))
        self.btn_go.setMaximumSize(QSize(50, 16777215))
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_go.setIcon(icon)

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.btn_go)

        self.btn_stop = QPushButton(Form)
        self.btn_stop.setObjectName(u"btn_stop")
        sizePolicy1.setHeightForWidth(self.btn_stop.sizePolicy().hasHeightForWidth())
        self.btn_stop.setSizePolicy(sizePolicy1)
        self.btn_stop.setMinimumSize(QSize(50, 0))
        self.btn_stop.setMaximumSize(QSize(50, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon1)

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.btn_stop)


        self.verticalLayout.addLayout(self.formLayout_3)

        self.line_2 = QFrame(Form)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.lbl_move = QLabel(Form)
        self.lbl_move.setObjectName(u"lbl_move")
        sizePolicy.setHeightForWidth(self.lbl_move.sizePolicy().hasHeightForWidth())
        self.lbl_move.setSizePolicy(sizePolicy)
        self.lbl_move.setFont(font)

        self.verticalLayout.addWidget(self.lbl_move)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.lbl_x_move = QLabel(Form)
        self.lbl_x_move.setObjectName(u"lbl_x_move")
        sizePolicy.setHeightForWidth(self.lbl_x_move.sizePolicy().hasHeightForWidth())
        self.lbl_x_move.setSizePolicy(sizePolicy)
        self.lbl_x_move.setMaximumSize(QSize(6, 16777215))
        self.lbl_x_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_x_move, 0, 0, 1, 1)

        self.btn_x_left_move = QPushButton(Form)
        self.btn_x_left_move.setObjectName(u"btn_x_left_move")
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/control-double-180.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_x_left_move.setIcon(icon2)

        self.gridLayout.addWidget(self.btn_x_left_move, 0, 1, 1, 1)

        self.btn_x_right_move = QPushButton(Form)
        self.btn_x_right_move.setObjectName(u"btn_x_right_move")
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/control-double.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_x_right_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_x_right_move, 0, 2, 1, 1)

        self.lbl_y_move = QLabel(Form)
        self.lbl_y_move.setObjectName(u"lbl_y_move")
        sizePolicy.setHeightForWidth(self.lbl_y_move.sizePolicy().hasHeightForWidth())
        self.lbl_y_move.setSizePolicy(sizePolicy)
        self.lbl_y_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_y_move, 1, 0, 1, 1)

        self.btn_y_left_move = QPushButton(Form)
        self.btn_y_left_move.setObjectName(u"btn_y_left_move")
        self.btn_y_left_move.setIcon(icon2)

        self.gridLayout.addWidget(self.btn_y_left_move, 1, 1, 1, 1)

        self.btn_y_right_move = QPushButton(Form)
        self.btn_y_right_move.setObjectName(u"btn_y_right_move")
        self.btn_y_right_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_y_right_move, 1, 2, 1, 1)

        self.lbl_z_move = QLabel(Form)
        self.lbl_z_move.setObjectName(u"lbl_z_move")
        sizePolicy.setHeightForWidth(self.lbl_z_move.sizePolicy().hasHeightForWidth())
        self.lbl_z_move.setSizePolicy(sizePolicy)
        self.lbl_z_move.setFont(font)

        self.gridLayout.addWidget(self.lbl_z_move, 2, 0, 1, 1)

        self.btn_z_left_move = QPushButton(Form)
        self.btn_z_left_move.setObjectName(u"btn_z_left_move")
        self.btn_z_left_move.setIcon(icon2)

        self.gridLayout.addWidget(self.btn_z_left_move, 2, 1, 1, 1)

        self.btn_z_right_move = QPushButton(Form)
        self.btn_z_right_move.setObjectName(u"btn_z_right_move")
        self.btn_z_right_move.setIcon(icon3)

        self.gridLayout.addWidget(self.btn_z_right_move, 2, 2, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.line_3 = QFrame(Form)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.lbl_status = QLabel(Form)
        self.lbl_status.setObjectName(u"lbl_status")
        sizePolicy.setHeightForWidth(self.lbl_status.sizePolicy().hasHeightForWidth())
        self.lbl_status.setSizePolicy(sizePolicy)
        self.lbl_status.setFont(font)

        self.verticalLayout.addWidget(self.lbl_status)

        self.formLayout_4 = QFormLayout()
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.formLayout_4.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.formLayout_4.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.formLayout_4.setLabelAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.formLayout_4.setFormAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.lbl_x_satus = QLabel(Form)
        self.lbl_x_satus.setObjectName(u"lbl_x_satus")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.lbl_x_satus.sizePolicy().hasHeightForWidth())
        self.lbl_x_satus.setSizePolicy(sizePolicy2)
        self.lbl_x_satus.setFont(font)

        self.formLayout_4.setWidget(0, QFormLayout.LabelRole, self.lbl_x_satus)

        self.icon_x_status = QLabel(Form)
        self.icon_x_status.setObjectName(u"icon_x_status")
        sizePolicy.setHeightForWidth(self.icon_x_status.sizePolicy().hasHeightForWidth())
        self.icon_x_status.setSizePolicy(sizePolicy)
        self.icon_x_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-connect.png"))

        self.formLayout_4.setWidget(0, QFormLayout.FieldRole, self.icon_x_status)

        self.lbl_y_status = QLabel(Form)
        self.lbl_y_status.setObjectName(u"lbl_y_status")
        sizePolicy2.setHeightForWidth(self.lbl_y_status.sizePolicy().hasHeightForWidth())
        self.lbl_y_status.setSizePolicy(sizePolicy2)
        self.lbl_y_status.setFont(font)

        self.formLayout_4.setWidget(1, QFormLayout.LabelRole, self.lbl_y_status)

        self.icon_y_status = QLabel(Form)
        self.icon_y_status.setObjectName(u"icon_y_status")
        sizePolicy.setHeightForWidth(self.icon_y_status.sizePolicy().hasHeightForWidth())
        self.icon_y_status.setSizePolicy(sizePolicy)
        self.icon_y_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-connect.png"))

        self.formLayout_4.setWidget(1, QFormLayout.FieldRole, self.icon_y_status)

        self.lbl_z_status = QLabel(Form)
        self.lbl_z_status.setObjectName(u"lbl_z_status")
        sizePolicy2.setHeightForWidth(self.lbl_z_status.sizePolicy().hasHeightForWidth())
        self.lbl_z_status.setSizePolicy(sizePolicy2)
        self.lbl_z_status.setFont(font)

        self.formLayout_4.setWidget(2, QFormLayout.LabelRole, self.lbl_z_status)

        self.icon_z_status = QLabel(Form)
        self.icon_z_status.setObjectName(u"icon_z_status")
        sizePolicy.setHeightForWidth(self.icon_z_status.sizePolicy().hasHeightForWidth())
        self.icon_z_status.setSizePolicy(sizePolicy)
        self.icon_z_status.setPixmap(QPixmap(u":/icons/qt_resources/plug-connect.png"))

        self.formLayout_4.setWidget(2, QFormLayout.FieldRole, self.icon_z_status)


        self.verticalLayout.addLayout(self.formLayout_4)

        self.line_4 = QFrame(Form)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.line_5 = QFrame(Form)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.VLine)
        self.line_5.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_5)

        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.slide_xz_z = QSlider(Form)
        self.slide_xz_z.setObjectName(u"slide_xz_z")
        self.slide_xz_z.setOrientation(Qt.Vertical)

        self.gridLayout_4.addWidget(self.slide_xz_z, 0, 0, 1, 1)

        self.plot_xz = MplCanvas(Form)
        self.plot_xz.setObjectName(u"plot_xz")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.plot_xz.sizePolicy().hasHeightForWidth())
        self.plot_xz.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.plot_xz, 0, 1, 1, 1)

        self.slide_xz_x = QSlider(Form)
        self.slide_xz_x.setObjectName(u"slide_xz_x")
        self.slide_xz_x.setOrientation(Qt.Horizontal)

        self.gridLayout_4.addWidget(self.slide_xz_x, 1, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.slide_xy_y = QSlider(Form)
        self.slide_xy_y.setObjectName(u"slide_xy_y")
        self.slide_xy_y.setOrientation(Qt.Vertical)

        self.gridLayout_3.addWidget(self.slide_xy_y, 0, 0, 1, 1)

        self.plot_xy = MplCanvas(Form)
        self.plot_xy.setObjectName(u"plot_xy")
        sizePolicy3.setHeightForWidth(self.plot_xy.sizePolicy().hasHeightForWidth())
        self.plot_xy.setSizePolicy(sizePolicy3)

        self.gridLayout_3.addWidget(self.plot_xy, 0, 1, 1, 1)

        self.slide_xy_x = QSlider(Form)
        self.slide_xy_x.setObjectName(u"slide_xy_x")
        self.slide_xy_x.setOrientation(Qt.Horizontal)

        self.gridLayout_3.addWidget(self.slide_xy_x, 1, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_3, 1, 0, 1, 1)

        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.slide_yz_y = QSlider(Form)
        self.slide_yz_y.setObjectName(u"slide_yz_y")
        self.slide_yz_y.setOrientation(Qt.Vertical)

        self.gridLayout_5.addWidget(self.slide_yz_y, 0, 0, 1, 1)

        self.plot_yz = MplCanvas(Form)
        self.plot_yz.setObjectName(u"plot_yz")
        sizePolicy3.setHeightForWidth(self.plot_yz.sizePolicy().hasHeightForWidth())
        self.plot_yz.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.plot_yz, 0, 1, 1, 1)

        self.slide_yz_z = QSlider(Form)
        self.slide_yz_z.setObjectName(u"slide_yz_z")
        self.slide_yz_z.setOrientation(Qt.Horizontal)

        self.gridLayout_5.addWidget(self.slide_yz_z, 1, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_5, 1, 1, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout_6)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lbl_cur_pos.setText(QCoreApplication.translate("Form", u"Current Position", None))
        self.lbl_x_cur_pos.setText(QCoreApplication.translate("Form", u"X", None))
        self.lbl_y_cur_pos.setText(QCoreApplication.translate("Form", u"Y", None))
        self.lbl_z_cur_pos.setText(QCoreApplication.translate("Form", u"Z", None))
        self.lbl_new_pos.setText(QCoreApplication.translate("Form", u"New Position", None))
        self.lbl_x_new_pos.setText(QCoreApplication.translate("Form", u"X", None))
        self.lbl_y_new_pos.setText(QCoreApplication.translate("Form", u"Y", None))
        self.lbl_z_new_pos.setText(QCoreApplication.translate("Form", u"Z", None))
        self.btn_go.setText(QCoreApplication.translate("Form", u"Go", None))
        self.btn_stop.setText(QCoreApplication.translate("Form", u"STOP", None))
        self.lbl_move.setText(QCoreApplication.translate("Form", u"Move", None))
        self.lbl_x_move.setText(QCoreApplication.translate("Form", u"X", None))
        self.btn_x_left_move.setText("")
        self.btn_x_right_move.setText("")
        self.lbl_y_move.setText(QCoreApplication.translate("Form", u"Y", None))
        self.btn_y_left_move.setText("")
        self.btn_y_right_move.setText("")
        self.lbl_z_move.setText(QCoreApplication.translate("Form", u"Z", None))
        self.btn_z_left_move.setText("")
        self.btn_z_right_move.setText("")
        self.lbl_status.setText(QCoreApplication.translate("Form", u"Status", None))
        self.lbl_x_satus.setText(QCoreApplication.translate("Form", u"X", None))
        self.icon_x_status.setText("")
        self.lbl_y_status.setText(QCoreApplication.translate("Form", u"Y", None))
        self.icon_y_status.setText("")
        self.lbl_z_status.setText(QCoreApplication.translate("Form", u"Z", None))
        self.icon_z_status.setText("")
    # retranslateUi

