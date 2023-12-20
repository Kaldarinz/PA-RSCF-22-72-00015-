# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'map_measure_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QWidget)

from ..widgets import (MplCanvas, QuantSpinBox)
from . import qt_resources_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(924, 639)
        self.btn_start = QPushButton(Form)
        self.btn_start.setObjectName(u"btn_start")
        self.btn_start.setGeometry(QRect(30, 420, 75, 23))
        font = QFont()
        font.setBold(True)
        self.btn_start.setFont(font)
        icon = QIcon()
        icon.addFile(u":/icons/qt_resources/control.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_start.setIcon(icon)
        self.btn_stop = QPushButton(Form)
        self.btn_stop.setObjectName(u"btn_stop")
        self.btn_stop.setGeometry(QRect(190, 420, 31, 23))
        icon1 = QIcon()
        icon1.addFile(u":/icons/qt_resources/control-stop-square.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_stop.setIcon(icon1)
        self.le_result = QLineEdit(Form)
        self.le_result.setObjectName(u"le_result")
        self.le_result.setGeometry(QRect(430, 210, 341, 20))
        self.scan = MplCanvas(Form)
        self.scan.setObjectName(u"scan")
        self.scan.setGeometry(QRect(430, 270, 350, 301))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scan.sizePolicy().hasHeightForWidth())
        self.scan.setSizePolicy(sizePolicy)
        self.lbl_cur_pos = QLabel(Form)
        self.lbl_cur_pos.setObjectName(u"lbl_cur_pos")
        self.lbl_cur_pos.setGeometry(QRect(100, 60, 97, 20))
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lbl_cur_pos.sizePolicy().hasHeightForWidth())
        self.lbl_cur_pos.setSizePolicy(sizePolicy1)
        self.lbl_cur_pos.setFont(font)
        self.lbl_cur_pos.setAlignment(Qt.AlignCenter)
        self.btn_start_2 = QPushButton(Form)
        self.btn_start_2.setObjectName(u"btn_start_2")
        self.btn_start_2.setGeometry(QRect(110, 420, 31, 23))
        icon2 = QIcon()
        icon2.addFile(u":/icons/qt_resources/control-pause.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_start_2.setIcon(icon2)
        self.btn_start_3 = QPushButton(Form)
        self.btn_start_3.setObjectName(u"btn_start_3")
        self.btn_start_3.setGeometry(QRect(150, 420, 31, 23))
        icon3 = QIcon()
        icon3.addFile(u":/icons/qt_resources/arrow-circle-225-left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_start_3.setIcon(icon3)
        self.line = QFrame(Form)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(110, 220, 118, 3))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.lbl_cur_pos_2 = QLabel(Form)
        self.lbl_cur_pos_2.setObjectName(u"lbl_cur_pos_2")
        self.lbl_cur_pos_2.setGeometry(QRect(130, 230, 97, 20))
        sizePolicy1.setHeightForWidth(self.lbl_cur_pos_2.sizePolicy().hasHeightForWidth())
        self.lbl_cur_pos_2.setSizePolicy(sizePolicy1)
        self.lbl_cur_pos_2.setFont(font)
        self.lbl_cur_pos_2.setAlignment(Qt.AlignCenter)
        self.comboBox = QComboBox(Form)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(160, 290, 69, 22))
        self.comboBox.setFrame(True)
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(80, 290, 81, 20))
        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(80, 320, 81, 20))
        self.comboBox_2 = QComboBox(Form)
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.comboBox_2.setGeometry(QRect(160, 320, 69, 22))
        self.comboBox_2.setFrame(True)
        self.checkBox = QCheckBox(Form)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(170, 260, 16, 17))
        self.label_6 = QLabel(Form)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(80, 260, 81, 20))
        self.comboBox_3 = QComboBox(Form)
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.comboBox_3.setGeometry(QRect(160, 350, 69, 22))
        self.comboBox_3.setFrame(True)
        self.label_7 = QLabel(Form)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(80, 350, 81, 20))
        self.line_2 = QFrame(Form)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(100, 390, 118, 3))
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.line_3 = QFrame(Form)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(100, 470, 118, 3))
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)
        self.widget = QWidget(Form)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(0, 90, 210, 86))
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)

        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.label_3 = QLabel(self.widget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 3, 1, 1)

        self.label_8 = QLabel(self.widget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)

        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)

        self.sb_x_new_pos = QuantSpinBox(self.widget)
        self.sb_x_new_pos.setObjectName(u"sb_x_new_pos")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.sb_x_new_pos.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos.setSizePolicy(sizePolicy2)
        self.sb_x_new_pos.setMinimumSize(QSize(0, 0))
        self.sb_x_new_pos.setDecimals(2)

        self.gridLayout.addWidget(self.sb_x_new_pos, 1, 1, 1, 1)

        self.sb_x_new_pos_3 = QSpinBox(self.widget)
        self.sb_x_new_pos_3.setObjectName(u"sb_x_new_pos_3")
        sizePolicy2.setHeightForWidth(self.sb_x_new_pos_3.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos_3.setSizePolicy(sizePolicy2)
        self.sb_x_new_pos_3.setMinimumSize(QSize(0, 0))

        self.gridLayout.addWidget(self.sb_x_new_pos_3, 1, 2, 1, 1)

        self.sb_x_new_pos_5 = QuantSpinBox(self.widget)
        self.sb_x_new_pos_5.setObjectName(u"sb_x_new_pos_5")
        sizePolicy2.setHeightForWidth(self.sb_x_new_pos_5.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos_5.setSizePolicy(sizePolicy2)
        self.sb_x_new_pos_5.setMinimumSize(QSize(0, 0))
        self.sb_x_new_pos_5.setDecimals(2)

        self.gridLayout.addWidget(self.sb_x_new_pos_5, 1, 3, 1, 1)

        self.label_9 = QLabel(self.widget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setFont(font)

        self.gridLayout.addWidget(self.label_9, 2, 0, 1, 1)

        self.sb_x_new_pos_2 = QuantSpinBox(self.widget)
        self.sb_x_new_pos_2.setObjectName(u"sb_x_new_pos_2")
        sizePolicy2.setHeightForWidth(self.sb_x_new_pos_2.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos_2.setSizePolicy(sizePolicy2)
        self.sb_x_new_pos_2.setMinimumSize(QSize(0, 0))
        self.sb_x_new_pos_2.setDecimals(2)

        self.gridLayout.addWidget(self.sb_x_new_pos_2, 2, 1, 1, 1)

        self.sb_x_new_pos_4 = QSpinBox(self.widget)
        self.sb_x_new_pos_4.setObjectName(u"sb_x_new_pos_4")
        sizePolicy2.setHeightForWidth(self.sb_x_new_pos_4.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos_4.setSizePolicy(sizePolicy2)
        self.sb_x_new_pos_4.setMinimumSize(QSize(0, 0))

        self.gridLayout.addWidget(self.sb_x_new_pos_4, 2, 2, 1, 1)

        self.sb_x_new_pos_6 = QuantSpinBox(self.widget)
        self.sb_x_new_pos_6.setObjectName(u"sb_x_new_pos_6")
        sizePolicy2.setHeightForWidth(self.sb_x_new_pos_6.sizePolicy().hasHeightForWidth())
        self.sb_x_new_pos_6.setSizePolicy(sizePolicy2)
        self.sb_x_new_pos_6.setMinimumSize(QSize(0, 0))
        self.sb_x_new_pos_6.setDecimals(2)

        self.gridLayout.addWidget(self.sb_x_new_pos_6, 2, 3, 1, 1)

        self.label_10 = QLabel(self.widget)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout.addWidget(self.label_10, 3, 0, 1, 1)

        self.radioButton_3 = QRadioButton(self.widget)
        self.radioButton_3.setObjectName(u"radioButton_3")

        self.gridLayout.addWidget(self.radioButton_3, 3, 1, 1, 1)

        self.radioButton_2 = QRadioButton(self.widget)
        self.radioButton_2.setObjectName(u"radioButton_2")

        self.gridLayout.addWidget(self.radioButton_2, 3, 2, 1, 1)

        self.radioButton = QRadioButton(self.widget)
        self.radioButton.setObjectName(u"radioButton")

        self.gridLayout.addWidget(self.radioButton, 3, 3, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.btn_start.setText(QCoreApplication.translate("Form", u"Scan", None))
        self.btn_stop.setText("")
        self.lbl_cur_pos.setText(QCoreApplication.translate("Form", u"Scan Area", None))
        self.btn_start_2.setText("")
        self.btn_start_3.setText("")
        self.lbl_cur_pos_2.setText(QCoreApplication.translate("Form", u"Parameters", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"Direction", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"Signal", None))
        self.checkBox.setText("")
        self.label_6.setText(QCoreApplication.translate("Form", u"Auto step", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"Plane", None))
        self.label.setText(QCoreApplication.translate("Form", u"Size", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Points", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Step", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"X", None))
        self.sb_x_new_pos.setSuffix(QCoreApplication.translate("Form", u" mm", None))
        self.sb_x_new_pos_5.setSuffix(QCoreApplication.translate("Form", u" mm", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"Y", None))
        self.sb_x_new_pos_2.setSuffix(QCoreApplication.translate("Form", u" mm", None))
        self.sb_x_new_pos_6.setSuffix(QCoreApplication.translate("Form", u" mm", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"Lock", None))
        self.radioButton_3.setText("")
        self.radioButton_2.setText("")
        self.radioButton.setText("")
    # retranslateUi

