# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'data_viewer.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
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
from PySide6.QtWidgets import (QApplication, QHeaderView, QLabel, QListView,
    QSizePolicy, QSplitter, QStackedWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget)

from ..widgets import TreeInfoWidget

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1152, 866)
        self.verticalLayout_4 = QVBoxLayout(Form)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.setOpaqueResize(False)
        self.splitter.setChildrenCollapsible(False)
        self.w_content = QWidget(self.splitter)
        self.w_content.setObjectName(u"w_content")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(10)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.w_content.sizePolicy().hasHeightForWidth())
        self.w_content.setSizePolicy(sizePolicy1)
        self.verticalLayout_2 = QVBoxLayout(self.w_content)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lbl_content_title = QLabel(self.w_content)
        self.lbl_content_title.setObjectName(u"lbl_content_title")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.lbl_content_title.sizePolicy().hasHeightForWidth())
        self.lbl_content_title.setSizePolicy(sizePolicy2)
        font = QFont()
        font.setBold(True)
        self.lbl_content_title.setFont(font)
        self.lbl_content_title.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.lbl_content_title)

        self.lv_content = QListView(self.w_content)
        self.lv_content.setObjectName(u"lv_content")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.lv_content.sizePolicy().hasHeightForWidth())
        self.lv_content.setSizePolicy(sizePolicy3)
        self.lv_content.setMinimumSize(QSize(150, 0))
        self.lv_content.setMaximumSize(QSize(16777215, 16777215))

        self.verticalLayout_2.addWidget(self.lv_content)

        self.splitter.addWidget(self.w_content)
        self.w_view = QWidget(self.splitter)
        self.w_view.setObjectName(u"w_view")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(75)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.w_view.sizePolicy().hasHeightForWidth())
        self.w_view.setSizePolicy(sizePolicy4)
        self.verticalLayout_3 = QVBoxLayout(self.w_view)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.lbl_data_title = QLabel(self.w_view)
        self.lbl_data_title.setObjectName(u"lbl_data_title")
        self.lbl_data_title.setFont(font)
        self.lbl_data_title.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.lbl_data_title)

        self.sw_view = QStackedWidget(self.w_view)
        self.sw_view.setObjectName(u"sw_view")
        sizePolicy5 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.sw_view.sizePolicy().hasHeightForWidth())
        self.sw_view.setSizePolicy(sizePolicy5)
        self.sw_view.setMinimumSize(QSize(500, 0))
        self.page = QWidget()
        self.page.setObjectName(u"page")
        sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy6.setHorizontalStretch(60)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.page.sizePolicy().hasHeightForWidth())
        self.page.setSizePolicy(sizePolicy6)
        self.sw_view.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        sizePolicy7 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy7.setHorizontalStretch(60)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.page_2.sizePolicy().hasHeightForWidth())
        self.page_2.setSizePolicy(sizePolicy7)
        self.widget = QWidget(self.page_2)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(30, 360, 800, 80))
        self.widget.setMinimumSize(QSize(800, 0))
        self.sw_view.addWidget(self.page_2)

        self.verticalLayout_3.addWidget(self.sw_view)

        self.splitter.addWidget(self.w_view)
        self.w_info = QWidget(self.splitter)
        self.w_info.setObjectName(u"w_info")
        sizePolicy8 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy8.setHorizontalStretch(15)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.w_info.sizePolicy().hasHeightForWidth())
        self.w_info.setSizePolicy(sizePolicy8)
        self.verticalLayout = QVBoxLayout(self.w_info)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_info_title = QLabel(self.w_info)
        self.lbl_info_title.setObjectName(u"lbl_info_title")
        sizePolicy2.setHeightForWidth(self.lbl_info_title.sizePolicy().hasHeightForWidth())
        self.lbl_info_title.setSizePolicy(sizePolicy2)
        self.lbl_info_title.setFont(font)
        self.lbl_info_title.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_info_title)

        self.tv_info = TreeInfoWidget(self.w_info)
        self.tv_info.setObjectName(u"tv_info")
        sizePolicy3.setHeightForWidth(self.tv_info.sizePolicy().hasHeightForWidth())
        self.tv_info.setSizePolicy(sizePolicy3)
        self.tv_info.setMinimumSize(QSize(250, 0))
        self.tv_info.setAnimated(True)

        self.verticalLayout.addWidget(self.tv_info)

        self.splitter.addWidget(self.w_info)

        self.verticalLayout_4.addWidget(self.splitter)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lbl_content_title.setText(QCoreApplication.translate("Form", u"Measurements", None))
        self.lbl_data_title.setText(QCoreApplication.translate("Form", u"Data", None))
        self.lbl_info_title.setText(QCoreApplication.translate("Form", u"Information", None))
        ___qtreewidgetitem = self.tv_info.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("Form", u"Value", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("Form", u"Parameter", None));
    # retranslateUi

