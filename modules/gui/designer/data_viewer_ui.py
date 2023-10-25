# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'data_viewer.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QLabel,
    QSizePolicy, QSplitter, QStackedWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1152, 866)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.setOpaqueResize(False)
        self.splitter.setChildrenCollapsible(False)
        self.w_content = QWidget(self.splitter)
        self.w_content.setObjectName(u"w_content")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(15)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.w_content.sizePolicy().hasHeightForWidth())
        self.w_content.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.w_content)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lbl_content_title = QLabel(self.w_content)
        self.lbl_content_title.setObjectName(u"lbl_content_title")
        font = QFont()
        font.setBold(True)
        self.lbl_content_title.setFont(font)
        self.lbl_content_title.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.lbl_content_title)

        self.tv_content = QTreeWidget(self.w_content)
        self.tv_content.setObjectName(u"tv_content")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tv_content.sizePolicy().hasHeightForWidth())
        self.tv_content.setSizePolicy(sizePolicy1)

        self.verticalLayout_2.addWidget(self.tv_content)

        self.splitter.addWidget(self.w_content)
        self.w_view = QWidget(self.splitter)
        self.w_view.setObjectName(u"w_view")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(65)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.w_view.sizePolicy().hasHeightForWidth())
        self.w_view.setSizePolicy(sizePolicy2)
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
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.sw_view.sizePolicy().hasHeightForWidth())
        self.sw_view.setSizePolicy(sizePolicy3)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.sw_view.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.sw_view.addWidget(self.page_2)

        self.verticalLayout_3.addWidget(self.sw_view)

        self.splitter.addWidget(self.w_view)
        self.w_info = QWidget(self.splitter)
        self.w_info.setObjectName(u"w_info")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(20)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.w_info.sizePolicy().hasHeightForWidth())
        self.w_info.setSizePolicy(sizePolicy4)
        self.verticalLayout = QVBoxLayout(self.w_info)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_info_title = QLabel(self.w_info)
        self.lbl_info_title.setObjectName(u"lbl_info_title")
        self.lbl_info_title.setFont(font)
        self.lbl_info_title.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_info_title)

        self.tv_info = QTreeWidget(self.w_info)
        __qtreewidgetitem = QTreeWidgetItem(self.tv_info)
        QTreeWidgetItem(__qtreewidgetitem)
        self.tv_info.setObjectName(u"tv_info")
        sizePolicy1.setHeightForWidth(self.tv_info.sizePolicy().hasHeightForWidth())
        self.tv_info.setSizePolicy(sizePolicy1)
        self.tv_info.setMinimumSize(QSize(200, 0))
        self.tv_info.setAnimated(True)

        self.verticalLayout.addWidget(self.tv_info)

        self.splitter.addWidget(self.w_info)

        self.horizontalLayout.addWidget(self.splitter)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lbl_content_title.setText(QCoreApplication.translate("Form", u"Measurements", None))
        ___qtreewidgetitem = self.tv_content.headerItem()
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("Form", u"Caption", None));
        self.lbl_data_title.setText(QCoreApplication.translate("Form", u"Data", None))
        self.lbl_info_title.setText(QCoreApplication.translate("Form", u"Information", None))
        ___qtreewidgetitem1 = self.tv_info.headerItem()
        ___qtreewidgetitem1.setText(1, QCoreApplication.translate("Form", u"Value", None));
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("Form", u"Parameter", None));

        __sortingEnabled = self.tv_info.isSortingEnabled()
        self.tv_info.setSortingEnabled(False)
        ___qtreewidgetitem2 = self.tv_info.topLevelItem(0)
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate("Form", u"Group1", None));
        ___qtreewidgetitem3 = ___qtreewidgetitem2.child(0)
        ___qtreewidgetitem3.setText(1, QCoreApplication.translate("Form", u"value1", None));
        ___qtreewidgetitem3.setText(0, QCoreApplication.translate("Form", u"Param1", None));
        self.tv_info.setSortingEnabled(__sortingEnabled)

    # retranslateUi

