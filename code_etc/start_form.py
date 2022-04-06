# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'start_form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_start_form(object):
    def setupUi(self, start_form):
        start_form.setObjectName("start_form")
        start_form.resize(700, 500)
        start_form.setMinimumSize(QtCore.QSize(700, 500))
        start_form.setMaximumSize(QtCore.QSize(700, 500))
        self.horizontalFrame = QtWidgets.QFrame(start_form)
        self.horizontalFrame.setGeometry(QtCore.QRect(100, 340, 500, 60))
        self.horizontalFrame.setMinimumSize(QtCore.QSize(0, 60))
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.image_BTN = QtWidgets.QPushButton(self.horizontalFrame)
        self.image_BTN.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.image_BTN.setFont(font)
        self.image_BTN.setObjectName("image_BTN")
        self.horizontalLayout.addWidget(self.image_BTN)
        self.video_BTN = QtWidgets.QPushButton(self.horizontalFrame)
        self.video_BTN.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.video_BTN.setFont(font)
        self.video_BTN.setObjectName("video_BTN")
        self.horizontalLayout.addWidget(self.video_BTN)
        self.horizontalFrame1 = QtWidgets.QFrame(start_form)
        self.horizontalFrame1.setGeometry(QtCore.QRect(100, 400, 500, 60))
        self.horizontalFrame1.setMinimumSize(QtCore.QSize(0, 60))
        self.horizontalFrame1.setObjectName("horizontalFrame1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalFrame1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.settings_BTN = QtWidgets.QPushButton(self.horizontalFrame1)
        self.settings_BTN.setMinimumSize(QtCore.QSize(200, 35))
        self.settings_BTN.setObjectName("settings_BTN")
        self.horizontalLayout_2.addWidget(self.settings_BTN)
        self.quit_BTN = QtWidgets.QPushButton(self.horizontalFrame1)
        self.quit_BTN.setMinimumSize(QtCore.QSize(200, 35))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.quit_BTN.setFont(font)
        self.quit_BTN.setObjectName("quit_BTN")
        self.horizontalLayout_2.addWidget(self.quit_BTN)

        self.retranslateUi(start_form)
        QtCore.QMetaObject.connectSlotsByName(start_form)

    def retranslateUi(self, start_form):
        _translate = QtCore.QCoreApplication.translate
        start_form.setWindowTitle(_translate("start_form", "Expression Recognizer"))
        self.image_BTN.setText(_translate("start_form", "IMAGE"))
        self.video_BTN.setText(_translate("start_form", "VIDEO"))
        self.settings_BTN.setText(_translate("start_form", "Settings"))
        self.quit_BTN.setText(_translate("start_form", "Quit"))

