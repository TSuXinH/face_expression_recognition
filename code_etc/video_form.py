# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_video_form(object):
    def setupUi(self, video_form):
        video_form.setObjectName("video_form")
        video_form.resize(1000, 800)
        video_form.setMinimumSize(QtCore.QSize(1000, 800))
        video_form.setMaximumSize(QtCore.QSize(1000, 800))
        self.horizontalFrame = QtWidgets.QFrame(video_form)
        self.horizontalFrame.setGeometry(QtCore.QRect(100, 600, 800, 100))
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.import_BTN = QtWidgets.QPushButton(self.horizontalFrame)
        self.import_BTN.setMinimumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.import_BTN.setFont(font)
        self.import_BTN.setObjectName("import_BTN")
        self.horizontalLayout.addWidget(self.import_BTN)
        self.play_BTN = QtWidgets.QPushButton(self.horizontalFrame)
        self.play_BTN.setMinimumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.play_BTN.setFont(font)
        self.play_BTN.setObjectName("play_BTN")
        self.horizontalLayout.addWidget(self.play_BTN)
        self.back_BTN = QtWidgets.QPushButton(self.horizontalFrame)
        self.back_BTN.setMinimumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.back_BTN.setFont(font)
        self.back_BTN.setObjectName("back_BTN")
        self.horizontalLayout.addWidget(self.back_BTN)
        self.player_WGT = QtWidgets.QWidget(video_form)
        self.player_WGT.setGeometry(QtCore.QRect(180, 100, 640, 480))
        self.player_WGT.setObjectName("player_WGT")

        self.retranslateUi(video_form)
        QtCore.QMetaObject.connectSlotsByName(video_form)

    def retranslateUi(self, video_form):
        _translate = QtCore.QCoreApplication.translate
        video_form.setWindowTitle(_translate("video_form", "Video Process"))
        self.import_BTN.setText(_translate("video_form", "IMPORT"))
        self.play_BTN.setText(_translate("video_form", "PLAY"))
        self.back_BTN.setText(_translate("video_form", "BACK"))

