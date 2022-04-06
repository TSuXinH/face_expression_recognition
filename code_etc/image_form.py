# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image_form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_image_form(object):
    def setupUi(self, image_form):
        image_form.setObjectName("image_form")
        image_form.resize(1000, 800)
        image_form.setMinimumSize(QtCore.QSize(1000, 800))
        image_form.setMaximumSize(QtCore.QSize(1000, 800))
        self.horizontalFrame = QtWidgets.QFrame(image_form)
        self.horizontalFrame.setGeometry(QtCore.QRect(200, 600, 600, 100))
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
        self.back_BTN = QtWidgets.QPushButton(self.horizontalFrame)
        self.back_BTN.setMinimumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.back_BTN.setFont(font)
        self.back_BTN.setObjectName("back_BTN")
        self.horizontalLayout.addWidget(self.back_BTN)
        self.image_label = QtWidgets.QLabel(image_form)
        self.image_label.setGeometry(QtCore.QRect(180, 70, 640, 480))
        self.image_label.setObjectName("image_label")

        self.retranslateUi(image_form)
        QtCore.QMetaObject.connectSlotsByName(image_form)

    def retranslateUi(self, image_form):
        _translate = QtCore.QCoreApplication.translate
        image_form.setWindowTitle(_translate("image_form", "Image Process"))
        self.import_BTN.setText(_translate("image_form", "IMPORT"))
        self.back_BTN.setText(_translate("image_form", "BACK"))
        self.image_label.setText(_translate("image_form", "TextLabel"))

