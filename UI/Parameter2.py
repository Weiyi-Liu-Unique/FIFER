# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Parameter2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(900, 750)
        Dialog.setMinimumSize(QtCore.QSize(900, 750))
        Dialog.setMaximumSize(QtCore.QSize(900, 750))
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(530, 670, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.Img_Label1 = QtWidgets.QLabel(Dialog)
        self.Img_Label1.setGeometry(QtCore.QRect(50, 50, 250, 250))
        self.Img_Label1.setMinimumSize(QtCore.QSize(250, 250))
        self.Img_Label1.setMaximumSize(QtCore.QSize(250, 250))
        self.Img_Label1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.Img_Label1.setScaledContents(True)
        self.Img_Label1.setObjectName("Img_Label1")
        self.Img_Label2 = QtWidgets.QLabel(Dialog)
        self.Img_Label2.setGeometry(QtCore.QRect(350, 50, 250, 250))
        self.Img_Label2.setMinimumSize(QtCore.QSize(250, 250))
        self.Img_Label2.setMaximumSize(QtCore.QSize(250, 250))
        self.Img_Label2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.Img_Label2.setScaledContents(True)
        self.Img_Label2.setObjectName("Img_Label2")
        self.Img_Label3 = QtWidgets.QLabel(Dialog)
        self.Img_Label3.setGeometry(QtCore.QRect(50, 350, 550, 250))
        self.Img_Label3.setMinimumSize(QtCore.QSize(550, 250))
        self.Img_Label3.setMaximumSize(QtCore.QSize(550, 250))
        self.Img_Label3.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.Img_Label3.setScaledContents(True)
        self.Img_Label3.setObjectName("Img_Label3")
        self.Label1 = QtWidgets.QLabel(Dialog)
        self.Label1.setGeometry(QtCore.QRect(50, 0, 250, 50))
        self.Label1.setObjectName("Label1")
        self.Label2 = QtWidgets.QLabel(Dialog)
        self.Label2.setGeometry(QtCore.QRect(350, 0, 250, 50))
        self.Label2.setObjectName("Label2")
        self.Label3 = QtWidgets.QLabel(Dialog)
        self.Label3.setGeometry(QtCore.QRect(50, 300, 550, 50))
        self.Label3.setObjectName("Label3")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(630, 40, 254, 261))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.P1 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.P1.setContentsMargins(0, 0, 0, 0)
        self.P1.setSpacing(0)
        self.P1.setObjectName("P1")
        self.P1_Label1 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.P1_Label1.setEnabled(True)
        self.P1_Label1.setMinimumSize(QtCore.QSize(250, 20))
        self.P1_Label1.setMaximumSize(QtCore.QSize(250, 20))
        self.P1_Label1.setObjectName("P1_Label1")
        self.P1.addWidget(self.P1_Label1)
        self.P1_Label2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.P1_Label2.setMinimumSize(QtCore.QSize(250, 20))
        self.P1_Label2.setMaximumSize(QtCore.QSize(250, 20))
        self.P1_Label2.setFrameShape(QtWidgets.QFrame.Box)
        self.P1_Label2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.P1_Label2.setObjectName("P1_Label2")
        self.P1.addWidget(self.P1_Label2)
        self.P1_Layout1 = QtWidgets.QHBoxLayout()
        self.P1_Layout1.setSpacing(0)
        self.P1_Layout1.setObjectName("P1_Layout1")
        self.P1_Slider1 = QtWidgets.QSlider(self.verticalLayoutWidget_3)
        self.P1_Slider1.setEnabled(True)
        self.P1_Slider1.setMinimumSize(QtCore.QSize(150, 40))
        self.P1_Slider1.setMaximumSize(QtCore.QSize(150, 40))
        self.P1_Slider1.setMinimum(1)
        self.P1_Slider1.setMaximum(10)
        self.P1_Slider1.setSingleStep(1)
        self.P1_Slider1.setPageStep(10)
        self.P1_Slider1.setProperty("value", 1)
        self.P1_Slider1.setOrientation(QtCore.Qt.Horizontal)
        self.P1_Slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.P1_Slider1.setObjectName("P1_Slider1")
        self.P1_Layout1.addWidget(self.P1_Slider1)
        self.P1_Box1 = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_3)
        self.P1_Box1.setMinimumSize(QtCore.QSize(100, 40))
        self.P1_Box1.setMaximumSize(QtCore.QSize(100, 40))
        self.P1_Box1.setDecimals(3)
        self.P1_Box1.setMinimum(0.005)
        self.P1_Box1.setMaximum(0.05)
        self.P1_Box1.setSingleStep(0.005)
        self.P1_Box1.setProperty("value", 0.005)
        self.P1_Box1.setObjectName("P1_Box1")
        self.P1_Layout1.addWidget(self.P1_Box1)
        self.P1.addLayout(self.P1_Layout1)
        self.P1_Label3 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.P1_Label3.setMinimumSize(QtCore.QSize(250, 20))
        self.P1_Label3.setMaximumSize(QtCore.QSize(250, 20))
        self.P1_Label3.setFrameShape(QtWidgets.QFrame.Box)
        self.P1_Label3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.P1_Label3.setObjectName("P1_Label3")
        self.P1.addWidget(self.P1_Label3)
        self.P1_Layout2 = QtWidgets.QHBoxLayout()
        self.P1_Layout2.setSpacing(0)
        self.P1_Layout2.setObjectName("P1_Layout2")
        self.P1_Slider2 = QtWidgets.QSlider(self.verticalLayoutWidget_3)
        self.P1_Slider2.setEnabled(True)
        self.P1_Slider2.setMinimumSize(QtCore.QSize(150, 40))
        self.P1_Slider2.setMaximumSize(QtCore.QSize(150, 40))
        self.P1_Slider2.setMinimum(1)
        self.P1_Slider2.setMaximum(5)
        self.P1_Slider2.setPageStep(0)
        self.P1_Slider2.setProperty("value", 2)
        self.P1_Slider2.setOrientation(QtCore.Qt.Horizontal)
        self.P1_Slider2.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.P1_Slider2.setObjectName("P1_Slider2")
        self.P1_Layout2.addWidget(self.P1_Slider2)
        self.P1_Box2 = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_3)
        self.P1_Box2.setMinimumSize(QtCore.QSize(100, 40))
        self.P1_Box2.setMaximumSize(QtCore.QSize(100, 40))
        self.P1_Box2.setDecimals(1)
        self.P1_Box2.setMinimum(0.3)
        self.P1_Box2.setMaximum(0.7)
        self.P1_Box2.setSingleStep(0.1)
        self.P1_Box2.setProperty("value", 0.4)
        self.P1_Box2.setObjectName("P1_Box2")
        self.P1_Layout2.addWidget(self.P1_Box2)
        self.P1.addLayout(self.P1_Layout2)
        self.P1_Label4 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.P1_Label4.setMinimumSize(QtCore.QSize(250, 20))
        self.P1_Label4.setMaximumSize(QtCore.QSize(250, 20))
        self.P1_Label4.setFrameShape(QtWidgets.QFrame.Box)
        self.P1_Label4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.P1_Label4.setObjectName("P1_Label4")
        self.P1.addWidget(self.P1_Label4)
        self.P1_Label5 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.P1_Label5.setMinimumSize(QtCore.QSize(250, 20))
        self.P1_Label5.setMaximumSize(QtCore.QSize(250, 20))
        self.P1_Label5.setFrameShape(QtWidgets.QFrame.Box)
        self.P1_Label5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.P1_Label5.setObjectName("P1_Label5")
        self.P1.addWidget(self.P1_Label5)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        self.P1_Slider1.valueChanged['int'].connect(Dialog.P1_Slider1_Changed)
        self.P1_Slider2.valueChanged['int'].connect(Dialog.P1_Slider2_Changed)
        self.P1_Box1.valueChanged['double'].connect(Dialog.P1_Box1_Changed)
        self.P1_Box2.valueChanged['double'].connect(Dialog.P1_Box2_Changed)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Matching Initial Feature Points"))
        self.Img_Label1.setText(_translate("Dialog", "TextLabel"))
        self.Img_Label2.setText(_translate("Dialog", "TextLabel"))
        self.Img_Label3.setText(_translate("Dialog", "TextLabel"))
        self.Label1.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Template</span></p></body></html>"))
        self.Label2.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Frame</span></p></body></html>"))
        self.Label3.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Matching Feature Points (RANSAC Filtration)</span></p></body></html>"))
        self.P1_Label1.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Matching Parameters</span></p></body></html>"))
        self.P1_Label2.setText(_translate("Dialog", "Alpha1"))
        self.P1_Label3.setText(_translate("Dialog", "J"))
        self.P1_Label4.setText(_translate("Dialog", "Match Number: "))
        self.P1_Label5.setText(_translate("Dialog", "<html><head/><body><p>Recommended Match Number: &gt; 5</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
