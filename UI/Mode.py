# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mode.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.WindowModal)
        Dialog.resize(500, 400)
        Dialog.setMinimumSize(QtCore.QSize(500, 400))
        Dialog.setMaximumSize(QtCore.QSize(500, 400))
        Dialog.setModal(True)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.Mode1 = QtWidgets.QRadioButton(self.groupBox)
        self.Mode1.setObjectName("Mode1")
        self.verticalLayout_4.addWidget(self.Mode1)
        self.Mode2 = QtWidgets.QRadioButton(self.groupBox)
        self.Mode2.setObjectName("Mode2")
        self.verticalLayout_4.addWidget(self.Mode2)
        self.horizontalLayout.addWidget(self.groupBox)
        self.Parameter = QtWidgets.QTableWidget(Dialog)
        self.Parameter.setEnabled(False)
        self.Parameter.setMinimumSize(QtCore.QSize(300, 300))
        self.Parameter.setMaximumSize(QtCore.QSize(300, 300))
        self.Parameter.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Parameter.setShowGrid(True)
        self.Parameter.setGridStyle(QtCore.Qt.SolidLine)
        self.Parameter.setWordWrap(False)
        self.Parameter.setCornerButtonEnabled(False)
        self.Parameter.setRowCount(8)
        self.Parameter.setColumnCount(1)
        self.Parameter.setObjectName("Parameter")
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Parameter.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 127))
        brush.setStyle(QtCore.Qt.NoBrush)
        item.setForeground(brush)
        self.Parameter.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 127))
        brush.setStyle(QtCore.Qt.NoBrush)
        item.setForeground(brush)
        self.Parameter.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 127))
        brush.setStyle(QtCore.Qt.NoBrush)
        item.setForeground(brush)
        self.Parameter.setItem(7, 0, item)
        self.horizontalLayout.addWidget(self.Parameter)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setMaximumSize(QtCore.QSize(500, 20))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)

        self.retranslateUi(Dialog)
        self.buttonBox.rejected.connect(Dialog.reject)
        self.buttonBox.accepted.connect(Dialog.accept)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Set Mode And Parameter"))
        self.groupBox.setTitle(_translate("Dialog", "Mode"))
        self.Mode1.setText(_translate("Dialog", "Normal"))
        self.Mode2.setText(_translate("Dialog", "DownScale"))
        self.Parameter.setSortingEnabled(False)
        item = self.Parameter.verticalHeaderItem(0)
        item.setText(_translate("Dialog", "Size"))
        item = self.Parameter.verticalHeaderItem(1)
        item.setText(_translate("Dialog", "r"))
        item = self.Parameter.verticalHeaderItem(2)
        item.setText(_translate("Dialog", "h"))
        item = self.Parameter.verticalHeaderItem(3)
        item.setText(_translate("Dialog", "Threshold"))
        item = self.Parameter.verticalHeaderItem(4)
        item.setText(_translate("Dialog", "R Square"))
        item = self.Parameter.verticalHeaderItem(5)
        item.setText(_translate("Dialog", "Alpha1"))
        item = self.Parameter.verticalHeaderItem(6)
        item.setText(_translate("Dialog", "Alpha2"))
        item = self.Parameter.verticalHeaderItem(7)
        item.setText(_translate("Dialog", "J"))
        item = self.Parameter.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Value"))
        __sortingEnabled = self.Parameter.isSortingEnabled()
        self.Parameter.setSortingEnabled(False)
        self.Parameter.setSortingEnabled(__sortingEnabled)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
