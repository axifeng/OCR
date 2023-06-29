# -*- coding: utf-8 -*-
import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from utils.const import const


class Ui_Form(QtWidgets.QMainWindow):
    signal_config_list = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(Ui_Form, self).__init__(parent)
        self.model_path = None
        self.dict_path = None
        self.configDict = None
        self.train_data = None
        self.test_data = None
        self.train_tag = None
        self.test_tag = None

        self.setupUi(self)
        self.getModelButton.clicked.connect(lambda: self.getModelPath())
        self.getDictButton.clicked.connect(lambda: self.getDictPath())
        self.startButton.clicked.connect(lambda: self.startActions())
        self.startButton.clicked.connect(lambda: self.getConfigInfo())
        self.cancelButton.clicked.connect(lambda: self.cancel())

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(450, 350)
        print(os.getcwd())
        MainWindow.setWindowIcon(QIcon(os.path.join(os.getcwd(), "UI/img/ocr.jpeg")))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.GetModelPathlineEdit = QtWidgets.QLineEdit(MainWindow)
        self.GetModelPathlineEdit.setGeometry(QtCore.QRect(200, 30, 160, 31))

        self.GetDictlineEdit = QtWidgets.QLineEdit(MainWindow)
        self.GetDictlineEdit.setGeometry(QtCore.QRect(200, 80, 160, 31))
        self.GetDictlineEdit.setObjectName("GetDictlineEdit")

        self.getbatchSizelineEdit = QtWidgets.QLineEdit(MainWindow)
        self.getbatchSizelineEdit.setGeometry(QtCore.QRect(200, 130, 160, 31))
        self.getbatchSizelineEdit.setObjectName("getbatchSizelineEdit")

        self.getEpochlineEdit = QtWidgets.QLineEdit(MainWindow)
        self.getEpochlineEdit.setGeometry(QtCore.QRect(200, 180, 160, 31))
        self.getEpochlineEdit.setObjectName("getEpochlineEdit")

        self.getLrlineEdit = QtWidgets.QLineEdit(MainWindow)
        self.getLrlineEdit.setGeometry(QtCore.QRect(200, 230, 160, 31))
        self.getLrlineEdit.setObjectName("getLrlineEdit")

        self.getModelButton = QtWidgets.QPushButton(MainWindow)
        self.getModelButton.setGeometry(QtCore.QRect(50, 30, 140, 31))
        self.getModelButton.setObjectName("getModelButton")
        self.getModelButton.setStyleSheet(
            '''QPushButton{background:#19232D;border-radius:5px;text-align:right}QPushButton:hover{background:blue;}''')

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(os.getcwd(), "UI\\img\\icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.getModelButton.setIcon(icon)
        self.getModelButton.setToolTip("点击选取模型路径")

        self.getDictButton = QtWidgets.QPushButton(MainWindow)
        self.getDictButton.setGeometry(QtCore.QRect(50, 80, 140, 31))

        self.getDictButton.setStyleSheet(
            '''QPushButton{background:#19232D;border-radius:0px;text-align:right}QPushButton:hover{background:blue;}''')
        self.getDictButton.setObjectName("getDictButton")
        self.getDictButton.setIcon(icon)
        self.getDictButton.setToolTip("点击选取字典路径")

        self.batchSizeButton = QtWidgets.QLabel(MainWindow)
        self.batchSizeButton.setGeometry(QtCore.QRect(50, 136, 140, 31))
        self.batchSizeButton.setObjectName("batchSizeButton")
        self.batchSizeButton.setText("批量大小")
        self.batchSizeButton.setAlignment(Qt.AlignRight)

        self.epochButton = QtWidgets.QLabel(MainWindow)
        self.epochButton.setGeometry(QtCore.QRect(50, 186, 140, 31))
        self.epochButton.setObjectName("epochButton")
        self.epochButton.setText("总轮次")
        self.epochButton.setAlignment(Qt.AlignRight)

        self.lrButton = QtWidgets.QLabel(MainWindow)
        self.lrButton.setGeometry(QtCore.QRect(50, 236, 140, 31))
        self.lrButton.setObjectName("lrButton")
        self.lrButton.setText("学习率")
        self.lrButton.setAlignment(Qt.AlignRight)

        self.startButton = QtWidgets.QPushButton(MainWindow)
        self.startButton.setGeometry(QtCore.QRect(70, 300, 70, 31))
        self.startButton.setObjectName("startButton")
        self.startButton.setText("确认")

        self.cancelButton = QtWidgets.QPushButton(MainWindow)
        self.cancelButton.setGeometry(QtCore.QRect(300, 300, 70, 31))
        self.cancelButton.setObjectName("getResultButton")
        self.cancelButton.setText("取消")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Form", "训练参数"))
        self.getModelButton.setText(_translate("Form", "预训练模型路径"))
        self.getDictButton.setText(_translate("Form", "字典路径"))
        self.startButton.setText(_translate("Form", "确认"))
        self.cancelButton.setText(_translate("Form", "取消"))

    def getModelPath(self):
        self.model_path = QtWidgets.QFileDialog.getOpenFileName(None,
                                                                "选择模型路径(.pdparams)",
                                                                "./")
        if self.model_path is None:
            return

        self.GetModelPathlineEdit.setText(str(self.model_path[0]))

    def getDictPath(self):
        self.dict_path = QtWidgets.QFileDialog.getOpenFileName(None,
                                                               "选择字典路径",
                                                               "./")
        if self.dict_path is None:
            return
        self.GetDictlineEdit.setText(str(self.dict_path[const.get_result_index]))

    def startActions(self):
        self.close()

    def getConfigInfo(self):
        modelPath = self.GetModelPathlineEdit.text()
        dictPath = self.GetDictlineEdit.text()
        epoch = self.getEpochlineEdit.text()
        lr = self.getLrlineEdit.text()
        batchSize = self.getbatchSizelineEdit.text()
        if modelPath == "" or dictPath == "" or epoch == "" or lr == "" or batchSize == "":
            return
        if not os.path.exists(modelPath) or not os.path.exists(dictPath):
            return

        self.configDict = {'modelPath': modelPath, 'dictPath': dictPath, 'epoch': epoch, 'lr': lr,
                           'batchSize': batchSize, 'train_data': self.train_data, 'test_data': self.test_data,
                           'train_tag': self.train_tag, 'test_tag': self.test_tag}
        self.signal_config_list.emit(self.configDict)

    def cancel(self):
        self.colse()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = Ui_Form()
    myWin.show()
    sys.exit(app.exec_())
