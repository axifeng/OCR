# -*- coding: utf-8 -*-

import sys

from PyQt5 import QtCore, QtGui

sys.path.insert(0, "..")
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget


class RoundProgress(QWidget):

    def __init__(self):
        super(RoundProgress, self).__init__()

        # self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        # self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.persent = 0

    def parameterUpdate(self, p):
        self.persent = p

    def paintEvent(self, event):
        rotateAngle = 360 * self.persent / 100
        # 绘制准备工作，启用反锯齿
        painter = QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QBrush(QColor("#5F89FF")))
        painter.drawEllipse(500, 400, 100, 100)  # 画外圆

        painter.setBrush(QBrush(QColor("#e3ebff")))
        painter.drawEllipse(500, 400, 96, 96)  # 画内圆

        gradient = QConicalGradient(50, 50, 91)
        gradient.setColorAt(0, QColor("#95BBFF"))
        gradient.setColorAt(1, QColor("#5C86FF"))
        self.pen = QPen()
        self.pen.setBrush(gradient)  # 设置画刷渐变效果
        self.pen.setWidth(8)
        # self.pen.setCapStyle(Qt.RoundCap)
        painter.setPen(self.pen)
        painter.drawArc(QtCore.QRectF(500, 400, 98, 98), (90 - 0) * 16, -rotateAngle * 16)  # 画圆环

        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QColor("#5481FF"))
        painter.drawText(QtCore.QRectF(500, 400, 98, 98), Qt.AlignCenter, "%d%%" % self.persent)  # 显示进度条当前进度
        self.update()


class LodingThread(QThread):
    my_signal = pyqtSignal(int)
    p = 0

    def __init__(self):
        super(LodingThread, self).__init__()

    def run(self):
        # log = program.__globals__['train_log_dist']
        while self.p < 100:
            self.p += 2
            self.my_signal.emit(self.p)
            self.msleep(100)
