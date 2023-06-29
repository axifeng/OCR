# coding:utf-8
import time

from PyQt5.QtGui import QPalette, QColor, QBrush, QPen
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *
import random




class Acc_window(QWidget):
    def __init__(self):
        super(Acc_window, self).__init__()
        # 开启获取新数据的子线程

        self.now_epoch = [0]
        self.point_list = []
        self.y_Aix = None
        self.x_Aix = None
        self.new_log_data = {}
        self.dict = None
        self.logInfo = {}
        self.init_line_acc()
        self.add_axis()

    def init_line_acc(self):
        # 实例折线对象
        self.series_1 = QLineSeries()  # 定义LineSerise，将类QLineSeries实例化
        # 折线初始数据了列表
        self.point_list = []

    def add_axis(self):
        self.x_Aix = QValueAxis()
        # self.x_Aix.setRange(0, 10)
        self.x_Aix.setTitleText("轮次")
        self.x_Aix.setTitleBrush(QBrush(QColor(255, 255, 255)))
        self.x_Aix.setLabelsColor(QColor(255,255,255))
        self.x_Aix.setLabelFormat("%0.2f")
        # self.x_Aix.setTickCount(21)  # 将0-10分成11份
        # self.x_Aix.setMinorTickCount(50)  # 设置每一份的分割数

        # 设置y轴
        self.y_Aix = QValueAxis()
        self.y_Aix.setTitleText("精度/acc")
        self.y_Aix.setRange(0.00, 1.00)
        self.y_Aix.setLabelFormat("%0.2f")
        self.y_Aix.setTickCount(11)
        self.y_Aix.setMinorTickCount(0.1)
        self.y_Aix.setTitleBrush(QBrush(QColor(255, 255, 255)))
        self.y_Aix.setLabelsColor(QColor(255,255,255))
        self.charView = QChartView(self)  # 定义charView，父窗体类型为 Window

        self.charView.setGeometry(20, 20,
                                  1200,
                                  400)  # 设置charView位置、大小

        self.charView.chart().addSeries(self.series_1)  # 添加折线实例
        self.charView.chart().setTitleBrush(QBrush(QColor(255, 255, 255)))
        self.charView.chart().setBackgroundBrush(QBrush(QColor(64, 64, 64)))
        self.charView.chart().setTitle('精度')
        self.charView.chart().setTitleBrush(QBrush(QColor(255, 255, 255)))
        self.series_1.setPointLabelsVisible(True)

        # 添加轴，Qt.AlignBottom表示底部，Qt.AlignLeft表示左边，Qt.AlignRight表示右边
        self.charView.chart().addAxis(self.x_Aix, Qt.AlignBottom)
        self.charView.chart().addAxis(self.y_Aix, Qt.AlignLeft)
        self.series_1.setUseOpenGL(True)
        pen = QPen(Qt.red, 3, Qt.SolidLine)
        pen.setWidth(3)
        self.series_1.setPen(pen)
        # 将折线与对应的y轴关联，这里只写了一个y轴，如果有多条折线可以关联不同的y轴
        self.series_1.attachAxis(self.y_Aix)

        # 隐藏或者显示折线 setVisible
        # self.series_1.setVisible(False)  # 隐藏折线

    def update(self):
        print("更新点的坐标")
        # 更新折线上的点的坐标

        if self.logInfo['epoch'] not in self.now_epoch:

            self.new_log_data = {'epoch': self.logInfo['epoch'], 'acc': self.dict['acc']}
            epoch = int(self.logInfo['epoch'])
            self.now_epoch.append(epoch)

            self.point_list.append(QPointF(self.new_log_data['epoch'], self.new_log_data['acc']))

            self.series_1.replace(self.point_list)
            print(self.point_list)
            self.series_1.attachAxis(self.x_Aix)

    def setLogInfo(self, dist):
        self.logInfo = dist
        str = self.logInfo['logs']
        self.dict = {}
        for item in str.split(", "):
            key, value = item.split(": ")
            self.dict[key] = float(value)

