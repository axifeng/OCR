import math
import os
import shutil
import sys
import random

import cv2
import qdarkstyle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize, QTimer, QPointF, QPoint
from PyQt5.QtGui import QIcon, QPixmap, QColor, QCursor, QImage, QMouseEvent
from PyQt5.QtWidgets import QComboBox, QListWidgetItem, QRadioButton, \
    QListWidget, QApplication, QMenu, QSlider, QFileDialog, QMessageBox, QProgressDialog

from UI.thread.recog_thread import MyThread

from UI.imageBox import ImageBox
from UI.thread.train_thread import trainThread
from UI.loding_window import RoundProgress, LodingThread
from UI.loss_chat_window import Loss_window
from UI.thread.getLog_thread import logThread
from UI.log_chat_window import Acc_window
from UI.utils.const import const
from util.export_model import export_inference


def split_data(data: list, scale):
    if not data:
        return
    n = len(data)
    m = 0 if math.ceil(n * scale) < 0 else math.ceil(n * scale)

    split_train = set(random.sample(data, m))
    test_data = split_train
    train_data = set(data) - test_data

    return train_data, test_data


def retranslateUi(MainWindow):
    _translate = QtCore.QCoreApplication.translate
    MainWindow.setWindowTitle(_translate("MainWindow", "PADDLEOCR文字检测识别平台"))


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.save_data_dir = None
        self.model_path_name = None
        self.progress = None
        self.val = None
        self.img_file_path = None
        self.export_file = None
        self.configInfo = None
        # 数据集制作
        self.img_item = None
        self.val_data = {}
        self.test_data_tag = {}
        self.train_data_tag = {}
        self.test_data_file = {}
        self.train_data_file = {}
        self.pred_list = None
        self.tagList = None
        self.view_list = []
        self.addImgList = None
        self.recog_list = None

        self.log_window = None
        self.updateTagText = None
        self.fileSearch = None
        self.file_dock = None

        self.open_select = None
        self.thread = None
        self.index = None
        self.viewCurrIndex = 0
        self.view_num = 0

        self.SPLIT_FLAG = False
        self.ADD_FLAG = False
        self.PRED_FLAG = False

        self.setupUi(self)
        self.init_slots()

    # 窗口UI
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.horizontalLayout_select = QtWidgets.QHBoxLayout()
        self.horizontalLayout_select.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        self.horizontalLayout_select.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_select.setObjectName("horizontalLayout_select")

        self.horizontalLayout_button = QtWidgets.QHBoxLayout()
        self.horizontalLayout_button.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        self.horizontalLayout_button.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_button.setObjectName("horizontalLayout_button")

        self.horizontalLayout_val = QtWidgets.QHBoxLayout()
        self.horizontalLayout_val.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        self.horizontalLayout_val.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_val.setObjectName("horizontalLayout_button")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        self.verticalLayout.setObjectName("verticalLayout")

        self.verticalLayout_middle = QtWidgets.QVBoxLayout()
        self.verticalLayout_middle.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        self.verticalLayout_middle.setObjectName("verticalLayout")

        self.horizontalLayout_add_data = QtWidgets.QHBoxLayout()
        self.horizontalLayout_add_data.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_add_data.setObjectName("horizontalLayout_add_data")

        self.horizontalLayout_pre_val = QtWidgets.QHBoxLayout()
        self.horizontalLayout_pre_val.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_pre_val.setObjectName("horizontalLayout_pre_val")

        self.verticalLayout_select = QtWidgets.QVBoxLayout()
        self.verticalLayout_select.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        self.verticalLayout_select.setObjectName("verticalLayout")
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)

        self.open_select = QComboBox()
        self.open_select.addItems(['导入数据集', '导入图片'])
        self.open_select.currentText().center(0)

        self.open_edit = QtWidgets.QLineEdit()
        self.open_edit.setReadOnly(True)
        self.open_edit.setAlignment(Qt.AlignCenter)
        self.open_select.setLineEdit(self.open_edit)
        self.open_select.activated.connect(self.fileSearchChanged)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged
        )

        self.delete_Menu = QMenu(self)
        self.delete_item = self.delete_Menu.addAction('删除图片')
        self.delete_item.triggered.connect(self.DeleteItem)
        self.fileListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fileListWidget.customContextMenuRequested.connect(self.showDeleteMenu)

        self.fileListWidget.keyPressEvent = self.keyPressEvent
        self.fileListWidget.setFocusPolicy(Qt.ClickFocus)

        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.addWidget(self.open_select)

        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget()

        self.file_dock.setFeatures(self.file_dock.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)

        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)
        self.verticalLayout.addLayout(fileListLayout)

        self.loding = RoundProgress()
        self.loding.setHidden(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loding.sizePolicy().hasHeightForWidth())
        self.loding.setSizePolicy(sizePolicy)
        self.loding.setMinimumSize(QtCore.QSize(1200, 800))
        self.loding.setMaximumSize(QtCore.QSize(1200, 800))
        self.horizontalLayout_2.addWidget(self.loding)

        self.box = ImageBox()
        print(self.box.width())
        print(self.box.height())
        self.box.setMinimumSize(QtCore.QSize(1200, 800))
        self.box.setMaximumSize(QtCore.QSize(1200, 800))
        self.horizontalLayout_2.addWidget(self.box)

        self.log_window = Acc_window()
        sizePolicy.setHeightForWidth(self.log_window.sizePolicy().hasHeightForWidth())
        self.log_window.setSizePolicy(sizePolicy)
        self.log_window.setMinimumSize(QtCore.QSize(1200, 400))
        self.log_window.setMaximumSize(QtCore.QSize(1200, 400))
        self.log_window.setHidden(True)

        self.log_window2 = Loss_window()
        sizePolicy.setHeightForWidth(self.log_window2.sizePolicy().hasHeightForWidth())
        self.log_window2.setSizePolicy(sizePolicy)
        self.log_window2.setMinimumSize(QtCore.QSize(1200, 400))
        self.log_window2.setMaximumSize(QtCore.QSize(1200, 400))
        self.log_window2.setObjectName("log_window2")
        self.log_window2.setHidden(True)
        self.verticalLayout_middle.addWidget(self.log_window)
        self.verticalLayout_middle.addWidget(self.log_window2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_middle)

        # 右侧图片填充区域
        self.labelRight = QtWidgets.QLabel()
        self.labelRight.setAlignment(Qt.AlignCenter)
        sizePolicy.setHeightForWidth(self.labelRight.sizePolicy().hasHeightForWidth())
        self.labelRight.setSizePolicy(sizePolicy)
        self.labelRight.setMinimumSize(QtCore.QSize(400, 200))
        self.labelRight.setMaximumSize(QtCore.QSize(400, 200))
        self.labelRight.setObjectName("labelRight")
        self.labelRight.setAlignment(Qt.AlignCenter)
        path = os.path.join(os.getcwd(), "UI/img/tag.png")
        self.setTitle(path)
        self.verticalLayout.setAlignment(Qt.AlignHCenter)
        self.labelRight.setStyleSheet("border: 1px solid white;")  # 添加显示区域边框
        self.verticalLayout.addWidget(self.labelRight)
        self.verticalLayout_select.addLayout(self.verticalLayout)

        self.btn_tag = QRadioButton("标注")
        self.btn_tag.setChecked(True)

        self.btn_tag.clicked.connect(self.on_btn_tag_toggled)
        self.btn_train = QRadioButton("训练")
        self.btn_train.clicked.connect(self.on_btn_train_toggled)
        self.btn_val = QRadioButton("验证结果")
        self.btn_val.clicked.connect(self.on_btn_val_toggled)

        self.horizontalLayout_select.addWidget(self.btn_tag)
        self.horizontalLayout_select.addWidget(self.btn_train)
        self.horizontalLayout_select.addWidget(self.btn_val)
        self.verticalLayout_select.addLayout(self.horizontalLayout_select)

        # 右下侧填充区域
        # ==================数据标注===============================
        self.text_tag = QtWidgets.QTextEdit(self.centralwidget)

        self.text_tag.setMinimumSize(QtCore.QSize(400, 80))
        self.text_tag.setMaximumSize(QtCore.QSize(400, 80))
        self.text_tag.setObjectName("text_tag")
        self.text_tag.setFontFamily("Agency FB")
        self.text_tag.setFontPointSize(20)
        self.verticalLayout_select.addWidget(self.text_tag)

        self.pushButton_pre_val = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_pre_val.setText("预训练验证")
        self.pushButton_pre_val.setEnabled(False)
        self.pushButton_pre_val.setStyleSheet('text-align:center')
        self.pushButton_pre_val.setSizePolicy(sizePolicy)
        self.pushButton_pre_val.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_pre_val.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_pre_val.setFont(font)

        self.pushButton_update = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_update.setShortcut("Alt+S")
        self.pushButton_update.setText("保存修改")
        self.pushButton_update.setEnabled(False)
        self.pushButton_update.setStyleSheet('text-align:center')
        self.pushButton_update.setSizePolicy(sizePolicy)
        self.pushButton_update.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_update.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_update.setFont(font)

        self.horizontalLayout_pre_val.addWidget(self.pushButton_update)
        self.horizontalLayout_pre_val.addWidget(self.pushButton_pre_val)
        self.verticalLayout_select.addLayout(self.horizontalLayout_pre_val)

        self.pushButton_add = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add.setText("新增数据")
        self.pushButton_add.setEnabled(False)
        self.pushButton_add.setStyleSheet('text-align:center')
        self.pushButton_add.setSizePolicy(sizePolicy)
        self.pushButton_add.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_add.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_add.setFont(font)
        self.pushButton_add.setObjectName("pushButton_add")
        self.horizontalLayout_add_data.addWidget(self.pushButton_add)

        self.pushButton_download = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_download.setText("导出数据")
        self.pushButton_download.setEnabled(False)
        self.pushButton_download.setStyleSheet('text-align:center')
        sizePolicy.setHeightForWidth(self.pushButton_download.sizePolicy().hasHeightForWidth())
        self.pushButton_download.setSizePolicy(sizePolicy)
        self.pushButton_download.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_download.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_download.setFont(font)
        self.pushButton_download.setObjectName("pushButton_download")
        self.horizontalLayout_add_data.addWidget(self.pushButton_download)
        self.verticalLayout_select.addLayout(self.horizontalLayout_add_data)

        self.slider_split_data = QSlider(Qt.Horizontal, self)  # 1
        self.slider_split_data.setEnabled(False)
        self.slider_split_data.setAutoFillBackground(True)
        self.slider_split_data.setValue(80)
        self.slider_split_data.setRange(0, 100)  # 2
        self.slider_split_data.valueChanged.connect(lambda: self.on_change_split_data())
        self.verticalLayout_select.addWidget(self.slider_split_data)

        self.text_split = QtWidgets.QTextEdit(self.centralwidget)
        sizePolicy.setHeightForWidth(self.text_tag.sizePolicy().hasHeightForWidth())
        self.text_split.setSizePolicy(sizePolicy)
        self.text_split.setMinimumSize(QtCore.QSize(400, 30))
        self.text_split.setMaximumSize(QtCore.QSize(400, 30))
        self.text_split.setObjectName("text_split")
        self.text_split.setText("训练集 80%" +
                                "                         " +
                                "验证集 20%")
        self.text_split.isReadOnly()
        self.verticalLayout_select.addWidget(self.text_split)

        # 自定义item中的widget 用来显示自定义的内容
        self.widget_view = QListWidget()
        self.widget_view.itemSelectionChanged.connect(
            self.viewSelectionChanged
        )
        self.widget_view.setSpacing(20)
        self.widget_view.resizeEvent = self.resizeEvent
        self.widget_view.setMovement(QtWidgets.QListView.Static)
        self.widget_view.setFlow(QtWidgets.QListView.LeftToRight)
        self.widget_view.setResizeMode(QtWidgets.QListView.Adjust)
        self.widget_view.setViewMode(QtWidgets.QListView.IconMode)
        self.widget_view.setModelColumn(0)

        self.contextMenu = QMenu(self)
        self.add_train = self.contextMenu.addAction('加入训练集')
        self.add_val = self.contextMenu.addAction('加入验证集')
        self.add_train.triggered.connect(self.add_train_data)
        self.add_val.triggered.connect(self.add_val_data)
        self.widget_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.widget_view.customContextMenuRequested.connect(self.showMenu)

        self.sliderScale = QtWidgets.QSlider()
        self.sliderScale.setOrientation(QtCore.Qt.Horizontal)
        self.sliderScale.setObjectName("sliderScale")
        self.sliderScale.valueChanged.connect(self.onSliderPosChanged)

        self.verticalLayout_select.addWidget(self.widget_view)
        self.verticalLayout_select.addWidget(self.sliderScale)
        self.horizontalLayout_2.addLayout(self.verticalLayout_select)

        # ==================训练===============================
        self.pushButton_config = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_config.setHidden(True)
        self.pushButton_config.setText("参数配置")
        self.pushButton_config.setStyleSheet('text-align:center')
        sizePolicy.setHeightForWidth(self.pushButton_config.sizePolicy().hasHeightForWidth())
        self.pushButton_config.setSizePolicy(sizePolicy)
        self.pushButton_config.setMinimumSize(QtCore.QSize(400, 30))
        self.pushButton_config.setMaximumSize(QtCore.QSize(400, 30))
        self.pushButton_config.setFont(font)
        self.pushButton_config.setObjectName("pushButton_config")
        self.verticalLayout_select.addWidget(self.pushButton_config)

        self.train = QtWidgets.QTextEdit(self.centralwidget)
        self.train.setHidden(True)
        self.train.setFontFamily("Agency FB")
        self.train.setFontPointSize(15)
        self.train.setText("训练信息")
        self.train.setFontPointSize(13)
        # 后期建议将字符串等常量值写进一个const中
        self.train.append(
            "\n" + "学习率:" + "\t\t\t" +
            "0.000000" + "\n" + "损失:" + "\t\t\t" +
            "0.000000" + "\n" + "已训练轮次:" + "\t\t" +
            "0")

        self.train.setReadOnly(True)
        sizePolicy.setHeightForWidth(self.train.sizePolicy().hasHeightForWidth())
        self.train.setSizePolicy(sizePolicy)
        self.train.setMaximumSize(QtCore.QSize(1000, 1000))
        self.train.setObjectName("train")

        self.pushButton_train = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_train.setHidden(True)
        self.pushButton_train.setText("训练")
        self.pushButton_train.setEnabled(False)
        self.pushButton_train.setStyleSheet('text-align:center')
        self.pushButton_train.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_train.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_train.setFont(font)
        self.pushButton_train.setObjectName("pushButton_train")
        self.horizontalLayout_button.addWidget(self.pushButton_train)

        self.pushButton_export = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_export.setHidden(True)
        self.pushButton_export.setText("导出模型")
        self.pushButton_export.setStyleSheet('text-align:center')
        self.pushButton_export.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_export.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_export.setFont(font)
        self.pushButton_export.setEnabled(False)
        self.pushButton_export.setObjectName("pushButton_export")
        self.horizontalLayout_button.addWidget(self.pushButton_export)

        self.pushButton_view = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_view.setHidden(True)
        self.pushButton_view.setText("查看图表")
        self.pushButton_view.setEnabled(False)
        self.pushButton_view.setStyleSheet('text-align:center')
        self.pushButton_view.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_view.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_view.setFont(font)
        self.pushButton_view.setObjectName("pushButton_view")

        self.pushButton_val = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_val.setHidden(True)
        self.pushButton_val.setText("验证")
        self.pushButton_val.setEnabled(False)
        self.pushButton_val.setStyleSheet('text-align:center')
        self.pushButton_val.setMinimumSize(QtCore.QSize(200, 30))
        self.pushButton_val.setMaximumSize(QtCore.QSize(200, 30))
        self.pushButton_val.setFont(font)
        self.pushButton_config.setObjectName("pushButton_val")

        self.verticalLayout_select.addWidget(self.train)
        self.verticalLayout_select.addLayout(self.horizontalLayout_button)
        self.horizontalLayout_val.addWidget(self.pushButton_view)
        self.horizontalLayout_val.addWidget(self.pushButton_val)
        self.verticalLayout_select.addLayout(self.horizontalLayout_val)
        self.horizontalLayout_2.addLayout(self.verticalLayout_select)

        # ==================验证===============================

        self.tagListWidget = QtWidgets.QListWidget()
        self.tagListWidget.setHidden(True)
        sizePolicy.setHeightForWidth(self.tagListWidget.sizePolicy().hasHeightForWidth())
        self.tagListWidget.setSizePolicy(sizePolicy)
        self.tagListWidget.setContentsMargins(0, 3, 0, 3)
        self.tagListWidget.setMaximumSize(QtCore.QSize(400, 700))

        self.tagListWidget.setObjectName("tagListWidget")
        self.tagListWidget.setStyleSheet("border: 1px solid white;")  # 添加显示区域边框
        self.tagListWidget.itemSelectionChanged.connect(self.tagSelectionChanged)
        self.verticalLayout_select.addWidget(self.tagListWidget)
        self.horizontalLayout_2.addLayout(self.verticalLayout_select)

        # 底部美化导航条
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # 槽函数的绑定与触发
    def init_slots(self):

        self.open_select.activated.connect(lambda: self.open_data())
        self.text_tag.textChanged.connect(lambda: self.on_tag_changed())
        self.pushButton_update.clicked.connect(lambda: self.button_update_tag())
        self.pushButton_config.clicked.connect(lambda: self.button_train_config())
        self.pushButton_train.clicked.connect(lambda: self.button_start_train())
        self.pushButton_view.clicked.connect(lambda: self.button_view_log())
        self.pushButton_export.clicked.connect(lambda: self.button_export_model())
        self.pushButton_val.clicked.connect(lambda: self.button_val_data())
        self.pushButton_pre_val.clicked.connect(lambda: self.button_pre_val_data())
        self.pushButton_add.clicked.connect(lambda: self.button_addData())
        self.pushButton_download.clicked.connect(lambda: self.button_export_data())

    # 数据划分的滑块触发函数，当滑动值改变，就会触发
    def on_change_split_data(self):
        # 获取值，然后进行数据集的划分
        train_data_per = self.slider_split_data.value()
        val_data_per = 100 - train_data_per

        self.text_split.setText("训练集 {}%".format(train_data_per) +
                                "                         " +
                                "验证集 {}%".format(val_data_per))

        # self.widget_view.clear()
        self.SPLIT_FLAG = True
        self.reload_split_data(val_data_per / 100)

    # 重写了上一页和下一页的快捷键，进行翻页
    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Down:
            self.button_next_image_open()
        if event.key() == Qt.Key_Up:
            self.button_pre_image_open()

    # 删除条目，鼠标右击之后会出现菜单”删除图片“
    def DeleteItem(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]
        index = self.fileListWidget.currentRow()
        viewCurrIndex = self.view_list.index(item.text())
        self.fileListWidget.takeItem(index)
        self.recog_list.pop(index)

        if self.ADD_FLAG:
            os.remove(os.path.join(self.add_img_file_path, item.text()))
        elif self.img_file_path:
            os.remove(os.path.join(self.img_file_path, item.text()))

        self.tagListWidget.takeItem(index)
        if self.tagList:
            self.tagList.pop(index)

        # 相应数据集中的删除
        text = self.widget_view.item(viewCurrIndex).text().split('-')[-1]
        if text == "验证":
            del self.test_data_file["{}".format(item.text())]
            del self.test_data_tag["{}".format(item.text())]
        if text == "训练":
            del self.train_data_file["{}".format(item.text())]
            del self.train_data_tag["{}".format(item.text())]
        self.widget_view.takeItem(viewCurrIndex)
        self.view_list.pop(viewCurrIndex)

    # def mousePressEvent(self, e: QMouseEvent) -> None:
    #     super(self.widget_view, self).mousePressEvent(e)
    #     if e.button() == Qt.RightButton and self.itemAt(e.pos()):
    #         self.contextMenu.exec(e.globalPos())
    # def on_key_press_event(self, event):
    #     # 判断用户按下的键是否是上箭头键
    #     if event.key() == Qt.Key_Up:
    #         # 获取当前选中的条目
    #         current_item = self.fileListWidget.currentItem()
    #         # 获取当前选中的条目的索引
    #         current_index = self.fileListWidget.row(current_item)
    #         print("当前选中的条目索引：",current_index)
    #
    #         # 判断当前选中的条目是否是第一个条目
    #         if current_index > 0:
    #             # 将当前选中的条目移动到上一个条目
    #             self.fileListWidget.item(current_index - 1).setSelected(True)
    #             self.tagListWidget.item(current_index - 1).setSelected(True)
    #             print("当前选中的条目",self.fileListWidget.currentItem())
    #             current_index = current_index - 1
    #             current_index *= 2
    #             self.index = current_index
    #             print("当前的索引：",self.index)
    #             self.show_img(self.index)
    #             # 将此时预测得到的文本放在文本框中
    #             # 获取当前索引相对应的预测文本
    #             text = self.recog_list[self.index + 1]
    #             self.text_tag.setText(text)
    #
    #
    #     # 判断用户按下的键是否是下箭头键
    #     elif event.key() == Qt.Key_Down:
    #         # 获取当前选中的条目
    #         current_item = self.fileListWidget.currentItem()
    #
    #         # 获取当前选中的条目的索引
    #         current_index = self.fileListWidget.row(current_item)
    #         # 判断当前选中的条目是否是最后一个条目
    #         if current_index < len(self.recog_list) / 2 - 1:
    #             self.fileListWidget.item(current_index + 1).setSelected(True)
    #             self.tagListWidget.item(current_index + 1).setSelected(True)
    #             current_index = current_index + 1
    #             current_index *= 2
    #             self.index = current_index
    #             self.show_img(self.index)
    #             text = self.recog_list[self.index + 1]
    #             self.text_tag.setText(text)

    # 选取文件夹或者图像文件，并进行initial操作；
    # 目前常量值直接是字符串，后期可以将其统一放在一个类中进行管理
    def open_data(self):

        text = self.open_select.currentText()
        self.init_pic(text)

        self.pushButton_pre_val.setEnabled(True)
        self.pushButton_add.setEnabled(True)

    # 初始化操作，将选取的文件或者文件夹中的图片展示在Qlabel上
    def init_pic(self, text):
        if self.ADD_FLAG:
            self.add_img_file_path = QFileDialog.getExistingDirectory(self,
                                                                      "选取数据集",
                                                                      "Images(*.png *.jpg)")  # 起始路径
            if self.Misjudgment(self.add_img_file_path):
                return

            self.index = 0
            # 确保再次导入时，不会扰乱之前已经分好的数据集
            self.addImgList = []
            self.rename(self.add_img_file_path, len(self.recog_list))
            add_img_file = os.listdir(self.add_img_file_path)

            for item in add_img_file:
                self.addImgList.append(os.path.basename(item))
                self.recog_list.append(os.path.basename(item))
            self.importDirImages()
            return
        if text == '导入数据集':
            self.img_file_path = QFileDialog.getExistingDirectory(self,
                                                                  "选取数据集",
                                                                  "Images(*.png *.jpg)")  # 起始路径
            if self.Misjudgment(self.img_file_path):
                return

        if text == '导入图片':
            self.image_path = QtWidgets.QFileDialog.getOpenFileNames(self,
                                                                     "选择图像路径",
                                                                     '.',
                                                                     "Images(*.webp *.jpg *.png)")
            if self.image_path[0].__len__() == 0:
                return

            path = os.path.join(os.getcwd(), 'selectImgs')
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            for item in self.image_path[0]:
                shutil.copy(item, path)
            self.img_file_path = path
        self.rename(self.img_file_path, 0)
        self.show_pred_img(self.img_file_path)

    def Misjudgment(self, path):
        if path == "":
            FALG_ERROR = True
            return FALG_ERROR
        for item in os.listdir(path):
            type = item.split('.')[-1]  # rfine从右侧查找

            if type != "jpg" and type != "png":
                FALG_ERROR = True

                QMessageBox.information(self, "提示", "数据有误，请重新上传！",
                                        QMessageBox.Ok, QMessageBox.Ok)
                return FALG_ERROR

    # 当选取不同文件夹时，会存在同名文件的情况，所以直接在导入图片之前进行rename操作
    def rename(self, img_path, i):
        img = os.listdir(img_path)
        num_img = i
        num = 1
        for item in img:
            new_img_name = "rgb_image_%05d" % num_img + ".png"
            #   将所有的图片文件内名取出来进行重命名
            new_file_name = os.path.join(img_path, new_img_name)
            if new_file_name in img:
                new_file_name = "rgb_image_%05d_num" % num_img + ".png"
                num += 1
            os.rename(os.path.join(img_path, item), new_file_name)
            num_img += 1

    # 新增数据
    def button_addData(self):
        self.ADD_FLAG = True
        text = 'add'
        self.init_pic(text)

    # 训练参数的配置
    def button_train_config(self):
        if self.val is None:
            QMessageBox.information(self, "提示", "请先导出数据集！",
                                    QMessageBox.Ok, QMessageBox.Ok)
            return
        from UI.config_window import Ui_Form
        self.config_select_window = Ui_Form()
        self.config_select_window.GetModelPathlineEdit.setText(self.model_path_name[0])
        # self.config_select_window.default_model = self.model_path_name[0]
        if self.configInfo:
            self.config_select_window.GetDictlineEdit.setText(self.configInfo['dictPath'])
            self.config_select_window.getLrlineEdit.setText(self.configInfo['lr'])
            self.config_select_window.getEpochlineEdit.setText(self.configInfo['epoch'])
            self.config_select_window.getbatchSizelineEdit.setText(self.configInfo['batchSize'])
        self.config_select_window.show()
        self.config_select_window.train_data = self.train_file
        self.config_select_window.test_data = self.test_file
        self.config_select_window.train_tag = self.train_tag
        self.config_select_window.test_tag = self.test_tag
        self.config_select_window.signal_config_list.connect(self.setConfigInfo)

        self.pushButton_train.setEnabled(True)

    # 窗口直接信息的传递
    def setConfigInfo(self, dict):
        self.configInfo = dict

    # 开始训练
    def button_start_train(self):

        # 点击训练按钮，会先进行数据加载，这个过程中，此时加载loding.
        self.box.setHidden(True)
        self.loding.setHidden(False)

        # loding界面的数据更新，新开启线程
        self.loding_thread = LodingThread()
        self.loding_thread.my_signal.connect(self.loding.parameterUpdate)
        self.loding_thread.start()
        self.loding_thread.finished.connect(self.lodingThreadStop)

        # 训练开始，在子线程中进行训练
        self.train_thread = trainThread()
        self.train_thread.start()
        self.train_thread.config_dict = self.configInfo
        self.train_thread.finished.connect(self.threadStop)

        self.pushButton_train.setEnabled(False)
        self.pushButton_pre_val.setEnabled(False)

    # 查看训练的图表
    def button_view_log(self):

        # 判断按钮点击的次数，可以在图表和labelmiddle进行切换
        if self.view_num % 2 == 1:
            self.log_window.setHidden(True)
            self.log_window2.setHidden(True)
            self.box.setHidden(False)
        else:
            self.box.setHidden(True)
            self.log_window.setHidden(False)
            self.log_window2.setHidden(False)

        self.view_num += 1

    def button_export_model(self):

        os.system(f'explorer /select, {self.export_file}')
        self.pushButton_val.setEnabled(True)

    def button_pre_val_data(self):
        self.model_path_name = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                     "选择模型路径(.pdparams)",
                                                                     "./")
        if self.model_path_name[0].__len__() == 0:
            return
        if self.model_path_name[0].split('.')[-1] != "pdparams":
            QMessageBox.information(self, "提示", "模型选取有误！",
                                    QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.ADD_FLAG:
            self.recog_pic(self.add_img_file_path, self.model_path_name[0])
        elif self.img_file_path:
            self.recog_pic(self.img_file_path, self.model_path_name[0])


    def recog_pic(self, img_path, model_path):

        self.PRED_FLAG = True
        self.thread = MyThread()
        self.thread.file_name = img_path
        self.progress = QProgressDialog(self)
        self.progress.setCancelButton(None)

        self.thread.model_path_name = model_path
        self.thread.start()
        self.progress.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        self.progress.setWindowTitle("请稍等")
        self.progress.setLabelText("模型验证中...")
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateProgress)
        self.timer.start(100)
        self.thread.show_pred_img_signal.connect(self.getPredInfo)

        self.thread.finished.connect(self.recog_threadStop)

    def updateProgress(self):
        self.thread.loading_value_signal.connect(self.setValue)

    def setValue(self, value):
        self.progress.setValue(value)
        if value == 100:
            self.progress.close()
            self.pushButton_download.setEnabled(True)

    def button_val_data(self):
        model_path = os.path.join(self.export_file,
                                  "latest.pdparams")
        self.recog_pic(self.val, model_path)

    def recog_threadStop(self):
        self.thread.quit()

    def getPredInfo(self, list):
        self.text_tag.setFontPointSize(20)

        self.pred_list = list
        self.tagList = [self.pred_list[tag] for tag in range(len(self.pred_list)) if tag % 2 == 1]
        self.predImgList = [self.pred_list[img] for img in range(len(self.pred_list)) if img % 2 == 0]

        for item in range(len(self.tagList)):
            img = self.predImgList[item]
            if self.train_data_tag.__contains__("{}".format(os.path.basename(img))):
                self.train_data_tag['{}'.format(format(os.path.basename(img)))] = self.tagList[item]
                self.text_tag.setText(self.tagList[item])

            else:
                self.test_data_tag['{}'.format(format(os.path.basename(img)))] = self.tagList[item]
                self.text_tag.setText(self.tagList[item])

            item = QtWidgets.QListWidgetItem(self.tagList[item])
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tagListWidget.addItem(item)

        self.tagListWidget.repaint()

    def threadStop(self):
        self.thread.quit()

    def button_export_data(self):
        self.saveData()
        if self.save_data_dir == "":
            return
        self.geneData()

        QMessageBox.information(self, "提示", "数据集已成功导出在\n{}！".format(self.save_data_dir),
                                QMessageBox.Ok, QMessageBox.Ok)

    def saveData(self):
        export_path = QFileDialog.getExistingDirectory(self,
                                                       "选取数据集存放路径",
                                                       "Images(*.png *.jpg)")
        if export_path == "":
            return
        self.save_data_dir = os.path.join(export_path, "train_data")

    def geneData(self):
        if not os.path.exists(self.save_data_dir):
            os.makedirs(self.save_data_dir)
        test_path_dir = os.path.join(self.save_data_dir, "test")
        train_path_dir = os.path.join(self.save_data_dir, "train")
        val_path_dir = os.path.join(self.save_data_dir, "val")

        test_tag_path_dir = os.path.join(self.save_data_dir, "rec_gt_test.txt")
        train_tag_path_dir = os.path.join(self.save_data_dir, "rec_gt_train.txt")
        if not os.path.exists(test_path_dir):
            os.makedirs(test_path_dir)
        if not os.path.exists(train_path_dir):
            os.makedirs(train_path_dir)
        if not os.path.exists(val_path_dir):
            os.makedirs(val_path_dir)
        self.clear(test_path_dir)
        self.clear(train_path_dir)
        self.clear(val_path_dir)

        for key, value in self.train_data_file.items():
            shutil.copy(value, train_path_dir)
            shutil.copy(value, val_path_dir)

        for key, value in self.test_data_file.items():
            shutil.copy(value, test_path_dir)
            shutil.copy(value, val_path_dir)

        with open(train_tag_path_dir, "a", encoding='utf-8') as f:
            f.truncate(0)
            for key, value in self.train_data_tag.items():
                f.write(key + "\t" + value + "\n")

        with open(test_tag_path_dir, "a", encoding='utf-8') as f:
            f.truncate(0)
            for key, value in self.test_data_tag.items():
                f.write(key + '\t' + value + "\n")
        self.val = val_path_dir
        self.train_file = train_path_dir
        self.test_file = test_path_dir
        self.train_tag = train_tag_path_dir
        self.test_tag = test_tag_path_dir

    def clear(self, path):
        for item in os.listdir(path):
            os.remove(os.path.join(path, item))

    def updateTrainInfo(self):
        print(self.log_window.logInfo)
        lr = str("{:.6f}".format(self.log_window.logInfo['lr']))
        epoch = str(self.log_window.logInfo['epoch'])
        loss = str("{:.6f}".format(self.log_window.logInfo['loss']))
        self.train.setText("训练信息")
        self.train.append(
            "\n" + "学习率:" + "\t\t\t" +
            lr + "\n" + "损失:" + "\t\t\t" +
            loss + "\n" + "已训练轮次:" + "\t\t" +
            epoch)
        if self.log_window.logInfo['epoch'] == self.log_window.logInfo['epoch_num']:
            self.timer.stop()
            QMessageBox.information(self, "提示", "训练完成！",
                                    QMessageBox.Ok, QMessageBox.Ok)
            self.export_file = self.log_window.logInfo['bestMetricPath']
            self.export_file = os.path.join(os.getcwd(), "output\\model_dir")
            export_inference(os.path.join(self.export_file, "latest.pdparams"),
                             os.path.join(os.getcwd(), "output\\inference_vision"))

            self.pushButton_export.setEnabled(True)
            self.pushButton_pre_val.setEnabled(True)

    def getInfo(self, dict):
        # 和主窗口传递消息，当子窗口得到之后，进行坐标的更新
        self.log_window.logInfo = dict
        self.log_window.setLogInfo(dict)
        self.log_window2.logInfo = dict
        self.log_window2.setLogInfo(dict)
        self.log_window.x_Aix.setRange(0, self.log_window.logInfo['epoch_num'])
        self.log_window2.x_Aix.setRange(0, self.log_window.logInfo['epoch_num'])

    def threadStop(self):
        # 退出线程
        self.train_thread.quit()

    def lodingThreadStop(self):
        # 退出线程
        self.loding_thread.quit()
        self.loding.setHidden(True)
        self.box.setHidden(False)
        self.pushButton_view.setEnabled(True)

        # loding加载数据之后，日志就开始加载
        self.log_thread = logThread()
        self.log_thread.start()
        # 和日志线程传递消息，获取当前的训练信息
        self.log_thread.logSignal.connect(self.getInfo)
        # 更新日志
        self.log_window.timer = QTimer()
        self.log_window.point_list.clear()
        self.log_window.point_list.append(QPointF(0, 0))
        self.log_window.timer.timeout.connect(self.log_window.update)
        self.log_window.timer.start(2000)

        self.log_window2.timer = QTimer()
        self.log_window2.point_list.clear()
        self.log_window2.point_list.append(QPointF(0, 0))
        self.log_window2.timer.timeout.connect(self.log_window2.update)
        self.log_window2.timer.start(2000)

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateTrainInfo)
        self.timer.start(1000)

    def fileSearchChanged(self):
        pass

    def on_tag_changed(self):
        self.updateTagText = self.text_tag.toPlainText()
        self.pushButton_update.setEnabled(True)

    def button_update_tag(self):

        # 获取到FileList中的图像名
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]
        print(item.text())
        viewCurrIndex = self.view_list.index(item.text())

        text = self.widget_view.item(viewCurrIndex).text().split('-')[-1]
        if text == "验证":
            self.test_data_tag["{}".format(item.text())] = self.updateTagText
        if text == "训练":
            self.train_data_tag["{}".format(item.text())] = self.updateTagText

        self.button_next_image_open()

    def importDirImages(self):

        self.fileListWidget.clear()

        for filename in self.recog_list:
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.fileListWidget.addItem(item)
        self.fileListWidget.repaint()
        self.reload_split_data(0.2)

    def reload_split_data(self, percentage):

        train_data, test_data = split_data(self.recog_list, percentage)
        if self.ADD_FLAG:
            train_data, test_data = split_data(self.addImgList, percentage)
            for i in train_data:
                i = os.path.join(self.add_img_file_path, i)
                data_type = "训练"
                data_name = os.path.basename(i)
                self.display(i, data_type, data_name)
                self.view_list.append(os.path.basename(i))
                self.img_item.setBackground(QColor("red"))

                self.train_data_tag["{}".format(os.path.basename(i))] = "单击鼠标左键开始标注"
                self.train_data_file["{}".format(os.path.basename(i))] = i
                self.text_tag.setText(self.train_data_tag["{}".format(os.path.basename(i))])

            for i in test_data:
                i = os.path.join(self.add_img_file_path, i)

                data_name = os.path.basename(i)
                data_type = "验证"
                self.display(i, data_type, data_name)
                self.img_item.setBackground(QColor("blue"))

                self.view_list.append(os.path.basename(i))
                self.test_data_tag["{}".format(os.path.basename(i))] = "单击鼠标左键开始标注"
                self.test_data_file["{}".format(os.path.basename(i))] = i
                self.text_tag.setText(self.test_data_tag["{}".format(os.path.basename(i))])
            QApplication.processEvents()
            return
        elif self.SPLIT_FLAG:
            self.widget_view.clear()
            self.view_list = []
            train_data, test_data = split_data(self.recog_list, percentage)
        self.val_data.clear()
        self.train_data_file.clear()
        self.train_data_tag.clear()
        self.test_data_tag.clear()
        self.test_data_file.clear()
        # 本部分代码后期优化
        for i in train_data:
            i = os.path.join(self.img_file_path, i)
            data_type = "训练"
            data_name = os.path.basename(i)
            self.display(i, data_type, data_name)
            # self.widget_view.addItem(self.img_item)
            self.view_list.append(os.path.basename(i))
            self.img_item.setBackground(QColor("red"))

            self.train_data_tag["{}".format(os.path.basename(i))] = "单击鼠标左键开始标注"
            self.train_data_file["{}".format(os.path.basename(i))] = i
            self.text_tag.setText(self.train_data_tag["{}".format(os.path.basename(i))])

        for i in test_data:
            i = os.path.join(self.img_file_path, i)

            data_name = os.path.basename(i)
            data_type = "验证"
            self.display(i, data_type, data_name)
            self.img_item.setBackground(QColor("blue"))

            self.view_list.append(os.path.basename(i))
            self.test_data_tag["{}".format(os.path.basename(i))] = "单击鼠标左键开始标注"
            self.test_data_file["{}".format(os.path.basename(i))] = i
            self.text_tag.setText(self.test_data_tag["{}".format(os.path.basename(i))])

        QApplication.processEvents()

        self.slider_split_data.setEnabled(True)

    def display(self, image, data_type, data_name):
        image = cv2.imread(image)
        height, width, channels = image.shape
        bytes_per_line = width * channels
        qImage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)

        self.img_item = QListWidgetItem(QIcon(pixmap), "{}----{}".format(data_name, data_type))
        self.widget_view.addItem(self.img_item)

    def resizeEvent(self, event):
        width = self.widget_view.contentsRect().width()
        self.sliderScale.setMaximum(width)
        self.sliderScale.setValue(width - 40)

    def onSliderPosChanged(self, value):
        self.widget_view.setIconSize(QSize(value, value))

    # 显示右键菜单
    def showMenu(self, pos):
        # pos 鼠标位置
        # 菜单显示前,将它移动到鼠标点击的位置
        self.contextMenu.exec_(QCursor.pos())  # 在鼠标位置显示

    def showDeleteMenu(self, pos):
        # pos 鼠标位置
        # 菜单显示前,将它移动到鼠标点击的位置
        self.delete_Menu.exec_(QCursor.pos())  # 在鼠标位置显示

    def add_train_data(self):
        items = self.widget_view.selectedItems()
        if not items:
            return
        item = items[0]
        currIndex = self.widget_view.row(item)

        img_name = str(item.text().split('-')[0])
        text = str(item.text().split('-')[-1])
        if text == "训练":
            return

        if self.train_data_tag.__contains__("{}".format(img_name)):
            tag = self.train_data_tag["{}".format(img_name)]
        else:
            tag = self.test_data_tag["{}".format(img_name)]

        self.widget_view.item(currIndex).setText("{}----{}".format(img_name, "训练"))
        self.widget_view.item(currIndex).setBackground(QColor("red"))

        # 更改了标记之后，从原来的验证集的中删除，加入到训练集中：
        self.train_data_tag["{}".format(img_name)] = tag
        self.train_data_file["{}".format(img_name)] = self.test_data_file["{}".format(img_name)]
        del self.test_data_file["{}".format(img_name)]
        del self.test_data_tag["{}".format(img_name)]

    def add_val_data(self):
        items = self.widget_view.selectedItems()
        if not items:
            return
        item = items[0]
        currIndex = self.widget_view.row(item)

        img_name = str(item.text().split('-')[0])
        text = str(item.text().split('-')[-1])

        if self.train_data_tag.__contains__("{}".format(img_name)):
            tag = self.train_data_tag["{}".format(img_name)]
        else:
            tag = self.test_data_tag["{}".format(img_name)]
        if text == "验证":
            return

        self.widget_view.item(currIndex).setText("{}----{}".format(img_name, "验证"))
        self.widget_view.item(currIndex).setBackground(QColor("blue"))

        # 更改了标记之后，从原来的验证集的中删除，加入到训练集中：
        self.test_data_tag["{}".format(img_name)] = tag
        self.test_data_file["{}".format(img_name)] = self.train_data_file["{}".format(img_name)]
        del self.train_data_file["{}".format(img_name)]
        del self.train_data_tag["{}".format(img_name)]

    def viewSelectionChanged(self):
        items = self.widget_view.selectedItems()
        if not items:
            return
        item = items[0]
        img_text = str(item.text().split('-')[0])
        viewCurrIndex = self.recog_list.index(img_text)

        self.fileListWidget.item(viewCurrIndex).setSelected(True)
        if self.train_data_tag.__contains__("{}".format(img_text)):
            self.text_tag.setText(self.train_data_tag["{}".format(img_text)])
        else:
            self.text_tag.setText(self.test_data_tag["{}".format(img_text)])

        if viewCurrIndex < len(self.recog_list):
            filename = self.recog_list[viewCurrIndex]
            if filename:
                self.loadFile(viewCurrIndex)

    def fileSelectionChanged(self):

        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]
        currIndex = self.fileListWidget.row(item)
        img_text = str(item.text())
        viewCurrIndex = self.view_list.index(img_text)

        if self.btn_val.isChecked():
            self.tagListWidget.item(currIndex).setSelected(True)
        if self.train_data_tag.__contains__("{}".format(img_text)):

            self.text_tag.setText(self.train_data_tag["{}".format(img_text)])
        else:
            self.text_tag.setText(self.test_data_tag["{}".format(img_text)])
        self.widget_view.scrollToItem(self.widget_view.item(viewCurrIndex))
        if currIndex < len(self.recog_list):
            filename = self.recog_list[currIndex]
            if filename:
                self.loadFile(currIndex)

    def tagSelectionChanged(self):
        items = self.tagListWidget.selectedItems()
        if not items:
            return
        item = items[0]
        tagCurrIndex = self.tagListWidget.row(item)
        self.fileListWidget.item(tagCurrIndex).setSelected(True)
        img_text = self.fileListWidget.item(tagCurrIndex).text()
        if self.train_data_tag.__contains__("{}".format(img_text)):

            self.text_tag.setText(self.train_data_tag["{}".format(img_text)])
        else:
            self.text_tag.setText(self.test_data_tag["{}".format(img_text)])
        if tagCurrIndex < len(self.tagList):
            tag = self.tagList[tagCurrIndex]
            if tag:
                self.loadFile(tagCurrIndex)

    def loadFile(self, currIndex):
        self.index = currIndex
        self.show_img(self.index)

    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def on_btn_tag_toggled(self):
        self.pushButton_config.setHidden(True)
        self.train.setHidden(True)
        self.text_tag.setHidden(False)
        self.widget_view.setHidden(False)
        self.tagListWidget.setHidden(True)
        self.pushButton_update.setHidden(False)
        self.pushButton_update.setEnabled(False)
        self.pushButton_add.setHidden(False)
        self.pushButton_download.setHidden(False)
        self.pushButton_train.setHidden(True)
        self.pushButton_view.setHidden(True)
        self.pushButton_val.setHidden(True)
        self.pushButton_export.setHidden(True)
        self.slider_split_data.setHidden(False)
        self.sliderScale.setHidden(False)
        self.text_split.setHidden(False)
        self.pushButton_pre_val.setHidden(False)
        path = os.path.join(os.getcwd(), "UI/img/tag.png")
        self.setTitle(path)

    def on_btn_train_toggled(self):
        self.text_tag.setHidden(True)
        self.tagListWidget.setHidden(True)
        self.pushButton_config.setHidden(False)
        self.train.setHidden(False)
        self.widget_view.setHidden(True)
        self.pushButton_update.setHidden(True)
        self.pushButton_add.setHidden(True)
        self.pushButton_download.setHidden(True)
        self.pushButton_train.setHidden(False)
        self.pushButton_export.setHidden(False)
        self.pushButton_view.setHidden(False)
        self.pushButton_val.setHidden(False)
        self.slider_split_data.setHidden(True)
        self.sliderScale.setHidden(True)
        self.text_split.setHidden(True)
        self.pushButton_pre_val.setHidden(True)
        path = os.path.join(os.getcwd(), "UI/img/train.png")
        self.setTitle(path)

    def on_btn_val_toggled(self):
        self.pushButton_config.setHidden(True)
        self.train.setHidden(True)
        self.text_tag.setHidden(False)
        self.tagListWidget.setHidden(False)
        self.widget_view.setHidden(True)
        self.pushButton_train.setHidden(True)
        self.pushButton_view.setHidden(True)
        self.pushButton_val.setHidden(True)
        self.pushButton_export.setHidden(True)
        self.pushButton_update.setHidden(True)
        self.pushButton_add.setHidden(True)
        self.pushButton_download.setHidden(True)
        self.slider_split_data.setHidden(True)
        self.sliderScale.setHidden(True)
        self.text_split.setHidden(True)
        self.pushButton_pre_val.setHidden(True)
        path = os.path.join(os.getcwd(), "UI/img/val.png")
        self.setTitle(path)

    def show_pred_img(self, path):
        self.index = 0
        # 初始化界面的数据加载，初始化时，需要一次性将所有的条目加载，后期就不应该再继续加载了。影响快捷键的操作
        self.recog_list = os.listdir(path)
        img_name = self.recog_list[self.index]

        self.img_file_name = os.path.dirname(img_name)
        self.importDirImages()
        self.show_img(self.index)

    def show_img(self, index):
        last_item = self.fileListWidget.item(self.index)
        self.fileListWidget.scrollToItem(last_item)
        img = self.recog_list[index]

        if self.train_data_file.__contains__("{}".format(img)):
            self.box.set_image(self.train_data_file["{}".format(img)])
        else:
            self.box.set_image(self.test_data_file["{}".format(img)])

        QApplication.processEvents()

    def button_next_image_open(self):

        if self.index == len(self.recog_list) - 1:
            return
        self.widget_view.item(self.viewCurrIndex).setBackground(QColor("0"))
        self.index += 1
        self.show_img(self.index)

        self.fileListWidget.item(int(self.index)).setSelected(True)
        img_text = self.fileListWidget.item(int(self.index)).text()
        if self.PRED_FLAG:
            self.tagListWidget.item(int(self.index)).setSelected(True)

        if self.train_data_tag.__contains__("{}".format(img_text)):
            self.text_tag.setText(self.train_data_tag["{}".format(img_text)])
        else:
            self.text_tag.setText(self.test_data_tag["{}".format(img_text)])

    def button_pre_image_open(self):
        if self.index == const.recog_imgIdx_init:
            return
        self.widget_view.item(self.viewCurrIndex).setBackground(QColor("0"))

        self.index -= const.recog_imgIdx
        self.show_img(self.index)
        last_item = self.fileListWidget.item(self.index)
        self.fileListWidget.item(self.index).setSelected(True)
        img_text = self.fileListWidget.item(int(self.index)).text()

        if self.PRED_FLAG:
            self.tagListWidget.item(self.index).setSelected(True)
            self.tagListWidget.scrollToItem(last_item)
        self.fileListWidget.scrollToItem(last_item)

        if self.train_data_tag.__contains__("{}".format(img_text)):
            self.text_tag.setText(self.train_data_tag["{}".format(img_text)])
        else:
            self.text_tag.setText(self.test_data_tag["{}".format(img_text)])

    def setTitle(self, path):
        pix = QPixmap(path)
        self.labelRight.setPixmap(pix)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap(os.path.join(os.getcwd(), "UI/img/ocr.jpeg")), QIcon.Normal, QIcon.Off)
    ui.setWindowIcon(icon)
    ui.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
