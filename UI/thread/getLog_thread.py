# -*- coding: utf-8 -*-

import sys


from PyQt5.QtCore import QThread, pyqtSignal

import util.program as program


class logThread(QThread):
    logSignal = pyqtSignal(dict)  # 定义自定义信号

    def __init__(self):
        super(logThread, self).__init__()

    def run(self):
        while True:
            self.now_epoch = []

            log = program.train_log_dist
            if log['epoch'] not in self.now_epoch:
                self.now_epoch.append(log['epoch'])
                self.logSignal.emit(log)  # 向主线程发送信号
            if log['epoch'] == log['epoch_num']:
                break
        return
