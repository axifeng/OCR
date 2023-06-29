from PyQt5.QtCore import QThread, pyqtSignal


class trainInfoThread(QThread):
    updateTrainInfo_signal = pyqtSignal(int)
    dict = {}

    def __init__(self):
        super(trainInfoThread, self).__init__()
        self.now_epoch = []
        self.epoch_num = 0

    def run(self):
        print(self.dict)
        while self.epoch_num < self.dict['epoch_num']:

            if self.dict['epoch'] not in self.now_epoch:
                epoch = int(self.dict['epoch'])
                self.now_epoch.append(epoch)

                self.updateTrainInfo_signal.emit(self.dict)
