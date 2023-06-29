# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread

from UI.ppocr.utils.utility import set_seed
import util.program as program
from util.train import main


def getTrainLog(config_dist):
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    print(vdl_writer)
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    set_seed(seed)

    config['Global']['pretrained_model'] = config_dist['modelPath']
    config['Train']['dataset']['data_dir'] = config_dist['train_data']
    config['Train']['dataset']['label_file_list'] = config_dist['train_tag']
    config['Eval']['dataset']['data_dir'] = config_dist['test_data']
    config['Eval']['dataset']['label_file_list'] = config_dist['test_tag']
    config['Global']['epoch_num'] = int(config_dist['epoch'])
    config['Global']['character_dict_path'] = config_dist['dictPath']
    config['Optimizer']['lr']['learning_rate'] = float(config_dist['lr'])
    config['Train']['loader']['batch_size_per_card'] = int(config_dist['batchSize'])

    main(config, device, logger, vdl_writer)


class trainThread(QThread):
    config_dict = None

    def run(self):
        '''
        获取日志的思路，由于训练是在子线程中，并且，当我们开始训练之后，需要一直等到训练结束，才能得到完整的日志信息
        每隔1秒，去读取日志信息。日志信息在program.train()中，使用一个字典存储，一种使用全局变量，一种使用set方法(舍弃)。
        '''

        getTrainLog(self.config_dict)
        return
