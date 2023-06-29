# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QThread

sys.path.insert(0, "..")

import numpy as np

import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))

from UI.ppocr.utils.logging import get_logger

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from UI.ppocr.data import create_operators, transform
from UI.ppocr.modeling.architectures import build_model
from UI.ppocr.postprocess import build_post_process
from UI.ppocr.utils.save_load import load_model
from UI.ppocr.utils.utility import get_image_file_list
from StyleText.utils.config import load_config



class MyThread(QThread):
    # 定义一个传输 str类型的信号
    # send_singal = pyqtSignal(int)
    # 处理完毕图像数据信号
    show_pred_img_signal = QtCore.pyqtSignal(list)
    loading_value_signal = QtCore.pyqtSignal(int)

    # 线程接收参数信号
    # to_start_image_process_thread_signal = QtCore.Signal(bytes, list)
    # 接受主线程传过来的文件夹名称
    file_name = None
    model_path_name = None

    def get_file_name(self, file_name):
        self.file_name = file_name
        return file_name

    def get_model_path_name(self, model_path_name):
        self.model_path_name = model_path_name
        return model_path_name

    # 重写 run() 函数
    def run(self):

        '''
        新增判空处理
        '''
        # 定义一个列表，存放图片地址以及预测文本
        list = []
        # 使用mmocr进行推理
        # ocr = MMOCR(det='', recog='SATRN')
        # # 推理===>组要将得到的results存入对应的文件夹，其次将得到的可视化识别结果进行展示
        # results = ocr.readtext(img, output=None, export=None)
        file_Name = self.get_file_name(self.file_name)
        if file_Name == "":
            return
        # 使用paddleocr进行推理
        model_path = self.get_model_path_name(self.model_path_name)
        print("获得的模型文件的路径", model_path)
        pre_text = self.getResult(file_Name, model_path)
        for img_name in os.listdir(file_Name):
            img_name = os.path.join(file_Name, img_name)
            list.append(img_name)
            pred_text = pre_text.get(img_name)
            list.append(pred_text)

        self.show_pred_img_signal.emit(list)

    def getResult(self, img_path, model_path):
        pred_info = self.main(img_path, model_path)
        return pred_info

    def main(self, recog_img, model_path):
        num = 0
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        save_model_dir = os.path.join(path, r"resources/output/rec_chinese_common_v2.0")
        character_dict_path = os.path.join(path, r"resources/UI/ppocr/utils/dict/en_dict.txt")
        config_path = os.path.join(path, r"resources/configs\rec\ch_ppocr_v2.0\rec_chinese_common_train_v2.0.yml")

        config = config_path
        config = load_config(config)
        global_config = config['Global']
        config['Global']['save_model_dir'] = save_model_dir
        config['Global']['character_dict_path'] = character_dict_path
        config['Global']['infer_img'] = recog_img
        config['Global']['pretrained_model'] = model_path

        # build post process
        post_process_class = build_post_process(config['PostProcess'],
                                                global_config)

        # build model
        if hasattr(post_process_class, 'character'):
            char_num = len(getattr(post_process_class, 'character'))
            if config['Architecture']["algorithm"] in ["Distillation",
                                                       ]:  # distillation model
                for key in config['Architecture']["Models"]:
                    if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                        out_channels_list = {}
                        if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                            char_num = char_num - 2
                        out_channels_list['CTCLabelDecode'] = char_num
                        out_channels_list['SARLabelDecode'] = char_num + 2
                        config['Architecture']['Models'][key]['Head'][
                            'out_channels_list'] = out_channels_list
                    else:
                        config['Architecture']["Models"][key]["Head"][
                            'out_channels'] = char_num
            elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head loss
                out_channels_list = {}
                if config['PostProcess']['name'] == 'SARLabelDecode':
                    char_num = char_num - 2
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                config['Architecture']['Head'][
                    'out_channels_list'] = out_channels_list
            else:  # base rec model
                config['Architecture']["Head"]['out_channels'] = char_num
        model = build_model(config['Architecture'])

        load_model(config, model)

        # create data ops
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        global_config['infer_mode'] = True
        ops = create_operators(transforms, global_config)

        save_res_path = config['Global'].get('save_res_path',
                                             "./output/rec/predicts_rec.txt")
        if not os.path.exists(os.path.dirname(save_res_path)):
            os.makedirs(os.path.dirname(save_res_path))

        model.eval()
        logger = get_logger()
        # 新建一个字典,存储我们最终的识别结果
        result_info = {}
        recog_num = get_image_file_list(config['Global']['infer_img']).__len__()
        with open(save_res_path, "w") as fout:
            for file in get_image_file_list(config['Global']['infer_img']):
                logger.info("infer_img: {}".format(file))
                with open(file, 'rb') as f:
                    img = f.read()
                    data = {'image': img}
                batch = transform(data, ops)

                images = np.expand_dims(batch[0], axis=0)
                images = paddle.to_tensor(images)

                preds = model(images)
                post_result = post_process_class(preds)

                info = None
                if isinstance(post_result, dict):
                    rec_info = dict()
                    for key in post_result:
                        if len(post_result[key][0]) >= 2:
                            rec_info[key] = {
                                "label": post_result[key][0][0],
                                "score": float(post_result[key][0][1]),
                            }
                    info = json.dumps(rec_info, ensure_ascii=False)
                else:
                    if len(post_result[0]) >= 2:
                        info = post_result[0][0] + "\t" + str(post_result[0][1])

                if info is not None:
                    logger.info("\t result: {}".format(info))
                    result_info[file] = post_result[0][0]
                    fout.write(file + "\t" + info + "\n")
                num += 1
                progress_num = num / recog_num * 100

                self.loading_value_signal.emit(progress_num)

        logger.info("success!")
        return result_info
