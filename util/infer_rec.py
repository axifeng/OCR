# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))

from ppocr.utils.logging import get_logger

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import util.program as program
from StyleText.utils.config import load_config





if __name__ == '__main__':
    #     cmd_args=[
    #     "-c ",
    #     "D:\PaddleOCR\configs\rec\ch_ppocr_v2.0\rec_chinese_common_train_v2.0.yml",
    #     "-o ",
    #     "Global.pretrained_model=D:\PaddleOCR\output\rec_chinese_common_v2.0\iter_epoch_450 Global.load_static_weights=false Global.infer_img=D:\PaddleOCR\train_data\VIN_DATA\train"
    # ]
    #     logger = get_logger(log_file=log_file)
    logger = get_logger()

    config, device, logger, vdl_writer = program.preprocess()
    # output = subprocess.Popen(cmd_args,stdout=subprocess.PIPE).communicate()[0]

    main()
