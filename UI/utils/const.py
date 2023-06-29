class const:
    # ------------------ main.py----------------------

    # 图片索引和标注信息索引之间的差值
    diff_index = 1

    # ------------------ data_itemwindow.py----------------------

    # 获取单个文件时，返回列表中第一个为路径，第二个为文件类型
    get_result_index = 0
    # 获取文件类型
    split_file_type = -1
    # 重新标注时，标注数据的路径索引
    re_tag_img_file_index = 0
    # 是否是重新标注标记的索引
    re_tag_flag_index = 1
    # 使用识别模型得到的标注信息的起始索引
    re_tag_pred_text_index = 2

    # ------------------ data_window.py----------------------
    # 图片显示的宽度
    resize_img_weight = 800
    # 图片显示的高度
    resize_img_height = 100
    # 图片宽度索引
    img_weight_index = 0
    # 图片高度索引
    img_height_index = 1
    # 图片索引和图片总数的差值
    diff_index_count = 1
    # 下一页图片索引加一
    next_img_index = 1
    # 标注信息列表索引
    list_tag_index = 1
    # 标注信息保存路径索引
    list_tag_resultPath_index = -1
    # 标注信息图片路径索引
    list_tag_imgPath_index = 0
    # 标注信息列表长度和标注信息索引之间的差值
    diff_textIndex_lenTagList = 1
    # 倒数第二张图片：使得最后一张的标注信息进行保存
    diff_seconde_to_last = 2
    # 图片的初始索引：判断当前图片是否为第一张
    img_index = 0
    # 图片索引减一：上一页
    pre_img_index = 1
    # 最后一个标注信息的索引
    last_tag_index = -1
    # 标注信息为空的时候
    list_tag_init_num = 0
    # 最后一个标注信息的文本索引
    list_tag_last_index = -1

    # ------------------recog_select_window.py----------------------
    # 获取模型路径
    model_path_index = 0
    # ------------------recog_window.py----------------------
    # 识别图片索引和识别列表的差值
    diff_imgIdx_recogListLen = 1
    # 识别图片的索引
    recog_imgIdx = 1
    # 识别图片的初始索引
    recog_imgIdx_init = 0
    # 识别的列表为0时
    recog_listLen = 2
    # 识别时，筛选的错误数据的存储位置
