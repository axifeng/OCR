import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
from paddle.jit import to_static

from UI.ppocr.modeling.architectures import build_model
from UI.ppocr.postprocess import build_post_process
from UI.ppocr.utils.save_load import load_model
from UI.ppocr.utils.logging import get_logger
from util.program import load_config, merge_config, ArgsParser


def export_single_model(model,
                        arch_config,
                        save_path,

                        input_shape=None,
                        quanter=None):
    if arch_config["algorithm"] == "SRN":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 64, 256], dtype="float32"), [
                paddle.static.InputSpec(
                    shape=[None, 256, 1],
                    dtype="int64"), paddle.static.InputSpec(
                    shape=[None, max_text_length, 1], dtype="int64"),
                paddle.static.InputSpec(
                    shape=[None, 8, max_text_length, max_text_length],
                    dtype="int64"), paddle.static.InputSpec(
                    shape=[None, 8, max_text_length, max_text_length],
                    dtype="int64")
            ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SAR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"),
            [paddle.static.InputSpec(
                shape=[None], dtype="float32")]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SVTR":
        if arch_config["Head"]["name"] == 'MultiHead':
            other_shape = [
                paddle.static.InputSpec(
                    shape=[None, 3, 48, -1], dtype="float32"),
            ]
        else:
            other_shape = [
                paddle.static.InputSpec(
                    shape=[None] + input_shape, dtype="float32"),
            ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "PREN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 512], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["model_type"] == "sr":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 16, 64], dtype="float32")
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "ViTSTR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 224, 224], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "ABINet":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 32, 128], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["NRTR", "SPIN"]:
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 32, 100], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "VisionLAN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 256], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "RobustScanner":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"), [
                paddle.static.InputSpec(
                    shape=[None, ], dtype="float32"),
                paddle.static.InputSpec(
                    shape=[None, max_text_length], dtype="int64")
            ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SEED":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 256], dtype="float32")
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["LayoutLM", "LayoutLMv2", "LayoutXLM"]:
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, 512, 4], dtype="int64"),  # bbox
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # token_type_ids
            paddle.static.InputSpec(
                shape=[None, 3, 224, 224], dtype="int64"),  # image
        ]
        if model.backbone.use_visual_backbone is False:
            input_spec.pop(4)
        model = to_static(model, input_spec=[input_spec])
    else:
        infer_shape = [3, -1, -1]
        if arch_config["model_type"] == "rec":
            infer_shape = [3, 32, -1]  # for rec model, H must be 32
            if "Transform" in arch_config and arch_config[
                "Transform"] is not None and arch_config["Transform"][
                "name"] == "TPS":
                infer_shape[-1] = 100
        elif arch_config["model_type"] == "table":
            infer_shape = [3, 488, 488]
            if arch_config["algorithm"] == "TableMaster":
                infer_shape = [3, 480, 480]
            if arch_config["algorithm"] == "SLANet":
                infer_shape = [3, -1, -1]
        model = to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + infer_shape, dtype="float32")
            ])

    if quanter is None:
        paddle.jit.save(model, save_path)
    else:
        quanter.save_quantized_model(model, save_path)

    return


def export_inference(model_path, inference_path):
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    config_path = os.path.join(path, r"resources/configs\rec\ch_ppocr_v2.0\rec_chinese_common_train_v2.0.yml")
    character_dict_path = os.path.join(path, r"resources/UI/ppocr/utils/dict/en_dict.txt")

    config = config_path
    config = load_config(config)
    global_config = config['Global']
    config['Global']['save_inference_dir'] = inference_path
    config['Global']['pretrained_model'] = model_path
    config['Global']['character_dict_path'] = character_dict_path

    post_process_class = build_post_process(config["PostProcess"],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if config["Architecture"]["Models"][key]["Head"][
                    "name"] == 'MultiHead':  # multi head
                    out_channels_list = {}
                    if config['PostProcess'][
                        'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"] = char_num
                # just one final tensor needs to exported for inference
                config["Architecture"]["Models"][key][
                    "return_all_feats"] = False
        elif config['Architecture']['Head'][
            'name'] == 'MultiHead':  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, 'character'))
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    # for sr algorithm
    if config["Architecture"]["model_type"] == "sr":
        config['Architecture']["Transform"]['infer_mode'] = True
    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config['Architecture']["model_type"])
    model.eval()

    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if arch_config["algorithm"] == "SVTR" and arch_config["Head"][
        "name"] != 'MultiHead':
        input_shape = config["Eval"]["dataset"]["transforms"][-2][
            'SVTRRecResizeImg']['image_shape']
    else:
        input_shape = None

    if arch_config["algorithm"] in ["Distillation", ]:  # distillation model
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(model.model_list[idx], archs[idx],
                                sub_model_save_path)
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(
            model, arch_config, save_path, input_shape=input_shape)


if __name__ == "__main__":
    export_inference(r"D:\OCR_DEBUG\resources\output\model_dir\latest.pdparams",r"D:\OCR_DEBUG\resources\output\inference")
