"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from models.yolo_demo import  Model
import models

from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import onnx

def load_model_new():
    save_path = './weights/yolov5_05_net_new1.pt'
    model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1)
    model.load_state_dict(torch.load(save_path))
    model.float().fuse().eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', nargs='+', type=int, default=[320, 320], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--simplify', action='store_true', default=False, help='simplify onnx')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    parser.add_argument('--onnx_infer', action='store_true', default=True, help='onnx infer test')
    #=======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', default=False, help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', default=False, help='fp16 infer')
    #================================================================
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    model = load_model_new()

    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid=[torch.zeros(1)] * 3 # nl=3 number of detection layers
    model.model[-1].export_cat = True

    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
    y = model(img)  # dry run
    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = "D:/project/python/study/yolov5face2/0516/yolov5-face/model/yolov5_0.5_new111.onnx"

    input_names=['input']
    output_names=['output']
    torch.onnx.export(model, img, f, verbose=True, opset_version=12,
        input_names=input_names,
        output_names=output_names)

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    # https://github.com/daquexian/onnx-simplifier
    if opt.simplify:
        try:
            import onnxsim
            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            onnx_model, check = onnxsim.simplify(onnx_model,
                dynamic_input_shape=opt.dynamic,
                input_shapes={'input': list(img.shape)} if opt.dynamic else None)
            assert check, "simplify check failed "
            onnx.save(onnx_model, f)
        except Exception as e:
            print(f"simplifer failure: {e}")

    print('ONNX export success, saved as %s' % f)
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))

    # onnx infer
    if opt.onnx_infer:
        import onnxruntime
        import numpy as np
        providers =  ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(f, providers=providers)
        im = img.cpu().numpy().astype(np.float32) # torch to numpy
        y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
        print("pred's shape is ",y_onnx.shape)
        print("max(|torch_pred - onnx_pred|） =",abs(y.cpu().detach().numpy()-y_onnx).max())


