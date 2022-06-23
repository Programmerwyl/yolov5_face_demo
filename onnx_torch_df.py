import torch
import onnxruntime
import numpy as np
import models
import torch.nn as nn
# from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from models.experimental import attempt_load

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

f = "./model/yolov5n-0.5_orinxx.onnx"

model = attempt_load('./model/yolov5n-0.5.pt', map_location=torch.device('cpu'))  # load FP32 model
delattr(model.model[-1], 'anchor_grid')
model.model[-1].anchor_grid = [torch.zeros(1)] * 3  # nl=3 number of detection layers
model.model[-1].export_cat = True
model.eval()
labels = model.names

# opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

# Input
# img = torch.randn(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
# img = torch.ones(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

img = torch.ones(1, 3,320,320)  # image size(1,3,320,192) iDetection


print("  img.shape  ", img.shape)
# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
        if isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
        elif isinstance(m.act, nn.SiLU):
            m.act = SiLU()
    # elif isinstance(m, models.yolo.Detect):
    #     m.forward = m.forward_export  # assign forward (optional)
    if isinstance(m, models.common.ShuffleV2Block):  # shufflenet block nn.SiLU
        for i in range(len(m.branch1)):
            if isinstance(m.branch1[i], nn.SiLU):
                m.branch1[i] = SiLU()
        for i in range(len(m.branch2)):
            if isinstance(m.branch2[i], nn.SiLU):
                m.branch2[i] = SiLU()
y = model(img)  # dry run




# img = torch.ones(1, 3,320,320)  # image size(1,3,320,192) iDetection
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = load_model("./model/yolov5n-0.5.pt", device)
# # model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
# delattr(model.model[-1], 'anchor_grid')
# model.model[-1].anchor_grid = [torch.zeros(1)] * 3  # nl=3 number of detection layers
# model.model[-1].export_cat = True
# model.eval()
#
# for k, m in model.named_modules():
#     m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
#     if isinstance(m, models.common.Conv):  # assign export-friendly activations
#         if isinstance(m.act, nn.Hardswish):
#             m.act = Hardswish()
#         elif isinstance(m.act, nn.SiLU):
#             m.act = SiLU()
#     # elif isinstance(m, models.yolo.Detect):
#     #     m.forward = m.forward_export  # assign forward (optional)
#     if isinstance(m, models.common.ShuffleV2Block):  # shufflenet block nn.SiLU
#         for i in range(len(m.branch1)):
#             if isinstance(m.branch1[i], nn.SiLU):
#                 m.branch1[i] = SiLU()
#         for i in range(len(m.branch2)):
#             if isinstance(m.branch2[i], nn.SiLU):
#                 m.branch2[i] = SiLU()
#
#
# y = model(img.to(device))



providers =  ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(f, providers=providers)
im = img.cpu().numpy().astype(np.float32) # torch to numpy
y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
print("pred's shape is ",y_onnx.shape)
print("max(|torch_pred - onnx_pred|） =",abs(y.cpu().numpy()-y_onnx).max())