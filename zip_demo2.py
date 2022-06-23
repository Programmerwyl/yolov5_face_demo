import torch
from models.experimental import attempt_load
import numpy as np
from models.yolo import  Model

# def load_model(weights, device):
#     model = torch.load(weights, map_location=device)['model'].eval()
#     return model

device = "cuda"
# ckpt = torch.load('./model/yolov5n-0.5.pt',device)
# model = Model(cfg="./models/yolov5n-0.5.yaml",ch=3,nc=1).cuda()
# model.load_state_dict(ckpt['model'].float().state_dict())


# model = torch.load('./model/yolov5n-0.5.pt', map_location=device)['model'].float().fuse().eval()
# model = torch.load('./model/yolov5n-0.5.pt', map_location=device)['model'].float().eval()

save_path = './weights/yolov5_05_net.pt'
model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
model.load_state_dict(torch.load(save_path))

input_data = torch.zeros(1, 3,320, 320).to(device)
torchscript_model=torch.jit.trace(model,input_data)
torchscript_model.save("yolov5n-0.5.zip")