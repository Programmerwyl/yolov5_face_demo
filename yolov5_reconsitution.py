import torch
from models.yolo_demo import  Model


save_path = './weights/yolov5_05_net.pt'
model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
model.load_state_dict(torch.load(save_path))

print(" load success ")
# for k, m in model.named_modules():
#     print(" k ",k)
#     print(" m ",m)