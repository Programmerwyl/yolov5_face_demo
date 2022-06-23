import torch
# from models.yolo import  Model
from models.yolo_demo import  Model

save_path = './weights/yolov5_05_net_new1.pt'


def save_model():
    device = "cuda"
    # model = torch.load('./model/yolov5n-0.5.pt', map_location=device)['model'].float().fuse().eval()
    model = torch.load('D:/project/python/study/yolov5face2/0516/yolov5-face/runs/train/exp34/weights/best.pt', map_location=device)
    torch.save(model['model'].state_dict(), save_path)


# save_model()


def load_model():
    model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
    model.load_state_dict(torch.load(save_path))

    # torch.save(model,'./model/yolov5n-0.52.pt')
    model.float().fuse().eval()


    # ckpt = torch.load(save_path, "cuda")
    # model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
    # model.load_state_dict(ckpt['model'].float().state_dict())



save_model()
print(" save success  ")
load_model()
print(" load success")