import torch
from models.experimental import attempt_load
import numpy as np

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"
model = load_model("./model/yolov5n-face.pt", device).eval()

# model = model.Long()
# input_data = torch.rand(1, 3,320, 320).float().to(device)
# input_data = torch.zeros(1, 3,320, 320).float().to(device)
input_data = torch.zeros(1, 3,320, 320,dtype=torch.float).to(device)
torchscript_model=torch.jit.trace(model,input_data)
torchscript_model.save("model.zip")