import cv2
import onnxruntime
import numpy as np
from utils.general import img_process

f1 = "D:/project/python/study/yolov5face2/0516/yolov5-face/runs/train/exp34/weights/best_111.onnx"
f2 = "D:/project/python/study/yolov5face2/0516/yolov5-face/runs/train/exp34/weights/best_error.onnx"

providers =  ['CPUExecutionProvider']


def get_onnx_out(f,im):
    session = onnxruntime.InferenceSession(f, providers=providers)
    y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
    return y_onnx


img_path = "F:/temp/wilder_face_data/2_Demonstration_Demonstration_Or_Protest_2_2.jpg"
img = cv2.imread(img_path)

# img = np.expand_dims(img,axis=0)

img = img_process(img, (320, 320)).to("cpu")

im = img.cpu().numpy().astype(np.float32) # torch to numpy

onnx1 = get_onnx_out(f1,im)
onnx2 = get_onnx_out(f2,im)

onnx1 = np.squeeze(onnx1)
onnx2 = np.squeeze(onnx2)


diff = onnx1-onnx2


print(" onnx1 ",onnx1.shape)
print(" onnx2 ",onnx2.shape)

print(" diff  ",np.sum(diff))