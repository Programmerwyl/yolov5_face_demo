import onnxruntime
import numpy as np
import torch
import cv2
from utils.general import non_max_suppression_face



# f = "./model/yolov5n-0.5orin.onnx"
f = "./model/yolov5n-0.5_orinxx.onnx"

providers =  ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(f, providers=providers)

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(320,320))

    show_frame = np.copy(frame)
    frame = np.transpose(frame,(2,0,1))
    frame = np.expand_dims(frame,axis=0)
    if not ret:
        break
    img = torch.from_numpy(frame)
    im = img.cpu().numpy().astype(np.float32) # torch to numpy
    y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
    # y_onnx = np.squeeze(y_onnx)

    # faces = non_max_suppression_face(torch.from_numpy(y_onnx[0]), conf_thres=0.3, iou_thres=0.5)
    faces = non_max_suppression_face(torch.from_numpy(y_onnx), conf_thres=0.3, iou_thres=0.5)
    # print("pred's shape is ",y_onnx.shape)
    for i, det in enumerate(faces):  # detections per imag
        if len(det):
            # print(det)

            for j in range(det.size()[0]):
                # print("  j  ", j)
                # if det[j, 4].cpu().numpy() < 0.6 or det[j, 4].cpu().numpy() >1.0 or det[j,15].cpu().numpy() < 0.5 or det[j,15].cpu().numpy()> 1.0:
                if det[j, 4].cpu().numpy() < 0.6 or det[j, 4].cpu().numpy() >1.0:
                    continue
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10)).view(-1).tolist()
                cv2.rectangle(show_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255),
                              thickness=2)
                # for i in range(5):
                #     cv2.circle(show_frame, (int(landmarks[2 * i]), int(landmarks[2 * i + 1])), 3, (0, 255, 0),
                #                thickness=2)
                # cv2.putText(show_frame, "conf " + str(conf), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX,
                #             1.0,
                #             (0, 50, 255), thickness=2)

    cv2.imshow("show_frame",show_frame)
    cv2.waitKey(3)
