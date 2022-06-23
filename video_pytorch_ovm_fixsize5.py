import torch
import cv2
import numpy as np
from utils.general import check_img_size, non_max_suppression_face,scale_coords,scale_coords_landmarks,img_process,non_max_suppression
from models.experimental import attempt_load
import time


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

if __name__ == '__main__':

    video_path ="E:/project/python/face/yolov5-face_simply/model/mouth.mp4"
    cap = cv2.VideoCapture(video_path)
    # save_path = './weights/yolov5_05_net_new1.pt'
    # model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
    # model.load_state_dict(torch.load(save_path))

    # model = load_model("D:/project/python/study/yolov5face2/0516/yolov5-face/runs/train/exp34/weights/best.pt", "cuda")
    # model = load_model("E:/project/python/face/yolov5-face_simply/model/best.pt", "cuda")
    model = load_model("E:/project/python/face/yolov5-face_simply/runs/train/exp14/weights/best.pt", "cuda")
    # model = load_model("E:/project/python/face/yolov5-face_simply/model/yolov5n-0.5.pt", "cuda")

    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid=[torch.zeros(1)] * 3 # nl=3 number of detection layers

    # print(" model ",model)

    while True:

        ret, frame = cap.read()
        print(" ret ",ret)
        if not ret:
            break
        t1 = time.time()
        show_frame = np.copy(frame)
        img = img_process(frame, (320, 320)).to("cuda")
        # print(" img ",img.shape)
        # pred = model(img)[0]
        pred = model(img)[0]

        pred = np.squeeze(pred)

        # pred = np.expand_dims(pred,axis=0)
        pred = torch.unsqueeze(pred,dim=0)
        print(" pred ", pred.shape)
        # (1,6300,16)
        faces = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5)
        for i, det in enumerate(faces):  # detections per imag
            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                # aligned_arr = np.zeros(shape=(det.size()[0], 3, 112, 112))
                # Rescale boxes from img_size to im0 size
                for j in range(det.size()[0]):
                    # print("  j  ", j)
                    # if det[j, 4].cpu().numpy() < 0.6:
                    #     continue
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    # landmarks = (det[j, 5:15].view(1, 10)).view(-1).tolist()
                    cv2.rectangle(show_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255),
                                  thickness=2)
                    # for i in range(5):
                    #     cv2.circle(show_frame, (int(landmarks[2 * i]), int(landmarks[2 * i + 1])), 3, (0, 255, 0),
                    #                thickness=2)
                    cv2.putText(show_frame, "conf " + str(conf), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 50, 255), thickness=2)

        t2 = time.time()
        print(" time ",(t2-t1))
        cv2.imshow('face', show_frame)
        cv2.waitKey(3)

    cap.release()
    cv2.destroyAllWindows()
