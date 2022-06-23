import torch
import cv2
import numpy as np
from utils.general import check_img_size, non_max_suppression_face,scale_coords,scale_coords_landmarks,img_process
from models.experimental import attempt_load
import time
from models.yolo import  Model

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_model("./model/yolov5n-0.5.pt", device)

    device = "cuda"
    # ckpt = torch.load('./model/yolov5n-0.5.pt', device)
    # model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
    # model.load_state_dict(ckpt['model'].float().state_dict())

    # model = torch.load('./model/yolov5n-0.5.pt', device)
    # model = torch.load('./model/yolov5n-0.5.pt', map_location=device)['model'].float().fuse().eval()
    save_path = './weights/yolov5_05_net.pt'
    model = Model(cfg="./models/yolov5n-0.5.yaml", ch=3, nc=1).cuda()
    model.load_state_dict(torch.load(save_path))
    model = model.float().fuse().eval()


    while True:

        ret, frame = cap.read()
        show_frame = np.copy(frame)
        if not ret:
            break
        t1 = time.time()
        img = img_process(frame, (512, 960)).to("cuda")
        print(" img ",img.shape)
        pred = model(img)[0]

        faces = non_max_suppression_face(pred, conf_thres=0.3, iou_thres=0.5)
        for i, det in enumerate(faces):  # detections per imag
            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                aligned_arr = np.zeros(shape=(det.size()[0], 3, 112, 112))
                # Rescale boxes from img_size to im0 size
                for j in range(det.size()[0]):
                    # print("  j  ", j)
                    # if det[j, 4].cpu().numpy() < 0.6:
                    #     continue
                    xyxy = det[j, :4].view(-1).tolist()
                    # conf = det[j, 4].cpu().numpy()
                    conf = det[j, 4].cpu().detach().numpy()

                    landmarks = (det[j, 5:15].view(1, 10)).view(-1).tolist()
                    cv2.rectangle(show_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255),
                                  thickness=2)
                    for i in range(5):
                        cv2.circle(show_frame, (int(landmarks[2 * i]), int(landmarks[2 * i + 1])), 3, (0, 255, 0),
                                   thickness=2)
                    cv2.putText(show_frame, "conf " + str(conf), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 50, 255), thickness=2)

        t2 = time.time()
        print(" time ",(t2-t1))
        cv2.imshow('face', show_frame)
        cv2.waitKey(3)

    cap.release()
    cv2.destroyAllWindows()
