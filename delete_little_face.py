import os

dir_path = "G:/dataset/face/face_alignment/wilderface/yolov5-face/train_big"
list_dir= os.listdir(dir_path)

for file_name in list_dir:
    if file_name.endswith("txt"):
        save_data = []
        current_tex = os.path.join(dir_path,file_name)
        current_img = os.path.join(dir_path,file_name.replace("txt",'jpg'))

        # print("current_tex  ",current_tex)
        # print("current_img  ",current_img)
        with open(current_tex,encoding="utf-8") as f:
            datas = f.readlines()
            for line in datas:
                line_datas = line.split(" ")
                face_size = float(line_datas[5])*float(line_datas[4])
                if face_size> 0.01:
                    save_data.append(line)

            if len(save_data) >0:
                for line_save in save_data:
                    print("line_save ")
                    f.write(line_save)
                    f.write("\n")
                    # print("line ",type(line))
            # else:
            #     os.remove(current_img)
            #     os.remove(current_tex)

