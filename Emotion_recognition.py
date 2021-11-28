import cv2
import dlib
import numpy as np
from skimage import io
# 使用特徵提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib的68點模型，使用作者訓練好的特徵預測器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 圖片所在路徑
img = io.imread("all.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height  = int(img.shape[0]*1.5)
width  = int(img.shape[1]*1.5)
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# 特徵提取器的例項化
dets = detector(img, 1)

print("人臉數：", len(dets))
for k, d in enumerate(dets):
    print("------")
    print("第", k+1, "個人臉d的座標：",
    "left:", d.left(),
    "right:", d.right(),
    "top:", d.top(),
    "bottom:", d.bottom())
    
    shape = predictor(img, d)
    # 標出68個點的位置
    for i in range(68):
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1)

        
        face_width = d.right() - d.left()
        
        line_brow_x = []
        line_brow_y = []
                
        mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴咧开程度
        mouth_higth = (shape.part(66).y - shape.part(62).y) / face_width  # 嘴巴张开程度

        
        # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
        brow_sum = 0  # 高度之和
        frown_sum = 0  # 两边眉毛距离之和
        for j in range(17, 21):
            brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
            frown_sum += shape.part(j + 5).x - shape.part(j).x
            line_brow_x.append(shape.part(j).x)
            line_brow_y.append(shape.part(j).y)
        

        tempx = np.array(line_brow_x)
        tempy = np.array(line_brow_y)
        z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
        brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的
        
        brow_hight = (brow_sum / 10) / face_width  # 眉毛高度占比
        brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比
        
        # 眼睛睁开程度
        eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                   shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
        eye_hight = (eye_sum / 4) / face_width
        
        # 分情况讨论
        # 张嘴，可能是开心或者惊讶

        if round(mouth_higth >= 0.04):
            if eye_hight >= 0.06:
                cv2.putText(img, "amazing"+str(k+1), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, 4)
            elif brow_k <= 0.07:
                cv2.putText(img, "angry"+str(k+1), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, 4)
            else:
                cv2.putText(img, "happy"+str(k+1), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, 4)
        
        # 没有张嘴，可能是正常和生气
        else:
            if brow_k <= 0.15:
                cv2.putText(img, "angry"+str(k+1), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, 4)
            elif eye_hight < 0.05:
                cv2.putText(img, "sad"+str(k+1), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, 4)
            else:
                cv2.putText(img, "nature"+str(k+1), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, 4)
                
    
    print("mouth_higth",mouth_higth)
    print("brow_k",brow_k)
    print("eye_hight",eye_hight)

# 顯示一下處理的圖片，然後銷燬視窗
cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()