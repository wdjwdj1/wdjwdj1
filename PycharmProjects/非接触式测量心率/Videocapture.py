import cv2

def getfirstROI(ROI,target):

    classifier_face = cv2.CascadeClassifier("haarcascade_frontalface_alt0.xml")
    img = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)
    faceRects_face = classifier_face.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))

    if len(faceRects_face) > 0:
        # 检测到人脸
        # for faceRect_face in faceRects_face:
        x, y, w, h = faceRects_face[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 0)
        # 获取图像x起点,y起点,宽，高
        h1 = int(float(h / 1.5))
        # 截取人脸区域高度的一半位置，以精确识别眼睛的位置
        intx = int(x + w/6)
        inty = int(y/1.5)
        intw = int(w*4/6)

        target.append(intx)
        target.append(int(h1 + inty))
        target.append(intw)
        target.append(int(inty/2))
        return True

    else:
        return False


