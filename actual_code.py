import cv2
import dlib
import numpy as np
import keras
from keras.preprocessing import image


model=keras.models.load_model('D:/Data science/dlib starting/new_model.h5')


def empty(a):
    pass



cv2.namedWindow("BGR COLOR")
cv2.resizeWindow("BGR COLOR",300,150)
cv2.createTrackbar("Blue","BGR COLOR",0,255,empty)
cv2.createTrackbar("Green","BGR COLOR",0,255,empty)
cv2.createTrackbar("Red","BGR COLOR",0,255,empty)




# cv2.namedWindow("QUALITY")
# cv2.resizeWindow("QUALITY",300,50)
cv2.createTrackbar("brightness","BGR COLOR",10,40,empty)

def create_box(img,points,b,g,r,scale=5):
    # b_box=cv2.boundingRect(points)
    # x,y,w,h=b_box
    # cropped_img=img[y:y+h,x:x+w]
    # cropped_img=cv2.resize(cropped_img,(0,0),None,scale,scale)
    # return cropped_img

    mask=np.zeros_like(img)
    # print(b,g,r)
    mask=cv2.fillPoly(mask,[points],(b,g,r))
    img=cv2.bitwise_and(img,mask)
    return img




detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cam=cv2.VideoCapture(0)
x=None
while True:
    ret,frame=cam.read()
    original_img=frame.copy()
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    b = cv2.getTrackbarPos("Blue", "BGR COLOR")
    g = cv2.getTrackbarPos("Green", "BGR COLOR")
    r = cv2.getTrackbarPos("Red", "BGR COLOR")
    for face in faces:

        x1,y1=face.left(),face.top()
        x2,y2=face.right(),face.bottom()
        new_img = frame[y1 - 30:y2 + 10, x1 - 30:x2 + 10]
        try:
            new_img1 = image.smart_resize(new_img, (150, 150)) / 255
            print("doing good")
            prediction=model.predict_classes(new_img1.reshape(1,150,150,3))
            prediction=prediction[0]
            print(prediction)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            landmarks = predictor(gray, face)
            if prediction==0:
                myPoints = []
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    # cv2.circle(frame, (x, y), 2, (0, 0, 255), cv2.FILLED)
                    # cv2.putText(frame, str(n), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
                    if n == 2:
                        x1, y1 = x, y
                    if n == 14:
                        x2, y2 = x, y
                    ra = (x2 - x1) // 8
                    c1 = (x1 + ra, y1)
                    c2 = (x2 - ra, y2)
                    myPoints.append([x, y])
                # print(myPoints)
                cheeks_img=np.zeros_like(frame)

                # b = cv2.getTrackbarPos("Blue", "BGR COLOR")
                # g = cv2.getTrackbarPos("Green", "BGR COLOR")
                # r = cv2.getTrackbarPos("Red", "BGR COLOR")
                # colour=((int(224+frame[c1[0],c1[1]][0])//2),int((217+frame[c1[0],c1[1]][1])//2),int((241+frame[c1[0],c1[1]][2])//2))
                # print(colour)
                cv2.circle(cheeks_img, c1, ra, (0,0,50),cv2.FILLED)
                cv2.circle(cheeks_img, c2, ra, (0,0,50), cv2.FILLED)
                cheeks_img = cv2.GaussianBlur(cheeks_img, (3, 3), 500)
                myPoints=np.array(myPoints)
                lips=create_box(frame,myPoints[48:61],b,g,r)
                lips = cv2.GaussianBlur(lips, (7, 7), 5)
                frame = cv2.addWeighted(frame, 1, lips, 0.4, 0)
                frame = cv2.addWeighted(frame, 1, cheeks_img, 0.4, 0)
        except:
            pass
    l=cv2.getTrackbarPos("brightness", "BGR COLOR")
    l=l/10
    cv2.imshow("modified image",(frame/255)*l)
    cv2.imshow("original image",original_img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        x=frame
        break
# print(x)
cam.release()
cv2.destroyAllWindows()