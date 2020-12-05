import keras
from keras.preprocessing import image
import dlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
detector=dlib.get_frontal_face_detector()
img=image.load_img('D:/Data science/dlib starting/chachi.jpg')
img=image.img_to_array(img)
img=img.astype('uint8')
faces=detector(img)
model=keras.models.load_model('D:/Data science/dlib starting/new_model.h5')
for face in faces:
    x1,y1=face.left(),face.top()
    x2,y2=face.right(),face.bottom()
    new_img=img[y1-30:y2+10,x1-30:x2+10]
    new_img1=image.smart_resize(new_img,(150,150))/255
    # plt.imshow(new_img/255)
    # plt.show()
    # plt.imshow(new_img1/255)
    # plt.show()
    # print(new_img.shape)
    # print(new_img1.shape)
    print(new_img1.shape)
    print(model.predict_classes(new_img1.reshape(1,150,150,3))[0])

