#Amal Nazeem
#Testing the basic face detection algorithm
#Lots of help taken from https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/

#also from the openCV tutorials themselves http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
#great resource for explaining haar cascades and Viola-Jones algorithm

import numpy as np
import cv2

#this link are where I got the haarcascades from
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#I decided to go ahead and use the smile one too to see if it would work

#This uses openCV to import record video from the webcam
camera = cv2.VideoCapture(0) #img = cv2.imread('sachin.jpg')


#This loop is where we find faces
while 1:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = img[y:y+h, x:x+w]

        #for eye detection within the face
        eyes = eye_cascade.detectMultiScale(face_roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

        #for smile detection
        smile =  smile_cascade.detectMultiScale(face_roi_gray)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(face_roi_color,(sx,sy),(sx+sw,sy+sh), (0,0,255),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()