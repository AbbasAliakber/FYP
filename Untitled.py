#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import cv2 
import os
import matplotlib.pyplot as plt 
import numpy as np 
import socket 
import random 
from time import sleep 


def sendDataToUnity(data):
    s= socket.socket()
    s.connect(("192.168.8.102",1755))
    s.send((str (data)).encode())
    s.close()

img_array = cv2.imread("M:/IIT stuff/Final Year Research/Fortensorflowmodel/Training/0/Training_3908.jpg")

img_array.shape #rgb

plt.imshow(img_array) #bgr

Datadirectory = "M:/IIT stuff/Final Year Research\Fortensorflowmodel/Training" ##training dataset

Classes = ["0","1","2","3","4","5","6"] ##list of classes 

for category in Classes :
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show
        break
    break 

img_size = 224
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()


new_array.shape

training_Data = [] ## data array 

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory,category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e :
                pass


create_training_Data()

print(len(training_Data))

temp = np.array(training_Data)

temp.shape

import random 

random.shuffle(training_Data)

x = []
y = []

for features,label in training_Data:
    x.append(features)
    y.append(label)
    
x = np.array(x).reshape(-1,img_size,img_size, 3) 
x.shape
x = x/255.0 
y[2001]
Y = np.array(y)
Y.shape


# #deep learning model for training - transfer learning 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.applications.MobileNetV2()
##model.summary()
# #transfer learning - Tuning,weights will start from last check point 

base_input = model.layers[0].input
base_output = model.layers[-2].output
base_output

final_output = layers.Dense(128)(base_output)
final_output = layers.Activation("relu")(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation("relu")(final_output)
final_output = layers.Dense(7,activation="softmax")(final_output)

final_output

new_model = keras.Model(inputs = base_input, outputs = final_output)


##new_model.summary()

##new_model.compile(loss="sparse_categorical_crossentropy",optimizer = "adam", metrics =["accuracy"])

##new_model.fit(x,Y,epochs = 30)

##new_model.save("final_model_95p07.h5")

new_model = tf.keras.models.load_model("final_model_95p07.h5",compile=False)
new_model.compile(loss="sparse_categorical_crossentropy",optimizer = "adam", metrics =["accuracy"])

frame = cv2.imread("M:/IIT stuff/Final Year Research/Fortensorflowmodel/happyboy.jpg")

frame.shape

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier ("haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gray.shape

faces = faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)
    facess= faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print ("face not detected")
    else:
        for (ex,ey,ew,eh) in facess :
            face_roi = roi_color[ey: ey+eh, ex: ex+ew]


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

plt.imshow(cv2.cvtColor(face_roi,cv2.COLOR_BGR2RGB))

final_image = cv2.resize(face_roi, (224,224))
final_image = np.expand_dims(final_image,axis=0)
final_image = final_image/255.0


predictions = new_model.predict(final_image)


predictions[0]


np.argmax(predictions)

print("Play game on unity")

import cv2
import numpy as np 
path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

rectangle_bgr = (255,255,255)
img = np.zeros((500,500))
text = "Some text in a box"
(text_width,text_height)= cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25

box_coords = ((text_offset_x, text_offset_y),(text_offset_x + text_width + 2, text_offset_y - text_height -2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale= font_scale, color=(0,0,0),thickness=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True : 
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)
        facess= faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print ("face not detected")
        else:
            for (ex,ey,ew,eh) in facess :
                face_roi = roi_color[ey: ey+eh, ex: ex+ew] ##cropping the face 
            
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image,axis=0)
    final_image = final_image/255.0
    
    font= cv2.FONT_HERSHEY_SIMPLEX
    
    predictions = new_model.predict(final_image)
    
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    
    if (np.argmax(predictions)==0):
        status = "Angry"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
        sendDataToUnity(np.argmax(predictions))

    elif (np.argmax(predictions)==1):
        status = "Disgust"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.PutText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        sendDataToUnity(np.argmax(predictions))

    elif (np.argmax(predictions)==2):
        status = "Fear"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        sendDataToUnity(status)

    elif (np.argmax(predictions)==3):
        status = "Happy"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        sendDataToUnity(status)

    elif (np.argmax(predictions)==4):
        status = "Neutral"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        sendDataToUnity(status)

    elif (np.argmax(predictions)==5):
        status = "Sad"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        
        sendDataToUnity(status)
    else:
        status = "Surprise"
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        
        sendDataToUnity(status)

    cv2.imshow("face Emotion Recogntion",frame)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
