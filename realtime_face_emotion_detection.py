# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:00:15 2021

@author: mdskh
"""

import cv2
import face_recognition  
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
  

webcam_video_stream = cv2.VideoCapture(0)
#LOAD THE MODEL AND LOAD  THE WEIGHTS
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#ldeclare the emotion labels
emotions_label = ('angry','disgust','fear','happy','sad','surprise','neutral')


#initialize the array variable to hold all face locations in the frame
all_face_locations = []

while True:
    #get the current frame from the video stream as an imageq
    ret,current_frame = webcam_video_stream.read()
 #resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    #arguments are image ,no_of_times_to_upsample=2
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get the four positional values
       top_pos,right_pos,bottom_pos,left_pos = current_face_location
       #change the position magnitude to fit the actual size video frame
       top_pos=top_pos*4
       right_pos=right_pos*4
       bottom_pos=bottom_pos*4
       left_pos=left_pos*4
       print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
       #extract the face from the frame , blur it and then paste it back to the frame
       #slicing the current face from the main iamge
       current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
       
       
       #draw rectangle around the face detected
       cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)  
       #preprocess inpur,convert it into an image like as the data in the dataset
       #convert greyscale
       current_face_image = cv2.cvtcolor(current_face_image,cv2.COLOR_BG2GRAY)
       #Resize to 48x48 px size
       current_face_image = cv2.resize(current_face_image,(48,48))
       #convert the PIL image into  a 3d numpy array
       img_pixels = image.img_to_array(current_face_image)
       #expand the shape of an array into single row multiple columns
       img_pixels = np.expand_dims(img_pixels,axis = 0)
       #pixels are in range of [0,255]. normalize all pixels in the range of [0,1] 
       img_pixels /= 255
       
       #do prediction using model, get thge prediction values for all 7 expressions
       exp_predictions =face_exp_model.predict(img_pixels)
       #find the max  indexed predictionn  value (0 till 7)
       max_index =  np.argmax(exp_predictions[0])
       #get corresponding lable from emotion label
       emoiton_label = emotions_label[max_index]
       #display the name as text in the image
       font = cv2.FONT.HERSHEY_DUPLEX
       cv2.putText(current_frame,emotions_label,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
       #showing the current face with rectangle 
    cv2.imshow("webcam video",current_frame)
       #press 'q' on keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #release the stream and cam
    #close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()   
       
                                                         
    