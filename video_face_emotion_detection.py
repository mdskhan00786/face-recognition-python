# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:00:15 2021

@author: mdskh
"""

import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)
  
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
       #blur the sliced face and save it to the same array itself
       current_face_image = cv2.GaussianBlur(current_face_image, (99,99),30)
       #paste the blurred face into the actual face
       current_frame[top_pos:bottom_pos,left_pos:right_pos] =  current_face_image
       
       #draw rectangle around the face detected
       cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
       #showing the current face with rectangle 
    cv2.imshow("webcam video",current_frame)
       #press 'q' on keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #release the stream and cam
    #close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()   
       
                                                         
    