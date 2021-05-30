#importingthe required libraries
import cv2
import face_recognition

#loading the image to detect
image_to_detect = cv2.imread('images/msk and rahul.jpg')
cv2.imshow("test",image_to_detect)
#detect all faces in the image
all_face_locations = face_recognition.face_locations(image_to_detect,model='hog') 
#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))
#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #splitting the tuple to get the four positional values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
   #slicing the current face from the main iamge
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    #showing the current face with dynamic title
    cv2.imshow("Face No."+str(index+1),current_face_image)
    