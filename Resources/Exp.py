import cv2
import numpy as np
import face_recognition

# imgElon = face_recognition.load_image_file('elon.png')
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

# # imgDhoni = face_recognition.load_image_file('dhoni.png')
# # imgDhoni = cv2.cvtColor(imgDhoni,cv2.COLOR_BGR2RGB)

imgVirat = face_recognition.load_image_file('virat.png')
imgVirat = cv2.cvtColor(imgVirat,cv2.COLOR_BGR2RGB)

imgViratTest = face_recognition.load_image_file('dhoni.png')
imgViratTest = cv2.cvtColor(imgViratTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgVirat)[0] # it peints four values (39, 163, 101, 100) top rigth bottom left
encodeVirat = face_recognition.face_encodings(imgVirat)[0]
print("This the first image",faceLoc)
cv2.rectangle(imgVirat,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest = face_recognition.face_locations(imgViratTest)[0] # it peints four values (39, 163, 101, 100) top rigth bottom left
encodeViratTest = face_recognition.face_encodings(imgViratTest)[0]
print("This is the Second image",faceLocTest)
cv2.rectangle(imgViratTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)

results = face_recognition.compare_faces([encodeVirat],encodeViratTest)
print(results," The encoding do not match")

# for the best match
faceDis = face_recognition.face_distance([encodeVirat],encodeViratTest)
print(faceDis," This is the faceDis")

cv2.putText(imgViratTest,f'{results} "Face matches"',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,255),1)




# cv2.imshow('elon',imgElon)
# cv2.imshow('Dhoni',imgDhoni)
cv2.imshow('Virat',imgVirat)
cv2.imshow('ViratTest',imgViratTest)
cv2.waitKey(10000) #// 10 seconds 


