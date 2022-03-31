import cv2
import slope
import perclos
import time
import vlc
import os
import numpy as np
import cv2 as cv

pCLOS = perclos.perclos()
frame=cv2.imread("photo6.png")

  
       
pCLOS.load_img(frame)
ear = pCLOS.calc_ear()
cv2.putText(frame, "EAR: {:.2f}".format(ear), (frame.shape[1]-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
  
if ear >= 0.215 and ear<=0.235 :
    cv2.putText(frame, "Happy", (frame.shape[1]-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
if ear >= 0.17 and ear<=0.21 :
    cv2.putText(frame, "Anger", (frame.shape[1]-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
if ear >= 0.335 and ear<=0.41 :
    cv2.putText(frame, "Surprise", (frame.shape[1]-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
if ear >= 0.235 and ear<=0.32 :
    cv2.putText(frame, "Sadness", (frame.shape[1]-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           

if type(pCLOS.leftEye) is not int or type(pCLOS.rightEye) is not int:
    leftEyeHull = cv2.convexHull(pCLOS.leftEye)
    rightEyeHull = cv2.convexHull(pCLOS.rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#Show Frame
cv2.imshow("Frame", frame)
cv.imwrite("output.png", frame)
   

   
    




