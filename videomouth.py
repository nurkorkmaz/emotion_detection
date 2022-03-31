import cv2
import slope
import perclos
import time
import vlc
import os
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
def smile(mouth):
        A = dist.euclidean(mouth[3], mouth[9])
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[4], mouth[8])
        avg = (A+B+C)/3
        D = dist.euclidean(mouth[0], mouth[6])
        mar=avg/D
        return mar
    
    
counter = 0
selfie_no = 0
    
shape_predictor= "shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
    
    
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
vs = cv2.VideoCapture(0)
    
while True:
    ret, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth= shape[mStart:mEnd]
        mar= smile(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
           
    
    
    
        cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        
    cv2.imshow('Live Capture', frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    
vs.release()
cv2.destroyAllWindows()