import cv2
import slope
import perclos
import time
import vlc
import os
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils

def main():
    
    pCLOS = perclos.perclos()
    
    vCap = cv2.VideoCapture(0)
   
    while True:
        ret, frame = vCap.read()
        frame = cv2.resize(frame, (640,480))
        if ret == True:
            
            pCLOS.load_img(frame)
            ear = pCLOS.calc_ear()
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if type(pCLOS.leftEye) is not int or type(pCLOS.rightEye) is not int:
                leftEyeHull = cv2.convexHull(pCLOS.leftEye)
                rightEyeHull = cv2.convexHull(pCLOS.rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Show Frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vCap.release()


if __name__ == '__main__':
    main()