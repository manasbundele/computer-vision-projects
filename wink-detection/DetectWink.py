'''
This is a python implementation for Wink Detection in both images and live video using Haar Cascades.

By: Manas Bundele
'''
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def doOverlap(l1,r1, l2, r2):
    if (l1[0] > r2[0] or l2[0] > r1[0]):
        return False 
    
    if (l1[1] > r2[1] or l2[1] > r1[1]):
        return False
  
    return True

def detectWink(frame, location, ROI, cascade):
    (h, w) = ROI.shape[:2]

    if h > 600:
        ROI = ROI[0:int(h*0.5),0:w]
    elif h < 150:
        ROI = ROI[0:int(h*0.6),0:w]
    else:
        ROI = ROI[0:int(h*0.56),0:w]

    eyes = cascade.detectMultiScale(
        ROI, 1.225, 5, 0|cv2.CASCADE_SCALE_IMAGE, (0, 0)) 

    eye_mod = []
    eye_mod_set = set([])
    for e in eyes:
        if len(eye_mod) == 0:
            eye_mod_set.add(tuple(e))
            eye_mod.append(e)
        else:
            for e1 in eye_mod:
                if e1.tolist() != e.tolist():
                    if doOverlap((e[0],e[1]),(e[0]+e[2],e[1]+e[3]),(e1[0],e1[1]),(e1[0]+e1[2],e1[1]+e1[3])) == False:
                        if tuple(e) not in eye_mod_set:
                            eye_mod.append(e)
                            eye_mod_set.add(tuple(e))
                    
    for e in eye_mod:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    return len(eye_mod) == 1    # number of eyes is one

def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    scaleFactor = 1.025 # range is from 1 to ..
    minNeighbors = 3   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (30,30) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)


    face_mod = []
    face_mod_set = set([])
    for f in faces:
        if len(face_mod) == 0:
            face_mod_set.add(tuple(f))
            face_mod.append(f)
        else:
            for f1 in face_mod:
                if f1.tolist() != f.tolist():
                    if doOverlap((f[0],f[1]),(f[0]+f[2],f[1]+f[3]),(f1[0],f1[1]),(f1[0]+f1[2],f1[1]+f1[3])) == False:
                        if tuple(f) not in face_mod_set:
                            face_mod.append(f)
                            face_mod_set.add(tuple(f))

    detected = 0
    for f in face_mod:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y+h, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()  

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)
