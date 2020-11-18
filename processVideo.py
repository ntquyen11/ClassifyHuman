import cv2
import numpy as np
import imutils
# import create_model
from LoadModel import predict

#read video 
cap=cv2.VideoCapture('classify.mp4')
count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
ret,frame=cap.read()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
background=gray
kernel=np.ones((5,5),np.uint8)

while (cap.isOpened()):
    ret,frame=cap.read()
    # convert RGB image to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    # frame difference
    foreward=cv2.absdiff(background,gray)
    ret,thresh=cv2.threshold(foreward,100,255,cv2.THRESH_BINARY)
   
    #find contour 
    d_thresh=cv2.dilate(thresh,kernel,iterations=1) #dilate image to apparent object
    contours,hierachy=cv2.findContours(d_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) <500:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        # Region of image
        ROI=gray[x:x+w,y:y+h]
        # Resize image to 64x64 image
        ROI_64=cv2.resize(ROI,(64,64))
        # Reshape image to corresponding image input model
        img=np.reshape(ROI_64,(1,64,64,1))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Predict ouput
        score=predict(img)
        if score==1:
            cv2.putText(frame,'Human',(x,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0))
        else:
            cv2.putText(frame,'Non Human',(x,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0))
    
    
    cv2.imshow('Ouput',frame)
    cv2.waitKey(0)
obj.release()
cv2.destroyAllWindows()