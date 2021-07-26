import cv2
import numpy as np
import pickle
cap = cv2.VideoCapture('final.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):  
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('final', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break
 
cap.release()
 
cv2.destroyAllWindows()
