import cv2
import numpy as np
import pickle
cap = cv2.VideoCapture('project_video.mp4')







if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  
  ret, frame = cap.read()
  if ret == True:
    src = np.float32([[535, 479], [739, 479], [1071,665], [235,670]])
    dst = np.float32([[230,0],[900,0],[900,720],[230,720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, frame.shape[1::-1], flags = cv2.INTER_LINEAR)
    (H,L,S) = cv2.split(cv2.cvtColor(warped, cv2.COLOR_RGB2HLS))
    #(H,L,S) = cv2.cvtColor(warpedImgColor, cv2.COLOR_RGB2HLS)
    #blur = cv2.GaussianBlur(S, (3, 3), 0)
    thresh1 = cv2.Canny(S,10,100)
    sobelx = cv2.Sobel(thresh1,cv2.CV_64F,1,0,ksize=1)
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(thresh1)
    s_binary[(S >= s_thresh_min) & (S<= s_thresh_max)] = 1
    combined_binary = np.zeros_like(sobelx)
    combined_binary[(s_binary == 1) | (sobelx == 1)] = 1
    #cv2.imshow('red',th3)
    #cv2.imshow('lol',  eh)
    cv2.imshow('v',combined_binary)
    cv2.imshow('a',warped)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break
 
cap.release()
 
cv2.destroyAllWindows()
