import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
from PIL import Image

def visualize(img1, img2, title1, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=30)

def distortionCorrection(img, K, D):
    undistImg = cv2.undistort(img, K, D, None, K)
    
    return undistImg

def calibAndUndist(testImg, objpoints, imgpoints):
    # Test undistortion on an image
    
    img_size = (testImg.shape[1], testImg.shape[0])

    # Do camera calibration given object points and image points
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = distortionCorrection(testImg, K, D)

    
    return K, D, dst

def generateImgPoints(cbSize, imgNamePattern, squareSize=0.03):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbSize[0]*cbSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1,2)
    objp = objp*squareSize
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = sorted(glob.glob(imgNamePattern))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, cbSize, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

        else:
            print('Image {} rejected.'.format(fname))

    return objpoints, imgpoints

imgPath = './camera_cal/'
imgNamePat = 'calibration*'
imgExt = '.jpg'
chessBoardCorners = (9,6)
objpoints, imgpoints = generateImgPoints(chessBoardCorners, imgPath+imgNamePat+imgExt)
K = None
D = None

img = np.array(Image.open(imgPath+'calibration1.jpg'))

K, D, undistImg = calibAndUndist(img, objpoints, imgpoints)
mpimg.imsave('output_images/'+'undist_checkerboard.jpg', undistImg)

# Visualize undistortion
visualize(img, undistImg, 'Original Image', 'Undistorted Image')
dist_pickle = {}
dist_pickle['K'] = K
dist_pickle['D'] = D
pickle.dump( dist_pickle, open(imgPath+'intrinsics.p', "wb" ) )
print(K)
print(D)
