#!/usr/bin/env python
# code from https://markhedleyjones.com/projects/calibration-checkerboard-collection
import os
import cv2
import glob
import pickle
import numpy as np
from datetime import datetime


def get_chessboard_mapping(img_fps, CHECKERBOARD, BLOCK_SIZE, imgs=None, imshow=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points and 2D points for each checkerboard image
    objpoints = []
    imgpoints = [] 

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= BLOCK_SIZE

    # Extracting path of individual image stored in a given directory
    if imgs is None:
        imgs = [cv2.imread(img_fp) for img_fp in img_fps]

    for img_fp, img in zip(img_fps, imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(img_fp, ret)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates (into float type not int) for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            if imshow:
              img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        if imshow:
            h, w = img.shape[:2]
            cv2.imshow('img', cv2.resize(img, (int(w/6), int(h/6))))
            cv2.waitKey(0)
    if imshow:
        cv2.destroyAllWindows()

    return np.array(objpoints), np.array(imgpoints)

def find_intrinsic_params(objpoints, imgpoints, img_shape):
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)

    cameraMatrix	Intrinsic camera matrix
    distCoeffs	Lens distortion coefficients. These coefficients will be explained in a future post.
    rvecs	Rotation specified as a 3×1 vector. The direction of the vector specifies the axis of rotation and the magnitude of the vector specifies the angle of rotation.
    tvecs	3×1 Translation vector.

    Camera Matrix: https://answers.opencv.org/question/89786/understanding-the-camera-matrix/
    distortion Coefficients: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    mtx = [[fx,  0, cx], 
           [ 0, fy, cy],
           [ 0,  0,  1]]
    fx, fy, cx, cy: is in pixel unit
    Distortion Coefficients: k1, k2, p1, p2, k3
    """
    h, w = img_shape[:2]
    ret, mtx_ori, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    mtx, roi = cv2.getOptimalNewCameraMatrix(mtx_ori, dist, (w, h), 1, (w, h), True)
    return ret, mtx_ori, dist, np.array(rvecs), np.array(tvecs), mtx, roi

def undistort(img, mtx_ori, dist, mtx):
    img_cal = cv2.undistort(img, mtx_ori, dist, None, mtx)
    return img_cal

def plot_3d_eval(rvecs, tvecs, mtx, objpoints, imgpoints, width, height, ax=None, scale_img=50, title='AeroTriangulation of All Calibraion Images'):
    """
    create your own ax:
        ```
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ```
    """
    from aero_triangulation import get_line_vecs, get_R, convert_imxyzs_to_npidxs, plot_aereo_triangulation_3d
    # prepare the data
    PXYZses = np.squeeze(objpoints)[0] # shape=(70, 3), use only one chessboard
    npidxs_imgs = [convert_imxyzs_to_npidxs(ps, mtx)[1] for ps in np.squeeze(imgpoints)] # shape=(n_img, 70, 2)
    plot_aereo_triangulation_3d(rvecs, tvecs, mtx, PXYZses, width, height)

if __name__ == '__main__':
    # Lucid camera spec>: https://thinklucid.com/product/triton-2-8-mp-imx429/
    # Len spec: https://vst.co.jp/zh-hant/machine-vision-lenses-zh-hant/sv-h-series/
    WIDTH, HEIGHT = 710, 710 # 2840, 2840
    PIXEL_SIZE = 0.00274 * 4 # mm (µm/1000)
    CHECKERBOARD = (7, 10)
    BLOCK_SIZE = 34 # mm
    pkl_dir = 'pkls'
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    img_dir = 'chessboard'
    img_fps = glob.glob(img_dir+r'\*.png')
    objpoints, imgpoints = get_chessboard_mapping(img_fps, CHECKERBOARD, BLOCK_SIZE, imshow=False)
    ret, mtx_ori, dist, rvecs, tvecs, mtx, roi = find_intrinsic_params(objpoints, imgpoints, (HEIGHT, WIDTH))

    pkl_fp = os.path.join(pkl_dir, os.path.basename(os.path.dirname(img_dir)) + "_" + os.path.basename(img_dir) + ".pkl")
    with open(pkl_fp, 'wb') as f:
        pickle.dump((ret, mtx_ori, dist, rvecs, tvecs, mtx, roi), f)
    
    fx, fy = mtx_ori[0][0] * PIXEL_SIZE, mtx_ori[1][1] * PIXEL_SIZE
    w_mv, h_mv = mtx_ori[0][2] - (WIDTH/2), mtx_ori[1][2] - (HEIGHT/2)
    print('uncalibrated')
    print("fx, fy", fx, fy)
    print("w_mv, h_mv", w_mv, h_mv)

    fx, fy = mtx[0][0] * PIXEL_SIZE, mtx[1][1] * PIXEL_SIZE
    w_mv, h_mv = mtx[0][2] - (WIDTH/2), mtx[1][2] - (HEIGHT/2)
    print('calibrated')
    print("fx, fy", fx, fy)
    print("w_mv, h_mv", w_mv, h_mv)


    
