# code from: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import os
import cv2
import numpy as np

if __name__ == '__main__':
    # evaluate
    import glob
    import pickle
    from intrinsic import undistort, get_chessboard_mapping, find_intrinsic_params
    
    WIDTH, HEIGHT = 2840, 2840
    PIXEL_SIZE = 0.00274 # mm (µm/1000)
    CHECKERBOARD = (7, 10)
    BLOCK_SIZE = 34 # mm

    pkl_dir = 'pkls'
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    serial_num_l = '212500080'
    img_dir = os.path.join('images', 'Stereo20210726', 'videocap_20210726170727') # 212500080
    pkl_fp_l = os.path.join(pkl_dir, os.path.basename(os.path.dirname(img_dir)) + "_" + os.path.basename(img_dir) + ".pkl")
    with open(pkl_fp_l, 'rb') as f:
        ret, mtx, dist_l, rvecs, tvecs, mtx_n_l, roi = pickle.load(f)

    serial_num_r = '212500081'
    img_dir = os.path.join('images', 'Stereo20210726', 'videocap_20210727152040') # 212500081
    pkl_fp_r = os.path.join(pkl_dir, os.path.basename(os.path.dirname(img_dir)) + "_" + os.path.basename(img_dir) + ".pkl")
    with open(pkl_fp_r, 'rb') as f:
        ret, mtx, dist_r, rvecs, tvecs, mtx_n_r, roi = pickle.load(f)

    img_dir = os.path.join('images', 'Stereo20210726', 'videocap_20210727152040') # 212500081
    img_fps = glob.glob(img_dir+r'\*.png')

    img_fps_l = [img_fp for img_fp in img_fps if serial_num_l in img_fp]
    img_fps_r = [img_fp for img_fp in img_fps if serial_num_r in img_fp]
    imgs_l = [undistort(cv2.imread(img_fp), mtx, dist_l, mtx_n_l) for img_fp in img_fps_l]
    imgs_r = [undistort(cv2.imread(img_fp), mtx, dist_r, mtx_n_r) for img_fp in img_fps_r]
    objpoints_l, imgpoints_l = get_chessboard_mapping(img_fps_l, CHECKERBOARD, BLOCK_SIZE, imgs=imgs_l, imshow=False)
    objpoints_r, imgpoints_r = get_chessboard_mapping(img_fps_r, CHECKERBOARD, BLOCK_SIZE, imgs=imgs_r, imshow=False)
    objpoints = objpoints_l + objpoints_r # objpoints_l.shape = [n_imgs, n_pnts, 3]
    imgpoints = imgpoints_l + imgpoints_r # imgpoints_r.shape = [n_imgs, n_pnts, 2]
    ret, mtx, dist, rvecs, tvecs, mtx_n, roi = find_intrinsic_params(objpoints, imgpoints, (HEIGHT, WIDTH))

    fx, fy = mtx[0][0] * PIXEL_SIZE, mtx[1][1] * PIXEL_SIZE
    w_mv, h_mv = mtx[0][2] - (WIDTH/2), mtx[1][2] - (HEIGHT/2)
    print("dist", dist)
    print("fx, fy", fx, fy)
    print("w_mv, h_mv", w_mv, h_mv)

    # rvecs
    # tvecs