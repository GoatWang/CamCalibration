# code from: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import os
import cv2
import numpy as np

if __name__ == '__main__':
    # evaluate
    import glob
    import pickle
    from intrinsic import undistort, get_chessboard_mapping, find_intrinsic_params, plot_3d_eval
    WIDTH, HEIGHT = 710, 710 # 2840, 2840
    PIXEL_SIZE = 0.00274 * 4 # mm (µm/1000)
    CHECKERBOARD = (7, 10)
    BLOCK_SIZE = 34 # mm
    pkl_dir = 'pkls'
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    img_dir = 'chessboard'
    img_fps = glob.glob(img_dir+r'\*.png')
    pkl_fp = os.path.join(pkl_dir, os.path.basename(os.path.dirname(img_dir)) + "_" + os.path.basename(img_dir) + ".pkl")
    with open(pkl_fp, 'rb') as f:
        ret, mtx_ori, dist_src, rvecs, tvecs, mtx, roi = pickle.load(f)

    # from matplotlib import pyplot as plt
    # img_fp = np.random.choice(img_fps)
    # img_src = cv2.imread(img_fp)
    # img_cal = undistort(img_src, mtx_ori, dist_src, mtx)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(img_src)
    # ax2.imshow(img_cal)
    # plt.show()

    imgs = [undistort(cv2.imread(img_fp), mtx_ori, dist_src, mtx) for img_fp in img_fps]
    objpoints, imgpoints = get_chessboard_mapping(img_fps, CHECKERBOARD, BLOCK_SIZE, imgs=imgs, imshow=False)
    ret, mtx, dist, rvecs, tvecs, mtx_n, roi = find_intrinsic_params(objpoints, imgpoints, (HEIGHT, WIDTH))
    print("uncalibrated dist", dist_src)
    print('calibrated dist', dist)
    
    # plot 3d
    plot_3d_eval(rvecs, tvecs, mtx, objpoints, imgpoints, WIDTH, HEIGHT, ax=None, scale_img=50, title='AeroTriangulation of All Calibraion Images')
