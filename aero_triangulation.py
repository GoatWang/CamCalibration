import os
import cv2
import glob
import pickle
import numpy as np
from intrinsic import undistort, get_chessboard_mapping, find_intrinsic_params
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac

def get_R(rvec):
    R = cv2.Rodrigues(rvec)[0] # index 0 to avoid jacobian
    return R

def convert_imxyzs_to_npidxs(imxyzs, mtx):
    """
    convert coords starts from center and in mm unit into pixel npidxs.
    idxs: numpy indeices.
    mtx: camera matrix. fx, fy, cx, cy are in pixel unit.
            [[fx,  0, cx], 
            [ 0, fy, cy], 
            [ 0,  0,  1]]
    row_idxs, col_idxs = (vs, us) in openCV
    """
    xs, ys = imxyzs.T[0], imxyzs.T[1]
    fx, fy, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]
    col_idxs = (fx * xs) + cx
    row_idxs = (fy * ys) + cy
    uvs = np.array([col_idxs, row_idxs]).T
    npidxs = np.array([row_idxs, col_idxs]).T
    return uvs, npidxs

def convert_npidxs_to_imxys(npidxs, mtx):
    """
    convert npidxs to coords starts from center and in mm unit.
    idxs: numpy indeices.
    mtx: camera matrix. fx, fy, cx, cy are in pixel unit.
            [[fx,  0, cx], 
            [ 0, fy, cy], 
            [ 0,  0,  1]]
    row_idxs, col_idxs = (vs, us) in openCV
    """
    idxs_trans = np.transpose(npidxs)
    row_idxs, col_idxs = idxs_trans[0, :], idxs_trans[1, :] 
    fx, fy, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]
    xs = (col_idxs - cx) / fx
    ys = (row_idxs - cy) / fy
    imxys = np.stack([xs, ys]).T
    return imxys

def project_XYZs_to_npidxs(XYZs, rvec, tvec, mtx):
    RtXYZs = np.dot(get_R(rvec), XYZs.T) + tvec.reshape(3, 1) # shape = (3, n_points)
    ks = RtXYZs[2]
    imxyzs = ((1/ks) * RtXYZs).T # shape = (n_points, 3), divide RtXYZs by Z (multiply 1/k)
    uvs, npidxs = convert_imxyzs_to_npidxs(imxyzs, mtx)
    return uvs, npidxs, ks

def project_npidxs_to_XYZs(npidxs, rvec, tvec, mtx, ks):
    imxys = convert_npidxs_to_imxys(npidxs, mtx) # shape = (n_points, 2)
    imxyzs = np.hstack([imxys, np.ones([len(imxys), 1])]) # shape = (n_points, 3)
    RtXYZs = ks * imxyzs.T # shape = (3, n_points)
    XYZs = np.dot(get_R(rvec).T, (RtXYZs - tvec.reshape(3, 1))).T
    return XYZs

def get_line_vecs(npidxs, rvec, tvec, mtx):
    imxys = convert_npidxs_to_imxys(npidxs, mtx) # shape = (n_points, 2)
    imxyzs = np.hstack([imxys, np.ones([len(imxys), 1])]) # shape = (n_points, 3)
    vecs = np.dot(get_R(rvec).T, imxyzs.T).T
    spt = - np.dot(get_R(rvec).T, tvec.reshape(3, 1)).T
    return vecs, spt

def cal_dist(vec1, spt1, vec2, spt2):
    spt1, spt2 = -spt1, -spt2
    p1, q1, r1 = spt1[:, 0], spt1[:, 1], spt1[:, 2]
    a1, b1, c1 = vec1[:, 0], vec1[:, 1], vec1[:, 2]
    p2, q2, r2 = spt2[:, 0], spt2[:, 1], spt2[:, 2]
    a2, b2, c2 = vec2[:, 0], vec2[:, 1], vec2[:, 2]
    a3 = a1*a2+b1*b2+c1*c2
    b3 = -(a1*a1+b1*b1+c1*c1)
    c3 = a1*(p2-p1)+b1*(q2-q1)+c1*(r2-r1)
    a4 = a2*a2+b2*b2+c2*c2
    b4 = -a3
    c4 = a2 * (p2-p1)+b2*(q2-q1)+c2*(r2-r1)
    t = (c3 * b4 - c4 * b3) / (a3 * b4 - a4 * b3)
    s = (a3 * c4 - a4 * c3) / (a3 * b4 - a4 * b3)
    xp = a1 * s - p1
    yp = b1 * s - q1
    zp = c1 * s - r1
    xq = a2 * t - p2
    yq = b2 * t - q2
    zq = c2 * t - r2
    dists_ecu=((xp-xq)**2+(yp-yq)**2+(zp-zq)**2)**(1/2)
    dists_blk = np.abs(xp-xq)+np.abs(yp-yq)+np.abs(zp-zq)
    pxyz1 = np.transpose(np.stack([xp, yp, zp]))
    pxyz2 = np.transpose(np.stack([xq, yq, zq]))
    return pxyz1, pxyz2, dists_ecu, dists_blk
    
def get_PQ(aereo_params1, aereo_params2, kp_npidxs1, kp_npidxs2):
    """
    depth: in meter
    """
    rvec1, tvec1, mtx1, width, height = aereo_params1
    rvec2, tvec2, mtx2, width, height = aereo_params2
    vecs1, LXYZ1 = get_line_vecs(kp_npidxs1, rvec1, tvec1, mtx1)
    vecs2, LXYZ2 = get_line_vecs(kp_npidxs2, rvec2, tvec2, mtx2)
    spts1, spts2 = np.tile(LXYZ1, [len(vecs1), 1]), np.tile(LXYZ2, [len(vecs2), 1])
    PXYZs1, PXYZs2, dists_ecu, dists_blk = cal_dist(vecs1, spts1, vecs2, spts2)
    uvs1, Pnpidxs1, ks1 = project_XYZs_to_npidxs(PXYZs1, rvec1, tvec1, mtx1)
    uvs2, Pnpidxs2, ks2 = project_XYZs_to_npidxs(PXYZs2, rvec2, tvec2, mtx2)
    return LXYZ1, LXYZ2, PXYZs1, PXYZs2, dists_ecu, dists_blk, ks1, ks2

def plot_aereo_triangulation_3d(rvecs, tvecs, mtx, PXYZses, width, height, ax=None, scale_img=50, title='AeroTriangulation of All Calibraion Images'):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    no_ax = ax is None
    if no_ax:
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    # draw the planes of each image
    for rvec, tvec in zip(rvecs, tvecs):
        npidxs = [(0, 0), (0, width), (height, width), (height, 0)]
        Plan_XYZs = project_npidxs_to_XYZs(npidxs, rvec, tvec, mtx, ks=scale_img)
        XYZ = (Plan_XYZs[[0,1,3,2], :].reshape(2, 2, -1).transpose(2, 0, 1))
        ax.plot_surface(*XYZ, alpha=0.3)

        LXYZ = - np.dot(get_R(rvec).T, tvec.reshape(3, 1)).T[0]
        ax.scatter(LXYZ[0], LXYZ[1], LXYZ[2], s=1, color='red')

    # draw the chessboard
    for idx, p in enumerate(PXYZses):
        c = np.random.rand(3)
        ax.scatter([p[0]], [p[1]], [p[2]], s=1, color=c)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if no_ax:
        plt.show()

if __name__  == '__main__':
    WIDTH, HEIGHT = 2840, 2840
    PIXEL_SIZE = 0.00274 # mm (µm/1000)
    CHECKERBOARD = (7, 10)
    BLOCK_SIZE = 34 # mm

    pkl_dir = 'pkls'
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    img_dir = os.path.join('images', 'Stereo20210726', 'videocap_20210727152040') # 212500081
    img_fps = glob.glob(img_dir+r'\*.png')
    img_idx = np.random.choice(range(len(img_fps)))
    img_fp = img_fps[img_idx]
    pkl_fp = os.path.join(pkl_dir, os.path.basename(os.path.dirname(img_dir)) + "_" + os.path.basename(img_dir) + ".pkl")
    with open(pkl_fp, 'rb') as f:
        ret, mtx_ori, dist, rvecs, tvecs, mtx, roi = pickle.load(f)
        rvec, tvec = rvecs[img_idx], tvecs[img_idx]
    img = undistort(cv2.imread(img_fp), mtx_ori, dist, mtx)
    objpoints, imgpoints = get_chessboard_mapping([img_fp], CHECKERBOARD, BLOCK_SIZE, imgs=[img], imshow=False)

    # get testing points
    n_point = [54, 60, 12, 26]
    imgpoints = np.squeeze(imgpoints)[n_point]
    objpoints = np.squeeze(objpoints)[n_point]

    # project_XYZs_to_npidxs
    imgpoints_proj, npidxs, ks = project_XYZs_to_npidxs(objpoints, rvec, tvec, mtx)
    print("imgpoints(cv2 vs self):\n", imgpoints[0], imgpoints_proj[0])
    print("diff of objpoints and imgpoints_proj:", np.mean(np.sqrt(np.sum(np.square(imgpoints - imgpoints_proj), axis=1))))

    # project_npidxs_to_XYZs
    objpoints_proj = project_npidxs_to_XYZs(npidxs, rvec, tvec, mtx, ks)
    print("objpoints(cv2 vs self):\n", objpoints[0], objpoints_proj.astype(np.float16)[0])
    print("diff of objpoints and objpoints_proj:", np.mean(np.sqrt(np.sum(np.square(objpoints - objpoints_proj), axis=1))))

    # self function to project npidxs to obj points
    img_idxs = np.random.choice(range(len(img_fps)), 2, replace=False)
    img_fps = np.array(img_fps)[img_idxs]
    with open(pkl_fp, 'rb') as f:
        ret, mtx_ori, dist, rvecs, tvecs, mtx, roi = pickle.load(f)
        rvecs, tvecs = rvecs[img_idxs], tvecs[img_idxs]
    imgs = [undistort(cv2.imread(img_fp), mtx_ori, dist, mtx) for img_fp in img_fps]
    objpoints, imgpoints = get_chessboard_mapping(img_fps, CHECKERBOARD, BLOCK_SIZE, imgs=imgs, imshow=False) # (n_img, 1, 70 points on chessboard, 3)

    # get intersection
    npidxs_imgs = [project_XYZs_to_npidxs(ops, rvec, tvec, mtx)[1] for rvec, tvec, ops in zip(rvecs, tvecs, np.squeeze(objpoints))]
    aereo_params_imgs = [(rvec, tvec, mtx, WIDTH, HEIGHT) for rvec, tvec in zip(rvecs, tvecs)]

    # self function to project npidxs to obj points
    PXYZses = np.squeeze(objpoints)[0] # shape=(70, 3), use only one chessboard
    kp_npidxs1, kp_npidxs2 = npidxs_imgs
    aereo_params1, aereo_params2 = aereo_params_imgs
    LXYZ1, LXYZ2, PXYZs1, PXYZs2, dists_ecu, dists_blk, ks1, ks2 = get_PQ(aereo_params1, aereo_params2, kp_npidxs1, kp_npidxs2)
    plot_aereo_triangulation_3d(rvecs, tvecs, mtx, PXYZses, WIDTH, HEIGHT)
