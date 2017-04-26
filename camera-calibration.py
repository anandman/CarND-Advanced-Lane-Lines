import os
import numpy as np
import cv2
import argparse

# based on code from
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html


def detect_corners(img_dir="camera_cal", visualize=False):
    """Finds corners in multiple chessboard images and returns objpoints and imgpoints"""

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cv2.startWindowThread()

    for file in os.scandir(img_dir):
        if not file.name.startswith('.') and file.is_file():
            # print("processing {0}...".format(file.name))

            # read in the image
            img = cv2.imread(file.path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                if visualize:
                    img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey()

    cv2.destroyAllWindows()

    return objpoints, imgpoints


def read_sample_image(img_dir):
    # read in a sample image
    for file in os.scandir(img_dir):
        if not file.name.startswith('.') and file.is_file():
            img = cv2.imread(file.path)
            break

    return img


def calibrate_camera(objpoints, imgpoints, width, height):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    # refine the camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height),
                                                      1, (width, height))

    return mtx, dist, newcameramtx, roi


def save_camera_calibration(filename, mtx, dist, newcameramtx, roi):
    np.savez_compressed(filename, cameraMatrix=mtx, distCoeffs=dist, newCameraMatrix=newcameramtx, validPixROI=roi)


def load_camera_calibration(filename):
    cal = np.load(filename)
    return cal['cameraMatrix'], cal['distCoeffs'], cal['newCameraMatrix'], cal['validPixROI']


def undistort_image(img, mtx, dist, newcameramtx=None, roi=None):
    # undistort sample image
    if newcameramtx:
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        if roi:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
    else:
        dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate Camera Distortion')
    parser.add_argument('--caldir', type=str, default="camera_cal",
                        help='directory containing chessboard calibration images')
    parser.add_argument('--npz', type=str, default="camera_cal.npz",
                        help='filename to save camera calibration matrix and distortion coefficients')
    parser.add_argument('--width', type=int, default=1280, help='width of camera in pixels')
    parser.add_argument('--height', type=int, default=720, help='height of camera in pixels')
    parser.add_argument('--show', type=bool, default=False, help='display test images')
    args = parser.parse_args()

    objpoints, imgpoints = detect_corners(args.caldir, visualize=args.show)

    # read in a sample image so we can get the width & height and check it against arguments
    img = read_sample_image(args.caldir)
    h, w = img.shape[:2]
    assert h == args.height, "image height %r is not equal to expected height %r" % (h, args.height)
    assert w == args.width, "image width %r is not equal to expected width %r" % (w, args.width)

    # calibrate the camera
    mtx, dist, newcameramtx, roi = calibrate_camera(objpoints, imgpoints, args.width, args.height)

    # save camera calibration data for future use
    save_camera_calibration(args.npz, mtx, dist, newcameramtx, roi)

    # load camera calibration data
    cameraMatrix, distCoeffs, newCameraMatrix, validPixROI = load_camera_calibration(args.npz)

    # undistort & show sample image
    if args.show:
        dst = undistort_image(img, cameraMatrix, distCoeffs, newCameraMatrix, validPixROI)
        cv2.imshow('dst', dst)
        cv2.waitKey()
