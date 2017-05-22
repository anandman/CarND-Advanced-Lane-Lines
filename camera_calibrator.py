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


def __read_sample_image(img_dir):
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
    if newcameramtx is not None:
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        if roi is not None:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
    else:
        dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


def get_birdseye_transforms():
    """
    generate perspective transform matrices for forward->birdseye & birdseye->forward
    :return m, minv: 
    """

    # TODO - calculate this based on camera calibration data
    # fixed for this car camera angle
    src = np.float32([[270, 665], [670, 665], [645, 465], [570, 465]])
    # focus on only lane lines all the way up to the top
    # dst = np.float32([[270, 665], [670, 665], [670, 465], [270, 465]])
    # lane lines only but crop out horizon, sky, etc.
    # dst = np.float32([[270, 665], [670, 665], [670, 0], [270, 0]])
    # see entire image
    # dst = np.float32([[570, 665], [645, 665], [645, 465], [570, 465]])
    # see entire image without horizon, sky, etc.
    dst = np.float32([[570, 665], [645, 665], [645, 0], [570, 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)

    return m, minv


def get_world_resolutions():
    """
    return dots per meter in X & Y directions
    :return dpmx, dpmy: 
    """

    # TODO: calculate this based on camera calibration data
    # use lane width of 12ft to calculate X dots per meter
    dpmx = 12 * 0.3048 / 140     # 12ft * 0.3048 m/ft / 140 pixels measured on screen across multiple sample pictures
    # use dashed lane length of 10ft to calculate Y dots per meter
    dpmy = 10 * 0.3048 / 110     # 10ft * 0.3048 m/ft / 110 pixels measured on screen across multiple sample pictures

    return dpmx, dpmy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate Camera Distortion')
    parser.add_argument('--caldir', type=str, metavar="DIRECTORY", default="camera_cal",
                        help='directory containing chessboard calibration images')
    parser.add_argument('--npz', type=str, metavar="FILENAME", default="camera_cal.npz",
                        help='filename to save camera calibration matrix and distortion coefficients')
    parser.add_argument('--width', type=int, default=1280, help='width of camera in pixels')
    parser.add_argument('--height', type=int, default=720, help='height of camera in pixels')
    parser.add_argument('--display', action="store_true", help='display test images')
    args = parser.parse_args()

    objpoints, imgpoints = detect_corners(args.caldir, visualize=args.display)

    # read in a sample image so we can get the width & height and check it against arguments
    img = __read_sample_image(args.caldir)
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
    if args.display:
        dst = undistort_image(img, cameraMatrix, distCoeffs, newCameraMatrix, validPixROI)
        cv2.imshow('dst', dst)
        cv2.waitKey()
