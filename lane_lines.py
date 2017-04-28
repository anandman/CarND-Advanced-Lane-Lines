"""
Lane Line Detector
   Will detect and add lane line visualizations to static image or a video
"""

import os
import sys
import imghdr
import numpy as np
import cv2
import argparse
import camera_calibrator as cc
from moviepy.editor import VideoFileClip, ImageClip


def display_images(*imgs, title='img', wait=True, scale=1.0):
    """uses OpenCV to display RGB image(s) on screen with optional scaling and keyboard wait"""

    stack_img = None
    # TODO: extend stack to be a grid if more than two images are passed in
    # convert from RGB to BGR since that's what cv2 wants
    for img in imgs:
        if len(img.shape) > 2:
            # assume RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # assume GRAY
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        if stack_img is None:
            stack_img = img
        else:
            stack_img = np.hstack((stack_img, img))

    # display stacked image
    cv2.imshow(title, stack_img)

    if wait:
        cv2.waitKey()

    cv2.destroyAllWindows()


def color_threshold_lanes(img):
    """takes in an RGB image and returns only white & yellow areas"""

    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # select white pixels
    lower_white = np.array([0, 0, 180], dtype=np.uint8)  # HSV (0-180, 0-255, 0-255)
    upper_white = np.array([255, 25, 255], dtype=np.uint8)  # HSV
    white = cv2.inRange(hsv, lower_white, upper_white)

    # select yellow pixels (US traffic yellow is 50/360 degrees, 100% sat, 98% value)
    lower_yellow = np.array([18, 120, 150], dtype=np.uint8)  # HSV
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)  # HSV
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # combine white & yellow lane lines
    color_mask = cv2.bitwise_or(white, yellow)  # could return this as gray, but...

    # instead, let's filter out some things we don't care about
    # by only taking bright lanes markers
    # who knows how it'll work in the rain or dark
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_mask = cv2.bitwise_and(gray_img, color_mask)
    _, gray = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY)

    return gray


def detect_lane_lines(img, mtx, dist):
    """takes in an RGB image, tries to find lane lines, and augments image with visualization of lane lines"""

    # undistort image
    dst = cc.undistort_image(img, mtx, dist)
    if args.display:
        # display images
        display_images(img, dst, title='distort', wait=True, scale=0.5)

    # color threshold image
    cth = color_threshold_lanes(dst)
    if args.display:
        # display images
        display_images(dst, cth, title='color threshold', wait=True, scale=0.5)

    # TODO: gradient threshold image
    # TODO: combine thresholded images to create binary image
    # TODO: transform perspective to top-down 2D
    # TODO: find lane lines using histograms in sliding windows
    # TODO: fit polynomial to detected lines
    # TODO: find radius of curvature
    # TODO: sanity check and save/load of lane line & curvature data
    # TODO: smooth over multiple frames if video
    # TODO: draw lane lines on image
    # TODO: transform perspective to forward-looking 3D

    return cth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cal', type=str, metavar="FILENAME", default="camera_cal.npz",
                        help='filename with camera calibration matrix & distortion coefficients (DEFAULT: camera_cal.npz)')
    parser.add_argument('--display', action="store_true", help='display augmented images/video')
    parser.add_argument('--debug', action="store_true", help=argparse.SUPPRESS)
    parser.add_argument('infile', type=str, metavar="INFILE",
                        help='input filename - either .jpg or .mp4 format')
    parser.add_argument('outfile', type=str, metavar="OUTFILE",
                        help='output filename - either .jpg or .mp4 format')
    args = parser.parse_args()

    # load camera calibration data
    cameraMatrix, distCoeffs, newCameraMatrix, validPixROI = cc.load_camera_calibration(args.cal)

    if imghdr.what(args.infile):
        # if input is an image file
        clip = ImageClip(args.infile)
    else:
        # assume it's a video
        clip = VideoFileClip(args.infile)

    # go through each frame in image (only 1 frame, obviously) or video
    outclip = clip.fl_image(lambda frame: detect_lane_lines(frame, cameraMatrix, distCoeffs))

    if imghdr.what(args.infile):
        # if input is an image file
        outclip.save_frame(args.outfile)
    else:
        # assume it's a video
        outclip.write_videofile(args.outfile, audio=False, progress_bar=True)
