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
from camera_calibrator import *
from moviepy.editor import VideoFileClip, ImageClip


def display_images(*imgs, title='img', captions=(), scale=1.0, wait=True, hstack=2, savefile=""):
    """
    uses OpenCV to display RGB image(s) on screen with optional scaling and keyboard wait
    assumes all imgs are the same size
    """

    # some quick checks on the parameters
    if captions:
        assert len(captions) == len(imgs)
    # TODO: assert that all images are the same shape so we don't run into problems later

    img_grid = None
    img_row = None
    for i in range(len(imgs)):
        img = imgs[i]
        if len(img.shape) > 2:
            # assume RGB & convert to BGR since that's what cv2 wants
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            captionColor = (0, 0, 0)
        else:
            # assume GRAY & convert to BGR since that's what cv2 wants
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            captionColor = (255, 255, 255)
        if captions:
            cv2.putText(img, captions[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, captionColor, 2, cv2.LINE_AA)

        # scale image as requested
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # place images into grid of hstack width
        if i % hstack:
            img_row = np.hstack((img_row, img))
        else:
            if i > 0:
                if i == hstack:
                    img_grid = img_row
                else:
                    img_grid = np.vstack((img_grid, img_row))
            img_row = img

    # if we have something other than a multiple of hstack, pad out the grid row
    if len(imgs) % hstack:
        pad_imgs = hstack - len(imgs) % hstack
        padding = pad_imgs * round(imgs[0].shape[1] * scale)
        img_row = cv2.copyMakeBorder(img_row, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # stack last row
    if img_grid is None:
        img_grid = img_row
    else:
        img_grid = np.vstack((img_grid, img_row))

    # display stacked image
    cv2.imshow(title, img_grid)
    # save file if requested
    if savefile:
        cv2.imwrite(savefile, img_grid)

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


def gradient_threshold_lanes(img, abs_kernel=3, mag_kernel=3, dir_kernel=3,
                                abs_thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2),
                                use_absx=True, use_absy=True, use_mag=True, use_dir=True):
    """takes in an RGB image and applies Sobel operator to find magnitude and direction of gradient"""

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=abs_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=abs_kernel)

    # Calculate the absolute gradient values
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Rescale to 8 bit
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx) )
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    # Create a binary image of ones where threshold is met, zeros otherwise
    gradx = np.ones_like(scaled_sobelx)
    grady = np.ones_like(scaled_sobely)
    if use_absx:
        gradx[(scaled_sobelx < abs_thresh[0]) | (scaled_sobelx > abs_thresh[1])] = 0
    if use_absy:
        grady[(scaled_sobely < abs_thresh[0]) | (scaled_sobely > abs_thresh[1])] = 0

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=mag_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=mag_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_output = np.ones_like(gradmag)
    if use_mag:
        mag_output[(gradmag < mag_thresh[0]) | (gradmag > mag_thresh[1])] = 0

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=dir_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=dir_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Rescale to 8 bit
    #scale_factor = np.max(absgraddir)/255
    #absgraddir = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    dir_output = np.ones_like(absgraddir)
    if use_dir:
        dir_output[(absgraddir < dir_thresh[0]) | (absgraddir > dir_thresh[1])] = 0

    # combine all Sobel gradients into one big gradient lovefest
    # peg binary image as white or black for ease of display
    sobel_output = np.zeros_like(img[:, :, 0])
    if (not use_absx and not use_absy):
        sobel_output[((mag_output == 1) & (dir_output == 1))] = 255
    elif (not use_mag and not use_dir):
        sobel_output[((gradx == 1) & (grady == 1))] = 255
    else:
        sobel_output[((gradx == 1) & (grady == 1)) | ((mag_output == 1) & (dir_output == 1))] = 255

    return sobel_output

def get_birdseye_transforms(img):
    """
    generate perspective transform matrices for forward->birdseye & birdseye->forward
    """

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


def detect_lane_lines(img, mtx, dist, display=False):
    """takes in an RGB image, tries to find lane lines, and augments image with visualization of lane lines"""

    # undistort image
    dst = undistort_image(img, mtx, dist)

    # transform perspective to bird's eye view
    to_birdseye, to_forward = get_birdseye_transforms(dst)
    bey = cv2.warpPerspective(dst, to_birdseye, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)

    # color threshold image
    cth = color_threshold_lanes(bey)

    # sobel gradient threshlold image
    # parameters based on interactive tuner in sobel_tuner.ipynb
    sth = gradient_threshold_lanes(bey, abs_kernel=9, mag_kernel=15, dir_kernel=15,
                                   abs_thresh=(20, 200), mag_thresh=(20, 200), dir_thresh=(0.7, 1.3),
                                   use_absx=True, use_absy=False, use_mag=True, use_dir=True)

    # combine thresholded images to create binary image
    bth = np.bitwise_or(cth, sth)

    # TODO: find lane lines using histograms in sliding windows
    # TODO: fit polynomial to detected lines
    # TODO: find radius of curvature
    # TODO: sanity check and save/load of lane line & curvature data
    # TODO: smooth over multiple frames if video
    # TODO: draw lane lines on image
    # TODO: transform perspective to forward-looking 3D

    retimg = bth
    
    if display:
        # display debug images
        display_images(img, dst, bey, cth, sth, bth, title='debug',
                       captions=("original", "undistorted", "birdseye",
                                 "color threshold", "sobel threshold", "binary threshold"),
                       scale=0.25, savefile="output_images/pipeline_images.png")

    return retimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cal', type=str, metavar="FILENAME", default="camera_cal.npz",
                        help='filename with camera calibration matrix & distortion coefficients (DEFAULT: camera_cal.npz)')
    parser.add_argument('--display', action="store_true", help='display augmented images/video')
    parser.add_argument('--debug', action="store_true", help=argparse.SUPPRESS)
    parser.add_argument('--outdir', type=str, metavar="OUTDIR", default=".",
                        help='output directory (DEFAULT: ".")')
    parser.add_argument('infiles', type=str, metavar="INFILE", nargs="+",
                        help='input filenames - either .jpg or .mp4 format')
    args = parser.parse_args()

    # TODO: add GUI using Tkinter

    # load camera calibration data
    cameraMatrix, distCoeffs, newCameraMatrix, validPixROI = load_camera_calibration(args.cal)

    # create output directory if needed
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for infile in args.infiles:
        print("Processing {0}...".format(infile))
        if imghdr.what(infile):
            # if input is an image file
            clip = ImageClip(infile)
        else:
            # assume it's a video
            clip = VideoFileClip(infile)

        # go through each frame in image (only 1 frame, obviously) or video
        outclip = clip.fl_image(lambda frame: detect_lane_lines(frame, cameraMatrix, distCoeffs, args.display))

        outfile = args.outdir + "/" + os.path.basename(infile)
        if imghdr.what(infile):
            # if input is an image file
            outclip.save_frame(outfile)
        else:
            # assume it's a video
            outclip.write_videofile(outfile, audio=False, progress_bar=True)
