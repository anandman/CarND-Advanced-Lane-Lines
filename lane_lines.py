"""
Lane Line Detector
   Will detect and add lane line visualizations to static image or a video
"""

import os
import imghdr
import numpy as np
import cv2
import argparse
from camera_calibrator import *
from moviepy.editor import VideoFileClip, ImageClip

# declare globals
_global_left_fits = np.zeros((5, 3))
_global_right_fits = np.zeros((5, 3))

def debug_images(*imgs, title='img', captions=(), scale=1.0, wait=True, hstack=2, display=True, savefile=""):
    """
    uses OpenCV to display RGB image(s) on screen with optional scaling and keyboard wait
    assumes all imgs are the same size
    :param imgs: 
    :param title: 
    :param captions: 
    :param scale: 
    :param wait: 
    :param hstack: 
    :param savefile: 
    :return: 
    """

    # some quick checks on the parameters
    if captions:
        # check to make sure we have a caption for every image
        assert len(captions) == len(imgs)
    # check to make sure all images are same shape
    img_size = imgs[0].shape
    assert all(i.shape == img_size for i in imgs)

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
            cv2.putText(img, captions[i], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, captionColor, 2, cv2.LINE_AA)

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

    if display:
        # display stacked image
        cv2.imshow(title, img_grid)
        # save file if requested
        if savefile:
            cv2.imwrite(savefile, img_grid)

        if wait:
            cv2.waitKey()

        cv2.destroyAllWindows()

    return cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB)


def color_threshold_lanes(img):
    """
    takes in an RGB image and returns only white & yellow areas
    :param img: 
    :return gray: 
    """

    # convert to HSV & LAB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # TODO: play with which morphology operations we use and where to place them in the pipeline
    # create structuring elements
    se_10x10ellipse = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
    se_2x2rect = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2, 2))

    # select white pixels
    # find using HSV threshold
    lower_white = np.array([0, 0, 200], dtype=np.uint8)  # HSV (0-180 - all hues, 0-25 = bottom 10%, 180-255 - top 20%)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)  # HSV
    white = cv2.inRange(hsv, lower_white, upper_white)
    # find using adaptive Rgb threshold
    # white = cv2.adaptiveThreshold(img[:, :, 0], 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               thresholdType=cv2.THRESH_BINARY, blockSize=15, C=-8)

    # select yellow pixels (US traffic yellow is 50/360 degrees, 100% sat, 98% value)
    # find using HSV thresholds
    # lower_yellow = np.array([18, 120, 150], dtype=np.uint8)  # HSV
    # upper_yellow = np.array([30, 255, 255], dtype=np.uint8)  # HSV
    # yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # find using laB threshold
    yellow = cv2.inRange(lab[:, :, 2], 150, 255)
    # find using adaptive laB threshold
    # yellow = cv2.adaptiveThreshold(lab[:, :, 2], 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                thresholdType=cv2.THRESH_BINARY, blockSize=35, C=-5)

    # combine white & yellow lane lines
    gray = cv2.bitwise_or(white, yellow)

    # erode away noise
    gray = cv2.erode(gray, se_2x2rect)
    # tophat away large objects
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_10x10ellipse)

    # filter out some things we don't care about
    # by only taking bright lanes markers
    # who knows how it'll work in the rain or dark
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray_mask = cv2.bitwise_and(gray_img, gray)
    # _, gray = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY)

    # chop off sides
    gray[:, 0:520] = 0
    gray[:, 760:1280] = 0

    return gray


def gradient_threshold_lanes(img, abs_kernel=3, mag_kernel=3, dir_kernel=3,
                                abs_thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2),
                                use_absx=True, use_absy=True, use_mag=True, use_dir=True,
                                erode=True, erode_ksize=4, tophat=True, tophat_ksize=10, chop=True):
    """
    takes in an RGB image and applies Sobel operator to find magnitude and direction of gradient
    :param img: 
    :param abs_kernel: 
    :param mag_kernel: 
    :param dir_kernel: 
    :param abs_thresh: 
    :param mag_thresh: 
    :param dir_thresh: 
    :param use_absx: 
    :param use_absy: 
    :param use_mag: 
    :param use_dir: 
    :param erode: 
    :param erode_ksize: 
    :param tophat: 
    :param tophat_ksize: 
    :param chop: 
    :return sobel_output: 
    """

    # TODO: play with which morphology operations we use and where to place them in the pipeline
    # create structuring elements
    se_tophat_ellipse = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(tophat_ksize, tophat_ksize))
    se_erode_rect = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(erode_ksize, erode_ksize))

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
    # Create a binary image of ones where threshold is met, zeros otherwise
    dir_output = np.ones_like(absgraddir)
    if use_dir:
        dir_output[(absgraddir < dir_thresh[0]) | (absgraddir > dir_thresh[1])] = 0

    # combine all Sobel gradients into one big gradient lovefest
    # peg binary image as white or black for ease of display
    sobel_output = np.zeros_like(img[:, :, 0])
    if not use_absx and not use_absy:
        sobel_output[((mag_output == 1) & (dir_output == 1))] = 255
    elif not use_mag and not use_dir:
        sobel_output[((gradx == 1) & (grady == 1))] = 255
    else:
        sobel_output[((gradx == 1) & (grady == 1)) | ((mag_output == 1) & (dir_output == 1))] = 255

    # erode away noise
    if erode:
        sobel_output = cv2.erode(sobel_output, se_erode_rect)
    # tophat away large objects
    if tophat:
        sobel_output = cv2.morphologyEx(sobel_output, cv2.MORPH_TOPHAT, se_tophat_ellipse)

    # chop off sides
    if chop:
        sobel_output[:, 0:520] = 0
        sobel_output[:, 760:1280] = 0

    return sobel_output


def find_lane_lines(img, last_left_fits, last_right_fits, isVideo=False, debug=False):
    """
    finds and displays lane lines
    :param img: 
    :param last_left_fits:
    :param last_right_fits:
    :param debug:
    :return out_img: 
    """

    # Assuming you have created a warped binary image
    # Take a histogram of the image
    histogram = np.sum(img, axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # get fit coefficients from last iteration
    if isVideo:
        last_left_fit = last_left_fits[0]
        last_right_fit = last_right_fits[0]
    else:
        last_left_fit = [0, 0, 0]
        last_right_fit = [0, 0, 0]

    # TODO: find lanes based on previous frame and new detection and determine which is more likely correct
    # if working on video and we have valid coefficients from the last iteration, use those to find new window
    if isVideo and any(c != 0 for c in last_left_fit) and any(c != 0 for c in last_right_fit):
        # search in a margin around the previous line position
        left_lane_inds = ((nonzerox > (last_left_fit[0] * (nonzeroy ** 2) + last_left_fit[1] * nonzeroy + last_left_fit[2] - margin))
                          & (nonzerox < (last_left_fit[0] * (nonzeroy ** 2) + last_left_fit[1] * nonzeroy + last_left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (last_right_fit[0] * (nonzeroy ** 2) + last_right_fit[1] * nonzeroy + last_right_fit[2] - margin))
                           & (nonzerox < (last_right_fit[0] * (nonzeroy ** 2) + last_right_fit[1] * nonzeroy + last_right_fit[2] + margin)))
    else:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            if debug:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    if debug:
        # visualize left as red and right as blue
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # TODO: use RANSAC spline fitting instead of simple polyfit to better ignore outliers
    if len(leftx) is 0:
        if all(c == 0 for c in last_left_fit):
            # eek! we have no lane line!
            print("ERROR: can't find left lane line")
            exit(-1)
        else:
            # use last good fit if no detected left points
            left_fit = last_left_fit
    else:
        # fit a second order polynomial to left side
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) is 0:
        if all(c == 0 for c in last_right_fit):
            # eek! we have no lane line!
            print("ERROR: can't find right lane line")
            exit(-1)
        else:
            # handle case of no detected right points
            right_fit = last_right_fit
    else:
        # fit a second order polynomial to rightside
        right_fit = np.polyfit(righty, rightx, 2)

    # TODO: handle lane merges or exits more gracefully rather than toss them out
    # take derivative of left & right polynomial curves to see if they are roughly the same at a few points
    yofs = 0
    for y in [0.25 * img.shape[0], 0.50 * img.shape[0], img.shape[0]]:
        left_dx = left_fit[0] * y + left_fit[1]
        right_dx = right_fit[0] * y + right_fit[1]
        if debug:
            cv2.putText(out_img, "dx at y = {:0.2f} : {:0.4f} {:0.4f} |{:0.4f}|".format(y, left_dx, right_dx,
                                                                                    abs(left_dx - right_dx)),
                        (50, 120+yofs), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)
            yofs += 50
        # lane lines are diverging too much from each other so use last found lines as a backup
        # NB: this will NOT work with lane merges or exits
        if abs(left_dx - right_dx) > 0.10:
            left_fit = last_left_fit
            right_fit = last_right_fit
            break

    # push the current polynomial coefficients onto front of queue
    # since we defined the np.array with a certain size,
    # the nth set of coefficients effectively drops off the back of queue
    # np.roll is computationally faster than using deque's, even though we want FIFO and nor ring buffer
    last_left_fits = np.roll(last_left_fits, 1, axis=0)
    last_right_fits = np.roll(last_right_fits, 1, axis=0)
    last_left_fits[0] = left_fit
    last_right_fits[0] = right_fit
    # smooth over multiple frames if video
    if isVideo:
        # take the average of the last few frames
        # TODO: consider using weighted average of frames (i.e. use higher weights for more recent frames)
        avg_left_fit = np.mean(last_left_fits, axis=0)
        avg_right_fit = np.mean(last_right_fits, axis=0)
    else:
        avg_left_fit = left_fit
        avg_right_fit = right_fit

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = avg_left_fit[0] * ploty ** 2 + avg_left_fit[1] * ploty + avg_left_fit[2]
    right_fitx = avg_right_fit[0] * ploty ** 2 + avg_right_fit[1] * ploty + avg_right_fit[2]

    if debug:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        window_img = np.zeros_like(out_img)
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # visualize polynomial fit lane lines
        left_pts = np.int32(np.column_stack((left_fitx, ploty)))
        right_pts = np.int32(np.column_stack((right_fitx, ploty)))
        cv2.polylines(out_img, [left_pts], isClosed=False, color=(255, 255, 0))
        cv2.polylines(out_img, [right_pts], isClosed=False, color=(255, 255, 0))
    else:
        zero = np.zeros_like(img).astype(np.uint8)
        out_img = np.dstack((zero, zero, zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))

    return out_img, last_left_fits, last_right_fits


def add_curvature_and_offset(img, left_fit, right_fit, dpmx, dpmy):
    """
    adds lane's radius of curvature and offset from center
    :param img: 
    :param leftx: 
    :param rightx: 
    :param dpmx: 
    :param dpmy: 
    :return overlay: 
    """

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * dpmy, left_fitx * dpmx, 2)
    right_fit_cr = np.polyfit(ploty * dpmy, right_fitx * dpmx, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * dpmy + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * dpmy + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    curverad = (left_curverad + right_curverad) / 2.0
    cv2.putText(img, "radius of curvature: {:0.4f} m".format(curverad), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    midx = img.shape[1] / 2
    dx1 = midx - left_fitx[-1]
    dx2 = right_fitx[-1] - midx
    offset = ((dx1 - dx2) / 2) * dpmx
    cv2.putText(img, "offset from center: {:0.4f} m".format(offset), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return img


def detect_lane_lines(img, mtx, dist, to_birdseye, to_forward, dpmx, dpmy, isVideo=False, debug=False, display=False):
    """
    takes in an RGB image, tries to find lane lines, and augments image with visualization of lane lines
    :param img: 
    :param mtx: 
    :param dist: 
    :param to_birdseye: 
    :param to_forward: 
    :param dpmx: 
    :param dpmy: 
    :param isVideo: 
    :param display: 
    :return retimg: 
    """

    # undistort image
    dst = undistort_image(img, mtx, dist)

    bey = cv2.warpPerspective(dst, to_birdseye, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)

    # color threshold image
    cth = color_threshold_lanes(bey)

    # TODO: tune Sobel thresholding and/or add Hough transform; currently, Sobel just adds unwanted noise
    use_sobel = False

    if use_sobel:
        # sobel gradient threshold image
        # parameters based on interactive tuner in sobel_tuner.ipynb
        sth = gradient_threshold_lanes(bey, abs_kernel=9, mag_kernel=15, dir_kernel=15,
                                       abs_thresh=(20, 200), mag_thresh=(20, 200), dir_thresh=(0.7, 1.3),
                                       use_absx=True, use_absy=False, use_mag=True, use_dir=True)
    else:
        sth = cth

    # combine thresholded images to create binary image
    bth = np.bitwise_or(cth, sth)

    global _global_left_fits, _global_right_fits

    lnl, _global_left_fits, _global_right_fits =\
        find_lane_lines(bth, last_left_fits=_global_left_fits, last_right_fits=_global_right_fits,
                        isVideo=isVideo, debug=debug)

    lnw = cv2.warpPerspective(lnl, to_forward, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)
    lna = cv2.addWeighted(dst, 1, lnw, 0.3, 0)

    coi = add_curvature_and_offset(lna, _global_left_fits[0], _global_right_fits[0], dpmx, dpmy)

    retimg = coi

    if debug or display:
        # display debug images
        '''
        debug_img = debug_images(img, dst, bey, cth, sth, bth, lnl, coi, title='debug',
                                 captions=("original", "undistorted", "birdseye",
                                           "color threshold", "sobel threshold", "binary threshold",
                                           "lane lines", "final"),
                                 scale=0.25, hstack=4, display=display, savefile="output_images/pipeline_images.png")
        '''
        debug_img = debug_images(img, cth, sth, bth, lnl, coi, title='debug',
                                 captions=("original",
                                           "color threshold", "sobel threshold", "binary threshold",
                                           "lane lines", "final"),
                                 scale=0.5, hstack=2, display=display, savefile="output_images/pipeline_images.png")
        if debug:
            retimg = debug_img

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

    # TODO: add GUI using Tkinter to display images/videos

    # load camera calibration data
    cameraMatrix, distCoeffs, newCameraMatrix, validPixROI = load_camera_calibration(args.cal)

    # calculate transforms to change perspective to/from bird's eye view
    to_birdseye, to_forward = get_birdseye_transforms()

    # get image space to world space resoltuon conversion
    dpmx, dpmy = get_world_resolutions()

    # create output directory if needed
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for infile in args.infiles:
        print("Processing {0}...".format(infile))
        if imghdr.what(infile):
            # if input is an image file
            clip = ImageClip(infile)
            video = False
        else:
            # assume it's a video
            clip = VideoFileClip(infile)
            video = True

        # go through each frame in image (only 1 frame, obviously) or video
        outclip = clip.fl_image(lambda frame: detect_lane_lines(frame, cameraMatrix, distCoeffs,
                                                                to_birdseye, to_forward, dpmx, dpmy,
                                                                isVideo=video, display=args.display, debug=args.debug))

        outfile = args.outdir + "/" + os.path.basename(infile)
        if imghdr.what(infile):
            # if input is an image file
            outclip.save_frame(outfile)
        else:
            # assume it's a video
            outclip.write_videofile(outfile, audio=False, progress_bar=True)
