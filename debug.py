"""
Debug Routines
"""

import numpy as np
import cv2


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
    # check to make sure all images are same shape (ignore color dimensions and only check X & Y)
    img_size = imgs[0].shape
    assert all(i.shape[0:1] == img_size[0:1] for i in imgs)

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

