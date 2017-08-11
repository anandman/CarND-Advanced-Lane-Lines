"""
Vehicle Detector
   Will detect and add visualizations around vehicles to static image or a video
"""

import os
import imghdr
import argparse
from debug import *
from camera_calibrator import *
from moviepy.editor import VideoFileClip, ImageClip
from imutils.video import FPS
import cv2
import h5py
from keras import backend as K
from keras.models import load_model
from keras import __version__ as keras_version
from yad2k.models.keras_yolo import yolo_eval, yolo_head


# some parameters to use to filter bounding boxes
SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.2
CLASSES_TO_SHOW = ['bus', 'car', 'motorbike', 'truck']


def detect_vehicles(img, fps, mtx, dist, sess, model, classes, anchors, isVideo=False, debug=False, display=False):
    """
    takes in an RGB image, tries to detect vehicles, and augments image with visualization of detected vehicles
    :param img:
    :param fps:
    :param mtx: 
    :param dist:
    :param model:
    :param isVideo:
    :param display: 
    :return retimg: 
    """

    # undistort image
    dst = undistort_image(img, mtx, dist)

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = cv2.resize(dst, model_image_size)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (dst.width - (dst.width % 32),
                          dst.height - (dst.height % 32))
        resized_image = cv2.resize(dst, new_image_size)
        image_data = np.array(resized_image, dtype='float32')

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # run image through model
    # TODO: model inference code

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(model.output, anchors, len(classes))
    input_image_shape = K.placeholder(shape=(2, ))
    o_boxes, o_scores, o_classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD)

    out_boxes, out_scores, out_classes = sess.run(
        [o_boxes, o_scores, o_classes],
        feed_dict={
            model.input: image_data,
            input_image_shape: [dst.shape[0], dst.shape[1]],
            K.learning_phase(): 0
        })

    box_img = dst

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = classes[c]
        box = out_boxes[i]
        score = out_scores[i]

        # only show if prediction is in CLASSES_TO_SHOW
        if predicted_class not in CLASSES_TO_SHOW:
            continue

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(dst.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(dst.shape[1], np.floor(right + 0.5).astype('int32'))

        # draw rectangle
        cv2.rectangle(box_img, (left, top), (right, bottom), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # write label
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 0.5
        fontthickness = 1
        textsize, _ = cv2.getTextSize(label, fontface, fontscale, fontthickness)
        cv2.putText(box_img, label, (left + 2, top + textsize[1] + 2),
                    fontface, fontScale=fontscale, color=(255, 255, 255),
                    thickness=fontthickness, lineType=cv2.LINE_AA)

    retimg = box_img

    if debug or display:
        # display debug images
        debug_img = debug_images(img, dst, box_img, title='debug',
                                 captions=("original", "undistorted", "annotated"),
                                 scale=0.5, hstack=2, display=display, savefile="output_images/pipeline_images.png")
        if debug:
            retimg = debug_img

    # update FPS timer
    fps.update()

    return retimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cal', type=str, metavar="FILENAME", default="camera_cal.npz",
                        help='filename with camera calibration matrix & distortion coefficients (DEFAULT: camera_cal.npz)')
    parser.add_argument('--model', type=str, metavar="FILENAME", default="yolo.h5",
                        help='filename with YOLO model and weights (DEFAULT: yolo.h5)')
    parser.add_argument('--anchors', type=str, metavar="FILENAME", default="yolo_anchors.txt",
                        help='filename with YOLO anchors (DEFAULT: yolo_anchors.txt)')
    parser.add_argument('--classes', type=str, metavar="FILENAME", default="yolo_classes.txt",
                        help='filename with YOLO anchors (DEFAULT: yolo_classes.txt)')
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

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    # load Keras model with weights
    model = load_model(args.model)

    # read classes file
    with open(args.classes) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]

    # read anchors file
    with open(args.anchors) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    # verify model, anchors, and classes are compatible
    num_classes = len(classes)
    num_anchors = len(anchors)
    model_output_channels = model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors and ' \
        '--classes flags.'

    # create output directory if needed
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # start FPS timer
    fps = FPS().start()

    # open TF session
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

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
        outclip = clip.fl_image(lambda frame: detect_vehicles(frame, fps, cameraMatrix, distCoeffs,
                                                              sess, model, classes, anchors,
                                                              isVideo=video, display=args.display, debug=args.debug))

        outfile = args.outdir + "/" + os.path.basename(infile)
        if video:
            # write video
            outclip.write_videofile(outfile, audio=False, progress_bar=True)
        else:
            # write image
            outclip.save_frame(outfile)

        del clip
        del outclip

    # print some FPS statistics
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # close TF session
    sess.close()

