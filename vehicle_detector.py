"""
Vehicle Detector
    Will detect and add visualizations around vehicles to static image or a video

    To setup darknet, follow these directions...
        1) git clone https://github.com/pjreddie/darknet
        2) cd darknet
        3) edit Makefile if using GPU, CUDNN, etc.
        4) make
        5) wget https://pjreddie.com/media/files/tiny-yolo-voc.weights
        6) edit cfg/voc.data to change "data/voc.names" to "darknet/data/voc.names"

    To setup YAD2K, follow these directions...
        1) Setup darknet as described above
        2) git clone https://github.com/allanzelener/yad2k.git
        3) cd yad2k
        4) python yad2k.py ../darknet/cfg/tiny-yolo-voc.cfg ../darknet/tiny-yolo-voc.weights model_data/tiny-yolo-voc.h5
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
from yad2k.yad2k.models.keras_yolo import yolo_eval, yolo_head
import darknet
import ctypes
from yolo_keras import *


# some parameters to use to crop and filter bounding boxes
CROP_YMIN = 300
CROP_YMAX = 719
CROP_XMIN = 0
CROP_XMAX = 1279
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
CLASSES_TO_SHOW = ['bus', 'car', 'motorbike', 'truck']

# declare globals
# declare it as empty array of objects to set the initial size
_global_last_boxes = np.empty(5, dtype=object)


def draw_boxes(boxes, img, model_size, crop_rect, color=(255, 255, 255), debug=False):
    """
    draws boxes onto an img after resizing from model coords to full image coords
    :param boxes:
    :param img:
    :param model_size:
    :param crop_rect:
    :param color:
    :param debug:
    :return retimg:
    """

    retimg = img.copy()
    [xmin, xmax] = crop_rect[0]
    [ymin, ymax] = crop_rect[1]
    crop_w = xmax - xmin
    crop_h = ymax - ymin
    [model_h, model_w] = model_size
    [img_h, img_w] = img.shape[0:2]

    for box in boxes:
        # only show if prediction is in CLASSES_TO_SHOW
        if box.cn not in CLASSES_TO_SHOW:
            if debug: print("[INFO] detected class", box.cn)
            continue

        label = '{} {:.2f}'.format(box.cn, box.prob)

        # convert bounding box to coordinates
        left = (box.x - box.w / 2)
        right = (box.x + box.w / 2)
        top = (box.y - box.h / 2)
        bottom = (box.y + box.h / 2)

        # scale up boxes to cropped image size
        left *= crop_w / model_w
        right *= crop_w / model_w
        top *= crop_h / model_h
        bottom *= crop_h / model_h

        # shift boxes from cropped to original image
        left += xmin
        right += xmin
        top += ymin
        bottom += ymin

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(img_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(img_w, np.floor(right + 0.5).astype('int32'))

        # draw rectangle
        cv2.rectangle(retimg, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)

        # write label
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 0.5
        fontthickness = 1
        textsize, _ = cv2.getTextSize(label, fontface, fontscale, fontthickness)
        cv2.putText(retimg, label, (left + 2, top + textsize[1] + 2),
                    fontface, fontScale=fontscale, color=color,
                    thickness=fontthickness, lineType=cv2.LINE_AA)

    return retimg


def init_yad2k(keras_version, model_file, classes_file, anchors_file):
    """
    initialize YAD2K model
    :param model_file:
    :param classes_file:
    :param anchors_file:
    :return model, classes, anchors:
    """

    f = h5py.File(model_file, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    # load Keras model with weights
    model = load_model(args.model)
    model.summary()

    # read classes file
    with open(classes_file) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]

    # read anchors file
    with open(anchors_file) as f:
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

    return model, classes, anchors


def init_keras(weights_file, classes_file, anchors_file):
    """
    initialize Keras model
    :param weights_file:
    :param classes_file:
    :param anchors_file:
    :return model, classes, anchors:
    """

    model = tiny_yolo_voc()
    load_weights(model, weights_file)
    model.summary()

    # read classes file
    with open(classes_file) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]

    # read anchors file
    with open(anchors_file) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    # verify model, anchors, and classes are compatible
    num_classes = len(classes)
    num_anchors = len(anchors)
    model_output_channels = model.layers[-1].output_shape[-2:]
    assert model_output_channels[0] == num_anchors, \
        'Mismatch between model and given anchor size. ' \
        'Specify matching anchors with --anchors flag.'
    assert model_output_channels[1] == (num_classes + 5), \
        'Mismatch between model and given class size. ' \
        'Specify matching classes with --classes flags.'

    return model, classes, anchors


def detect_vehicles_keras(img, fps, mtx, dist, model, classes, anchors,
                          isVideo=False, returnOverlay=False, debug=False, display=False):
    """
    takes in an RGB image, tries to detect vehicles, and augments image with visualization of detected vehicles
    :param img:
    :param fps:
    :param mtx:
    :param dist:
    :param model:
    :param classes:
    :param anchors:
    :param isVideo:
    :param returnOverlay:
    :param display:
    :return retimg:
    """

    # undistort image
    dst = undistort_image(img, mtx, dist)

    if returnOverlay:
        box_img = np.zeros_like(dst)
    else:
        box_img = dst

    # crop to road area
    dst_cropped = dst[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]

    # check if model is fully convolutional, assuming channel last order.
    model_image_size = model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = cv2.resize(dst_cropped, model_image_size)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (dst_cropped.width - (dst_cropped.width % 32),
                          dst_cropped.height - (dst_cropped.height % 32))
        resized_image = cv2.resize(dst_cropped, new_image_size)
        image_data = np.array(resized_image, dtype='float32')

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # run image through model
    netout = model.predict(image_data)

    global _global_last_boxes

    # extract bounding boxes and filter based on thresholds and classes
    pboxes = prediction_bboxes(netout, classes, anchors, SCORE_THRESHOLD)

    # do non-maximal suppression (NMS) to eliminate similar boxes
    boxes = non_maximal_suppression(pboxes, NMS_THRESHOLD)

    # we also use NMS to average across frames if it's a video
    if isVideo:
        # push the current pboxes onto front of queue
        # since we defined the np.array with a certain size,
        # the nth set of pboxes effectively drops off the back of queue
        # np.roll is computationally faster than using deque's, even though we want FIFO and nor ring buffer
        _global_last_boxes = np.roll(_global_last_boxes, 1, axis=0)
        _global_last_boxes[0] = boxes
        fboxes = np.hstack(_global_last_boxes.flat)    # flatten into 1D array of BoundBox boxes
        fboxes = fboxes[fboxes != np.array(None)]  # remove None from queue if any
        boxes = average_boxes(fboxes, NMS_THRESHOLD)

    if isVideo and debug:
        box_img = draw_boxes(fboxes, box_img, model_image_size, ((CROP_XMIN, CROP_XMAX), (CROP_YMIN, CROP_YMAX)),
                             color=(255, 0, 0))
    box_img = draw_boxes(boxes, box_img, model_image_size, ((CROP_XMIN, CROP_XMAX), (CROP_YMIN, CROP_YMAX)))

    retimg = box_img

    if debug or display:
        # display debug images
        debug_img = debug_images(dst, box_img, title='debug',
                                 captions=("undistorted", "annotated"),
                                 scale=0.5, hstack=1, display=display, savefile="output_images/vd/pipeline_images.png")
        if debug:
            retimg = debug_img

    # update FPS timer
    fps.update()

    return retimg


def detect_vehicles_yad2k(img, fps, mtx, dist, sess, model, classes, anchors, isVideo=False, debug=False, display=False):
    """
    takes in an RGB image, tries to detect vehicles, and augments image with visualization of detected vehicles
    :param img:
    :param fps:
    :param mtx: 
    :param dist:
    :param sess:
    :param model:
    :param classes:
    :param anchors:
    :param isVideo:
    :param display: 
    :return retimg: 
    """

    # undistort image
    dst = undistort_image(img, mtx, dist)

    box_img = dst

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
    # generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(model.output, anchors, len(classes))
    input_image_shape = K.placeholder(shape=(2, ))
    o_boxes, o_scores, o_classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=NMS_THRESHOLD)

    out_boxes, out_scores, out_classes = sess.run(
        [o_boxes, o_scores, o_classes],
        feed_dict={
            model.input: image_data,
            input_image_shape: [dst.shape[0], dst.shape[1]],
            K.learning_phase(): 0
        })

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
                                 scale=0.5, hstack=2, display=display, savefile="output_images/vd/pipeline_images.png")
        if debug:
            retimg = debug_img

    # update FPS timer
    fps.update()

    return retimg


def detect_vehicles_darknet(img, fps, mtx, dist, net, meta, isVideo=False, debug=False, display=False):
    """
    takes in an RGB image, tries to detect vehicles, and augments image with visualization of detected vehicles
    :param img:
    :param fps:
    :param mtx:
    :param dist:
    :param net:
    :param meta:
    :param isVideo:
    :param display:
    :return retimg:
    """

    # undistort image
    dst = undistort_image(img, mtx, dist)

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = (darknet.lib.network_width(net), darknet.lib.network_height(net))
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

    image_data = image_data.transpose(2, 0, 1)
    c, h, w = image_data.shape[0], image_data.shape[1], image_data.shape[2]
    # print w, h, c
    image_data = image_data.ravel() / 255.0
    image_data = np.ascontiguousarray(image_data, dtype=np.float32)

    dnimg = darknet.IMAGE(ctypes.c_int(w), ctypes.c_int(h), ctypes.c_int(c),
                          image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    detections = darknet.detect_np(net, meta, dnimg,
                                   thresh=SCORE_THRESHOLD, hier_thresh=SCORE_THRESHOLD, nms=NMS_THRESHOLD)

    box_img = dst

    for det in detections:
        predicted_class = str(det[0], 'utf-8')
        score = det[1]
        bbox = det[2]

        # only show if prediction is in CLASSES_TO_SHOW
        if predicted_class not in CLASSES_TO_SHOW:
            continue

        label = '{} {:.2f}'.format(predicted_class, score)

        # convert bounding box to coordinates
        left = (bbox[0] - bbox[2] / 2);
        right = (bbox[0] + bbox[2] / 2);
        top = (bbox[1] - bbox[3] / 2);
        bottom = (bbox[1] + bbox[3] / 2);

        # scale up boxes to original image size
        left *= dst.shape[1] / model_image_size[0]
        right *= dst.shape[1] / model_image_size[0]
        top *= dst.shape[0] / model_image_size[1]
        bottom *= dst.shape[0] / model_image_size[1]

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
                                 scale=0.5, hstack=2, display=display, savefile="output_images/vd/pipeline_images.png")
        if debug:
            retimg = debug_img

    # update FPS timer
    fps.update()

    return retimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cal', type=str, metavar="FILENAME", default="camera_cal.npz",
                        help='filename with camera calibration matrix & distortion coefficients (DEFAULT: camera_cal.npz)')
    parser.add_argument('--method', type=str, choices=['yad2k', 'darknet', 'keras'], default='keras',
                        help='which detection method to use (DEFAULT: keras')
    parser.add_argument('--cfg', type=str, metavar="FILENAME", default="darknet/cfg/tiny-yolo-voc.cfg",
                        help='Darknet configuration file (DEFAULT: darknet/cfg/tiny-yolo-voc.cfg)')
    parser.add_argument('--meta', type=str, metavar="FILENAME", default="darknet/cfg/voc.data",
                        help='Darknet metadata file (DEFAULT: darknet/cfg/voc.data)')
    parser.add_argument('--weights', type=str, metavar="FILENAME", default="darknet/tiny-yolo-voc.weights",
                        help='Darknet weights file (DEFAULT: darknet/tiny-yolo-voc.weights)')
    parser.add_argument('--model', type=str, metavar="FILENAME", default="yad2k/model_data/tiny-yolo-voc.h5",
                        help='filename with YAD2K model and weights (DEFAULT: yad2k/model_data/tiny-yolo-voc.h5)')
    parser.add_argument('--anchors', type=str, metavar="FILENAME", default="yad2k/model_data/tiny-yolo-voc_anchors.txt",
                        help='filename with YOLO anchors (DEFAULT: yad2k/model_data/tiny-yolo-voc_anchors.txt)')
    parser.add_argument('--classes', type=str, metavar="FILENAME", default="yad2k/model_data/pascal_classes.txt",
                        help='filename with YOLO class names (DEFAULT: yad2k/model_data/pascal_classes.txt)')
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
    if args.method == 'yad2k':
        model, classes, anchors = init_yad2k(keras_version, args.model, args.classes, args.anchors)

        # open TF session
        sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    elif args.method == 'darknet':
        net = darknet.load_net(args.cfg.encode('utf-8'), args.weights.encode('utf-8'), 0)
        meta = darknet.load_meta(args.meta.encode('utf-8'))
    elif args.method == 'keras':
        model, classes, anchors = init_keras(args.weights, args.classes, args.anchors)

    # create output directory if needed
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # start FPS timer
    fps = FPS().start()

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

        # reset global between files
        _global_last_boxes = np.empty(5, dtype=object)

        # go through each frame in image (only 1 frame, obviously) or video
        if args.method == 'yad2k':
            outclip = clip.fl_image(lambda frame: detect_vehicles_yad2k(frame, fps, cameraMatrix, distCoeffs,
                                                                  sess, model, classes, anchors,
                                                                  isVideo=video, display=args.display, debug=args.debug))
        elif args.method == 'darknet':
            outclip = clip.fl_image(lambda frame: detect_vehicles_darknet(frame, fps, cameraMatrix, distCoeffs,
                                                                          net, meta,
                                                                          isVideo=video, display=args.display, debug=args.debug))
        elif args.method == 'keras':
            outclip = clip.fl_image(lambda frame: detect_vehicles_keras(frame, fps, cameraMatrix, distCoeffs,
                                                                        model, classes, anchors,
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
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if args.method == 'yad2k':
        # close TF session
        sess.close()

