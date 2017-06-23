## Advanced Lane Lines

### OpenCV-based traffic lane line detection algorithm

*I plan to add the vehicle detection project to this code base so one code base can show both.*

### Project Description
---

**The goals / steps of this project are the following:**

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Files & Running the Code
---
**The project includes the following files:**

- [camera_calibrator.py](camera_calibrator.py) module/program to calibrate and transform images between camera and real world
- [camera_cal.npz](camera_cal.npz) saved camera distortion coefficients
- [camera_cal/](camera_cal) directory of checkerboard images taken with camera
- [sobel_tuner.ipynb](sobel_tuner.ipynb) interactive Sobel thresholding parameter tuner
- [lane_lines.py](lane_lines.py) lane line detection module/program
- [test_images/](test_images) sample input images
- [project_video.mp4](project_video.mp4) sample input video (*signoff criteria*)
- [challenge_video.mp4](challenge_video.mp4) challenging input video
- [harder_challenge_video.mp4](harder_challenge_video.mp4) very challenging input video
- [output_images/](output_images) output images and videos after running through lane line detection algorithm

**To run the code, the following resources are needed:**

- a full Python 3.x, OpenCV 3.x environment
	- if using conda, run `conda env create -f conda-env.yml`
	- here are environment files for [macOS](conda-macos.yml) and [AWS Ubuntu 16.04 p2.xlarge AMI](conda-aws-p2.yml)

**To calibrate the camera:**

- if using conda environment above, run `source activate highgain`
- run the calibrator as follows:
```sh
python camera_calibrator.py [--caldir DIRECTORY] [--npz FILENAME] [--width WIDTH] [--height HEIGHT] [--display]

  --caldir DIRECTORY  directory containing chessboard calibration images (DEFAULT: camera_cal)
  --npz FILENAME      filename to save camera calibration matrix and distortion coefficients (DEFAULT: camera_cal.npz)
  --width WIDTH       width of camera in pixels (DEFAULT: 1280)
  --height HEIGHT     height of camera in pixels (DEFAULT: 720)
  --display           display test images
```

** To run the lane line detection program:**

- if using the conda environment above, run `source activate highgain`
- run the lane line detector on one or more images or videos as follows (e.g. INFILE can be `project_video.mp4` and OUTDIR can be `output_images`)
```sh
python lane_lines.py [--cal FILENAME] [--display] [--outdir OUTDIR] INFILE [INFILE ...]

  INFILE           input filenames - either .jpg or .mp4 format
  --cal FILENAME   filename with camera calibration matrix & distortion coefficients (DEFAULT: camera_cal.npz)
  --display        display augmented images/video
  --outdir OUTDIR  output directory (DEFAULT: ".")
```
  
### Camera Calibration
---

The camera calibration module [camera_calibrator.py](camera_calibrator.py) has many helper functions that can be used to get data to map from what the camera sees to the real world (and vice versa in some cases). Currently, this includes:
    1. finding camera distortion effects, calculating distortion coefficients to undistort the image, and loading/saving/using those coefficients
    2. calculating transforms to convert between forward and birds-eye top-down views (*currently hardcoded*)
    3. calculating pixel to real world resolutions to enable distance calcluations (*currently hardcoded*)
    
For #1 above, the algorithm is to use multiple checkerboard images and find the corners in both image space and expected 2D space. Using those differences, OpenCV has simple functions to calculate the distortion coefficients. The images below show the detected corners and a sample undistorted image.

<img src="examples/detect_corners.png" width="400" height="225"> <img src="examples/undistorted.png" width="400" height="225">

For #2 & #3, they are hardcoded based on data measured from sample images. In future work, this can be automated by getting data from known calibration test images.

### Lane Line Detection Pipeline
---

The lane line detection module [lane_lines.py](lane_lines.py) consists of the following steps:

- load the camera distortion coefficients
- correct for camera distortion in images or incoming video frames
- transform perspective from forward camera view to top-down/birdseye view
- use color thresholds to find yellow and white lane lines
- optionally, use Sobel gradient thresholds to find lane lines (*this is disabled right now due to excessive noise introduced by the algorithm*)
- combine the thresholded images to create a single binary threshold image with lane pixels
- detect lane pixels (red for left/blue for right) and find left and right polynomials that fit the majority of those lane pixels
- determine curvature of the lane and how far offset from center the vehicle/camera is
- draw lane data and curvature/offset onto picture
- transform perspective back from birdseye view to forward view
- output the image or video

For this test image and output, the intermediate pipeline images, with some debug information, is shown below.

<img src="test_images/test6.jpg" width="400" height="225"> <img src="output_images/test6.jpg" width="400" height="225">
<img src="output_images/pipeline_images.png" width="800">

Here is the output of [lane_lines.py](lane_lines.py) on the sample video. Click on the previews to see the full video.
<a href="project_video.mp4"><img src="examples/project_video.gif"></a> <a href="output_images/project_video.mp4"><img src="examples/project_video_out.gif"></a>

### Discussion
---

Though this pipeline works well for the test images and the first sample video, it is not robust. Here are some issues that will need to be addressed. A lot of these are listed in the TODO comments in the code.

- The birdseye and distance transforms are currently hardcoded. In a proper implementation, this could be dynamic using calibration images, ala the checkerboards for the distortion.
- Color thresholding works well on its own, but will likely fail in poor lighting conditions or when there are white or yellow like colors nearby (e.g., the wheatish grass in the [harder challenge video](harder_challenge_video.mp4)). This will need to be supplemented with something else like Sobel gradient thresholding.
- Sobel gradient thresholding does not work well due to lots of "noise". This may be tuned (see [sobel_tuner.ipynb](sobel_tuner.ipynb)) to work better, but the likely issue is that there will always be too much spurious data that needs to be filtered either before or after the gradient thresholding. I've attempted to do so using some morphology operations (erosion and tophat) but further work is needed here.
- Currently, we speed up the algorithm by searching for lane line pixels only around the previous frame's lane lines if we have found valid lines. This works well for most situations except when there is a sudden change in lane direction which can happen from time to time. The solution to this would be also re-search the entire image for the new lane lines and then do a probabilistic guess on which search is more accurate. However, this increases computation time quite a bit.
- Once the lane line pixels are found, the current algorithm uses a simple 2nd order polynomial fit to determine lane lines. A more robust implementation would be to use the RANSAC (RANdom SAmple Consensus) algorithm to do spline fitting. This would be much better at ignoring outliers and only considering high probability pixels.
- The current algorithm filters out bad lane fits if one side of the lane diverges from the other by too much. This effectively throws out lane merges and exits and continues on the current trajectory which is usually safe but not necessarily optimal. Future work will have to gracefully handle this.
- The current algorithm averages across the last 5 frames equally to smooth out the path. We may want to do a weighted average across more frames, favoring more recent lane data.
- The interactive iPython tuner for Sobel threshold parameters is quite useful but limited and cumbersome. In the future, we could implement a Tkinter-based GUI to tune those and many other parameters and show the results real-time.
- Overall, the OpenCV based implementation will have lots of limits as new scenarios are found. A much better approach would be to create a deep learning neural network to handle these and new situations.
