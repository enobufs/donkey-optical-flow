"""
test2.py
"""

import cv2 as cv
import numpy as np

WIDTH = 160         # With of input image
HEIGHT = 120        # Height of input image

INPUT_PATH="dkcar.mp4"

CAM = cv.VideoCapture(INPUT_PATH)
UBASE_WIDTH = 60    # Upper-base width
LBASE_WIDTH = 320   # Lower-base width
UOFFSET = 45        # Upper-base margin
LOFFSET = 20        # Lower-base margin
MAX_ACC = 0.2       # Max possible acceleration

SRC_UL = [(WIDTH - UBASE_WIDTH) / 2, UOFFSET]
SRC_LL = [(WIDTH - LBASE_WIDTH) / 2, HEIGHT - LOFFSET]
SRC_UR = [(WIDTH + UBASE_WIDTH) / 2, UOFFSET]
SRC_LR = [(WIDTH + LBASE_WIDTH) / 2, HEIGHT - LOFFSET]

DST_UL = [0, 0]
DST_LL = [0, HEIGHT]
DST_UR = [WIDTH, 0]
DST_LR = [WIDTH, HEIGHT]

VELOCITY_CUTOFF_PCT = 67

def make_velocity_detector():
    """Velocity detector factory."""

    pts1 = np.float32([SRC_UL, SRC_LL, SRC_UR, SRC_LR])
    pts2 = np.float32([DST_UL, DST_LL, DST_UR, DST_LR])

    M = cv.getPerspectiveTransform(pts1, pts2)

    prev = None
    v_last = 0.0

    def detect_velocity(image):
        """Detect velocity from images"""
        nonlocal prev, v_last
        curr_bgr = cv.warpPerspective(image, M, (160, 120))
        curr = cv.cvtColor(curr_bgr, cv.COLOR_BGR2GRAY)

        if prev is None:
            prev = curr
            v_last = 0.0
            return v_last, curr_bgr, np.zeros_like(image)

        flow = cv.calcOpticalFlowFarneback(
            prev,   # Previous image
            curr,   # Current image
            None,   # Computed flow image that has the same size oas prev and type CV_32FC2.
            0.5,    # Specifies the image scale (<1) to build pyramids for each image.
            3,      # Number of pyramid layers including the initial image.
            15,     # winsize, averaging windows size.
            3,      # iterations, number of iterations the algorithm does at each pyramid level.
            5,      # standard deviation of the Gaussian that is used to smooth derivative
            1.5,
            0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        v = mag * np.sin(ang)

        ######################
        ## Histogram for mag
        ar = np.arange(-20.0, 20.0, 0.50, dtype=np.float)
        his = np.histogram(v, bins=ar)

        for i, n in enumerate(his[0]):
            bgr = (255, 255, 0)
            if his[1][i] < 0:
                bgr = (0, 255, 255)

            #print('[{}] {} - {}'.format(i, n, his[1][i]))
            cv.rectangle(   image, #curr_bgr,
                            (i*2, HEIGHT),
                            (i*2, HEIGHT - int(n / 10)),
                            bgr, #(0, 255, 255),
                            cv.FILLED)

        hsv = np.zeros_like(image)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(np.abs(v), None, 0, 255, cv.NORM_MINMAX)
        hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        ##
        ######################

        v_abs = np.absolute(v)
        v = v[v_abs >= np.percentile(v_abs, VELOCITY_CUTOFF_PCT)]

        v_max = v_last + MAX_ACC
        v_min = v_last - MAX_ACC
        v = np.clip(v, v_min, v_max)
        if v.size > 0:
            v_avg = v.mean()
        else:
            if v_last > 0:
                v_avg = max(v_last - MAX_ACC, 0)
            elif v_last < 0:
                v_avg = min(v_last + MAX_ACC, 0)
            else:
                v_avg = 0

        prev = curr
        v_last = v_avg
        return v_last, curr_bgr, hsv_bgr

    return detect_velocity

get_velocity = make_velocity_detector()

while 1:

    ret, image = CAM.read()
    if not ret:
        break

    velocity, top_view, hsv_bgr = get_velocity(image)
    print('velocity:', velocity)

    if velocity >= 0:
        cv.rectangle(   image,
                        (80, 0),
                        (int(velocity * 8) + 80, 4),
                        (0, 0, 255),
                        cv.FILLED)
    else:
        cv.rectangle(   image,
                        (80, 0),
                        (int(velocity * 8) + 80, 4),
                        (255, 0, 0),
                        cv.FILLED)

    vis = np.concatenate((image, top_view, hsv_bgr), axis=1)
    cv.imshow('Velocity Detection using Optical Flow', vis)

    k = cv.waitKey(30) & 0xff
    if k == 27: # if ESC
        break


CAM.release()
cv.destroyAllWindows()
