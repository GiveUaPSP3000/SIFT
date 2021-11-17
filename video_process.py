# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema

# Setting fixed threshold criteria
USE_THRESH = False
# fixed threshold value
THRESH = 0.6
# Setting fixed threshold criteria
USE_TOP_ORDER = False
# Setting local maxima criteria
USE_LOCAL_MAXIMA = True
# Number of top sorted frames
NUM_TOP_FRAMES = 20
# smoothing window size
len_window = int(25)


def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    """
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


# Class to hold information about each frame

class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
    if (max(a, b) != 0):
        x = (b - a) / max(a, b)
    else:
        return 0
    return x


def write_frames(filename):
    save_name = []
    file_name = filename.split('.')[0]
    if not os.path.exists('images/' + file_name):
        os.mkdir('images/' + file_name)

    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("value"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            name = "frame_" + str(keyframe.id) + ".jpg"
            cv2.imwrite("images/" + file_name + '/' + name, keyframe.frame)

    if USE_THRESH:
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].value), np.float(frames[i].value)) >= THRESH):
                name = "frame_" + str(frames[i].id) + ".jpg"
                cv2.imwrite("images/" + file_name + '/' + name, frames[i].frame)

    if USE_LOCAL_MAXIMA:
        # frame_diffs记录每两帧之间差异的和
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            name = "frame_" + str(frames[i - 1].id) + ".jpg"
            save_name.append(name)
            cv2.imwrite("images/" + file_name + '/' + name, frames[i - 1].frame)
    return save_name

frame_diffs = []
frames = []


def all_path(videopath):
    cap = cv2.VideoCapture('videos/' + str(videopath))

    curr_frame = None
    prev_frame = None

    # read a frame in the video
    ret, frame = cap.read()
    i = 1

    while (ret):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # 找寻前后两帧图的差异
            diff = cv2.absdiff(curr_frame, prev_frame)
            count = np.sum(diff)
            frame_diffs.append(count)
            frame = Frame(i, frame, count)
            frames.append(frame)

        prev_frame = curr_frame
        i = i + 1
        # 读下一张图片
        ret, frame = cap.read()

    save_files = write_frames(videopath)
    cap.release()
    frame_diffs.clear()
    frames.clear()
    return save_files


def image_de(videopath, images):
    file_name = videopath.split('.')[0]
    image_clear = []
    for f in images:
        real_name = 'images/' + file_name + '/' + f
        imag = cv2.imread(real_name)
        grayImag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        # get the canny value
        canny = cv2.Canny(grayImag, 200, 200)
        value = canny.var()

        # get the Laplacian value
        lapla = cv2.Laplacian(grayImag, cv2.CV_8U)
        imageVar = lapla.var()
        if value >= 400 or imageVar >= 100:
            image_clear.append(real_name)
    return image_clear
