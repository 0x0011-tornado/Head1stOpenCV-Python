from __future__ import print_function

import cv2 as cv
import matplotlib
import numpy as np
import sys
from matplotlib import pyplot as plt
import inspect
from PySide6.QtWidgets import QApplication, QWidget
import argparse

matplotlib.use("QtAgg", force=True)
plt.ion()


def function():
    current_function_name = inspect.currentframe().f_code.co_name
    print("current_function_name:", current_function_name)
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()
    if args.algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


if __name__ == "__main__":
    app = QApplication(sys.argv)

    function()

    sys.exit(app.exec())
