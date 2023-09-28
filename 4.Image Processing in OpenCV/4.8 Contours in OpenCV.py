import cv2 as cv
import matplotlib
import numpy as np
import sys
from matplotlib import pyplot as plt
import inspect
from PySide6.QtWidgets import QApplication, QWidget

matplotlib.use("QtAgg", force=True)
plt.ion()


def contours_in_OpenCV():
    img = cv.imread('../data/white_rect_in_black.png')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    outline = img.copy()
    cv.drawContours(outline, contours, -1, (0, 255, 0), 2)

    # cv.drawContours(img, contours, 3, (0, 255, 0), 3)
    # cnt = contours[4]
    # cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    plt.figure(1)
    plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('img'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(cv.cvtColor(outline, cv.COLOR_BGR2RGB))
    plt.title('outline'), plt.xticks([]), plt.yticks([])

    plt.show()


def moments(pic):  # ../data/white_rect_in_black.png #../4.8.0/star_filled@2x.png
    img = cv.imread(pic, 0)
    img_clor = cv.imread(pic, 1)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    outline = img_clor.copy()
    cv.drawContours(outline, contours, -1, (0, 255, 0), 2)

    cnt = contours[0]
    M = cv.moments(cnt)
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print(cx)
    print(cy)

    # 2. 轮廓面积
    area = cv.contourArea(cnt)
    print("area: ", area)

    # 3. 轮廓周长
    perimeter = cv.arcLength(cnt, True)
    print("perimeter: ", perimeter)

    # 4. 轮廓近似 根据我们指定的精度，它可以将轮廓形状近似为顶点数量较少的其他形状。它是Douglas-Peucker算法的实现
    epsilon = 0.1 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    print("epsilon: ", epsilon)
    print("approx: ", approx)

    plt.figure(2)
    plt.subplot(331), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('img'), plt.xticks([]), plt.yticks([])

    plt.subplot(332), plt.imshow(cv.cvtColor(thresh, cv.COLOR_BGR2RGB))
    plt.title('thresh'), plt.xticks([]), plt.yticks([])

    plt.subplot(333), plt.imshow(cv.cvtColor(outline, cv.COLOR_BGR2RGB))
    plt.title('outline'), plt.xticks([]), plt.yticks([])

    outline4 = img_clor.copy()
    cv.drawContours(outline4, approx, -1, (0, 255, 0), 20, cv.LINE_8)
    cv.drawContours(outline4, [approx], -1, (0, 255, 0), 2, cv.LINE_8)

    plt.subplot(334), plt.imshow(cv.cvtColor(outline4, cv.COLOR_BGR2RGB))
    plt.title('approx'), plt.xticks([]), plt.yticks([])


# 5. Convex Hull
def convex_hull():
    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)

    contours_in_OpenCV()
    moments('../4.8.0/electric.jpg')
    convex_hull()

    sys.exit(app.exec())
