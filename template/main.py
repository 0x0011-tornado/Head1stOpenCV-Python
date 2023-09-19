import cv2
import numpy
def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')
    img = cv2.imread("github_icon.png")
    cv2.imshow("WindowTitle", img)
    cv2.waitKey(0)
