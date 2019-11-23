'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-11-23 17:26:05
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

import os
from os import listdir
import re


def is_image_file(file_name):
    return any(
        file_name.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def gray_loader(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def hsv_loader(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


class Dataset():
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def otsu_adaptive(self, index):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片", index)
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            c_image = default_loader(image_path)
            image = gray_loader(image_path)

            otsu = cv2.adaptiveThreshold(image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

            cv2.imwrite(str(*image_name) + " Otsu.png", otsu)

    def otsu_histogram(self, index):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片", index)
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            c_image = default_loader(image_path)
            image = gray_loader(image_path)

            plt.hist(image.ravel(), 256, color="black")
            plt.title("Otsu Gray Scale Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " Otsu Gray Scale Histogram.png")

            # 使用 Otsu 算法自动求解双峰中的谷底
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.max()
            Q = hist_norm.cumsum()

            bins = np.arange(256)

            fn_min = np.inf
            thresh = -1

            for i in range(1, 256):
                p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
                q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
                b1, b2 = np.hsplit(bins, [i])  # weights

                # finding means and variances
                m1, m2 = np.sum(p1 * b1) / (q1 + 1), np.sum(p2 * b2) / (q2 + 1)
                v1, v2 = np.sum(((b1 - m1)**2) * p1) / (q1 + 1), np.sum(
                    ((b2 - m2)**2) * p2) / (q2 + 1)

                # calculates the minimization function
                fn = v1 * q1 + v2 * q2
                if fn < fn_min:
                    fn_min = fn
                    thresh = i

            # find otsu's threshold value with OpenCV function
            ret, otsu = cv2.threshold(image, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(
                str(*image_name) + " ret " + str(int(ret)) + " Otsu.png", otsu)
