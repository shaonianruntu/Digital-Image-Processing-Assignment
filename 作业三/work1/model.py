'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-12-02 16:41:47
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
        for extension in [".png", ".jpg", ".jpeg", '.PNG', '.JPG', '.JPEG'])


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

        self.sobel_kernal_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 2, 1]])
        self.sobel_kernal_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.prewitt_kernal_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.prewitt_kernal_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        self.roberts_kernal_x = np.array([[0, -1], [1, 0]])
        self.roberts_kernal_y = np.array([[-1, 0], [0, 1]])

        self.laplacian_kernal = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    def edge_detection(self, index, modal="sobel", dir="x"):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片", index)
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            c_image = default_loader(image_path)
            image = gray_loader(image_path)

            # [b, g, r] = cv2.split(image)

            if (modal == "sobel"):
                if (dir == "x"):
                    kernal = self.sobel_kernal_x
                elif (dir == "y"):
                    kernal = self.sobel_kernal_y
            elif (modal == "prewitt"):
                if (dir == "x"):
                    kernal = self.prewitt_kernal_x
                elif (dir == "y"):
                    kernal = self.prewitt_kernal_y
            elif (modal == "roberts"):
                if (dir == "x"):
                    kernal = self.roberts_kernal_x
                elif (dir == "y"):
                    kernal = self.roberts_kernal_y
            elif (modal == "laplacian"):
                kernal = self.laplacian_kernal

            # b_lap = signal.convolve2d(b, kernal, boundary="symm", mode='same')
            # g_lap = signal.convolve2d(g, kernal, boundary="symm", mode='same')
            # r_lap = signal.convolve2d(r, kernal, boundary="symm", mode='same')

            # b_lap = np.absolute(b_lap)
            # g_lap = np.absolute(g_lap)
            # r_lap = np.absolute(r_lap)

            lap = signal.convolve2d(image,
                                    kernal,
                                    boundary="symm",
                                    mode='same')

            lap = np.absolute(lap)

            # lap = cv2.merge((b_lap, g_lap, r_lap))
            lap = np.array(lap, dtype=np.uint8)
            if (modal == "laplacian"):
                cv2.imwrite(
                    str(*image_name) + " " + str(modal.title()) + ".png", lap)
            else:
                cv2.imwrite(
                    str(*image_name) + " " + str(dir) + " " +
                    str(modal.title()) + ".png", lap)
