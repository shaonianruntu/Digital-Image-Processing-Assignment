'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-11-22 21:54:42
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

import os
from os import listdir
import re

import skimage


def is_image_file(file_name):
    return any(
        file_name.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg"])


def is_noise_image_file(file_name, index, noise):
    return any(
        file_name.endswith(extension) for extension in [
            str(index) + " " + noise.title() + ".png",
            str(index) + " " + noise.title() + ".jpg",
            str(index) + " " + noise.title() + ".jpeg"
        ])


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
        self.lap_kernal = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.sobel_kernal_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        self.sobel_kernal_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        self.fangnan_kernal_left = np.array([[3, -2, 0], [-2, 2, -2],
                                             [0, -2, 3]])
        self.fangnan_kernal_right = np.array([[0, -2, 3], [-2, 2, -2],
                                              [3, -2, 0]])

    def Laplacian(self, index):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片", index)
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            c_image = default_loader(image_path)
            image = gray_loader(image_path)

            # [b, g, r] = cv2.split(image)

            # b_lap = signal.convolve2d(b,
            #                           self.lap_kernal,
            #                           boundary="symm",
            #                           mode='same')
            # g_lap = signal.convolve2d(g,
            #                           self.lap_kernal,
            #                           boundary="symm",
            #                           mode='same')
            # r_lap = signal.convolve2d(r,
            #                           self.lap_kernal,
            #                           boundary="symm",
            #                           mode='same')

            # b_lap = np.absolute(b_lap)
            # g_lap = np.absolute(g_lap)
            # r_lap = np.absolute(r_lap)

            lap = signal.convolve2d(image,
                                    self.lap_kernal,
                                    boundary="symm",
                                    mode='same')

            lap = np.absolute(lap)

            # lap = cv2.merge((b_lap, g_lap, r_lap))
            lap = np.array(lap, dtype=np.uint8)
            cv2.imwrite(str(*image_name) + " Lap.png", lap)

            # 与原图做叠加
            stacked_lap = np.stack((lap, ) * 3, axis=-1)
            added_img = cv2.add(c_image, stacked_lap)
            cv2.imwrite(str(*image_name) + " Lap ADD.png", added_img)

    def Sobel(self, index, dir="x"):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片", index)
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            c_image = default_loader(image_path)
            image = gray_loader(image_path)

            # [b, g, r] = cv2.split(image)

            if (dir == "x"):
                kernal = self.sobel_kernal_x
            elif (dir == "y"):
                kernal = self.sobel_kernal_y

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
            cv2.imwrite(str(*image_name) + " " + str(dir) + " Sobel.png", lap)

            # 与原图做叠加
            stacked_lap = np.stack((lap, ) * 3, axis=-1)
            added_img = cv2.add(c_image, stacked_lap)
            cv2.imwrite(
                str(*image_name) + " " + str(dir) + " Sobel ADD.png",
                added_img)

    def fangnan(self, index, fdir="left"):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片", index)
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            c_image = default_loader(image_path)
            image = gray_loader(image_path)

            if (fdir == "left"):
                kernal = self.fangnan_kernal_left
            elif (fdir == "right"):
                kernal = self.fangnan_kernal_right

            lap = signal.convolve2d(image,
                                    kernal,
                                    boundary="symm",
                                    mode='same')

            lap = np.absolute(lap)

            lap = np.array(lap, dtype=np.uint8)
            cv2.imwrite(
                str(*image_name) + " " + str(fdir) + " Fangnan.png", lap)

            # 与原图做叠加
            stacked_lap = np.stack((lap, ) * 3, axis=-1)
            added_img = cv2.add(c_image, stacked_lap)
            cv2.imwrite(
                str(*image_name) + " " + str(fdir) + " Fangnan ADD.png",
                added_img)
