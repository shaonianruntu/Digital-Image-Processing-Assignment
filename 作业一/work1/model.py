'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-11-22 19:54:14
'''
import numpy as np
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


def hsv_loader(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


class Dataset():
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def rgb_equalization(self, index):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            image = default_loader(image_path)

            [b, g, r] = cv2.split(image)

            fig = plt.figure(figsize=(10, 6), num="Red Color Histogram")
            plt.hist(r.flatten(), 256, [0, 256], color="red")
            plt.title("Red Color Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " Red Color Histogram.png")

            fig = plt.figure(figsize=(10, 6), num="Green Color Histogram")
            plt.hist(g.flatten(), 256, [0, 256], color="green")
            plt.title("Green Color Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " Green Color Histogram.png")

            fig = plt.figure(figsize=(10, 6), num="Blue Color Histogram")
            plt.hist(b.flatten(), 256, [0, 256], color="blue")
            plt.title("Blue Color Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " Blue Color Histogram.png")

            # 直方图均衡化处理
            equ_r = cv2.equalizeHist(r)
            equ_g = cv2.equalizeHist(g)
            equ_b = cv2.equalizeHist(b)

            fig = plt.figure(figsize=(10, 6),
                             num="Red Color Histogram (Equed)")
            plt.hist(r.flatten(), 256, [0, 256], color="red")
            plt.title("Red Color Histogram (Equed)", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " Red Color Histogram (Equed).png")

            fig = plt.figure(figsize=(10, 6),
                             num="Green Color Histogram (Equed)")
            plt.hist(g.flatten(), 256, [0, 256], color="green")
            plt.title("Green Color Histogram (Equed)", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(
                str(*image_name) + " Green Color Histogram (Equed).png")

            fig = plt.figure(figsize=(10, 6),
                             num="Blue Color Histogram (Equed)")
            plt.hist(b.flatten(), 256, [0, 256], color="blue")
            plt.title("Blue Color Histogram (Equed)", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " Blue Color Histogram (Equed).png")

            equ = cv2.merge((equ_b, equ_g, equ_r))
            cv2.imwrite(str(*image_name) + " RGB Equ.png", equ)

    def hsv_equalization(self, index):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行HSV处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            image = default_loader(image_path)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            [h, s, v] = cv2.split(hsv)

            fig = plt.figure(figsize=(10, 6), num="HSV Hue Histogram")
            plt.hist(h.flatten(), 256, [0, 256], color="blue")
            plt.title("HSV Hue Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " HSV Hue Histogram.png")

            fig = plt.figure(figsize=(10, 6), num="HSV Saturation Histogram")
            plt.hist(s.flatten(), 256, [0, 256], color="green")
            plt.title("HSV Saturation Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " HSV Saturation Histogram.png")

            fig = plt.figure(figsize=(10, 6), num="HSV Value Histogram")
            plt.hist(v.flatten(), 256, [0, 256], color="red")
            plt.title("HSV Value Histogram", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " HSV Value Histogram.png")

            # 直方图均衡化处理
            equ_h = cv2.equalizeHist(h)
            equ_s = cv2.equalizeHist(s)
            equ_v = cv2.equalizeHist(v)

            fig = plt.figure(figsize=(10, 6), num="HSV Hue Histogram (Equed)")
            plt.hist(equ_h.flatten(), 256, [0, 256], color="blue")
            plt.title("HSV Hue Histogram (Equed)", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " HSV Hue Histogram (Equed).png")

            fig = plt.figure(figsize=(10, 6),
                             num="HSV Saturation Histogram (Equed)")
            plt.hist(equ_s.flatten(), 256, [0, 256], color="green")
            plt.title("HSV Saturation Histogram (Equed)", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(
                str(*image_name) + " HSV Saturation Histogram (Equed).png")

            fig = plt.figure(figsize=(10, 6),
                             num="HSV Value Histogram (Equed)")
            plt.hist(equ_v.flatten(), 256, [0, 256], color="red")
            plt.title("HSV Value Histogram (Equed)", fontsize=24)
            plt.xlim([0, 256])
            plt.tick_params(labelsize=14)
            plt.savefig(str(*image_name) + " HSV Value Histogram (Equed).png")

            equ = cv2.merge((equ_h, equ_s, equ_v))
            cv2.imwrite(str(*image_name) + " HSV Equ.png", equ)