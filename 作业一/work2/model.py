'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-11-24 13:49:07
'''
import numpy as np
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


def hsv_loader(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


class Dataset():
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def add_noise(
            self,
            index,
            noise="gaussian",
    ):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行添加噪声处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            image = default_loader(image_path)

            noise_image = skimage.util.random_noise(image, mode=noise) * 256

            cv2.imwrite(
                str(*image_name) + " " + noise.title() + ".png", noise_image)

    def delete_noise(self,
                     index,
                     noise="gaussian",
                     modal="mean",
                     kernal_size=3):
        self.local_image_list = [
            x for x in listdir(".") if is_noise_image_file(x, index, noise)
        ]
        for image in self.local_image_list:
            image_name = re.findall(r'(.+?)\.', image)
            img = default_loader(image)

            if (modal == "mean"):
                image_blur = cv2.blur(img, (kernal_size, kernal_size))
                cv2.imwrite(
                    str(image_name[0]) + " " + str(modal.title()) +
                    str(kernal_size) + ".png", image_blur)
            elif (modal == "middle"):
                image_blur = cv2.medianBlur(img, kernal_size)
                cv2.imwrite(
                    str(image_name[0]) + " " + str(modal.title()) +
                    str(kernal_size) + ".png", image_blur)
