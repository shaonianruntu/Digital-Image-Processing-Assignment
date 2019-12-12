'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-12-02 16:41:29
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

    def fft_lowpass(self, index, radius=250):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            image = default_loader(image_path)

            [b, g, r] = cv2.split(image)

            # 使用快速傅里叶变换求频域
            r_f = np.fft.fft2(r)
            g_f = np.fft.fft2(g)
            b_f = np.fft.fft2(b)
            # 将低频信息移到图像的中间位置
            r_fshift = np.fft.fftshift(r_f)
            g_fshift = np.fft.fftshift(g_f)
            b_fshift = np.fft.fftshift(b_f)

            # 创建蒙版
            rows, cols, _ = image.shape
            crow, ccol = rows // 2, cols // 2
            if radius > crow or radius > ccol:
                radius = min(crow, ccol)
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

            r_ffshift = r_fshift * mask
            g_ffshift = g_fshift * mask
            b_ffshift = b_fshift * mask

            ffshift = cv2.add(r_ffshift, g_ffshift, b_ffshift)

            # 构建振幅图
            r_magnitude_spectrum = 20 * np.log(np.abs(r_ffshift) + 1) / 3
            g_magnitude_spectrum = 20 * np.log(np.abs(g_ffshift) + 1) / 3
            b_magnitude_spectrum = 20 * np.log(np.abs(b_ffshift) + 1) / 3

            magnitude_spectrum = cv2.add(b_magnitude_spectrum,
                                         g_magnitude_spectrum,
                                         r_magnitude_spectrum)

            cv2.imwrite(
                str(*image_name) + " " + str(radius) + " Lowpass (FD).png",
                magnitude_spectrum)

            # 逆向傅里叶变换
            r_fishift = np.fft.ifftshift(r_ffshift)
            g_fishift = np.fft.ifftshift(g_ffshift)
            b_fishift = np.fft.ifftshift(b_ffshift)

            r_i = np.fft.ifft2(r_fishift)
            g_i = np.fft.ifft2(g_fishift)
            b_i = np.fft.ifft2(b_fishift)

            r_i = np.abs(r_i)
            g_i = np.abs(g_i)
            b_i = np.abs(b_i)

            image_back = cv2.merge((b_i, g_i, r_i))

            cv2.imwrite(
                str(*image_name) + " " + str(radius) + " Lowpass.png",
                image_back)

    def fft_highpass(self, index, radius=50):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])
            image = default_loader(image_path)

            [b, g, r] = cv2.split(image)

            # 使用快速傅里叶变换求频域
            r_f = np.fft.fft2(r)
            g_f = np.fft.fft2(g)
            b_f = np.fft.fft2(b)
            # 将低频信息移到图像的中间位置
            r_fshift = np.fft.fftshift(r_f)
            g_fshift = np.fft.fftshift(g_f)
            b_fshift = np.fft.fftshift(b_f)

            # 创建蒙版
            rows, cols, _ = image.shape
            crow, ccol = rows // 2, cols // 2
            if radius > crow or radius > ccol:
                radius = min(crow, ccol)
            mask = np.ones((rows, cols), np.uint8)
            mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

            r_ffshift = r_fshift * mask
            g_ffshift = g_fshift * mask
            b_ffshift = b_fshift * mask

            # 构建振幅图
            r_magnitude_spectrum = 20 * np.log(np.abs(r_ffshift) + 1) / 3
            g_magnitude_spectrum = 20 * np.log(np.abs(g_ffshift) + 1) / 3
            b_magnitude_spectrum = 20 * np.log(np.abs(b_ffshift) + 1) / 3

            magnitude_spectrum = cv2.add(b_magnitude_spectrum,
                                         g_magnitude_spectrum,
                                         r_magnitude_spectrum)

            cv2.imwrite(
                str(*image_name) + " " + str(radius) + " Highpass (FD).png",
                magnitude_spectrum)

            # 逆向傅里叶变换
            r_fishift = np.fft.ifftshift(r_ffshift)
            g_fishift = np.fft.ifftshift(g_ffshift)
            b_fishift = np.fft.ifftshift(b_ffshift)

            r_i = np.fft.ifft2(r_fishift)
            g_i = np.fft.ifft2(g_fishift)
            b_i = np.fft.ifft2(b_fishift)

            r_i = np.abs(r_i)
            g_i = np.abs(g_i)
            b_i = np.abs(b_i)

            image_back = cv2.merge((b_i, g_i, r_i))

            cv2.imwrite(
                str(*image_name) + " " + str(radius) + " Highpass.png",
                image_back)
