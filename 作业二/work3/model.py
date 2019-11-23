'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-11-23 16:06:18
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

    def dct8(self, index, color="rgb"):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])

            if (color == "rgb"):
                image = default_loader(image_path)

                height, width = image.shape[:2]

                if height % 8 != 0 or width % 8 != 0:
                    image = np.pad(image, ((0, (8 - height % 8) % 8),
                                           (0, (8 - width % 8) % 8), (0, 0)),
                                   "edge")

                height, width = image.shape[:2]

                [b, g, r] = cv2.split(image)

                image_dct = []
                image_back = []

                for img in [b, g, r]:

                    f_patches = []
                    fi_patches = []

                    h_patches = np.vsplit(img, height // 8)

                    for i in range(height // 8):
                        wh_patches = np.hsplit(h_patches[i], width // 8)

                        f_patch = []
                        fi_patch = []
                        for j in range(width // 8):
                            # DCT 变换
                            patch_dct = cv2.dct(wh_patches[j].astype(np.float))
                            # IDCT 变换
                            patch_idct = cv2.idct(patch_dct)

                            f_patch.append(patch_dct)
                            fi_patch.append(patch_idct)

                        f_patchs = np.hstack(f_patch)
                        f_patches.append(f_patchs)

                        fi_patchs = np.hstack(fi_patch)
                        fi_patches.append(fi_patchs)

                    img_dct = np.vstack(f_patches)
                    img_back = np.vstack(fi_patches).astype(np.uint8)

                    image_dct.append(img_dct)
                    image_back.append(img_back)

                image_dct = np.moveaxis(image_dct, 0, 2)
                image_back = np.moveaxis(image_back, 0, 2)

                cv2.imwrite(str(*image_name) + " DCT.png", image_dct)
                cv2.imwrite(str(*image_name) + " IDCT.png", image_back)

            elif (color == "gray"):

                image = gray_loader(image_path)

                height, width = image.shape[:2]

                if height % 8 != 0 or width % 8 != 0:
                    image = np.pad(image, ((0, (8 - height % 8) % 8),
                                           (0, (8 - width % 8) % 8)), "edge")

                height, width = image.shape[:2]

                f_patches = []
                fi_patches = []

                h_patches = np.vsplit(image, height // 8)

                for i in range(height // 8):
                    wh_patches = np.hsplit(h_patches[i], width // 8)

                    f_patch = []
                    fi_patch = []
                    for j in range(width // 8):
                        # DCT 变换
                        patch_dct = cv2.dct(wh_patches[j].astype(np.float))
                        # IDCT 变换
                        patch_idct = cv2.idct(patch_dct)

                        f_patch.append(patch_dct)
                        fi_patch.append(patch_idct)

                    f_patchs = np.hstack(f_patch)
                    f_patches.append(f_patchs)

                    fi_patchs = np.hstack(fi_patch)
                    fi_patches.append(fi_patchs)

                image_dct = np.vstack(f_patches)
                image_back = np.vstack(fi_patches).astype(np.uint8)

                cv2.imwrite(str(*image_name) + " DCT.png", image_dct)
                cv2.imwrite(str(*image_name) + " IDCT.png", image_back)