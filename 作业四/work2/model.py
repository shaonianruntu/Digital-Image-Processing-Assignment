'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-12-12 21:44:40
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

import os
from os import listdir
import re
from itertools import groupby
import sys


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


def ycrcb_loader(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)


class RLE:
    """
    行程编码
    """
    def matrix2list(self, matirx):
        """ 
        按照行程编码样式将2维数组展开为一维数组 
        """
        mrows, mcols = matirx.shape[:2]
        mrows -= 1
        mcols -= 1
        mlen = min(mrows, mcols)

        rmatrix = []
        rmatrix.append(matirx[0][0])

        rmatrix.extend(self.first_encode(matirx, mlen))

        if mcols > mrows:
            rmatrix.extend(
                self.colmore_middle_encode(matirx, mlen, mcols, mrows))
            rmatrix.extend(self.colmore_last_encode(matirx, mlen, mcols,
                                                    mrows))
        else:
            rmatrix.extend(
                self.rowmore_middle_encode(matirx, mlen, mcols, mrows))
            rmatrix.extend(self.rowmore_last_encode(matirx, mlen, mcols,
                                                    mrows))

        rmatrix.append(matirx[-1][-1])

        return rmatrix

    def first_encode(self, matirx, mlen):
        rmatrix = []
        for len in range(1, mlen + 1):
            if (len % 2 == 1):
                for i in range(0, len + 1):
                    rmatrix.append(matirx[i][len - i])
            else:
                for i in range(0, len + 1):
                    rmatrix.append(matirx[len - i][i])
        return rmatrix

    def colmore_middle_encode(self, matirx, mlen, mcols, mrows):
        rmatrix = []
        if mlen % 2 == 0:
            for extra in range(mcols - mrows):
                if extra % 2 == 0:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[i][mlen - i + extra + 1])
                else:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[mlen - i][i + extra + 1])
        else:
            for extra in range(mcols - mrows):
                if extra % 2 == 1:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[i][mlen - i + extra + 1])
                else:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[mlen - i][i + extra + 1])
        return rmatrix

    def colmore_last_encode(self, matirx, mlen, mcols, mrows):
        rmatrix = []
        if mcols % 2 == 0:
            for len in range(0, mlen - 1):
                if len % 2 == 0:
                    for i in range(mlen - len):
                        rmatrix.append(
                            matirx[mlen - (mlen - 1 - len - i)][mlen - i +
                                                                mcols - mrows])
                else:
                    for i in range(mlen - len):
                        rmatrix.append(matirx[mlen -
                                              i][mlen - (mlen - 1 - len - i) +
                                                 mcols - mrows])
        else:
            for len in range(0, mlen - 1):
                if len % 2 == 1:
                    for i in range(mlen - len):
                        rmatrix.append(
                            matirx[mlen - (mlen - 1 - len - i)][mlen - i +
                                                                mcols - mrows])
                else:
                    for i in range(mlen - len):
                        rmatrix.append(matirx[mlen -
                                              i][mlen - (mlen - 1 - len - i) +
                                                 mcols - mrows])
        return rmatrix

    def rowmore_middle_encode(self, matirx, mlen, mcols, mrows):
        rmatrix = []
        if mlen % 2 == 0:
            for extra in range(mrows - mcols):
                if extra % 2 == 1:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[mlen - i + extra + 1][i])
                else:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[i + extra + 1][mlen - i])
        else:
            for extra in range(mrows - mcols):
                if extra % 2 == 0:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[mlen - i + extra + 1][i])
                else:
                    for i in range(mlen + 1):
                        rmatrix.append(matirx[i + extra + 1][mlen - i])

        return rmatrix

    def rowmore_last_encode(self, matirx, mlen, mcols, mrows):
        rmatrix = []
        if mrows % 2 == 0:
            for len in range(0, mlen - 1):
                if len % 2 == 0:
                    for i in range(mlen - len):
                        rmatrix.append(matirx[mlen - (mlen - 1 - len - i) +
                                              mrows - mcols][mlen - i])
                else:
                    for i in range(mlen - len):
                        rmatrix.append(matirx[mlen - i + mrows -
                                              mcols][mlen -
                                                     (mlen - 1 - len - i)])
        else:
            for len in range(0, mlen - 1):
                if len % 2 == 1:
                    for i in range(mlen - len):
                        rmatrix.append(matirx[mlen - (mlen - 1 - len - i) +
                                              mrows - mcols][mlen - i])
                else:
                    for i in range(mlen - len):
                        rmatrix.append(matirx[mlen - i + mrows -
                                              mcols][mlen -
                                                     (mlen - 1 - len - i)])
        return rmatrix

    def encode(self, lst):
        lst_encode = np.array([(len(list(group)), name)
                               for name, group in groupby(lst)])
        return lst_encode.flatten()

    def decode(self, lst_encode):
        lst = []
        for i in range(0, len(lst_encode), 2):
            print(lst_encode[i])
            length = int(lst_encode[i])
            for j in range(length):
                lst.append(lst_encode[i + 1])
        return lst

    def compress(self, image):
        return self.encode(self.matrix2list(image)).astype(np.uint8)


class JPEGEncode:
    """
    JPEG 格式压缩
    """
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def compress(self, index, q_factor):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])

            # Step 1: convert rgb image space tp YCrCb space
            image = ycrcb_loader(image_path)

            # 图像尺寸调整，以适应分块
            height, width = image.shape[:2]
            if height % 8 != 0 or width % 8 != 0:
                image = np.pad(image, ((0, (8 - height % 8) % 8),
                                       (0, (8 - width % 8) % 8), (0, 0)),
                               "edge")
            height, width = image.shape[:2]
            size = sys.getsizeof((image.flatten()))

            print("Image {}:".format(index))
            print("Origin Image's Size is {:.2f} KB.".format(size / 1024))

            [y, cr, cb] = cv2.split(image)

            # Step 2: DCT decomposition, transform from time-domain to
            # frequency-domain, and choose 8*8 block
            image_dct = []

            for img in [y, cr, cb]:
                f_patches = []
                fi_patches = []

                # 图像分块
                h_patches = np.vsplit(img, height // 8)

                for i in range(height // 8):
                    wh_patches = np.hsplit(h_patches[i], width // 8)

                    f_patch = []
                    fi_patch = []
                    for j in range(width // 8):
                        # DCT 变换
                        patch_dct = cv2.dct(wh_patches[j].astype(np.float))
                        f_patch.append(patch_dct)

                    f_patchs = np.hstack(f_patch)
                    f_patches.append(f_patchs)

                img_dct = np.vstack(f_patches)
                image_dct.append(img_dct)

            image_dct = np.moveaxis(image_dct, 0, 2)

            # Step 3: 量化
            image_dct = np.around(image_dct / q_factor)

            # Step 4: 行程编码，转换为一维数组
            rle = RLE()
            [d_y, d_cr, d_cb] = cv2.split(image_dct)
            image_rle = []
            for dct in [d_y, d_cr, d_cb]:
                dct_rle = rle.compress(dct)
                image_rle.append(dct_rle)

            # 图像大小计算，压缩比计算
            r_size = sys.getsizeof((image_rle))
            print("After Run JPEG Compress Image's Size is  {:.2f} KB.\
                    \nCompressed Image's size is {:.2%} of Origin Image.".
                  format(r_size / 1024, r_size / size))

            # Step 5: Huffman Transformation
            ## 不想写了，Step 4 和 Step 5 直接省略，
            ## 如果需要实现，可以参考 作业四的 work1，
            ## 直接在量化之后进行 IDCT 还原

            image_iq = image_dct * q_factor

            [r_y, r_cr, r_cb] = cv2.split(image_iq)

            image_back = []

            for img in [r_y, r_cr, r_cb]:
                f_patches = []

                # 图像分块
                h_patches = np.vsplit(img, height // 8)

                for i in range(height // 8):
                    wh_patches = np.hsplit(h_patches[i], width // 8)

                    f_patch = []
                    fi_patch = []
                    for j in range(width // 8):
                        # IDCT 变换
                        patch_dct = cv2.idct(wh_patches[j].astype(np.float))
                        f_patch.append(patch_dct)

                    f_patchs = np.hstack(f_patch)
                    f_patches.append(f_patchs)

                img_back = np.vstack(f_patches).astype(np.uint8)
                image_back.append(img_back)

            image_back = np.moveaxis(image_back, 0, 2)

            # YCrCb 空间转换回 RGB 空间 
            image_back = cv2.cvtColor(image_back, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite(
                str(*image_name) + " " + str(q_factor) + " IDCT.png",
                image_back)

            # 计算均方根误差
            mse = ((image - image_back)**2).mean()
            print("Compressed Image's MSE is {:.2f}".format(mse))

            print()
