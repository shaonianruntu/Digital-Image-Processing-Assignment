'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-12-01 17:21:04
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
        for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def gray_loader(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def hsv_loader(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Run Length Encoding 行程编码
class RLE():
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def matrix2list(self, matirx):
        """ 按照行程编码样式将2维数组展开为一维数组 """
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

    def test(self):
        img3 = np.array(range(1, 9 + 1)).reshape(3, 3)
        img5 = np.array(range(1, 25 + 1)).reshape(5, 5)
        img4 = np.array(range(1, 16 + 1)).reshape(4, 4)

        img = np.array([[5, 4, 1, 2], [4, 3, 2, 1], [3, 3, 2, 1], [2, 3, 1,
                                                                   0]])

        col_img46 = np.array(range(1, 24 + 1)).reshape(4, 6)
        col_img45 = np.array(range(1, 20 + 1)).reshape(4, 5)
        col_img35 = np.array(range(1, 15 + 1)).reshape(3, 5)

        row_img53 = np.array(range(1, 15 + 1)).reshape(5, 3)
        row_img54 = np.array(range(1, 20 + 1)).reshape(5, 4)
        row_img64 = np.array(range(1, 24 + 1)).reshape(6, 4)
        # code = self.encode(self.matrix2list(col_img))
        # print(self.decode(code))

        print(self.matrix2list(row_img64))

    def compress(self, index):
        try:
            image_path = os.path.join(self.path, self.image_list[index - 1])
        except:
            print("ERROR！ 并不包含你想要进行RGB处理的这张图片")
        else:
            image_name = re.findall(r'(.+?)\.', self.image_list[index - 1])

            image = default_loader(image_path)

            size = sys.getsizeof((image.flatten()))

            print("Image {}:".format(index))

            print("Origin Image's Size is {} B.".format(size))

            [b, g, r] = cv2.split(image)

            r_b = self.encode(self.matrix2list(b))
            r_g = self.encode(self.matrix2list(g))
            r_r = self.encode(self.matrix2list(r))

            # # 通过打印下面的语句，可以证明最终的存储比例超过200%，是因为对于
            # # 多维度图片的每个维度通道新建不同的数组进行保存的时候，开辟新数组
            # # 空间导致的过大的存储消耗，与压缩算法本身无关。
            # print(b.flatten().shape, r_b.shape)
            # print(g.flatten().shape, r_g.shape)
            # print(r.flatten().shape, r_r.shape)
            # print(sys.getsizeof(np.array([])))

            r_size = sys.getsizeof((r_b)) + sys.getsizeof(
                (r_g)) + sys.getsizeof((r_r))

            print(
                "After Run Length Encoding Image's Size is  {} B.\nCompressed Image's size is {:%} of Origin Image."
                .format(r_size, r_size / size))

            print("Origin Image's Array Size is {} .".format(
                len(image.flatten())))
            print(
                "After Run Length Encoding Image's Array Size is  {} .\nCompressed Image's Array size is {:%} of Origin Image."
                .format(
                    len(r_b) + len(r_g) + len(r_r),
                    (len(r_b) + len(r_g) + len(r_r)) / len(image.flatten())))

            print("")
