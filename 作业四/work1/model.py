'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 09:40:18
@LastEditors: fangn
@LastEditTime: 2019-12-12 18:04:37
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


# Run Length Encoding 行程编码
class RLE:
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

            print("Origin Image's Size is {:.2f} KB.".format(size / 1024))

            [b, g, r] = cv2.split(image)

            r_b = self.encode(self.matrix2list(b)).astype(np.uint8)
            r_g = self.encode(self.matrix2list(g)).astype(np.uint8)
            r_r = self.encode(self.matrix2list(r)).astype(np.uint8)

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
                "After Run Length Encoding Image's Size is  {:.2f} KB.\nCompressed Image's size is {:.2%} of Origin Image."
                .format(r_size / 1024, r_size / size))

            # print("Origin Image's Array Size is {} .".format(
            #     len(image.flatten())))
            # print(
            #     "After Run Length Encoding Image's Array Size is  {} .\nCompressed Image's Array size is {:%} of Origin Image."
            #     .format(
            #         len(r_b) + len(r_g) + len(r_r),
            #         (len(r_b) + len(r_g) + len(r_r)) / len(image.flatten())))

            print()


class HuffmanLetter:
    def __init__(self, letter, freq):
        self.letter = letter
        self.freq = freq
        self.bitstring = ""

    def __repr__(self):
        return f"{self.letter}"


class HuffmanTreeNode:
    def __init__(self, freq, left, right):
        self.freq = freq
        self.left = left
        self.right = right


class Huffman:
    """
    Huffman coding compress for rgb image,
    using variable-length binary replace fixed-length coding
    to reduce image size.
    """
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def byte_cut(self, image):
        """
        Split the image according to the length of the Byte (8 bits).
        """
        image_list = image.flatten()
        chars = {}
        for c in image_list:
            chars[c] = chars[c] + 1 if c in chars.keys() else 1
        return sorted([HuffmanLetter(c, f) for c, f in chars.items()],
                      key=lambda l: l.freq)

    def build_tree(self, letters):
        """
        Build huffman tree structure according to original character segment.
        """
        while len(letters) > 1:
            left = letters.pop(0)
            right = letters.pop(0)
            total_freq = left.freq + right.freq
            node = HuffmanTreeNode(total_freq, left, right)
            letters.append(node)
            letters.sort(key=lambda l: l.freq)
        return letters[0]

    def traverse_tree(self, root, bitstring):
        """
        Re-encoding according to the huffman tree structure, 
        getting huffman code.
        """
        if type(root) is HuffmanLetter:
            root.bitstring = bitstring
            return [root]
        letters = []
        letters += self.traverse_tree(root.left, bitstring + "0")
        letters += self.traverse_tree(root.right, bitstring + "1")
        return letters

    def test(self):
        test_image = np.array(np.random.randint(0, 25, size=[5, 5]))
        print(test_image.flatten())
        letters_list = self.byte_cut(test_image)
        print(letters_list)
        root = self.build_tree(letters_list)
        letters = self.traverse_tree(root, "")

        dict = {}
        for letter in letters:
            dict[letter.letter] = letter.bitstring
        # print(dict)

        compress = ""
        for bs in test_image.flatten():
            compress += dict[bs]
            # print(bs)
            # print(dict[bs])

        # for c in test_image:
        #     compress += letter.bitstring

        print(sys.getsizeof(test_image.flatten()))
        print(sys.getsizeof(compress))

    def huffman_change(self, image):
        letters_list = self.byte_cut(image)
        root = self.build_tree(letters_list)
        letters = self.traverse_tree(root, "")

        dict = {}
        for letter in letters:
            dict[letter.letter] = letter.bitstring

        compress = ""
        for bs in image.flatten():
            compress += dict[bs]

        return compress, dict

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

            print("Origin Image's Size is {:.2f} KB.".format(size / 1024))

            [b, g, r] = cv2.split(image)

            r_b, r_b_dict = self.huffman_change(b)
            r_g, r_g_dict = self.huffman_change(g)
            r_r, r_r_dict = self.huffman_change(r)

            r = []
            r.append(r_b)
            r.append(r_g)
            r.append(r_r)

            # print(r)

            r_size = sys.getsizeof(r)
            r_dict_size = sys.getsizeof(r_b_dict) + sys.getsizeof(
                r_g_dict) + sys.getsizeof(r_r_dict)
            r_size_all = r_size + r_dict_size

            print("After Huffman Encoding Image's Size is  {:.2f} KB.\
                    \nCompressed Image's Huffman coding size is {:.2f} KB.\
                    \nCompressed Image's Huffman coding dictonary size is {:.2f} KB.\
                    \nCompressed Image's size is {:.2%} of Origin Image.".
                  format(r_size_all / 1024, r_size / 1024, r_dict_size / 1024,
                         r_size_all / size))

            print()


class PredictCode:
    """
    Linear prediction coding
    """
    def __init__(self, path):
        self.path = path
        self.image_list = [x for x in listdir(path) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

    def predict_f(self, x):
        """
        Prediction function: y = x
        """
        return x

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

    def predict(self, image):
        # Using RLE to improve prediction coding efficiency.
        image_list = self.matrix2list(image)
        predict_list = []
        update_list = []
        predict_list.append(image_list[0])
        update_list.append(0)
        for c in image_list[1:]:
            pred = self.predict_f(predict_list[-1])
            e = c - pred
            predict_list.append(pred)
            update_list.append(e)
        return np.int8(predict_list), np.int8(update_list)

    def test(self):
        test_image = np.array(np.random.randint(0, 25, size=[5, 5]))
        predict_list, update_list = self.predict(test_image)
        print(predict_list)
        print(predict_list)

        print(sys.getsizeof(test_image.flatten()))
        # print(sys.getsizeof(predict_list))
        print(sys.getsizeof(update_list))

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

            print("Origin Image's Size is {:.2f} KB.".format(size / 1024))

            [b, g, r] = cv2.split(image)

            r_b_pred, r_b_update = self.predict(b)
            r_g_pred, r_g_update = self.predict(g)
            r_r_pred, r_r_update = self.predict(r)

            r = []
            r.append(r_b_update)
            r.append(r_g_update)
            r.append(r_r_update)

            # print(r_b_update)
            # print(r)

            r_size = sys.getsizeof(r)

            print("After Predict Encoding Image's Size is  {:.2f} KB.\
                    \nCompressed Image's size is {:.2%} of Origin Image.".
                  format(r_size / 1024, r_size / size))

            print()
