'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-29 17:20:44
'''
from os import listdir

from model import RLE
from option import args

image_path = "../test_image/"

rel = RLE(image_path)
rel.test()