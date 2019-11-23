'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-23 16:07:03
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (int(args.rgb) == 0):
    for i in range(len(listdir(image_path))):
        dataset.dct8(i + 1, color=args.color)
else:
    dataset.dct8(int(args.rgb), color=args.color)
