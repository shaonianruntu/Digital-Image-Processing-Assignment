'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-23 17:25:31
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (int(args.rgb) == 0):
    for i in range(len(listdir(image_path))):
        dataset.hough_line(i + 1)
else:
    dataset.hough_line(int(args.rgb))
