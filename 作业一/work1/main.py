'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-22 20:07:24
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (args.rgb):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.rgb_equalization(i + 1)
    else:
        dataset.rgb_equalization(int(args.rgb))
elif (args.hsv):
    if (int(args.hsv) == 0):
        for i in range(len(listdir(image_path))):
            dataset.hsv_equalization(i + 1)
    else:
        dataset.hsv_equalization(int(args.hsv))
else:
    print("ERROR！你需要输入你要进行的操作 'RGB' or 'HSV' 的直方图均衡化处理")
