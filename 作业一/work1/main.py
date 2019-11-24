'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-24 13:45:21
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
    print("ERROR！你需要输入你要进行的操作:")
    print("输入 '--rgb [0~5]' 选择你需要进行RGB直方图处理的图片（0表示选择全部）;")
    print("输入 '--hsv [0~5]' 选择你需要进行HSV直方图处理的图片（0表示选择全部）;")
