'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-23 17:25:44
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (args.modal == "adaptive"):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.otsu_adaptive(i + 1)
    else:
        dataset.otsu_adaptive(int(args.rgb))
elif (args.modal == "histogram"):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.otsu_histogram(i + 1)
    else:
        dataset.otsu_histogram(int(args.rgb))
else:
    print("ERROR！你需要选择你的 Otsu 操作方式 'adaptive' or 'histogram'")
