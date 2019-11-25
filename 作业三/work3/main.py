'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-25 09:54:17
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
    print("ERROR！你需要输入你要进行的操作：")
    print("输入 '--rgb [0~5]' 选择你需要进行处理的图片（0表示选择全部）；")
    print(
        "输入 '--modal [adaptive, histogram]' 选择你的 Otsu 操作方式【Otsu自动阈值法，直方图阈值法】；")
