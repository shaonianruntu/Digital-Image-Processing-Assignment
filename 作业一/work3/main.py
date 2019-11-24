'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 19:17:58
@LastEditors: fangn
@LastEditTime: 2019-11-24 14:23:40
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (args.lap):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.Laplacian(i + 1)
    else:
        dataset.Laplacian(int(args.rgb))
elif (args.sobel):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.Sobel(i + 1, args.dir)
    else:
        dataset.Sobel(int(args.rgb), args.dir)
elif (args.fangnan):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.fangnan(i + 1, args.fdir)
    else:
        dataset.fangnan(int(args.rgb), args.fdir)
else:
    print("ERROR！你需要输入你要进行的操作:")
    print("输入 '--rgb [0~5]' 选择你需要进行处理的图片（0表示选择全部）;")
    print(
        "输入 '--[lap, sobel, fangnan]' 选择你需要进行图片锐化操作的算子【Laplacian算子，Sobel算子，我的方楠算子】;"
    )
    print("输入 '--dir [x, y]' 选择你需要进行Sobel算子边缘检测的方向【x方向，y方向】;")
    print("输入 '--fdir [left, right]' 选择你需要进行方楠算子边缘检测的方向【正对角线方向，副对角线方向】;")
