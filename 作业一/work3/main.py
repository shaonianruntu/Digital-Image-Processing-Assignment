'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 19:17:58
@LastEditors: fangn
@LastEditTime: 2019-11-22 20:29:09
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
    print("ERROR！你需要选择你想要的算子 'lap', 'sobel' or 'fangnan'")
