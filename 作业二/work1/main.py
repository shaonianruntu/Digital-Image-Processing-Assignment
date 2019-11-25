'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-25 08:47:18
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (args.low):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.fft_lowpass(i + 1, int(args.r))
    else:
        dataset.fft_lowpass(int(args.rgb), int(args.r))
elif (args.high):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.fft_highpass(i + 1, int(args.r))
    else:
        dataset.fft_highpass(int(args.rgb), int(args.r))
else:
    print("ERROR！你需要输入你要进行的操作：")
    print("输入 '--rgb [0~5]' 选择你需要进行处理的图片（0表示选择全部）；")
    print("输入 '--[low, high]' 选择【低通，高通】滤波器来对图像进行滤波操作；")
    print("输入 '--r [5, 20, 50, 80, 250]' 选择滤波器的半径；")