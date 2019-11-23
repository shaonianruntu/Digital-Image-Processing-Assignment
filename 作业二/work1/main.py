'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-22 22:19:10
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
    print("ERROR！你需要输入你要进行的操作 'low' or 'high' 的低通高通滤波器")