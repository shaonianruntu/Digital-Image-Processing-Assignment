'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-11-23 16:26:42
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (int(args.rgb) == 0):
    for i in range(len(listdir(image_path))):
        dataset.edge_detection(i + 1, args.modal, args.dir)
else:
    dataset.fft_lowpass(int(args.rgb), args.modal, args.dir)
