'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-12-12 21:34:13
'''
from os import listdir

from model import JPEGEncode
from option import args

image_path = "../test_image/"

modal = JPEGEncode(image_path)
if (int(args.rgb) == 0):
    for i in range(len(listdir(image_path))):
        modal.compress(i + 1, args.q_factor)
else:
    modal.compress(int(args.rgb), args.q_factor)
