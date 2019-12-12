'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 16:44:33
@LastEditors: fangn
@LastEditTime: 2019-12-12 17:31:36
'''
from os import listdir

from model import RLE, Huffman, PredictCode
from option import args

image_path = "../test_image/"

if args.modal == "rle":
    modal = RLE(image_path)
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            modal.compress(i + 1)
    else:
        modal.compress(int(args.rgb))
elif args.modal == "huffman":
    modal = Huffman(image_path)
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            modal.compress(i + 1)
elif args.modal == "predict":
    modal = PredictCode(image_path)
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            modal.compress(i + 1)
