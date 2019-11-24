'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-24 13:28:06
'''
import argparse

parser = argparse.ArgumentParser(description="RGB和HSV图像的直方图均衡化处理")

parser.add_argument('--hsv', help="选择你需要进行HSV直方图均衡化处理的图片(1~5)")
parser.add_argument('--rgb', help="选择你需要进行RGB直方图均衡化处理的图片(1~5)")

args = parser.parse_args()
