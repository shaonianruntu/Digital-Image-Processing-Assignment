'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-22 22:15:52
'''
import argparse

parser = argparse.ArgumentParser(description="RGB和HSV图像的直方图均衡化处理")

parser.add_argument('--rgb', default=0, help="选择你需要进行RGB直方图均衡化处理的图片(1~5)")
parser.add_argument('--r', default='150', help="选择你的滤波器的通过半径")
parser.add_argument('--low', action='store_true', help="使用低通滤波器")
parser.add_argument('--high', action='store_true', help="使用高通滤波器")

args = parser.parse_args()
