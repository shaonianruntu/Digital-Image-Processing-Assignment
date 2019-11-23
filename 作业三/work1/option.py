'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-23 16:29:27
'''
import argparse

parser = argparse.ArgumentParser(description="RGB和HSV图像的直方图均衡化处理")

parser.add_argument('--rgb', default=0, help="选择你需要进行RGB直方图均衡化处理的图片(1~5)")
parser.add_argument(
    '--modal',
    default="sobel",
    help="选择你的边缘检测算子 'sobel' , 'prewitt' , 'roberts' or 'laplacian'")
parser.add_argument('--dir', default="x", help="选择你的滤波器的算子方向 'x' or 'y'")

args = parser.parse_args()
