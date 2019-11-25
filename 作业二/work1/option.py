'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-25 09:25:54
'''
import argparse

parser = argparse.ArgumentParser(description="彩色图像的频域滤波器")

parser.add_argument('--rgb', default=0, help="选择你需要进行处理的图片(0~5)")
parser.add_argument('--r', default='80', help="选择你的滤波器的通过半径")
parser.add_argument('--low', action='store_true', help="使用低通滤波器")
parser.add_argument('--high', action='store_true', help="使用高通滤波器")

args = parser.parse_args()
