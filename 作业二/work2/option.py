'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-25 09:25:34
'''
import argparse

parser = argparse.ArgumentParser(description="灰度和彩色图像的快速傅立叶变换")

parser.add_argument('--rgb', default=0, help="选择你需要进行处理的图片(0~5)")
parser.add_argument('--color', default='rgb', help="选择你的图片色彩样式【rgb, gray】")

args = parser.parse_args()
