'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-23 17:25:51
'''
import argparse

parser = argparse.ArgumentParser(description="RGB和HSV图像的直方图均衡化处理")

parser.add_argument('--rgb', default=0, help="选择你需要进行RGB直方图均衡化处理的图片(1~5)")
parser.add_argument('--modal',
                    default="adaptive",
                    help="选择你的 Otsu 操作方式 'adaptive' or 'histogram'")

args = parser.parse_args()
