'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-25 09:53:46
'''
import argparse

parser = argparse.ArgumentParser(description="采用阈值处理方法进行图像分割")

parser.add_argument('--rgb', default=0, help="选择你需要进行处理的图片(0~5)")
parser.add_argument('--modal',
                    default="adaptive",
                    help="选择你的 Otsu 操作方式 'adaptive' or 'histogram'")

args = parser.parse_args()
