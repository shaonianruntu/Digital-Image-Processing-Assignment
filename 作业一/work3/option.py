'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-25 09:31:01
'''
import argparse

parser = argparse.ArgumentParser(description="图像锐化")

parser.add_argument('--lap', action='store_true', help='使用拉普拉斯算子进行边缘检测')
parser.add_argument('--sobel', action='store_true', help='使用Sobel算子进行边缘检测')
parser.add_argument('--fangnan',
                    action='store_true',
                    help='使用我的fangnan算子进行边缘检测')
parser.add_argument('--dir', default='x', help='Sobel算子的边缘检测方向 x or y')
parser.add_argument('--fdir',
                    default='left',
                    help='我的fangnan算子的边缘检测方向 left or right')
parser.add_argument('--rgb', default="0", help="选择你需要进行RGB直方图均衡化处理的图片(1~5)")

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False