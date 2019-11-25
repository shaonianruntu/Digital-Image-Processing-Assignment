'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-11-25 09:30:49
'''
import argparse

parser = argparse.ArgumentParser(description="彩色图像的去噪")

parser.add_argument('--rgb', default="0", help="选择你需要进行处理的图片(1~5)")
parser.add_argument('--noise', default="gaussian", help="选择你需要处理的噪声类型")
parser.add_argument('--modal', default="mean", help="选择你需要删除噪声的滤波方式")
parser.add_argument('--kernal_size', default="3", help="选择你需要删除噪声的滤波器的卷积核大小")
parser.add_argument('--add', action='store_true', help='进行添加噪声操作')
parser.add_argument('--delete', action='store_true', help='进行剔除噪声操作')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False