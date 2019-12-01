'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-12-01 17:01:10
'''
import argparse

parser = argparse.ArgumentParser(description="使用一阶和二阶检测算子（导数）进行图像的边缘检测")

parser.add_argument('--rgb', default=0, help="选择你需要进行处理的图片(0~5)")
parser.add_argument(
    '--modal',
    default="rle",
    help="选择你的压缩算法 'rle（行程编码）' , 'prewitt' , 'roberts' or 'laplacian'")

args = parser.parse_args()
