'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:10:27
@LastEditors: fangn
@LastEditTime: 2019-12-12 21:37:03
'''
import argparse

parser = argparse.ArgumentParser(description="使用一阶和二阶检测算子（导数）进行图像的边缘检测")

parser.add_argument('--rgb', default=0, help="选择你需要进行处理的图片(0~5)")
parser.add_argument('--q_factor',
                    type=int,
                    default=20,
                    choices=[20, 60, 80],
                    help="选择你的 JPEG 压缩的质量因子，可选项为【20，60，80】")

args = parser.parse_args()
