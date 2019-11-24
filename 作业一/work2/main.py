'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:24:35
@LastEditors: fangn
@LastEditTime: 2019-11-24 14:23:00
'''
from os import listdir

from model import Dataset
from option import args

image_path = "../test_image/"

dataset = Dataset(image_path)

if (args.add):
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.add_noise(i + 1, noise=args.noise)
    else:
        dataset.add_noise(int(args.rgb), noise=args.add_noise)
elif args.delete:
    if (int(args.rgb) == 0):
        for i in range(len(listdir(image_path))):
            dataset.delete_noise(i + 1,
                                 noise=args.noise,
                                 modal=args.modal,
                                 kernal_size=int(args.kernal_size))
    else:
        dataset.delete_noise(int(args.rgb),
                             noise=args.noise,
                             modal=args.modal,
                             kernal_size=int(args.kernal_size))
else:
    print("ERROR！你需要输入你要进行的操作：")
    print("输入 '--rgb [0~5]' 选择你需要进行处理的图片（0表示选择全部）;")
    print("输入 '--add' 表示进行添加噪声操作;")
    print("输入 '--delete' 表示进行剔除噪声操作;")
    print("输入 '--noise [gaussian, salt]' 选择你需要处理的噪声类型【高斯噪声，椒盐噪声】;")
    print("输入 '--modal [mean, middle]' 选择你需要降噪的滤波方式【均值滤波，中值滤波】;")
    print("输入 '--kernal_size [e.p. 3,5,7,...]' 选择你的滤波器的卷积核大小;")