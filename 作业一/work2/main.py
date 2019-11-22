'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-22 17:24:35
@LastEditors: fangn
@LastEditTime: 2019-11-22 20:03:37
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
    print("ERROR！你需要输入你要进行的操作 'add' or 'delete'")
