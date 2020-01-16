import os
from os import path as opath

from model.inference import loadModel, predict


'''核心函数有:extract、loadModel、predict'''

def extract(pth: str):
    '''反正提取vgg特征就是用下面这个文件，但是具体指令容易因相对路径而
    变得混乱。那个文件需要feature底下的几个文件。
    在wav的路径下创建一个和wav同名的pk文件。'''
    b = opath.abspath(pth)
    os.system("python feature/extract_pickle.py --wav_file %s 2>nul" % b)


TARGET, OUT = 'aaa.wav', 'aaa.pk'
extract(TARGET)
loadModel('settings/model.pth', 0.11647819887652171, 0.21648931333751797)
print(predict(OUT))
