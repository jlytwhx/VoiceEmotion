'''这个文件是用来寻找推断正确的音频'''

import csv
from os.path import join as pjoin

from model.inference import loadModel, predict

B = r'D:\Python\localatt_emorecog\Datas\data'
reader = csv.reader(open(r'D:\Python\localatt_emorecog\Datas\full_dataset.csv', newline=''))
next(reader)
loadModel('model.pth', 0.11647819887652171, 0.21648931333751797)
OUT = open('corrects.txt', 'w')
for name, id, sex, ses, emo, is_vgg in reader:
    is_vgg = is_vgg == 'yes'
    ses = int(ses)
    if is_vgg and 15 < ses < 24:
        dest = pjoin(B, id + '.pk')
        # print(emo,predict(dest))
        if emo == predict(dest):
            print(name + '.wav', emo, file=OUT)
