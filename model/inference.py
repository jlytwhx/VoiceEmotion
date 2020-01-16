import os.path as opath
import pickle

import torch as tc

from .Normalizer import GlobalGaussNorm
from .classic import localatt

Net = None

MEAN, STD = ..., ...

labels = {0: "N", 1: "H", 2: "A", 3: "S"}
emo_map = {"N": "0", "H": "1", "A": "2", "S": "3"}


def getData(pth: str) -> [tc.Tensor]:
    # pth为数据文件路径
    # 数据文件里是一个numpy数组
    assert opath.exists(pth)
    assert MEAN != ... and STD != ...
    norm = GlobalGaussNorm(mean=MEAN, std=STD)
    a: [tc.Tensor] = [tc.from_numpy(pickle.load(open(pth, 'rb')))]
    # a = a.unsqueeze(0)
    # print(a.shape)
    norm(a)
    return a


def loadModel(pth: str, mean: float, std: float):
    # 加载模型
    # pth为参数文件的路径
    global Net, MEAN, STD
    assert opath.exists(pth)
    net = localatt(128, 256, 256, 4)
    state_dict = tc.load(open(pth, 'rb'), map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()
    Net = net
    MEAN, STD = mean, std

#必须先使用loadModel
def predict(input_path: str):
    # input_path为输入路径
    assert Net is not None
    DATA = getData(input_path)
    with tc.no_grad():
        result: tc.Tensor = Net(DATA)
        # print(result)
        result: tc.Tensor = result[0]
    # print(result)
    return labels[result.argmax(0).item()]


if __name__ == '__main__':
    loadModel('../model.pth', 0.11647819887652171, 0.21648931333751797)
    print(predict('../aaa.pk'))
