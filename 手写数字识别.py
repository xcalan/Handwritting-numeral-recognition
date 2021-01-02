#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/1/2 10:35
@Author  : Xie Cheng
@File    : 手写数字识别.py
@Software: PyCharm
@desc: Mnist手写数字识别代码
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000


# 1. 准备数据集
def get_dataloader(train=True, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = MNIST(root="./data", train=train, transform=transform_fn)  # 如果数据没下载的话加上这句话 download=True
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# 2. 构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        """
        :param input: [batch_size,1,28,28]
        :return:
        """
        # 1. 修改形状
        x =input.view([input.size(0), 1*28*28])
        # input.view([-1, 1*28*28])
        # 2. 进行全连接的操作
        x = self.fc1(x)
        # 3. 进行激活函数的处理，形状没有变化
        x = F.relu(x)
        # 4. 输出层
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)


model = MnistModel()
optimizer = Adam(model.parameters(), lr=0.001)

if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict((torch.load("./model/optimizer.pkl")))


def train(epoch):
    """实现训练过程"""
    data_loader = get_dataloader()
    for idx, (input, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)  # 调用模型得到预测值
        loss = F.nll_loss(output, target)  # 得到损失
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度的更新
        if idx % 10 == 0:
            print(epoch, idx, loss.item())

        # 模型的保存
        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")


def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
    for idx, (input, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss)
            # 计算准确率
            # output [batch_size,10] target:[batch_size]
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()  # eq判断相等，输出为bool
            acc_list.append(cur_acc)

    print("平均准确率，平均损失：", np.mean(acc_list), np.mean(loss_list))


if __name__ == '__main__':
    # # 训练
    for i in range(5):
        train(i)

    # 测试
    # test()