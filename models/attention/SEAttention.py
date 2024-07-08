import numpy as np
import torch
from torch import nn
from torch.nn import init

# 定义SE注意力机制的类
class SEAttention(nn.Module):

    # 初始化函数，定义网络结构
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，输出大小为1x1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 全连接层，降维
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 全连接层，恢复维度
            nn.Sigmoid()  # Sigmoid激活函数
        )

    # 初始化权重函数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 初始化卷积层权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化卷积层偏置
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 初始化批归一化层权重
                init.constant_(m.bias, 0)  # 初始化批归一化层偏置
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 初始化全连接层权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化全连接层偏置

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入张量的尺寸
        y = self.avg_pool(x).view(b, c)  # 平均池化并展平
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层并恢复到原始维度
        return x * y.expand_as(x)  # 将注意力权重应用到输入张量

# 测试代码
if __name__ == '__main__':
    input=torch.randn(50,512,7,7)  # 生成随机输入张量
    se = SEAttention(channel=512,reduction=8)  # 创建SEAttention实例
    output=se(input)  # 前向传播
    print(output.shape)  # 打印输出张量的形状