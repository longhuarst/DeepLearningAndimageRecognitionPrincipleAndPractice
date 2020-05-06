import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root = '/ml/pymnist', #选择数据的根目录
                           train = True, # 选择训练集
                           transform = None, #不考虑使用任何数据预处理
                           download = True) # 从网络上download图片
test_dataset = dsets.MNIST(root = '/ml/pymnist', #选择数据的根目录
                           train = False, # 选择测试集
                           transform = None, #不考虑使用任何数据预处理
                           download = True) # 从网络上download图片
#加载数据
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)