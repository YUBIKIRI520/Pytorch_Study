import torch
import torch.nn.functional
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameter
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# 打印一下看看
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# 用于github测试
# git add -A