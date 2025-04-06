import torch
from torch.backends import cudnn

print(torch.__version__)               # 输出Pytorch版本
print(torch.version.cuda)              # 输出cuda版本
print(torch.cuda.is_available())       # 输出为True，则cuda安装成功
print(torch.backends.cudnn.version())  # 输出cuDNN版本
print(cudnn.is_available())            # 输出为True，则cuDNN安装成功