'''
@ project: Neural-Network
@ file: main
@ user: 罗申申
@ email: luoshenshen@buaa.edu.cn
@ tool: PyCharm
@ time: 2021/5/26 14:00
'''
import numpy
import torch
import numpy as np

print(torch.__version__)

a = torch.tensor([1,2,3])
print(type(a))
a = torch.Tensor([1,2,3])
print(type(a))
arr = np.arange(12).reshape(3,4)
print(arr)

a = torch.Tensor(arr)
print(a)

a = torch.empty(3,4)
print(a)

a = torch.ones(3,4)
print(a)

a = torch.zeros(3,4)
print(a)

a = torch.rand(3,4)
print(a)

a = torch.randint(low=1,high=2,size=[3,4])
print(a)

a = torch.randn([3,4])
print(a)

a = torch.Tensor([[[[1]]]])
print(a)
print(a.item())
print(a.size())
print(a.numpy())

a = a.view(1,-1)
print(a)

a = torch.tensor([3,4])
print(a.dim())

a = torch.randn(3,4)
print(a)
print(a.dim())
print(a.max())
print(a.min())
print("-----------------------------------------------")
a = torch.Tensor(numpy.arange(24).reshape(2,3,4))
print(a)
a.permute(1,2,0)
print(a)

a = torch.Tensor(numpy.arange(24).reshape(4,6))
print(a)
a = a.t()
print(a)

a = torch.tensor(np.arange(28).reshape(4,7),dtype=torch.float64,)
print(type(a))
print(a.dtype)

a = torch.LongTensor(np.arange(12).reshape(3,4))
print(a)
print(a.dtype)

a = torch.DoubleTensor([1,2])
print(a.dtype)

a = torch.float64
print(a)

a = torch.Tensor(1)
print(a)
print(type(a))
print(a.dtype)

a = torch.Tensor(np.arange(8).reshape(2,2,2))
print(a)
a = a.transpose(1,2)
print(a)

a = torch.tensor([3,4],dtype=torch.float64)
b = torch.tensor([3,4],dtype=torch.float64)
print(a+b)
a = a.new_ones(4,4,dtype=torch.float16)
print(a)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cup")
a.to(device)
print(a)















