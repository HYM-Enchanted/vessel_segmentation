import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import vnet
import torch.nn.functional as F
from numpy import array

class convvvvv(nn.Module):
    def __init__(self):
        super(convvvvv,self).__init__()
        self.conv1 = nn.Conv3d(1,4,kernel_size=1,stride=2)

    def forward(self, x):
        x = self.conv1(x)
        return x


# net = convvvvv()
# net.cuda()
# data1 = [[[1,2,3,4,5,6,7,8,9,3,2,1],
#          [2,3,4,5,6,3,2,1,5,6,7,7],
#          [6,7,8,0,4,2,5,3,1,2,3,5],
#          [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1],
#          [2, 3, 4, 5, 6, 3, 2, 1, 5, 6, 7, 7],
#          [6, 7, 8, 0, 4, 2, 5, 3, 1, 2, 3, 5],
#          [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1],
#          [2, 3, 4, 5, 6, 3, 2, 1, 5, 6, 7, 7],
#          [6, 7, 8, 0, 4, 2, 5, 3, 1, 2, 3, 5],
#          [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1],
#          [2, 3, 4, 5, 6, 3, 2, 1, 5, 6, 7, 7],
#          [6, 7, 8, 0, 4, 2, 5, 3, 1, 2, 3, 5],
#          ]]    #12*12
# data1 = np.array(data1)
# data2 = data1 +10
# data = [data1,data2]
# data = np.array(data)
# print(data.shape)
'''
data = torch.tensor(data[:,np.newaxis,:])
data = data.type(torch.FloatTensor)
data = data.cuda()
print('inpu size',data.shape)
res = net(data)                  #3维网络卷积
print('output size',res.shape)
# print(res)
#可歌可泣，跑通了！
# (2, 1, 12, 12)
# inpu size torch.Size([2, 1, 1, 12, 12])
# output size torch.Size([2, 4, 1, 6, 6])
'''

# aaa = [[1,2,3,4,5,6]]
# # bbb = [[1],[2],[3],[4],[5],[6]]
# # aaa= [[1,1,1,1,1,1]]
# bbb = [[3],[3],[3],[3],[3],[3]]
# aaa = np.array(aaa)
# bbb = np.array(bbb)
# aaa = torch.from_numpy(aaa)
# bbb = torch.from_numpy(bbb)
# aaa = aaa.type(torch.FloatTensor)
# bbb = bbb.type(torch.FloatTensor)
#
# ccc = torch.add(aaa,bbb)
# print(ccc)
#
# ddd = torch.mean(ccc,dim= 0)
# print(ddd)
#
# aaa= np.array([0,1,0,1,0,0,0,0,1])
# print(np.mean(aaa))
#
#
# aaa =array([[0.1,0.9],
#       [0.2,0.8],
#       [0.6,0.4],
#       [0.3,0.7],
#       [0.9,0.1]
#       ])
# print(aaa.shape)
# bbb = aaa.min(1)
# print(bbb)
# print(bbb[1])
# print(torch.ones([64]))

# aaa1 = torch.tensor([[-0.2,0.34],
#                     [0.4,0.93],
#                     [0.5,-0.34],
#                     [0.05,0.7]])
# print(aaa.shape)
# bbb = torch.t(aaa)
# ccc = bbb[0]
# print(ccc)
# aaa = torch.tensor(torch.randn(4))
# print(aaa)
# b = torch.nn.Sigmoid()
# c = torch.nn.Softmax()
# d = torch.nn.LogSigmoid()
# bbb = b(aaa1)
# b1 = F.sigmoid(aaa1)
# print(bbb)
# print('=====')
# print(b1)
# e,r = bbb.max(1)
# print(r)
# q = bbb[:,1:]
# print(q)
# ccc = c(aaa1)
# print(ccc)
# ddd = d(aaa1)
# print(ddd)
# _,res = ddd.max(1)
# print(res)


# aaa1 = torch.tensor([[[-0.2,0.34],
#                     [0.4,0.93],
#                     [0.5,-0.34],
#                     [0.05,0.7]],
#                      [[1,2],
#                       [0.5,9],
#                       [4.3,6.5],
#                       [0.9,7.8]],
#                      [[0.9,-9.8],
#                       [3,5],
#                       [-8.7,9],
#                       [4,5]]])
# # print(aaa1)
# print(aaa1.shape)
# ddd = aaa1.view(aaa1.numel()//2,2)
# print(ddd)
# bbb = torch.tensor([[1,2,3,4],
#                     [3,4,5,7],
#                     [6,8,3,4]])
# print(bbb.shape)
# print(bbb.view(bbb.numel(),1))
# print('===========')
# ccc = ddd.clone().detach()
# print(ccc)
# bb = torch.mean(aaa1,dim=0)
# print(bb)
# print(bb.shape)

# a = torch.tensor([[[1,2,3,4],[4,4,4,4],[6,6,6,6],[2,7,5,8],[0,8,6,4],[3,9,1,3]]])
# b = torch.tensor([[[10,20,30,40]],[[10,20,30,40]],[[10,20,30,40]]])#,[[6,1,3,8]],[[2,5,1,2]],[[6,7,3,1]]])
# d = torch.tensor([[[10,20,30,40]],[[10,20,30,40]],[[10,20,30,40]],
#                   [[10,20,30,40]],[[10,20,30,40]],[[10,20,30,40]]])
# print(a.shape,b.shape)
# c= torch.add(a,b)
# print(c.shape)
# print(c)
# print('#######3')
# e = torch.add(a,d)
# print(e.shape)
# print(e)
#
# a = torch.tensor([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
# b = torch.tensor([[100,200,300,400]])
# c = torch.add(a,b)
# print(c)

a = np.array([[[1,-2,-3,4],[4,-4,4,-4],[-6,6,6,6],[2,-7,5,8],[0,8,-6,4],[-3,9,-1,3]]])
b = np.array([[[1,0,1,0],[0,1,1,1],[1,0,1,1],[1,1,1,1],[0,1,0,0],[1,0,1,0]]])
print(sum(sum(sum(b))))
a[a>6] = 6
a[a<0] = 0
print(a)
c = a*b
c[c>0]=1
print(c)
print(sum(sum(sum(c))))