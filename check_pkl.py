import torch
import vnet
import numpy as np

model = vnet.VNet(elu=False, nll=False)
# model.load_state_dict(torch.load('artery_params_1/vnet_params11.pkl'))
# model.load_state_dict(torch.load('artery_params_3/vnet_params9.pkl'))
model.load_state_dict(torch.load('artery_params_9/vnet_params0.pkl'))
model.cuda()
print(model.state_dict())
print('=========================================')

model1 = vnet.VNet(elu=False, nll=False)
# model.load_state_dict(torch.load('artery_params_1/vnet_params11.pkl'))
# model.load_state_dict(torch.load('artery_params_3/vnet_params9.pkl'))
model1.load_state_dict(torch.load('artery_params_9/vnet_params1.pkl'))
model1.cuda()
# print(model1.state_dict())

# aaa = model.state_dict()
# bbb = model1.state_dict()
# print(type(aaa))

# if aaa == bbb:
#     print('yep')
# else :
#     print('no')



