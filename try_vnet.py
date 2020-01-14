import vnet
import load_data
import numpy as np
import torch
import torch.nn as nn
import losses
import def_loss
import torch.nn.functional as F
import tensorboardX
import os
import random

deviceid = [0]
model = vnet.VNet(elu=False, nll=False)
# model = nn.parallel.DataParallel(model, device_ids=deviceid)
# model.load_state_dict(torch.load('model_save4/vnet_params16.pkl'))  #finetune
model = model.cuda()
LR = 0.001
choose_optimizer = 'SGD'
log = tensorboardX.SummaryWriter('./logs_one/log5')  #dice loss only

url = '/home/hym/Documents/vessel_data/train_data/19540105M354G'
dcmpath = url + '/original1'
maskpath = url + '/vein'
a,b,c = load_data.load_dicom(dcmpath)   #input
print('a size',a.shape)
a = load_data.normalize_ct_array(a)
a = load_data.set_window(a,256)
d = load_data.crop(a)
# crop_img = []
# for di in d:
#     crop_img.append(di)
# print(len(crop_img))     #109
e,f,g = load_data.load_dicom(maskpath)  #target
h = load_data.crop(e)
train_data = []
train_label = []
for i in range(len(d)):
    ccc = d[i]
    check = sum(sum(sum(h[i])))
    print('su7m',check)
    if check != 0:   #如果这个batch有object，则放入训练
        train_data.append(d[i])
        train_label.append(h[i])
    elif check == 0 and random.randint(0,9) > 5: #如果这个batch内没有object，则按照一定概率放入训练
        print(i)
        train_data.append(d[i])
        train_label.append(h[i])

print(len(train_data),len(train_label))   #finish crop data
# check input data
# for h in range(len(train_data)):
#     # print(train_data[h])
#     ddd = train_data[h]
#     eee = train_label[h]
#     per = np.mean(eee)
#     print(per)
#     print('===============')

# breakpoint()
crop_nums = len(train_data)
# edim = np.nonzero(e)
# print('edim',edim[0],'\n',edim[1],'\n',edim[2])
# print('bbox',np.max(edim[0]),np.min(edim[0]))
# print('bbox',np.max(edim[1]),np.min(edim[1]))
# print('bbox',np.max(edim[2]),np.min(edim[2]))
mask_sum = np.sum(e)
lz,lx,ly = e.shape
box = lz*lx*ly
print('标签占比per {} / {} = {:.4f} %'.format(mask_sum,box,(mask_sum/box)*100))
p_weight = torch.tensor((1 - (mask_sum/box))/(mask_sum/box))
p_weight = p_weight.cuda()
print('ppppweight',p_weight)

torch.set_grad_enabled(True)
# model optimizer
if choose_optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=LR)

# h = load_data.crop(e)
# crop_mask = []
# for hi in h:
#     crop_mask.append(hi)
# print(len(crop_mask))    #109

# data = crop_img[50]  #[0:4,:,:]
# data = data[np.newaxis,:]
# data = data[np.newaxis,:]
# input = torch.from_numpy(data)
# input = input.type(torch.FloatTensor)
# input = input.cuda()
#
# print('cccshape',crop_mask[50].shape)
# label = crop_mask[50]  #[0:4,:,:]
# print('hhhhhahahah',label.shape)
# print('checcccccc',np.sum(label))
# nll loss weight
wei = np.mean(e) * 100
print('wei',wei)
target_mean = wei
bg_weight = target_mean / (1. + target_mean)   #background
fg_weight = 1. - bg_weight            #foreground
class_weights = torch.FloatTensor([bg_weight, fg_weight])
class_weights = class_weights.cuda()
print('weiweiweii',class_weights)

##  不按顺序训练
alist = list(range(0,len(train_data)))
print('alist',alist)
random.shuffle(alist)
print('asss',alist)
# for a in range(len(alist)):
#     n = alist[a]
#     data = train_data[n]
#     label = train_label[n]
#     ...
##
# breakpoint()
model.train()
for epoch in range(100000):

    #make train data in batch
    # for j in range(len(train_label)):   #按顺序训练
    for k in range(len(alist)):
        j = alist[k]
        data = train_data[j]
        data = data[np.newaxis,:]
        data = data[np.newaxis,:]
        data = torch.from_numpy(data).type(torch.FloatTensor).cuda()

        label = train_label[j]
        target = torch.from_numpy(label)
        target = target.type(torch.FloatTensor)
        target = target.long()
        target = target.cuda()

        #train
        optimizer.zero_grad()
        # with torch.no_grad():
        output = model(data)
        # print('finish',output.shape)  #input 1*[16,128,128] -> 2*16*[16*128*128]
                                        #两个维度一个是前景一个是背景，两个数经过softmax后加起来和为1
        object = output    #[16*128*128,2]   #两个分类
        # print('=====',object.shape)
        # print('model output',object)
        out = torch.t(output)
        o1 = out[0]
        o2 = out[1]

        target = target.view(target.numel())  #[16*128*128]一维  #object
        """
        # print('target shape = ',target.shape)
        one = torch.ones(target.shape).long().cuda()
        secs = torch.sub(one,target)      #backgound
        label = torch.stack((target,secs),0)
        label = torch.t(label).float()
        # bceloss = F.binary_cross_entropy(object,label)   #bce loss
        creterion = torch.nn.BCELoss(weight=p_weight)
        creterion2 = torch.nn.BCELoss()
        bceloss = creterion(object,label)
        log.add_scalar('bce loss',float(bceloss),k + epoch * crop_nums)
        t1 = target.float()
        t2 = secs.float()
        bce_o1 = creterion(o1,t1)
        bce_o2 = creterion(o2,t2)
        log.add_scalar('bce object loss',float(bce_o1),k + epoch * crop_nums)
        log.add_scalar('bce backgound loss',float(bce_o2),k + epoch * crop_nums)

        # bce_oo1 = creterion2(o1, t1)
        # bce_oo2 = creterion2(o2, t2)
        # log.add_scalar('bce object loss2', float(bce_oo1), k + epoch * crop_nums)
        # log.add_scalar('bce backgound loss2', float(bce_oo2), k + epoch * crop_nums)

        print(bce_o1,'====',bce_o2)
        # print(bce_o2,'====',bce_oo2)
        """

        nll_loss = F.nll_loss(object,target,weight=class_weights)
        log.add_scalar('nll loss',float(nll_loss),k + epoch * crop_nums)
        out1 = output
        dice =losses.dice_error(out1,target)             #dice
        dice_loss = def_loss.dice_coef(out1,target)      #dice loss
        log.add_scalar('dice loss',float(dice_loss),k + epoch * crop_nums)
        # loss_func = bceloss + dice_loss
        # loss_func = dice_loss
        loss_func = nll_loss
        loss_func.backward()
        optimizer.step()
        print('nll loss',nll_loss)
        # print('bce loss',bceloss)
        print('dice',dice)
        pred = output.data.max(1)[1]   #预测点，第1维是背景0，第二维是前景1
        # pred = output.data.min(1)[1]   #预测点，第1维是背景0，第二维是前景1
        #预测，按第一维度的结果作为预测结果，如果大于0.5，则预测为object，否则为背景
        pred_dot = torch.sum(pred).item()
        print('pred dot',pred_dot)
        target_sum = torch.sum(target).item()
        incorrect = pred.ne(target.data).cpu().sum()
        correct = pred.eq(target.data).cpu().sum()
        log.add_scalars('pred and target',{'pred':pred_dot,'target':target_sum},k + epoch * crop_nums)
        err = 100. * incorrect / target.numel()
        print('error {} / {} = {}%'.format(incorrect,target.numel(),err))

        ccc = 100 * correct / target_sum
        log.add_scalar('correct pred',float(ccc),k + epoch * crop_nums)
        print('correct precent {} / {} = {}%'.format(correct,target_sum,ccc))

    if epoch % 10 == 0 :
        params = model.state_dict()
        model_save_path = 'onemodel5'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # torch.save(model,'onemodel/onevnet{}.pkl'.format(i))
        torch.save(params,model_save_path +'/onevnet{}.pkl'.format(epoch))
        log.add_scalar('model dice', float(dice), epoch)
        log.add_scalar('model BCE loss', float(loss_func), epoch)
        LR /= 5