import vnet
import load_data
import numpy as np
import torch
import torch.nn as nn
import losses
import torch.nn.functional as F
import SimpleITK as sitk
import os


url = '/home/hym/Documents/vessel_data/train_data/19540105M354G'
# url = '/home/hym/Documents/vessel_data/train_data1/19381024F23'
dcmpath = url + '/original1'
maskpath = url + '/vein'
a,b,c = load_data.load_dicom(dcmpath)   #input
size_z,size_x,size_y = a.shape
print('a size',a.shape)
a = load_data.normalize_ct_array(a)
a = load_data.set_window(a,255)    #统一灰度值范围
d = load_data.crop(a,mode = 'test')
crop_img = []
for di in d:
    crop_img.append(di)
print(len(crop_img))   #crop data
e,f,g = load_data.load_dicom(maskpath)  #target
# sitk.WriteImage(sitk.GetImageFromArray(e),'mask.vtk')
# breakpoint()

mask_all = torch.from_numpy(e[:,152:360,88:424]).long()  #check the predict and the mask intersection

# check the size of mask
edim = np.nonzero(e)
print('edim',edim[0],'\n',edim[1],'\n',edim[2])
print('bbox',np.min(edim[0]),np.max(edim[0]))
print('bbox',np.min(edim[1]),np.max(edim[1]))
print('bbox',np.min(edim[2]),np.max(edim[2]))
####
h = load_data.crop(e,mode = 'test')
crop_mask = []
for hi in h:
    crop_mask.append(hi)
print(len(crop_mask))   #crop label
crop_nums = len(crop_img)

#load model
model = vnet.VNet(elu=False, nll=False).cuda()
# model.load_state_dict(torch.load('onemodel/onevnet600.pkl'))    #one batch[50] the best model
# model.load_state_dict(torch.load('model_save9/vnet_params18.pkl'))  #record  the best : 18 and 21
# model.load_state_dict(torch.load('model_save11/vnet_params1-1.pkl'))    #record the best :1-1
model.load_state_dict(torch.load('model_save21/vnet_params87.pkl'))
model.eval()  #测试模式

pred_res = torch.zeros(size_z,size_x,size_y)
nummmm = 0
# for data in crop_img:
for i in range(len(crop_img)):
    data = crop_img[i]
    # sitk.WriteImage(sitk.GetImageFromArray(data), "vtk/ori{}.vtk".format(i))
    # mask = crop_mask[i]
    # sitk.WriteImage(sitk.GetImageFromArray(mask), "vtk/mask{}.vtk".format(i))

    data = data[np.newaxis,:]
    data = data[np.newaxis,:]
    input = torch.from_numpy(data)
    input = input.type(torch.FloatTensor)
    if torch.cuda.is_available():
        input = input.cuda()

    label = crop_mask[i]
    target = torch.from_numpy(label)
    target = target.type(torch.FloatTensor).long().cuda()

    #predict
    with torch.no_grad():
        output = model(input)
        # _,pred = output.data.min(1)   #预测点，第1维是背景0，第二维是前景1
        _,pred = output.data.max(1)   #预测点，第1维是背景0，第二维是前景1
        re_pred = torch.reshape(pred,(4,208,336))

        # write each batch's vtk
        # model_save_path = 'vtk21'
        # if not os.path.exists(model_save_path):
        #     os.makedirs(model_save_path)
        # hua = re_pred.cpu()
        # sitk.WriteImage(sitk.GetImageFromArray(hua), model_save_path + "/pred{}.vtk".format(i))

        if nummmm + 4 < size_z:
            pred_res[nummmm:nummmm + 4,152:360,88:424] = re_pred
            nummmm += 4
        else:
            pred_res[-4:,152:360,88:424] = re_pred
        # if nummmm == 0:
        #     pred_res = re_pred.cpu()
        #     nummmm += 1
        # elif nummmm == crop_nums - 2:
        #     l = len(pred_res)
        #     print('now len',l)
        #     start = (size_z - l) * -1
        #     pred_res = torch.cat((pred_res,re_pred[start:,:,:].cpu()),0)
        # else :
        #     pred_res = torch.cat((pred_res,re_pred.cpu()),0)
        #     nummmm += 1


print('predict shape',pred_res.shape)
# result = torch.zeros(size_z,size_x,size_y)
result = pred_res
print('check 512',result.shape)
sitk.WriteImage(sitk.GetImageFromArray(result),'whole7.vtk')
sum_pred = torch.sum(pred_res)
print('sum predict dots',sum_pred)
# incorrect = pred_res.ne(mask_all.data).cpu().sum()
# all = torch.sum(mask_all)
# all_size = mask_all.numel()
# error = 100 * incorrect / all
# print('error {} / {} = {}'.format(incorrect,all_size,error))

