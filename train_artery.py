import torchsnooper
import torch
import tensorboardX
import os
import vnet
import random
import torch.optim as optim
import load_data
import numpy as np
import torch.nn.functional as F
import def_loss
import losses
import SimpleITK as sitk

def normalize_data(dicom_array,normalize = 'max_min'):
    dicom_array[dicom_array < -100] = -100
    dicom_array[dicom_array > 600] = 600
    if normalize == 'gaosi':
        avg = np.mean(dicom_array)
        std = np.std(dicom_array)      #标准差
        dicom_array = (dicom_array - avg) / std
    elif normalize == 'max_min':
        max = np.max(dicom_array)
        min = np.min(dicom_array)
        dicom_array = (dicom_array - min) / (max - min)
    else:
        pass
    return dicom_array

def make_data(url,normalize = 'max_min'):
    image = sitk.ReadImage(url)
    dicom_array = sitk.GetArrayFromImage(image)
    # origin = np.array(image.GetOrigin())
    # spacing = np.array(image.GetSpacing())
    dicom_array = normalize_data(dicom_array)
    url_mask = url.replace('volume','segmentation')
    url_mask = url_mask.replace('all','all_label')
    mask = sitk.ReadImage(url_mask)
    mask_array = sitk.GetArrayFromImage(mask)
    maskkk = np.zeros(mask_array.shape)
    maskkk[mask_array == 3] = 1
    # print(sum(sum(sum(maskkk))))
    i_crop = load_data.crop(dicom_array,mode='train')
    m_crop = load_data.crop(maskkk,mode='train')
    result = []
    for j in range(len(i_crop)):
        ij = i_crop[j][np.newaxis,:]
        mj = m_crop[j][np.newaxis,:]
        if np.mean(mj) != 0:
            result.append((ij,mj))
        elif np.mean(mj) == 0 and random.randint(0,9) > 6:
            result.append((ij,mj))
    print('这套数据分成{}个batch'.format(len(result)))
    return result


# @torchsnooper.snoop()
def train(this_epoch,train_list,model,optimizer):
    global log_step
    model.train()
    nums_data = len(train_list)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # print('sadadadadad',train_list)
    for i,name in enumerate(train_list):
        print('name',name)
        x_y_data = make_data(name)
        random.shuffle(x_y_data)  #打乱一套数据的crop
        check = np.array(x_y_data)
        print('x y data ====== ', check.shape)
        a_dice = 0
        for batch_i,x_y in enumerate(x_y_data):
            data_x = x_y[0][np.newaxis,:]
            data_y = x_y[1][np.newaxis,:]
            image = torch.from_numpy(data_x)
            image = image.type(torch.FloatTensor).cuda()
            target = torch.from_numpy(data_y)
            target = target.type(torch.FloatTensor).cuda()
            optimizer.zero_grad()  # 梯度清零
            output = model(image)  # n*2维
            # output = output.clone().detach().requires_grad_(True)  # 可求导
            target = target.view(target.numel())

            # focal loss
            pred_ = output[..., 1]
            # pred = pred_.clone().detach().requires_grad_(True)
            print('safadfada',pred_.sum(), target.sum())
            f_loss = def_loss.focal_loss(pred_, target)
            print('focal loss', f_loss)
            log.add_scalar('focal loss', float(f_loss), log_step)
            log_step += 1
            # print('global step',log_step)
            # a_focal_loss += f_loss

            dice = losses.dice_loss(output, target)
            a_dice += dice
            print('dice', dice)
            # log.add_scalar('dice1', float(dice),batch_i + i * len(name_list) + this_epoch *(batch_i*len(x_y_data)*len(name_list)))

            # loss_func = nll_loss
            # loss_func = f_loss
            # loss_func.backward()
            f_loss.backward()
            optimizer.step()

        log.add_scalar('dice', a_dice, i + this_epoch * nums_data)
        a_dice = 0
    params = model.state_dict()
    torch.save(params, model_save_path + '/vnet_params{}.pkl'.format(this_epoch))
    # torch.save(model,model_save_path + '/my_model{}.pth'.format(this_epoch))


def make_test_data(path,normalize = 'max_min'):
    image = sitk.ReadImage(path)
    dicom_array = sitk.GetArrayFromImage(image)
    # origin = np.array(image.GetOrigin())
    # spacing = np.array(image.GetSpacing())
    dicom_array = normalize_data(dicom_array)
    url_mask = path.replace('volume','segmentation')
    url_mask = url_mask.replace('test','test_label')
    mask = sitk.ReadImage(url_mask)
    mask_array = sitk.GetArrayFromImage(mask)
    maskkk = np.zeros(mask_array.shape)
    maskkk[mask_array == 3] = 1
    i_crop = load_data.crop(dicom_array,mode='test')
    m_crop = load_data.crop(maskkk,mode='test')
    result = []
    for j in range(len(i_crop)):
        ij = i_crop[j][np.newaxis, :]
        mj = m_crop[j][np.newaxis, :]
        if np.mean(mj) != 0:
            result.append((ij, mj))
        elif np.mean(mj) == 0 and random.randint(0, 9) > 6:
            result.append((ij, mj))
    print('这套数据分成{}个batch'.format(len(result)))
    return result


def for_try(this_epoch,datalist,model):
    global test_step
    print('====================test=========================')
    model.eval()
    nums_test_data = len(datalist)
    for j, path in enumerate(datalist):
        print('path',path)
        xydata = make_test_data(path)
        random.shuffle(xydata)
        t_nll_loss = 0
        t_dice = 0
        for batch_k, x_y in enumerate(xydata):
            data_x = x_y[0][np.newaxis, :]
            data_y = x_y[1][np.newaxis, :]
            image = torch.from_numpy(data_x)
            image = image.type(torch.FloatTensor).cuda()
            # target = data_y.type(torch.FloatTensor).long().cuda()
            target = torch.from_numpy(data_y)
            target = target.type(torch.FloatTensor).cuda()
            target = target.view(target.numel())
            optimizer.zero_grad()  # 梯度清零
            with torch.no_grad():  # 不求梯度,训练是不能写这句的，测试的时候才写
                output = model(image)  # n*2维   #sigmoid -> focal loss
            # focal loss
            pred_ = output[..., 1]
            pred = pred_.clone().detach().requires_grad_(True)
            f_loss = def_loss.focal_loss(pred, target)
            print('test focal loss', f_loss)
            log.add_scalar('test focal loss', float(f_loss), test_step)
            test_step += 1
            # print('global step', test_step)

            dice = losses.dice_loss(output, target)
            t_dice += dice
            print('test dice', dice)

        print('this data dice ', t_dice)
        log.add_scalar('test dice', t_dice, j + this_epoch * nums_test_data)
        t_dice = 0

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES = 1
    MIN_BOUND = -100
    MAX_BOUND = 600
    CROP_SIZE = [4, 208, 336]
    EPOCH = 100
    LR = 0.001 * 5
    MOMENTUM = 0.5
    model = vnet.VNet(elu=False, nll=False)
    # model.load_state_dict(torch.load('artery_params_1/vnet_params11.pkl'))
    # model.load_state_dict(torch.load('artery_params_2/vnet_params0.pkl'))
    model.cuda()

    # weight_decay = float(1e-8)


    log = tensorboardX.SummaryWriter('./logs_artery/log10')
    log_step = 0
    test_step = 0

    train_url = '/home/hym/Documents/yankai/dataset/all'
    test_url = '/home/hym/Documents/yankai/dataset/test'

    names = os.listdir(train_url)
    url_list = [train_url + '/' + name for name in names]
    print(url_list)
    print('训练数据{}套'.format(len(url_list)))

    test_ = os.listdir(test_url)
    test_url_list = [test_url + '/' + tt for tt in test_]
    # print(test_url_list)
    print('测试数据{}套'.format(len(test_url_list)))

    model_save_path = 'artery_params_10'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    for epo in range(EPOCH):
        model.train()
        print('================================================================')
        print('epoch {}'.format(epo))
        if epo % 5 == 0:
            LR /= 2
            # MOMENTUM *= 1.2
            log.add_scalar('learning rate',float(LR),epo)
        optimizer = optim.SGD(model.parameters(), lr=LR)  #,momentum=0.5,weight_decay=1e-8)    #sgd
        # optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-8)    #adam
        random.shuffle(url_list)  # 每一个epoch之前打乱数据顺序
        this_list = url_list
        train(epo,this_list,model,optimizer)
        # params = model.state_dict()
        # torch.save(params, model_save_path + '/vnet_params{}.pkl'.format(epo))

        #test
        testlist = test_url_list
        for_try(epo,testlist,model)