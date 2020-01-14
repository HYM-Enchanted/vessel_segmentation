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

def make_data(name):
    url_data = name + '/original1'
    url_mask = name + '/vein'
    a,b,c = load_data.load_dicom(url_data)
    a = load_data.normalize_ct_array(a)
    a = load_data.set_window(a,255)      #对原始数据做normalized
    e,_,_ = load_data.load_dicom(url_mask)
    a_crop = load_data.crop(a)     #crop之后，框出了肺部，切成了小个小个的batch[4,208,336]
    e_crop = load_data.crop(e)
    result = []
    for i in range(len(a_crop)):
        ai = a_crop[i][np.newaxis,:]
        ei = e_crop[i][np.newaxis,:]
        # print('ssasa',np.mean(ei))
        if np.mean(ei) != 0:  #说明有object
            result.append((ai,ei))
        elif np.mean(ei) == 0 and random.randint(0,9) > 6: #如果没有object则按照一定比例放进去训练
            result.append((ai,ei))
    print('这套数据分成{}个batch'.format(len(result)))
    return result

# @torchsnooper.snoop()
def train(this_epoch,name_list,model):
    global log_step
    model.train()
    nll_weight = torch.tensor([1,100]).float().cuda()
    nums_data = len(name_list)
    for i,name in enumerate(name_list):
        #load data and label
        x_y_data = make_data(name)   #n,2,1,4,208,336
        random.shuffle(x_y_data)
        check = np.array(x_y_data)
        print('x y data ====== ',check.shape)
        a_nll_loss = 0
        a_dice = 0
        a_focal_loss = 0
        for batch_i,x_y in enumerate(x_y_data):
            data_x = x_y[0][np.newaxis,:]  #data
            data_y = x_y[1][np.newaxis,:]  #label
            # image = data_x.type(torch.FloatTensor).cuda()
            image = torch.from_numpy(data_x)
            image = image.type(torch.FloatTensor).cuda()
            # target = data_y.type(torch.FloatTensor).long().cuda()
            target = torch.from_numpy(data_y)
            target = target.type(torch.FloatTensor).cuda()
            optimizer.zero_grad()  # 梯度清零
            # with torch.no_grad():   #不求梯度,训练是不能写这句的，测试的时候才写
            output = model(image)  # n*2维
            # output = torch.tensor(output,requires_grad=True)
            output = output.clone().detach().requires_grad_(True)  # 可求导
            target = target.view(target.numel())
            #nll loss
            """
            outputcopy = output
            output0 = outputcopy[:,0]
            output1 = outputcopy[:,1] *100
            sss = torch.stack((output0,output1),0)
            sss = torch.t(sss)
            nll_loss = F.nll_loss(sss,target.long()) #,weight=nll_weight)  #weight就没起作用
            a_nll_loss += nll_loss
            print('nll loss   weight',nll_loss)
            # log.add_scalar('nll lose1', float(nll_loss),batch_i + i * len(name_list) + this_epoch *(batch_i*len(x_y_data)*len(name_list)))

            nll_loss1 = F.nll_loss(output, target.long()) #, weight=nll_weight)  # weight就没起作用
            # a_nll_loss += nll_loss1
            print('nll loss1  no weight', nll_loss1)
            """
            #bce loss + weight
            #sigmoid
            #weight
            # bceloss_func = torch.nn.CrossEntropyLoss()

            #focal loss
            pred_ = output[...,1]
            pred = pred_.clone().detach().requires_grad_(True)
            f_loss = def_loss.focal_loss(pred,target)
            print('focal loss',f_loss)
            log.add_scalar('focal loss',float(f_loss),log_step)
            log_step += 1
            # print('global step',log_step)
            # a_focal_loss += f_loss

            dice = losses.dice_loss(output,target)
            a_dice += dice
            print('dice',dice)
            # log.add_scalar('dice1', float(dice),batch_i + i * len(name_list) + this_epoch *(batch_i*len(x_y_data)*len(name_list)))

            # loss_func = nll_loss
            loss_func = f_loss
            loss_func.backward()
            optimizer.step()

        # log.add_scalar('nll loss',a_nll_loss,i + this_epoch*nums_data)
        # a_nll_loss = 0
        # log.add_scalar('focal loss',a_focal_loss, i +this_epoch*nums_data)
        # a_focal_loss = 0
        log.add_scalar('dice',a_dice,i +this_epoch*nums_data)
        a_dice = 0


def make_test_data(url):
    url_data = url + '/original1'
    url_mask = url + '/vein'
    a,b,c = load_data.load_dicom(url_data)
    a = load_data.normalize_ct_array(a)
    a = load_data.set_window(a,255)
    e,_,_ = load_data.load_dicom(url_mask)
    a_crop = load_data.crop(a,mode='test')
    e_crop = load_data.crop(e,mode='test')
    result = []
    for k in range(len(a_crop)):
        ak = a_crop[k][np.newaxis,:]
        ek = e_crop[k][np.newaxis,:]
        result.append((ak,ek))
    print('这套测试数据分成{}个batch'.format(len(result)))
    return result


def test(this_epoch,datalist,model):
    global test_step
    print('====================test=========================')
    model.eval()
    nums_test_data = len(datalist)
    for j,path in enumerate(datalist):
        xydata = make_test_data(path)
        random.shuffle(xydata)
        t_nll_loss = 0
        t_dice = 0
        for batch_k,x_y in enumerate(xydata):
            data_x = x_y[0][np.newaxis,:]
            data_y = x_y[1][np.newaxis,:]
            image = torch.from_numpy(data_x)
            image = image.type(torch.FloatTensor).cuda()
            # target = data_y.type(torch.FloatTensor).long().cuda()
            target = torch.from_numpy(data_y)
            target = target.type(torch.FloatTensor).cuda()
            target = target.view(target.numel())
            optimizer.zero_grad()  # 梯度清零
            with torch.no_grad():   #不求梯度,训练是不能写这句的，测试的时候才写
                output = model(image)  # n*2维   #sigmoid -> focal loss
            #nll loss
            """
            outputcopy = output
            output0 = outputcopy[:, 0]
            output1 = outputcopy[:, 1] * 1000
            sss = torch.stack((output0, output1), 0)
            sss = torch.t(sss)
            nll_loss = F.nll_loss(sss, target.long())  # ,weight=nll_weight)  #weight就没起作用
            t_nll_loss += nll_loss
            print('test  nll loss   weight', nll_loss)

            nll_loss1 = F.nll_loss(output, target.long())  # , weight=nll_weight)  # weight就没起作用
            # a_nll_loss += nll_loss1
            print('nll loss1  no weight', nll_loss1)
            """
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

        # print('this data nll loss',t_nll_loss)
        # log.add_scalar('test nll loss', t_nll_loss, j + this_epoch * nums_test_data)
        # t_nll_loss = 0
        print('this data dice ',t_dice)
        log.add_scalar('test dice', t_dice, j + this_epoch * nums_test_data)
        t_dice = 0

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES = 1
    MIN_BOUND = 0
    MAX_BOUND = 800
    CROP_SIZE = [4, 208, 336]
    EPOCH = 100
    LR = 0.0001 * 5
    model = vnet.VNet(elu=False, nll=False)
    # model.load_state_dict(torch.load('param_save4/vnet_params5.pkl'))
    # model.load_state_dict(torch.load('param_save6/vnet_params36.pkl'))
    # model.load_state_dict(torch.load('param_save8/vnet_params16.pkl'))
    model.load_state_dict(torch.load('param_save9/vnet_params8.pkl'))
    model.cuda()

    # log = tensorboardX.SummaryWriter('./logssss/logtry')
    log = tensorboardX.SummaryWriter('./logssss/log10')
    log_step = 0
    test_step = 0
    # global log_step
    #文件list，并打乱
    train_Data_list = '/home/hym/hym/vessel_train_data'   #all in
    # train_Data_list ='/home/hym/Documents/vessel_data/train_data'  #一套数据
    names = os.listdir(train_Data_list)
    url_list = [train_Data_list + '/' + name for name in names]
    print(url_list)
    print('训练数据{}套'.format(len(url_list)))

    #test list
    test_Data_list = '/home/hym/Documents/vessel_data/test_data'
    test_ = os.listdir(test_Data_list)
    test_url = [test_Data_list + '/' + tt for tt in test_]
    print('测试数据{}套'.format(len(test_url)))


    #train
    # model_save_path = 'param_save_try'
    model_save_path = 'param_save10'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    for epo in range(EPOCH):
        model.train()
        print('================================================================')
        print('epoch {}'.format(epo))
        if epo % 5 == 0:
            LR /= 2
            log.add_scalar('learning rate',float(LR),epo)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        random.shuffle(url_list)  # 每一个epoch之前打乱数据顺序
        this_list = url_list
        train(epo,this_list,model)
        params = model.state_dict()
        torch.save(params, model_save_path + '/vnet_params{}.pkl'.format(epo))

        #test
        testlist = test_url
        test(epo,testlist,model)