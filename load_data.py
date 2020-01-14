import numpy as np
import os
import SimpleITK as sitk
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
import torch
import torch.optim as optim
import vnet
import losses
import def_loss
import torch.nn.functional as F
import torch.nn as nn
import tensorboardX
import random
import gc

gc.collect()
import torchsnooper


MIN_BOUND = 0
MAX_BOUND = 900

CROP_SIZE = [4, 208, 336]

def normalize_ct_array(array):
    array[array<MIN_BOUND] = MIN_BOUND
    array[array>MAX_BOUND] = MAX_BOUND
    return array

def set_window(image,out_max):  #设置统一灰度范围
    s_max = np.max(image)
    s_min = np.min(image)
    image_out = (image-s_min)/(s_max-s_min)
    image_out = image_out * out_max
    return image_out

def load_dicom(file):  #读一整套dicom文件的原始array，spacing和origin
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(file)
    dicom_names = reader.GetGDCMSeriesFileNames(file, seriesIDs[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    image_array = sitk.GetArrayFromImage(image)  # z, y, x

    origin = image.GetOrigin()  # x, y, z
    spacing = image.GetSpacing()  # x, y, z
    return image_array,origin,spacing

def cal_weight(mask_mean):
    '''
    计算loss函数的weight
    :param mask_mean: 所有传入训练的mask的平均值，是一个小数
    :return: weight
    '''
    # bg_weight = mask_mean / (1. + mask_mean)  # background
    # fg_weight = 1. - bg_weight  # foreground
    # print(bg_weight)
    # class_weights = torch.FloatTensor([bg_weight, fg_weight])
    # class_weights = class_weights.cuda()
    # print('class weight', class_weights)
    # print('mask mean',mask_mean)
    '''
       t_mean = my_train_data.mask_mean
       print('tmean:',t_mean)
       p_weight = torch.tensor((1-t_mean)/t_mean)
       p_weight = p_weight.cuda()
       print('pos weight',p_weight)
       '''
    rate = 1 / mask_mean
    print('rate',rate)  #365
    rate3 = rate    #3倍，1000多
    # w1 = 1/rate3
    nll_weight = torch.tensor([1,rate])
    print('weight',nll_weight)
    return nll_weight

def make_data(url,mode):  #load all train data and test data and their mask
    data_list = []
    image_array_crop = []
    # image_dict ={}
    mask_array_crop = []
    # mask_dict ={}
    info_dict ={}
    cal_mean = []
    for file in os.listdir(url):
        print('file name',file)
        data_list.append(file)
        path = os.path.join(url,file) + '/original1'
        #process origin array
        dcm_array,origin,spacing = load_dicom(path)   #读取数据
        # image_dict[file] = dcm_array
        info_dict[file] = [origin, spacing]
        dcm_array = normalize_ct_array(dcm_array)  #截断[0,800]
        dcm_array = set_window(dcm_array, 255)  # 统一灰度值范围到0-255
        res = crop(dcm_array,mode=mode)    #切成一个batch一个batch
        crop_len = len(res)
        # print(file + '切成{}块'.format(crop_len))
        for i in range(0,crop_len):
            # series = file + '_{}'.format(i)  # new key for crop is the image file name + series
            # image_dict[series] = res[i]
            a = res[i][np.newaxis,:]     #加一个维度，Channel为1 ,shape:[1,cropsize0,cropsize1,cropsize2]
            image_array_crop.append(a)


        #process label
        mask_path = os.path.join(url,file) + '/vein'
        mask_array,origin,spacing = load_dicom(mask_path)   #读标签
        #calculate the bias of foreground and background
        target_mean = np.mean(mask_array) #[:,152:360,88:424])
        cal_mean.append(target_mean)      #object占总体体积的比率
        # mask_dict[file] = mask_array
        res_label = crop(mask_array,mode=mode)  #标签切成一个一个batch
        crop_len1 = len(res_label)
        # assert crop_len == crop_len1
        # print(file + '标签切成{}块'.format(crop_len1))
        for j in range(0,crop_len1):
            # series = file + '_{}'.format(j)  #new key for crop is the image file name + series
            # mask_dict[series] = res[j]
            b = res_label[j][np.newaxis,:]  #channel
            mask_array_crop.append(b)
    cal_mean = np.array(cal_mean)
    final_mean = np.mean(cal_mean)     #所有数据的object占比
    result = []
    for i in range(len(image_array_crop)):
        x_data = image_array_crop[i]
        y_data = mask_array_crop[i]
        print('mask sum',sum(sum(sum(sum(y_data)))))  #计算
        this_sum = sum(sum(sum(sum(y_data))))
        if this_sum != 0:   #如果这个crop有object，则加入训练集
            result.append((x_data,y_data))    #result每一个位上存放对应的原始数据和label
        elif this_sum == 0 and random.randint(0,9) > 5:  #如果这个crop没有object，则按照一定概率加入训练集
            print(i,' add into')
            result.append((x_data, y_data))
    print('{}一共切成{}块'.format(mode,len(result)))  #1020个batch
    # print(len(result[0]),'----',len(result[9]))
    #check data 的维度
    # rrr = np.array(result)
    # print('dataset shape',rrr.shape)   #n,2,1,8,208,336  ,n个batch
    #释放内存
    del image_array_crop
    del mask_array_crop
    del data_list
    del info_dict
    gc.collect()

    return result,final_mean


def crop(array,mode = 'train'):   #将数据crop成一个一个batch
    # print('crop now')
    c,w,h = CROP_SIZE[0],CROP_SIZE[1],CROP_SIZE[2]  #[4, 208, 336]
    # c = 16
    # w , h = 128,192
    # print(c,w,h)
    if mode != 'train':
        stride = c
    else:
        stride = int(c/2)
    crop_result = []
    z,x,y = array.shape
    start = 0
    l1 = int((512 / 2) - (w / 2))
    l2 = int((512 / 2) + (w / 2))
    l3 = int((512 / 2) - (h / 2))
    l4 = int((512 / 2) + (h / 2))

    while start < (z-c):
        # print('start',start)
        temp = array[start:start + c,l1:l2,l3:l4]
        # print(temp.shape)   #size = CROP_SIZE
        crop_result.append(temp)
        start += stride
    if start > (z-c) and start != z:
        # print('start',start)
        temp = array[z-c:,l1:l2,l3:l4]
        crop_result.append(temp)
    # print('cropppppp',len(crop_result))
    return crop_result


class vessel_data(data.Dataset):
    def __init__(self,mode = 'train',train_root = '/home/hym/Documents/vessel_data/train_data',
                 test_root = '/home/hym/Documents/vessel_data/test_data'):
        if mode == 'train': #训练模式
            x_y_data,m_mean = make_data(train_root,mode='train')
            self.len = len(x_y_data)    #所有数据一共分成了多少个crop
            self.data = x_y_data
            self.mask_mean = m_mean
        elif mode == 'test':   #test 模式的mask mean用训练data的mask mean
            test_x_y_data,mask_mean = make_data(test_root,mode='test')
            self.len = len(test_x_y_data)
            self.data = test_x_y_data

    def  __getitem__(self,index):
        return self.data[index]

    def __len__(self):  #data len
        return self.len

@torchsnooper.snoop()
def train(epoch,model,trainLoader,optimizer,posweight):
    model.train()   #训练模式
    train_dice_loss = 0
    train_bce_loss = 0
    train_incorrect = 0
    train_nll_loss = 0
    nums_data = len(trainLoader)
    print('train batch ',nums_data)
    for batch_i, data in enumerate(trainLoader):
        print('========================================================================')
        img, mask = data
        # print('check train shape',img.shape,mask.shape)
        # print('mask.type',mask.type)
        image = img.type(torch.FloatTensor).cuda()
        target = mask.type(torch.FloatTensor).long().cuda()
        optimizer.zero_grad()   #梯度清零
        # with torch.no_grad():   #不求梯度,训练是不能写这句的，测试的时候才写
        output = model(image)   #n*2维
        # output = torch.tensor(output,requires_grad=True)
        output = output.clone().detach().requires_grad_(True)   #可求导
        print('=====',output.shape)
        target = target.view(target.numel())   #一维
        print('-----',target.shape)
        """
        #todo 把target从一维张量变成2维张量
        one = torch.ones(target.shape).long().cuda()
        secs = torch.sub(one,target)
        label = torch.stack((target,secs),0)  #max 所以第二维作为预测值  没改
        label = torch.t(label).float()   #转置，变成 n*2维
        label = label.clone().detach().requires_grad_(False)   #bceloss 的target不能有梯度
        # label = label.cuda()
        ########## bceloss = F.binary_cross_entropy(output,label)
        ##bce loss + weights
        creterion = torch.nn.BCELoss(weight=posweight)  #vent 网络最后一层是softmax，不能再用sigmoid的loss
        # creterion = torch.nn.BCEWithLogitsLoss(pos_weight=posweight)   #加上sigmoid的
        # bceloss = creterion(output,label)
        """
        #
        aa = torch.zeros(target.shape).cuda()
        print(aa.shape,"23qrfsdrfgs",target.shape)
        bb = torch.cat((aa,target),0)
        tt = torch.t(bb)
        print('asdfadf',tt.shape)
        # tt = target.view(target.numel(),1)
        output[tt==1] *= 100
        nll_loss = F.nll_loss(output,target,weight=posweight)
        print('=========',nll_loss)
        ######focal loss sigmoid output
        # _,aaa = output.data.max(1)
        # f_loss_func = def_loss.FocalLoss(aaa,target)
        # f_loss_func.backward()
        ##
        dice = losses.dice_error(output,target)   #dice 他的loss=1-dice
        dice_loss = def_loss.dice_coef(output,target)   #dice loss
        log.add_scalar('dice',float(dice),batch_i + epoch*nums_data)
        # log.add_scalar('BCE loss',float(bceloss),batch_i + epoch*nums_data)
        ##log.add_scalar('dice loss',float(dice_loss),batch_i + epoch * nums_data)
        # loss_func = bceloss + dice_loss                                     #损失函数定义为bce loss + dice loss
        # loss_func = dice_loss   #learning rate = 0.0001 log13
        # loss_func = bceloss    #learning rate = 0.0001  log12  误分割的很多，巨多，分割结果一层一层的，跟crop data一样，一层一层的
        loss_func = nll_loss
        log.add_scalar('LOSS',float(loss_func),batch_i + epoch * nums_data)
        loss_func.backward()
        optimizer.step()
        _,pred = output.data.max(1)   #每个像素点的预测值，按dim1来看是这个点属于前景，背景的概率，返回概率大的点的index，若为0，则是前景，若为1则是背景
        # pred = output.data.min(1)[1]   #每个像素点的预测值，按dim1来看是这个点属于前景，背景的概率，返回概率小的点的index，若为0，则是背景，若为1，则是前景
        predsum = torch.sum(pred).item()
        should = torch.sum(target).item()
        log.add_scalars('pred and label dots',{'predict':predsum,'target':should},batch_i + epoch * nums_data)

        print(' pred sum',torch.sum(pred).item())
        print(' target sum',torch.sum(target).item())   #target！=0的error才有意义
        incorrect = pred.ne(target.data).cpu().sum()   #incorrect   #不相等的
        correct = pred.eq(target.data).cpu().sum()    #相等的点
        print('check incorrect',incorrect.item(),'correct',correct.item())#'pred',pred,'\n target',target)
        err = 100. * incorrect / target.numel()  # 分错的点的数目
        # err = 100. * incorrect / torch.sum(target).item() # 分错的点的数目
        print('Epoch {} / batch {}  Dice :{:.3f}  error：{} / {} = {}%'.format(epoch,batch_i,dice,incorrect,target.numel(),err))

        #
        pred_fl = pred.view(-1,1).cuda()
        target_fl = mask.view(-1,1).cuda().long()
        tp = torch.sum(torch.mul(pred_fl,target_fl))  #分对的点
        fp = torch.sum((torch.sub(pred_fl,target_fl))==1)  #误分
        fn = torch.sum((torch.sub(target_fl,pred_fl))==1) #漏分
        print(tp,fp,fn)
        if should == 0:
            should = 1
        a = 100.* tp / should
        b = 100.* fp / target.numel()
        c = 100.* fn / target.numel()
        print('分对占比{} / {} = {}%'.format(tp,should,a))
        print('误分占比{} / {} = {}%'.format(fp,target.numel(),b))
        print('漏分占比{} / {} = {}%'.format(fn,target.numel(),c))

        train_dice_loss += dice.data.item()
        # train_bce_loss += bceloss.data.item()
        train_incorrect += incorrect
        train_nll_loss += nll_loss

        # 每个batch中间存个model，每隔400存一次
        # if batch_i % 400 == 0 and batch_i != 0:
        #     t = batch_i // 400
        #     params = model.state_dict()
        #     torch.save(params, model_save_path + '/vnet_params{}-{}.pkl'.format(epoch,t))
    log.add_scalar('epoch nll loss',float(train_nll_loss),epoch)
    train_nll_loss = 0
    log.add_scalar('epoch dice loss ',float(train_dice_loss),epoch)
    train_dice_loss = 0
    train_incorrect = 0
    epcoh_dice_loss = train_dice_loss / len(trainLoader)
    epoch_bce_loss = train_bce_loss / len(trainLoader)
    # epoch_correct =    #有重复的，所以不知道要怎么算了。。。。
    print('whole epoch {} , Dice {}, BCE loss{}'.format(epoch,epcoh_dice_loss,epoch_bce_loss))


def test(model,testLoader,optimizer,b_f_weights):
    model.eval()
    test_loss = 0
    dice_loss = 0
    incorrect = 0
    numel = 0
    for i,data in enumerate(testLoader):
        test_x,test_y = data
        img = test_x.type(torch.FloatTensor).cuda()
        # img = img.cuda()
        target = test_y.type(torch.FloatTensor).long().cuda()
        # with torch.no_grad():
        output = model(img)
        target = target.view(target.numel())
        one = torch.ones(target.shape).long().cuda()
        secs = torch.sub(one, target)
        label = torch.stack((target, secs), 0)
        label = torch.t(label)
        label = label.float()
        bceloss = F.binary_cross_entropy(output,label)
        dice_l = 1 - losses.dice_error(output,target)
        dice_loss += dice_l
        test_loss += bceloss
        # test_loss += F.nll_loss(output,target,weight=b_f_weights)

        numel += target.numel()    #所有该分出来的点
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()  #不正确分出来的点
    test_loss /= len(testLoader)
    dice_loss /= len(testLoader)
    err = 100. * incorrect / numel
    print('Test Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
        test_loss, incorrect, numel, err, dice_loss))
    return err

if __name__ == '__main__':
    train_url = '/home/hym/Documents/vessel_data/pick'
    # train_url = '/home/hym/hym/vessel_train_data'
    test_url = '/home/hym/Documents/vessel_data/test_data'
    CUDA_VISIBLE_DEVICES = 1
    MIN_BOUND = 0
    MAX_BOUND = 800
    CROP_SIZE = [4, 208, 336]
    EPOCH = 100
    LR = 0.001 * 5
    log = tensorboardX.SummaryWriter('./logs/log23')  #log6 bce + dice loss ,without finetune

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    my_train_data = vessel_data(mode = 'train',train_root=train_url,test_root=test_url)
    trainLoader = DataLoader(my_train_data, batch_size=1, shuffle=True,num_workers=1, pin_memory=True)
    #load test data
    # my_test_data = vessel_data(mode='test',train_root=train_url,test_root=test_url)
    # testLoader = DataLoader(my_test_data,batch_size=1,shuffle=True,num_workers=1, pin_memory=True)

    model = vnet.VNet(elu=False, nll=False)    #如果要用到nll loss ，最后一层要用log softmax作为输出
    # model.load_state_dict(torch.load('model_save18/vnet_params59.pkl'))   #finetune BY parameter
    # model = torch.load('model_save/vnet0.pkl')    #finetune
    model.cuda()
    # optimizer = optim.Adam(model.parameters(),lr=LR)
    # '''
    t_mean = my_train_data.mask_mean
    class_weight = cal_weight(t_mean).cuda()

    #train
    model_save_path = 'model_save22'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    for epoch in range(EPOCH):
        if epoch % 5 == 0:
            LR = LR * 0.2
        log.add_scalar('learning rate',LR,epoch)
        optimizer = optim.Adam(model.parameters(), lr=LR)   #考虑将优化器换成SGD的
        train(epoch,model,trainLoader,optimizer,class_weight)
        params = model.state_dict()
        torch.save(params, model_save_path + '/vnet_params{}.pkl'.format(epoch))
    #test 一个epoch做一次test  不做测试先，内存不够
        # print('test now')
        # test_err = test(model,testLoader,optimizer,class_weights)
        # print('test err',test_err)
        # if epoch == 0:
        #     best_err = test_err
        # else:
        #     if test_err < best_err:
        #         best_err = test_err
        #         print('epoch {} model is the best !!!!!!!!'.format(epoch))

    log.close()

