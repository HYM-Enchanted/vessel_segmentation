import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

def dice_coef(input,target):
    smooth = 0.000001
    # _,res = input.min(1)  # input输入为2维，按第一维度的结果作为预测
    _,res = input.max(1)  # input输入为2维，按第一维度的结果作为预测
    # print('resssss',res)
    res = torch.squeeze(res)  # 拉成一维
    result = torch.cuda.FloatTensor(res.size())
    target = torch.cuda.FloatTensor(target.size())
    input_flat = torch.flatten(result)
    target_flat = torch.flatten(target)
    print('match',input_flat.shape,target_flat.shape)
    # print('?',input_flat.requires_grad)
    # print('?',target_flat.requires_grad)
    input_flat = input_flat.clone().detach().requires_grad_(True)
    target_flat = target_flat.clone().detach().requires_grad_(True)
    # print('?', input_flat.requires_grad)
    # print('?', target_flat.requires_grad)
    # input_flat = torch.tensor(input_flat, requires_grad=True)
    # target_flat = torch.tensor(target_flat, requires_grad=True)
    intersection = torch.dot(input_flat, target_flat)      #input 和target重叠的部分
    # print('===========', intersection,)
    # print('ssususu',input_flat.sum(),target_flat.sum())
    union = input_flat.sum() + target_flat.sum()
    dice = 2 * (intersection.sum() + smooth) / (union + smooth)
    print('dice 2 * {} / {} = {}'.format(intersection, union, dice))
    dice_loss = 1 - dice
    print('ddddddd', dice_loss)
    return dice_loss


def deneralized_dice_loss(input,target):
    smooth = 0.000001
    _, res = input.min(1)  # input输入为2维，按第一维度的结果作为预测
    # print('resssss',res)
    res = torch.squeeze(res)  # 拉成一维
    result = torch.cuda.FloatTensor(res.size())
    target = torch.cuda.FloatTensor(target.size())
    input_flat = torch.flatten(result)
    target_flat = torch.flatten(target)




def sigmoid_focal_loss(pred,target,weight = None,
                       gamma = 2.0,alpha = 0.25,
                       reduction = 'mean',
                       avg_factor = None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha)* (1-target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none') * focal_weight
    loss = weight_reduce_loss(loss,weight,reduction,avg_factor)
    return loss

# @torchsnooper.snoop()
def focal_loss(pred,target,gamma=2.0,alpha_ = 0.25,size_average = False):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)

    pred = torch.cat((1 - pred, pred), dim=1)  # 将模型预测的正负概率都算出来
    class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
    class_mask.scatter_(1, target.data.long(), 1.)  #按照标签写成两维
    # class_mask.scatter_(1, target.view(-1,1).long(), 1.)
    # print('////',class_mask)
    # probs = (pred.float() * class_mask.float()).sum(dim=1).view(-1, 1)
    probs = (pred * class_mask).sum(dim=1).view(-1, 1)
    print('probs',probs)    #sum???????
    probs = probs.clamp(min=0.0001, max=1.0)  # 区间限定函数，
    log_p = probs.log()

    alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
    alpha[:, 0] = alpha[:, 0] * (1 - alpha_)
    alpha[:, 1] = alpha[:, 1] * alpha_
    alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

    batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p

    if size_average:
        loss = batch_loss.mean()
    else:
        loss = batch_loss.sum()
    return loss


class FocalLoss(nn.Module):
    def __init__(self,alpha = 0.25, gamma = 2, size_average = True):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self,pred, target):
        # pred = nn.Sigmoid(pred)

        pred = pred.view(-1,1)
        target = target.view(-1,1)

        pred = torch.cat((1-pred,pred),dim=1)  #将模型预测的政府概率都算出来
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        class_mask.scatter_(1,target.view(-1,1).long(),1.)

        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min = 0.0001,max = 1.0)    #区间限定函数，
        log_p = probs.log()

        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1 - self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        batch_loss = -alpha*(torch.pow((1-probs),self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else :
            loss = batch_loss.sum()
        return loss


def generalized_dice_loss(pred,target):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    nc1 = pred.shape[-1]
    w = torch.zeros(shape=(nc1,))
    w = torch.sum(target,axis=())