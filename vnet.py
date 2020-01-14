import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

# @torchsnooper.snoop()
def passthrough(x, **kwargs):
    return x

# @torchsnooper.snoop()
def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# @torchsnooper.snoop()
# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)    #raise error

    def forward(self, input):
        self._check_input_dim(input)       #raise error
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

# @torchsnooper.snoop()
class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)

# @torchsnooper.snoop()
#step1
class InputTransition(nn.Module):  #1*[128,128,64] -> 16*[128,128,64] ====== 1*[8,512,512] -> 16*[8*512,512]
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))   #input [1,1,16,128,128]--> output [1,16,16,128,128]
        # out = self.conv1(x)   #input [1,1,16,128,128]--> output [1,16,16,128,128]
        # print('//////',out.shape)
        # split input in to 16 channels
        # x16 = torch.cat((x, x, x, x, x, x, x, x,
        #                  x, x, x, x, x, x, x, x), 0)
        x16 = torch.cat((x, x, x, x,x,x,x,x), 0)         #######x input 有做bn，后面做add，所以x要不要做bn？
        # print('x16 size',x16.shape)     #[16, 1, 16, 128, 128]
        ## out = self.relu1(torch.add(out, x16))  #[16,16,16,128,128]
        ####
        aa = torch.add(out,x16)
        out = self.relu1(aa)

        # print('final out size',out.shape)
        return out

# @torchsnooper.snoop()
#step2
class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False,dilation = (1,1,1),padding = (0,0,0),stride = 2):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride= stride,dilation=dilation,padding=padding)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d(p = 0.1)                    # null -> 0.5
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        x1 = self.down_conv(x)
        x2 = self.bn1(x1)
        down = self.relu1(x2)
        # down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out

# @torchsnooper.snoop()
class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False,padding = (0,0,0),dilation = (1,1,1),stride=2):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride= stride,padding = padding,dilation=dilation)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d(p = 0.1)               #null -> 0.5
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        # print('check input size',x.shape)  #[1,15,15]
        out = self.do1(x)    #[1,15,15]
        # print('check up size',out.shape)
        skipxdo = self.do2(skipx)    #[2,31,31]

        a1 = self.up_conv(out)  #[2,30,30]    #[4,63,63]
        # print('a1,size',a1.shape)
        a2 = self.bn1(a1)   #[2,30,30]
        # print('a2 size',a2.shape)
        out = self.relu1(a2)   #[2,30,30]
        # out = self.relu1(self.bn1(self.up_conv(out)))  #[2,30,30]
        # print('check up sample size',out.shape,'\n',skipxdo.shape)
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

# @torchsnooper.snoop()
class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        # print('last input ',x.shape)    #[16, 32, 16, 128, 128]
        out = self.relu1(self.bn1(self.conv1(x)))
        # print('check ffffist',out.shape)    #[16, 2, 16, 128, 128]
        out = self.conv2(out)
        # print('check first',out.shape)    #[16, 2, 16, 128, 128]
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # print('check there',out.shape)    #[16, 16, 128, 128, 2]
        # flatten
        # out = out[0]
        out = torch.mean(out, dim=0)    #dim
        # print('check shape',out.shape)
        out1 = out[...,0]
        out2 = out[...,1]
        # print('shashaiodfha',out1.shape,out2.shape)
        out1 = out1.view(out1.numel(),1)
        out2 = out2.view(out2.numel(),1)
        # print('out1 shape', out1.shape)
        # print('out2 shape', out2.shape)

        out = torch.cat((out1,out2),1)
        # print('check shape', out.shape)
        # out = torch.mean(out,dim=0)    #[16,128,128,2]       #add！！！！！！
        # out = out.view(out.numel() // 2, 2)   #out.numel()所有元素的个数  #shape: [  ,2]
        # print('///////',out.shape)
        # out = self.softmax(out,dim=1)    #加了dim=1，softmax是判断这个点是属于前景还是背景，后续如果要用到nll loss则最后一层要用log softmax
        log_softmax_out = F.log_softmax(out)
        # print('output logsoftmax',log_softmax_out)
        softmax_out = F.softmax(out,dim=1)
        # print('output1',softmax_out)
        s = torch.nn.Sigmoid()
        sigmoid_out = s(out)
        # print('return out ',sigmoid_out)
        # treat channel 0 as the predicted output
        output = softmax_out
        return output

# @torchsnooper.snoop()
class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu) #1*[128,128,64] -> 16*[128,128,64] ====== 1*[8,512,512] -> 16*[8*512,512]
        self.down_tr32 = DownTransition(16, 1, elu)  #16*[128,128,64]
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True,dilation = (2,2,2),padding=(1,1,1),stride = 1)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True,dilation= (2,2,2),padding=(1,1,1),stride = 1)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True,dilation=(2,2,2),padding = (1,1,1),stride = 1)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True,dilation=(2,2,2),padding=(1,1,1),stride = 1)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)   #outchan 32
        self.out_tr = OutputTransition(32, elu, nll)   #传入的nll要为true，因为后面要用到nll loss，nll loss最后一层要用log softmax

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        # print('input size',x.shape)
        out16 = self.in_tr(x)
        # print('layer 1 dim',out16.shape)
        out32 = self.down_tr32(out16)
        # print('layer 2 dim', out32.shape)
        out64 = self.down_tr64(out32)
        # print('layer 3 dim', out64.shape)
        out128 = self.down_tr128(out64)
        # print('layer 4 dim', out128.shape)
        out256 = self.down_tr256(out128)
        # print('layer 5 dim', out256.shape)
        out = self.up_tr256(out256, out128)
        # print('uplayer 1 dim', out.shape)
        out = self.up_tr128(out, out64)
        # print('uplayer 2 dim', out.shape)
        out = self.up_tr64(out, out32)
        # print('uplayer 3 dim', out.shape)
        out = self.up_tr32(out, out16)
        # print('uplayer 4 dim', out.shape)
        out = self.out_tr(out)
        # print('uplayer 5 dim', out.shape)
        return out