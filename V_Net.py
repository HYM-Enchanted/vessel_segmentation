import torch
import torch.nn as nn
import torch.nn.functional as F

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):  #input size = output size
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class step1(nn.Module):
    def __init__(self,elu):
        super(step1,self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            padding=2
        )
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu,16)
    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out

class step2(nn.Module):
    pass