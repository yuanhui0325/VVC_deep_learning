import torchvision
import torch

import torch.nn as nn
import argparse

# # Neuron Attention # #
class Neuron_Attention(nn.Module):
    def __init__(self, channel):
        super(Neuron_Attention, self).__init__()
        self.dw_conv = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.pw_conv = nn.Conv2d(channel, channel, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.relu(self.dw_conv(x))
        y = self.pw_conv(y)
        y = self.sigmoid(y)
        return x * y

class MSRB_NA_Block(nn.Module):
    def __init__(self, n_feats):
        super( MSRB_NA_Block, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        self.conv_3_1=nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=(kernel_size_1 // 2), bias=True)
        #self.conv_3_2=nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_1, padding=(kernel_size_1 // 2), bias=True)
        self.NA1=Neuron_Attention(n_feats*2)
        self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size_2,padding=(kernel_size_2 // 2), bias=True)
        #self.conv_5_2=nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_2,padding=(kernel_size_2 // 2), bias=True)
        self.NA2 = Neuron_Attention(n_feats)
        self.confusion = nn.Conv2d(n_feats*2 , n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        #output_3_2 = self.relu(self.conv_3_2(input_2))
        output_3_2=self.NA1(input_2)
        #output_5_2 = self.relu(self.conv_5_2(input_2))
        #output_5_2 = self.NA2(input_2)
        #input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(output_3_2)
        output += x
        return output


class MSRB_NA(nn.Module):
    def __init__(self, args):
        super(MSRB_NA, self).__init__()

        n_feats = 64
        n_blocks = 8
        kernel_size = 3
        self.n_blocks = n_blocks
        #define shallow module
        self.modules_shallow1=nn.Conv2d(args.n_colors, n_feats, 3, padding=(kernel_size // 2), bias=True)
        self.modules_shallow2 = nn.Conv2d(n_feats, n_feats, 3, padding=(kernel_size // 2), bias=True)

        # define head module
        modules_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB_NA_Block(n_feats=n_feats))

        # define tail module
        modules_tail = nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        #define reconstruction layer
        self.modules_rec1 = nn.Conv2d(n_feats, n_feats, 3, padding=(kernel_size // 2), bias=True)
        self.modules_rec2 = nn.Conv2d(n_feats, 1, 3, padding=(kernel_size // 2), bias=True)

    def forward(self, x1,x2):
        x=torch.cat([x1,x2],1)
        x_shallow1 = self.modules_shallow1(x)
        x_shallow2 = self.modules_shallow2(x_shallow1 )
        res =  x_shallow2

        MSRB_out = []
        for i in range(self.n_blocks):
            x_shallow2= self.body[i] (x_shallow2)
            MSRB_out.append( x_shallow2)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out, 1)
        y = self.tail(res)
        y_rec1=torch.relu(self.modules_rec1(y))
        y_rec2 = torch.tanh(self.modules_rec2( y_rec1))
        return y_rec2
parser = argparse.ArgumentParser(description="PyTorch MAR_NA")
parser.add_argument("--n_colors", default=2, type=float, help="channel nums: 2")
global opt
opt = parser.parse_args()
model1=MSRB_NA(opt)
model1.eval()
#print(model1.state_dict())#输出网络模型的参数
model1.load_state_dict(torch.load('G:\\idea2_质量增强\\result\\MSR_NA_luma\\89net_param.pkl',map_location='cpu'))
#8x8
#example_y = torch.ones(1, 1, 4, 4)
#example_ref=torch.ones(1,3,9,1)
#4x4
#example_y = torch.ones(1, 1, 2, 2)
#example_ref=torch.ones(1,3,5,1)
#16x16
#example_y = torch.ones(1, 1, 8, 8)
#example_ref=torch.ones(1,3,17,1)
#32x32
#example_y = torch.ones(1, 1, 16, 16)
#example_ref=torch.ones(1,3,33,1)
#64x64
#example_y = torch.ones(1, 1, 32, 32)
#example_ref=torch.ones(1,3,66,2)
#128x128
inputy = torch.ones(1, 1, 128, 128)
qp=torch.ones(1, 1, 128, 128)

out1=model1(inputy,qp)
print(out1)
traced_script_module = torch.jit.trace(model1, (inputy,qp))
#print(traced_script_module )
out2=traced_script_module(inputy,qp)
print(out2)
print(out1==out2)
traced_script_module.save("MSR_NA_89th.pt")
#print(model1.load_state_dict)'''








