import torch
import torch.nn as nn
from math import sqrt
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import argparse
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
class YUV:
    def __init__(self,height,width,y,u,v):
        self.height=height
        self.width=width
        self.y=y
        self.u=u
        self.v=v
def YUVread(path,h,w):
    y_all=np.uint8([])
    u_all=np.uint8([])
    v_all=np.uint8([])
    with open(path,'rb')as file:
        y=np.uint8(list(file.read(w*h)))
        u= np.uint8(list(file.read(w * h>>2)))
        v = np.uint8(list(file.read(w * h>>2)))
        y_all=np.concatenate([y_all,y])
        u_all=np.concatenate([u_all,u])
        v_all=np.concatenate([v_all,v])
        y_all=np.reshape(y_all,[1,h,w])
        u_all=np.reshape(u_all,[1,h>>1,w>>1])
        v_all = np.reshape(v_all, [1, h >> 1, w >> 1])
    return y_all,u_all,v_all
def cut_patch(video_in,patch_size,center_x_Y,center_y_Y ,center_x_UV, center_y_UV):
    patch_Y=video_in.y[0,center_y_Y:center_y_Y+patch_size,center_x_Y:center_x_Y+patch_size]
    patch_Y = np.reshape(patch_Y, [1, patch_size,patch_size])
    patch_Y = torch.from_numpy(patch_Y)
    patch_Y= patch_Y.float()
    patch_uv=patch_size//2
    patch_U=video_in.u[0,center_y_UV:center_y_UV+patch_uv,center_x_UV:center_x_UV+patch_uv]
    patch_U=np.reshape(patch_U,[1,patch_uv,patch_uv])
    patch_U=torch.from_numpy(patch_U)
    patch_U = patch_U.float()
    patch_V = video_in.v[0,center_y_UV:center_y_UV + patch_uv, center_x_UV:center_x_UV + patch_uv]
    patch_V=np.reshape(patch_V,[1,patch_uv,patch_uv])
    patch_V = torch.from_numpy(patch_V)
    patch_V = patch_V.float()
    return patch_Y, patch_U, patch_V
def cut_ref_left(video_in,center_x_Y,center_y_Y,center_x_UV,center_y_UV,rec_size,patchsize):
    patch_y_left=video_in.y[0,center_y_Y-rec_size:center_y_Y+patchsize,center_x_Y-rec_size:center_x_Y]

    patch_u_left=video_in.u[0,center_y_UV-rec_size//2:center_y_UV+patchsize//2,center_x_UV-rec_size//2:center_x_UV]
    patch_v_left = video_in.v[0, center_y_UV - rec_size // 2:center_y_UV + patchsize // 2,center_x_UV - rec_size // 2:center_x_UV]
    return patch_y_left,patch_u_left,patch_v_left
def cut_ref_up(video_in,center_x_Y,center_y_Y,center_x_UV,center_y_UV,rec_size,patchsize):
    patch_y_up=video_in.y[0,center_y_Y-rec_size:center_y_Y,center_x_Y:center_x_Y+patchsize]
    patch_u_up=video_in.u[0,center_y_UV-rec_size//2:center_y_UV,center_x_UV:center_x_UV+patchsize//2]
    patch_v_up = video_in.v[0, center_y_UV - rec_size // 2:center_y_UV ,center_x_UV :center_x_UV+patchsize//2]
    return patch_y_up,patch_u_up,patch_v_up
def YUVwrite(path,y,u,v):

    if type(y) is not np.ndarray:
        y=y.numpy()
        y = y.astype(np.uint8)
    if type(u) is not np.ndarray:
        u=u.numpy()
        u = u.astype(np.uint8)########必须是uint8才能正常显示
    if type(v) is not np.ndarray:
        v=v.detach().numpy()
        v = v.astype(np.uint8)

    with open(path, 'wb') as file:
        for fn in range(1):
            file.write(y.tobytes())
            file.write(u.tobytes())
            file.write(v.tobytes())
class Imagedata(Dataset):
    def __init__(self,img_path,label_path,transform):
        self.img_path=img_path
        self.label_path=label_path
        self.transform=transform
        imgs=os.listdir(img_path)
        labels=os.listdir(label_path)
        imgs=sorted(imgs,key=lambda x:int(os.path.splitext(x)[0][0:4]))
        labels=sorted(labels,key=lambda x:int(os.path.splitext(x)[0][0:4]))
        self.img_label=list(zip(imgs,labels))
    def __getitem__(self, index):
        x_path,y_path=self.img_label[index]
        ##################################################
        #根据命名获取长度

        begin1 = x_path.rfind('_')
        string2 = x_path[10:begin1]
        between = string2.find('x')
        w = int(string2[0:between])
        h = int(string2[between + 1:])
        ################################################
        start = x_path.rfind('_')
        end = x_path.rfind('.')
        theQP = int(x_path[start + 1:end])
        QP = torch.ones([1, 128, 128]) * theQP / 63.0
        total_x_path=self.img_path+x_path
        total_y_label=self.label_path+y_path
        patchsize=128
        y_input,u_input,v_input=YUVread(total_x_path,h,w)
        y_label,u_label,v_label=YUVread(total_y_label,h,w)
        input_video = YUV(h, w, y_input, u_input, v_input)
        label_video=YUV(h,w,y_label,u_label,v_label)
        #生成patch的左上角随机点
        center_y_Y=random.randrange(8,input_video.height-patchsize,2)
        center_x_Y=random.randrange(8,input_video.width-patchsize,2)
        center_y_UV=center_y_Y//2
        center_x_UV=center_x_Y//2
        #获取待预测块的和待预测块的真实值
        patch_y_input,patch_u_input,patch_v_input=cut_patch(input_video,patchsize,center_x_Y,center_y_Y,center_x_UV,center_y_UV)
        #YUVwrite('G:\\思路\\intra\\idea1_train\\参考图片\\' + str(index) + '_input.yuv', patch_y_input,patch_u_input,patch_v_input)
        patch_y_label,patch_u_label,patch_v_label=cut_patch(label_video, patchsize, center_x_Y,center_y_Y ,center_x_UV, center_y_UV)
        input_y= patch_y_input/255.0
        label_y=patch_y_label/255.0
        #label_uv=torch.from_numpy(label_uv)
        #label_uv=label_uv.float()

        return input_y,QP, label_y
    def __len__(self):
        return len(self.img_label)






def loadtraindata():
    transform=transforms.Compose([transforms.ToTensor])
    trainset=Imagedata("G:\\idea2_质量增强\\质量增强数据集\\trainDataSet\\input\\","G:\\idea2_质量增强\\质量增强数据集\\trainDataSet\\label\\",transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,num_workers=8,drop_last = True)
    #"num_workers"这个参数设置的是CPU进程数，若不修改，可能不能充分利用GPU
    return trainloader

def loadtestdata():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = Imagedata('G:\\idea2_质量增强\\质量增强数据集\\testDataSet\\input\\', 'G:\\idea2_质量增强\\质量增强数据集\\testDataSet\\label\\',transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True,num_workers=8,drop_last = True)  # 对数据集分batch
    return trainloader
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def trainandsave():
    parser = argparse.ArgumentParser(description="PyTorch MAR_NA")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
    parser.add_argument("--step", type=int, default=30,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument("--n_colors", default=2, type=float, help="channel nums: 2")
    global opt
    opt = parser.parse_args()
    trainloader = loadtraindata()
    testload = loadtestdata()
    net=MSRB_NA(opt)
    print(net)
    net=net.to(device)
    criterion = nn.MSELoss()
    #criterion=nn.SmoothL1Loss()
    '''
     optimizer = optim.Adam([
        {'params': net.conv1.parameters()},
        {'params': net.conv2.parameters()},
        {'params': net.conv3.parameters(), 'lr': 1e-5}
    ], lr=1e-4)
    '''

    optimizer=optim.Adam(net.parameters(),lr=opt.lr)
    #optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    Loss_list=[]
    totalloss=0
    Loss_test_list = []
    totalonetest_epoch = 0
    path='G:\\idea2_质量增强\\result\\MSR_NA_luma\\'
    for epoch in range(90):
        bar=tqdm(trainloader)
        lr = adjust_learning_rate(optimizer, epoch - 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for input_uv,QP,label_uv in bar:
            print(input_uv.size())
            input_uv=input_uv.to(device)
            QP = QP.to(device)
            label_uv=label_uv.to(device)
            output=net(input_uv,QP)
            print(output.size())
            loss=criterion(output,label_uv)
            totalloss=totalloss+loss
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm(net.parameters(), opt.clip)
            optimizer.step()
            bar.set_description("epoch:%d/%d loss:%f"%(epoch,150,loss.item()))
        Loss_list.append(totalloss/200)
        totalloss=0
        for  input_uv,QP,label_uv  in testload:
            input_uv = input_uv.to(device)
            QP=QP.to(device)
            label_uv = label_uv.to(device)
            with torch.no_grad():  # 不计算梯度也不进行反向传播
                output=net( input_uv,QP)  # 把数据输进CNN网络net
                totalonetest_epoch=totalonetest_epoch+criterion(output,label_uv)
        Loss_test_list.append(totalonetest_epoch/24)
        np.savetxt(path+"Loss.txt", Loss_list)
        np.savetxt(path+"Loss_test.txt", Loss_test_list)
        totalonetest_epoch=0
        torch.save(net,path+str(epoch)+"net.pkl")
        torch.save(net.state_dict(),path+str(epoch)+"net_param.pkl")
    print("Finished Training")
    x1=range(0,90)
    y1=Loss_list
    y2= Loss_test_list
    plt.plot(x1,y1,'.-',label="train loss")
    plt.plot(x1,y2,'ro-',label="test loss")
    plt.xlabel("loss vs. epoches")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path+"train loss.png")
    plt.show()
if __name__=='__main__':
    trainandsave()


