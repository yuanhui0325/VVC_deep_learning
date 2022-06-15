import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Netref(nn.Module):
    def __init__(self):
        super(Netref,self).__init__()
        self.linear1=nn.Linear(1584,128)
        self.linear2=nn.Linear(512,256)
        self.linear3=nn.Linear(256,128)
        self.senet=SE_Module(128)
    def forward(self,x):
        b, c, _, _ = x.size()
        x1=x.view(b,-1)#32*(33*4*4*3)
        linear1=self.linear1(x1)
        #linear2=self.linear2(linear1)
        #linear3=self.linear3(linear2)
        linear1=linear1.view(b,128,1,1)
        tile=linear1.repeat(1,1,64,64)#32x128x64x64
        senet=self.senet(tile)

        return senet
class NetY(nn.Module):
    def __init__(self):
        super(NetY,self).__init__()
        self.conv1_1=nn.Conv2d(1,32,3,1,1 )
        self.conv1_1norm=nn.BatchNorm2d(32,affine=False)
        self.conv1_2=nn.Conv2d(32,64,3,1,1)
        self.conv1_2norm=nn.BatchNorm2d(64,affine=False)
        self.conv1_3=nn.Conv2d(64,128,3,1,1)
        self.conv1_3norm=nn.BatchNorm2d(16,affine=False)
    def forward(self,x):
        conv1_1=F.relu(self.conv1_1(x))
        #conv1_1norm=self.conv1_1norm(conv1_1)
        conv1_2=F.relu(self.conv1_2(conv1_1))
        conv1_3 = F.relu(self.conv1_3(conv1_2))
        #conv1_2norm=self.conv1_2norm(conv1_2)
        #conv1_3=F.relu(self.conv1_3(conv1_2norm))
        #conv1_3norm=self.conv1_3norm(conv1_3)#16x64x64
        return conv1_3
class SE_Module(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)
class AllNet(nn.Module):
    def __init__(self):
        super(AllNet, self).__init__()
        self.NetY=NetY()
        self.Netref=Netref()
        self.conv1=nn.Conv2d(128,128,1,1,0)

        self.conv2=nn.Conv2d(128,64,1,1,0)
        self.conv3 = nn.Conv2d(64, 2, 1, 1, 0)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
    def forward(self,x1,x2):
        convy_1=self.NetY(x1)
        convref_1=self.Netref(x2)
        conv=convy_1+convref_1#对应位置相加实现特征融合
        conv1=F.relu(self.conv1(conv))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = self.conv3(conv2)



        pred_ab = (torch.tanh(conv3) + 1)/2

        return pred_ab

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
    #patch_Y = np.reshape(patch_Y, [1, patch_size,patch_size])
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


        #string1 = x_path[8:]
        #begin1 = string1.find('_')
        #string2=string1[0:begin1]
        #between = string2.find('x')
        #w = int(string2[0:between])
        #h = int(string2[between + 1:])
        w=256
        h=256
        ################################################
        total_x_path=self.img_path+x_path
        total_y_label=self.label_path+y_path
        patchsize=128
        y_input,u_input,v_input=YUVread(total_x_path,h,w)
        y_label,u_label,v_label=YUVread(total_y_label,h,w)
        input_video = YUV(h, w, y_input, u_input, v_input)
        label_video=YUV(h,w,y_label,u_label,v_label)
        #生成patch的左上角随机点
        #center_y_Y=random.randrange(8,input_video.height-patchsize,2)
        #center_x_Y=random.randrange(8,input_video.width-
        #center_x_Y = random.randrange(8, input_video.width - patchsize, 2)
        center_y_Y=128
        center_x_Y=128
        #center_y_Y=128
        #center_x_Y=128
        center_y_UV=center_y_Y//2
        center_x_UV=center_x_Y//2
        #获取待预测块的和待预测块的真实值
        patch_y_input,patch_u_input,patch_v_input=cut_patch(input_video,patchsize,center_x_Y,center_y_Y,center_x_UV,center_y_UV)
        #YUVwrite('G:\\思路\\intra\\idea1_train\\参考图片\\' + str(index) + '_input.yuv', patch_y_input,patch_u_input,patch_v_input)
        patch_y_label,patch_u_label,patch_v_label=cut_patch(label_video, patchsize, center_x_Y,center_y_Y ,center_x_UV, center_y_UV)
        #获取参考像素的值：左边相邻的8x8的块和上面相邻8x8的块。
        rec_size=8
        ref_left_y,ref_left_u,ref_left_v=cut_ref_left(input_video,center_x_Y,center_y_Y,center_x_UV,center_y_UV,rec_size,patchsize)
        #YUVwrite('G:\\思路\\intra\\idea1_train\\参考图片\\' + str(index) + '_refleft.yuv',ref_left_y,ref_left_u,ref_left_v)
        ref_up_y,ref_up_u,ref_up_v=cut_ref_up(input_video,center_x_Y,center_y_Y,center_x_UV,center_y_UV,rec_size,patchsize)
        #YUVwrite('G:\\思路\\intra\\idea1_train\\参考图片\\' + str(index) + '_refup.yuv',ref_up_y,ref_up_u,ref_up_v)
        ###Y分量的降采样
        down_ref_up_y=np.zeros([rec_size//2,patchsize//2])
        row=0
        col=0
        for i in range(rec_size):
            if i%2==0:
                for j in range(patchsize):
                    if(j%2==0):
                        down_ref_up_y[row][col]=ref_up_y[i][j]
                        col=col+1
                col=0
                row=row+1
        down_ref_left_y=np.zeros([(rec_size+patchsize)//2,rec_size//2])
        row1=0
        col1=0
        for i in range(rec_size+patchsize):
            if i%2==0:
                for j in range(rec_size):
                    if j%2==0:
                        down_ref_left_y[row1][col1]=ref_left_y[i][j]
                        col1=col1+1
                col1=0
                row1=row1+1
        row2=0
        col2=0
        down_y_input=np.zeros([patchsize//2,patchsize//2])
        for i in range(patchsize):
            if i % 2 == 0:
                for j in range(patchsize):
                    if j % 2 == 0:
                        down_y_input[row2][col2] = patch_y_input[i][j]
                        col2 = col2 + 1
                col2 = 0
                row2 = row2 + 1
        down_y_input=np.reshape(down_y_input,[1,patchsize//2,patchsize//2])
        #参考垂直方向的拼接
        down_ref_up_y=down_ref_up_y.T
        ref_up_u=ref_up_u.T
        ref_up_v=ref_up_v.T
        ref_y=np.concatenate((down_ref_up_y,down_ref_left_y),axis=0)#132x4
        ref_u=np.concatenate((ref_up_u,ref_left_u),axis=0)#132x4
        ref_v=np.concatenate((ref_up_v,ref_left_v),axis=0)#132x4
        # 参考通道的拼接
        ref_yuv=np.array([ref_y,ref_u,ref_v],dtype='float64')
        #ref_yuv=np.concatenate((ref_y,ref_u,ref_v),dim=0)#3x132x4
        #ref_yuv=ref_yuv.tensor()
        #ref_yuv=ref_yuv.astype(float)
        ref_yuv=torch.from_numpy(ref_yuv)
        ref_yuv=ref_yuv.float()
        ref_yuv=ref_yuv/255
        #待预测快的Y分量
        #down_y_input=self.transform(down_y_input)
        down_y_input=torch.from_numpy(down_y_input)
        down_y_input=down_y_input.float()
        down_y_input=down_y_input/255
        #真实值UV的拼接
        label_uv=torch.cat([patch_u_label,patch_v_label],dim=0)
        label_uv=label_uv/255
        #label_uv=np.concatenate((patch_u_label,patch_v_label),dim=0)#2x64x64
        #label_uv=np.array([patch_u_label,patch_v_label],dtype='float64')
        #label_uv=torch.from_numpy(label_uv)
        #label_uv=label_uv.float()
        return down_y_input,ref_yuv,label_uv
    def __len__(self):
        return len(self.img_label)




def loadtraindata():
    transform=transforms.Compose([transforms.ToTensor])
    trainset=Imagedata("G:\\AIDataset\\QP22_DIV2K_train\\input\\","G:\\AIDataset\\QP22_DIV2K_train\\label\\",transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,drop_last = True)
    return trainloader

def loadtestdata():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = Imagedata('G:\\AIDataset\\QP22_DIV2K_test\\input\\', 'G:\\AIDataset\\QP22_DIV2K_test\\label\\',transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True,drop_last = True)  # 对数据集分batch
    return trainloader


def trainandsave():
    trainloader = loadtraindata()
    testload = loadtestdata()
    #net=AllNet()
    net=torch.load("G:\\liuyao\\帧内结果\\idea7改进1_patch128_DIV2KQP22\\9net.pkl")
    print(net)
    net=net.to(device)
    criterion = nn.MSELoss()
    #criterion=nn.SmoothL1Loss()
    optimizer=optim.Adam(net.parameters(),lr=1e-5)
    Loss_list=[]
    totalloss=0
    Loss_test_list = []
    totalonetest_epoch = 0
    for epoch in range(10):
        bar=tqdm(trainloader)
        for down_y_input,ref_yuv,label_uv in bar:
            print(down_y_input.size())

            down_y_input=down_y_input.to(device)
            ref_yuv=ref_yuv.to(device)
            label_uv=label_uv.to(device)
            output=net(down_y_input,ref_yuv)
            print(output.size())
            loss=criterion(output,label_uv)
            totalloss=totalloss+loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_description("epoch:%d/%d loss:%f"%(epoch,10,loss.item()))
        Loss_list.append(totalloss/1585)
        totalloss=0
        for  down_y_input,ref_yuv,label_uv in testload:
            down_y_input = down_y_input.to(device)
            ref_yuv = ref_yuv.to(device)
            label_uv = label_uv.to(device)
            with torch.no_grad():  # 不计算梯度也不进行反向传播
                output=net(down_y_input,ref_yuv)  # 把数据输进CNN网络net
                totalonetest_epoch=totalonetest_epoch+criterion(output,label_uv)
        Loss_test_list.append(totalonetest_epoch/201)
        np.savetxt("G:\\liuyao\\帧内结果\\idea7改进1_patch128_DIV2KQP22\\Loss2.txt", Loss_list)
        np.savetxt("G:\\liuyao\\帧内结果\\idea7改进1_patch128_DIV2KQP22\\Loss_test2.txt", Loss_test_list)
        totalonetest_epoch=0
        torch.save(net,"G:\\liuyao\\帧内结果\\idea7改进1_patch128_DIV2KQP22\\"+str(epoch+10)+"net.pkl")
        torch.save(net.state_dict(),"G:\\liuyao\\帧内结果\\idea7改进1_patch128_DIV2KQP22\\"+str(epoch+10)+"net_param.pkl")
    print("Finished Training")
    x1=range(0,10)
    y1=Loss_list
    y2= Loss_test_list
    plt.plot(x1,y1,'.-',label="train loss")
    plt.plot(x1,y2,'ro-',label="test loss")
    plt.xlabel("loss vs. epoches")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("G:\\liuyao\\帧内结果\\idea7改进1_patch128_DIV2KQP22\\train loss2.png")
    plt.show()
if __name__=='__main__':
    trainandsave()

