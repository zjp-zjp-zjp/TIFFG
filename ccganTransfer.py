from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from Data import CDdataset
import functools
from CDmodel import MyEncoder
from CrossAttention import Attention
#single
# class UnetGenerator(nn.Module):
#     """Create a Unet-based generator"""

#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet generator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             output_nc (int) -- the number of channels in output images
#             num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
#                                 image of size 128x128 will become of size 1x1 # at the bottleneck
#             ngf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer

#         We construct the U-Net from the innermost layer to the outermost layer.
#         It is a recursive process.
#         """
#         super(UnetGenerator, self).__init__()
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
#         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         # gradually reduce the number of filters from ngf * 8 to ngf
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)

# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """

#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,crossAttention=None):
#         """Construct a Unet submodule with skip connections.

#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.crossAttention=crossAttention
#         self.outer_nc=outer_nc
#         self.inner_nc=inner_nc
#         self.input_nc=input_nc
#         self.submodule=submodule
#         self.innermost=innermost
#         self.norm_layer=norm_layer
#         self.use_dropout=use_dropout
#         self.outermost = outermost
#         if type(self.norm_layer) == functools.partial:
#             use_bias = self.norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = self.norm_layer == nn.InstanceNorm2d
#         if self.input_nc is None:
#             self.input_nc = self.outer_nc
#         self.downconv = nn.Conv2d(self.input_nc, self.inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         self.downrelu = nn.LeakyReLU(0.2, True)
#         self.downnorm = self.norm_layer(self.inner_nc)
#         self.uprelu = nn.ReLU(True)
#         self.upnorm = self.norm_layer(self.outer_nc)
#         if self.outermost:
#             self.upconv = nn.ConvTranspose2d(self.inner_nc * 2, self.outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             self.down=nn.Sequential(self.downconv)
#             self.up=nn.Sequential(self.uprelu, self.upconv, nn.Tanh())
#         elif self.innermost:
#             self.upconv = nn.ConvTranspose2d(self.inner_nc, self.outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             self.down=nn.Sequential(self.downrelu, self.downconv)
#             self.up=nn.Sequential(self.uprelu, self.upconv, self.upnorm)
#         else:
#             self.upconv = nn.ConvTranspose2d(self.inner_nc * 2, self.outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             self.down=nn.Sequential(self.downrelu, self.downconv, self.downnorm)
#             self.up=nn.Sequential(self.uprelu, self.upconv, self.upnorm)
        

#     def forward(self, x):
#         x_origin=x
#         y=None

#         if self.outermost:
#             x=self.down(x)
#             x=self.submodule(x)
#             y=self.up(x)
#         elif self.innermost:
#             x=self.down(x)
#             # print(fusion.shape)
#             y=self.up(x)
#         else:

#             x=self.down(x)
#             x=self.submodule(x)
#             y=self.up(x)
#         # print(x.shape,y.shape)
#         if self.outermost:
#             return y
#         else:   # add skip connections
#             return torch.cat([x_origin, y], 1)



























#double
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.crossAttention=Attention(2048)
        self.myEncoder=MyEncoder()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,crossAttention=self.crossAttention,myEncoder=self.myEncoder)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input,target):
        """Standard forward"""
        return self.model(input,target)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,crossAttention=None,myEncoder=None):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.myEncoder=myEncoder
        self.crossAttention=crossAttention
        self.outer_nc=outer_nc
        self.inner_nc=inner_nc
        self.input_nc=input_nc
        self.submodule=submodule
        self.innermost=innermost
        self.norm_layer=norm_layer
        self.use_dropout=use_dropout
        self.outermost = outermost
        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm2d
        if self.input_nc is None:
            self.input_nc = self.outer_nc
        self.downconv = nn.Conv2d(self.input_nc, self.inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.downnorm = self.norm_layer(self.inner_nc)
        self.uprelu = nn.ReLU(True)
        self.upnorm = self.norm_layer(self.outer_nc)
        if self.outermost:
            self.upconv = nn.ConvTranspose2d(self.inner_nc * 2, self.outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            self.down=nn.Sequential(self.downconv)
            self.up=nn.Sequential(self.uprelu, self.upconv, nn.Tanh())
        elif self.innermost:
            self.upconv = nn.ConvTranspose2d(self.inner_nc, self.outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down=nn.Sequential(self.downrelu, self.downconv)
            self.up=nn.Sequential(self.uprelu, self.upconv, self.upnorm)
        else:
            self.upconv = nn.ConvTranspose2d(self.inner_nc * 2, self.outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down=nn.Sequential(self.downrelu, self.downconv, self.downnorm)
            self.up=nn.Sequential(self.uprelu, self.upconv, self.upnorm)
        

    def forward(self, x,target):
        x_origin=x
        y=None

        if self.outermost:
            x=self.down(x)
            x=self.submodule(x,target)
            y=self.up(x)
        elif self.innermost:
            x=self.down(x)
            B, C, W, H =x.shape
            # print(x.shape)
            target_feature,pic_feature_target=self.myEncoder(target)
            x=torch.flatten(x,start_dim=2).transpose(-1, -2)
            target_feature=torch.flatten(target_feature,start_dim=2).transpose(-1, -2)
            fusion=self.crossAttention(x,target_feature)
            fusion = fusion.transpose(-1, -2).reshape((B, C, W, H))
            # print(fusion.shape)
            y=self.up(fusion)
        else:

            x=self.down(x)
            x=self.submodule(x,target)
            y=self.up(x)
        # print(x.shape,y.shape)
        if self.outermost:
            return y
        else:   # add skip connections
            return torch.cat([x_origin, y], 1)
cuda = torch.cuda.is_available()
# Tensor = torch.Tensor
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device=torch.device('cuda:4')
transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
generator='unet'

resnet=models.resnet50()
num_ftrs=resnet.fc.in_features
resnet.fc=nn.Linear(num_ftrs,3)
resnet.load_state_dict(torch.load('transfer.pth',map_location=device))
resnet.to(device)
#ccgan
# thisepoch=200
# # model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/results/ccGAN(unet)NoSeason_weights/epoch_200_AB.pth')
# # target_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_noclassify/train/A','/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_noclassify/test/A','/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_noclassify/val/A']
# model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)snow_weights/epoch_200_AB.pth',map_location=device)
# nosnow_model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)nosnow_weights/epoch_195_AB.pth',map_location=device)
# target_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify'+str(thisepoch)+'/train','/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify'+str(thisepoch)+'/test','/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify'+str(thisepoch)+'/val']

#cyclegan
# thisepoch=200
# norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
# model=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)
# model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/cyclegan_snow_weights/epoch_200_AB.pth',map_location=device)
# nosnow_model=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)
# nosnow_model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/cyclegan_nosnow_weights/epoch_200_AB.pth',map_location=device)
# target_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_classify'+str(thisepoch)+'/train','/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_classify'+str(thisepoch)+'/test','/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_classify'+str(thisepoch)+'/val']

#cyclegansingle
# thisepoch=200
# norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
# model=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)
# model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/cyclegan_all_weights/epoch_200_AB.pth',map_location=device)
# target_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_single'+str(thisepoch)+'/train','/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_single'+str(thisepoch)+'/test','/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_single'+str(thisepoch)+'/val']

#ccgansingle
thisepoch=200
norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
model=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)
model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)all_weights/epoch_200_AB.pth',map_location=device)
target_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/ccgan_single'+str(thisepoch)+'/train','/archive/hot17/zjp/Cross-Seasonal-CD/ccgan_single'+str(thisepoch)+'/test','/archive/hot17/zjp/Cross-Seasonal-CD/ccgan_single'+str(thisepoch)+'/val']


model=model.to(device=device)
# nosnow_model=nosnow_model.to(device=device)
data_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/test','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val']
def restore_picture(picture):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    picture=picture.squeeze(0)
    for i in range(3):
        picture[i] = picture[i] * std[i] + mean[i]
    to_pil = transforms.ToPILImage()
    restored_image = to_pil(picture)
    return restored_image
def judge(pictureA_input,pictureB_input,classify_model):
    pictureA_input=pictureA_input.unsqueeze(0)
    pictureA_input=pictureA_input.to(device)
    pictureA_output=classify_model(pictureA_input)
    Aarray=pictureA_output.data.cpu().numpy()[0]
    resultA=np.argmax(Aarray, axis=None)

    pictureB_input=pictureB_input.unsqueeze(0)
    pictureB_input=pictureB_input.to(device)
    pictureB_output=classify_model(pictureB_input)
    Barray=pictureB_output.data.cpu().numpy()[0]
    resultB=np.argmax(Barray, axis=None)
    return resultA,resultB
def transfer(data_path,name,transform,model,target_path):
    Apath=data_path+'/A/'+name
    Bpath=data_path+'/B/'+name
    pictureA=transform(Image.open(Apath)).unsqueeze(0)
    pictureB=transform(Image.open(Bpath)).unsqueeze(0)
    real_A = Variable(pictureA.type(Tensor))
    real_B = Variable(pictureB.type(Tensor))
    fake_B=model(real_A,real_B).transpose(1,2).transpose(2,3)
    fake_B_numpy=fake_B.cpu().detach().numpy()
    fake_B_numpy=((fake_B_numpy-fake_B_numpy.min())/(fake_B_numpy.max()-fake_B_numpy.min())*255)
    picture=fake_B_numpy[0]
    cv2.imwrite(target_path+'/'+name,picture)

def transfer_classify(data_path,name,transform,model,target_path,classify_model):
    Apath=data_path+'/A/'+name
    Bpath=data_path+'/B/'+name
    Asavepath=target_path+'/A/'+name
    if not os.path.exists(target_path+'/A/'):
        os.makedirs(target_path+'/A/')
    if not os.path.exists(target_path+'/B/'):
        os.makedirs(target_path+'/B/')
    Bsavepath=target_path+'/B/'+name
    resultA,resultB=judge(transform(Image.open(Apath)),transform(Image.open(Bpath)),classify_model=classify_model)
    if resultA==2 and resultB!=2:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==2 and resultA!=2:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultA==1 and resultB==0:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=nosnow_model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==1 and resultA==0:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=nosnow_model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    else:
        A=Image.open(Apath)
        B=Image.open(Bpath)
        A.save(Asavepath)
        B.save(Bsavepath)

def transfer_classify_single(data_path,name,transform,model,target_path,classify_model):
    Apath=data_path+'/A/'+name
    Bpath=data_path+'/B/'+name
    Asavepath=target_path+'/A/'+name
    if not os.path.exists(target_path+'/A/'):
        os.makedirs(target_path+'/A/')
    if not os.path.exists(target_path+'/B/'):
        os.makedirs(target_path+'/B/')
    Bsavepath=target_path+'/B/'+name
    resultA,resultB=judge(transform(Image.open(Apath)),transform(Image.open(Bpath)),classify_model=classify_model)
    if resultA==2 and resultB!=2:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==2 and resultA!=2:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultA==1 and resultB==0:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==1 and resultA==0:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A,real_B)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    else:
        A=Image.open(Apath)
        B=Image.open(Bpath)
        A.save(Asavepath)
        B.save(Bsavepath)

def transfer_cyclegan(data_path,name,transform,model,target_path,classify_model):
    Apath=data_path+'/A/'+name
    Bpath=data_path+'/B/'+name
    Asavepath=target_path+'/A/'+name
    if not os.path.exists(target_path+'/A/'):
        os.makedirs(target_path+'/A/')
    if not os.path.exists(target_path+'/B/'):
        os.makedirs(target_path+'/B/')
    Bsavepath=target_path+'/B/'+name
    resultA,resultB=judge(transform(Image.open(Apath)),transform(Image.open(Bpath)),classify_model=classify_model)
    if resultA==2 and resultB!=2:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==2 and resultA!=2:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultA==1 and resultB==0:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=nosnow_model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==1 and resultA==0:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=nosnow_model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    else:
        A=Image.open(Apath)
        B=Image.open(Bpath)
        A.save(Asavepath)
        B.save(Bsavepath)

def transfer_cyclegan_single(data_path,name,transform,model,target_path,classify_model):
    Apath=data_path+'/A/'+name
    Bpath=data_path+'/B/'+name
    Asavepath=target_path+'/A/'+name
    if not os.path.exists(target_path+'/A/'):
        os.makedirs(target_path+'/A/')
    if not os.path.exists(target_path+'/B/'):
        os.makedirs(target_path+'/B/')
    Bsavepath=target_path+'/B/'+name
    resultA,resultB=judge(transform(Image.open(Apath)),transform(Image.open(Bpath)),classify_model=classify_model)
    if resultA==2 and resultB!=2:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==2 and resultA!=2:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultA==1 and resultB==0:
        # print(name)
        A=Image.open(Apath)
        B=Image.open(Bpath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    elif resultB==1 and resultA==0:
        # print(name)
        A=Image.open(Bpath)
        B=Image.open(Apath)
        pictureA=transform(A).unsqueeze(0)
        pictureB=transform(B).unsqueeze(0)
        # real_A = Variable(pictureA.type(Tensor))
        # real_B = Variable(pictureB.type(Tensor))
        real_A = pictureA
        real_B = pictureB
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        fake_B=model(real_A)
        fake_B_numpy=fake_B
        fake_B_picture=restore_picture(fake_B_numpy)
        fake_B_picture.save(Asavepath)
        B.save(Bsavepath)
    else:
        A=Image.open(Apath)
        B=Image.open(Bpath)
        A.save(Asavepath)
        B.save(Bsavepath)

for i in range(3):
    data_path=data_paths[i]
    target_path=target_paths[i]
    namelist=os.listdir(data_path+'/A/')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for i in tqdm(range(len(namelist))):
        name=namelist[i]
        # transfer(data_path,name,transform,model,target_path)
        # transfer_classify(data_path,name,transform,model,target_path,resnet)
        # transfer_cyclegan(data_path,name,transform,model,target_path,resnet)
        # transfer_cyclegan_single(data_path,name,transform,model,target_path,resnet)
        transfer_classify_single(data_path,name,transform,model,target_path,resnet)






