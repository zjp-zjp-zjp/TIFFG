from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn

device=torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

def save(A,B,OUT,name,target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    targetA_path=target_path+'/A/'+name
    targetB_path=target_path+'/B/'+name
    targetOUT_path=target_path+'/OUT/'+name
    if not os.path.exists(os.path.join(target_path,'A')):
        os.makedirs(os.path.join(target_path,'A'))
    if not os.path.exists(os.path.join(target_path,'B')):
        os.makedirs(os.path.join(target_path,'B'))
    if not os.path.exists(os.path.join(target_path,'OUT')):
        os.makedirs(os.path.join(target_path,'OUT'))
    A.save(targetA_path)
    B.save(targetB_path)
    OUT.save(targetOUT_path)

data_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train',
            '/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/test',
            '/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val']
save_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/seasonalGANdataset/train',
            '/archive/hot17/zjp/Cross-Seasonal-CD/seasonalGANdataset/test',
            '/archive/hot17/zjp/Cross-Seasonal-CD/seasonalGANdataset/val']
transform_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/seasonalgan_snow_weights/epoch_200_AB.pth'


resnet=models.resnet50()
num_ftrs=resnet.fc.in_features
resnet.fc=nn.Linear(num_ftrs,3)
resnet.load_state_dict(torch.load('transfer.pth',map_location=device))
resnet.to(device)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def judge(pictureA,pictureB,judgeModel):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    pictureA_input=transform(pictureA)
    pictureA_input=pictureA_input.unsqueeze(0)
    pictureA_input=pictureA_input.to(device)
    pictureA_output=judgeModel(pictureA_input)
    Aarray=pictureA_output.data.cpu().numpy()[0]
    resultA=np.argmax(Aarray, axis=None)

    pictureB_input=transform(pictureB)
    pictureB_input=pictureB_input.unsqueeze(0)
    pictureB_input=pictureB_input.to(device)
    pictureB_output=judgeModel(pictureB_input)
    Barray=pictureB_output.data.cpu().numpy()[0]
    resultB=np.argmax(Barray, axis=None)
    #2雪冬
    #1无雪冬
    #0非冬
    return resultA,resultB

def transfer_1(picture,transform_model):
    #一个输入一个输出
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    input=transform(picture).unsqueeze(0)
    input=input.to(device)
    output=transform_model(input)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    output=output.squeeze(0)
    for i in range(3):
        output[i] = output[i] * std[i] + mean[i]
    to_pil = transforms.ToPILImage()
    restored_image = to_pil(output)
    return restored_image
def transfer_2(picture,target,transform_model):
    #两个输入一个输出
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    input=transform(picture).unsqueeze(0)
    input=input.to(device)
    input2=transform(target).unsqueeze(0)
    input2=input2.to(device)
    output=transform_model(input,input2)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    output=output.squeeze(0)
    for i in range(3):
        output[i] = output[i] * std[i] + mean[i]
    to_pil = transforms.ToPILImage()
    restored_image = to_pil(output)
    return restored_image
# transform_model=GeneratorResNet((3,256,256), 9)
transform_model=torch.load(transform_model_path,map_location=device)
#一个输入一个输出：
# for j in range(3):
#     data_path=data_paths[j]
#     target_path=save_paths[j]
#     namelist=os.listdir(data_path+'/A/')
#     for i in tqdm(range(len(namelist))):
#         name=namelist[i]

#         pictureA=Image.open(data_path+'/A/'+name)
#         pictureB=Image.open(data_path+'/B/'+name)
#         pictureOUT=Image.open(data_path+'/OUT/'+name)
#         resultA,resultB=judge(pictureA=pictureA,pictureB=pictureB,judgeModel=resnet)
#         if resultA==2:
#             pictureA=transfer_1(picture=pictureA,transform_model=transform_model)
#         if resultB==2:
#             pictureB=transfer_1(picture=pictureB,transform_model=transform_model)
#         save(pictureA,pictureB,pictureOUT,'%05d.jpg'%i,target_path=target_path)
#两个输入一个输出：
for j in range(3):
    data_path=data_paths[j]
    target_path=save_paths[j]
    namelist=os.listdir(data_path+'/A/')
    for i in tqdm(range(len(namelist))):
        name=namelist[i]

        pictureA=Image.open(data_path+'/A/'+name)
        pictureB=Image.open(data_path+'/B/'+name)
        pictureOUT=Image.open(data_path+'/OUT/'+name)
        resultA,resultB=judge(pictureA=pictureA,pictureB=pictureB,judgeModel=resnet)
        if resultA==2:
            pictureA=transfer_2(picture=pictureA,target=pictureB,transform_model=transform_model)
        if resultB==2:
            pictureB=transfer_2(picture=pictureB,target=pictureA,transform_model=transform_model)
        save(pictureA,pictureB,pictureOUT,'%05d.jpg'%i,target_path=target_path)