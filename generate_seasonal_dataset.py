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

data_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/test','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val']
snow_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/train','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/val']
nosnow_paths=['/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/train','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test','/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/val']


device=torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

resnet=models.resnet50()
num_ftrs=resnet.fc.in_features
resnet.fc=nn.Linear(num_ftrs,3)
resnet.load_state_dict(torch.load('transfer.pth',map_location=device))
resnet.to(device)

#A为冬天，B为夏天

transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

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

for i in range(3):
    snow=0
    nosnow=0
    data_path=data_paths[i]
    snow_path=snow_paths[i]
    nosnow_path=nosnow_paths[i]
    namelist=os.listdir(data_path+'/A/')
    for i in tqdm(range(len(namelist))):
        name=namelist[i]

        pictureA=Image.open(data_path+'/A/'+name)
        pictureB=Image.open(data_path+'/B/'+name)
        pictureOUT=Image.open(data_path+'/OUT/'+name)

        pictureA_input=transform(pictureA)
        pictureA_input=pictureA_input.unsqueeze(0)
        pictureA_input=pictureA_input.to(device)
        pictureA_output=resnet(pictureA_input)
        Aarray=pictureA_output.data.cpu().numpy()[0]
        resultA=np.argmax(Aarray, axis=None)

        pictureB_input=transform(pictureB)
        pictureB_input=pictureB_input.unsqueeze(0)
        pictureB_input=pictureB_input.to(device)
        pictureB_output=resnet(pictureB_input)
        Barray=pictureB_output.data.cpu().numpy()[0]
        resultB=np.argmax(Barray, axis=None)

        # 2 2
        # 1 1
        # 0 0

        if resultA==2 and resultB!=2:
            save(pictureA,pictureB,pictureOUT,'%05d.jpg'%snow,snow_path)
            snow+=1
        elif resultB==2 and resultA!=2:
            save(pictureB,pictureA,pictureOUT,'%05d.jpg'%snow,snow_path)
            snow+=1
        elif resultA==1 and resultB==0:
            save(pictureA,pictureB,pictureOUT,'%05d.jpg'%nosnow,nosnow_path)
            nosnow+=1
        elif resultB==1 and resultA==0:
            save(pictureB,pictureA,pictureOUT,'%05d.jpg'%nosnow,nosnow_path)
            nosnow+=1
        elif resultA==2 and resultB==2:
            pass
            # save(pictureB,pictureA,pictureOUT,'%05d.jpg'%snow,snow_path)
            # snow+=1
        else:
            save(pictureB,pictureA,pictureOUT,'%05d.jpg'%nosnow,nosnow_path)
            nosnow+=1