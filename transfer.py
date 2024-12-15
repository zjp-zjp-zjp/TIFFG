from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

model=torch.load('/archive/cold0/zjp/Cross-Seasonal-CD/My_model/model/weights/epoch_100_AB.pth')
model=model.to(device=device)
data_paths=['/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset/train','/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset/test','/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset/val']
target_paths=['/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset_after_seasonalGAN_noseasonclassify/train/A','/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset_after_seasonalGAN_noseasonclassify/test/A','/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset_after_seasonalGAN_noseasonclassify/val/A']

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

for i in range(3):
    data_path=data_paths[i]
    target_path=target_paths[i]
    namelist=os.listdir(data_path+'/A/')
    for i in tqdm(range(len(namelist))):
        name=namelist[i]
        transfer(data_path,name,transform,model,target_path)
