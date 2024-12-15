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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

model=torch.load('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/results/CDmodelv2(unet&focalloss)NoSeason_weights/epoch_200.pth',map_location=device)
target_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/results/CDmodelv2(unet&focalloss)NoSeason_weights/pictures'
model=model.to(device=device)
data_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/test'
def transfer(data_path,name,transform,model,target_path):
    Apath=data_path+'/A/'+name
    Bpath=data_path+'/B/'+name
    pictureA=transform(Image.open(Apath)).unsqueeze(0)
    pictureB=transform(Image.open(Bpath)).unsqueeze(0)
    real_A = Variable(pictureA.type(Tensor))
    real_B = Variable(pictureB.type(Tensor))
    change_map,fake_B=model(real_A,real_B)
    fake_B=fake_B.transpose(1,2).transpose(2,3)
    fake_B_numpy=fake_B.cpu().detach().numpy()
    fake_B_numpy=((fake_B_numpy-fake_B_numpy.min())/(fake_B_numpy.max()-fake_B_numpy.min())*255)
    picture=fake_B_numpy[0]
    cv2.imwrite(target_path+'/B_fake_'+name,picture)
    cv2.imwrite(target_path+'/A_'+name,np.array(Image.open(Apath)))
    cv2.imwrite(target_path+'/B_'+name,np.array(Image.open(Bpath)))

namelist=os.listdir(data_path+'/A/')
if not os.path.exists(target_path):
    os.makedirs(target_path)
for i in range(10):
    import random
    random_num = random.randint(0, 2999)
    name=namelist[random_num]
    transfer(data_path,name,transform,model,target_path)
