from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CDdataset(Dataset):
    def __init__(self,img_path):
        self.Apath=os.path.join(img_path,'A')
        self.Bpath=os.path.join(img_path,'B')
        self.OUTpath=os.path.join(img_path,'OUT')
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.onetransform=transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        all=os.listdir(self.Apath)
        return len(all)
    
    def __getitem__(self, index):
        name='%05d.jpg'%index
        Aname=os.path.join(self.Apath,name)
        Bname=os.path.join(self.Bpath,name)
        OUTname=os.path.join(self.OUTpath,name)
        pictureA=self.transform(Image.open(Aname))
        pictureB=self.transform(Image.open(Bname))
        pictureOUT=self.onetransform(Image.open(OUTname))
        pictureOUT=(pictureOUT!=0).float()
        return pictureA,pictureB,pictureOUT

class SWAUdataset(Dataset):
    def __init__(self,img_path):
        self.Apath=os.path.join(img_path,'A')
        self.Bpath=os.path.join(img_path,'B')
        self.OUTpath=os.path.join(img_path,'OUT')
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.onetransform=transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        all=os.listdir(self.Apath)
        return int(len(all)/2)
    
    def __getitem__(self, index):
        name='%05d.jpg'%index
        #'%05d_%05d_1.00.png'%(index,index)
        Aname=os.path.join(self.Apath,name)
        Bname=os.path.join(self.Bpath,name)
        OUTname=os.path.join(self.OUTpath,name)
        pictureA=self.transform(Image.open(Aname))
        pictureB=self.transform(Image.open(Bname))
        pictureOUT=self.onetransform(Image.open(OUTname))
        pictureOUT=(pictureOUT!=0).float()
        return pictureA,pictureB,pictureOUT

class SWAUtestdataset(Dataset):
    def __init__(self,img_path):
        self.Apath=os.path.join(img_path,'A')
        self.Bpath=os.path.join(img_path,'B')
        self.OUTpath=os.path.join(img_path,'OUT')
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.onetransform=transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        all=os.listdir(self.Apath)
        return len(all)
    
    def __getitem__(self, index):
        name='%05d.jpg'%(index+41)
        #'%05d_%05d_1.00.png'%(index,index)
        Aname=os.path.join(self.Apath,'%05d_%05d_1.00.png'%(index+41,index+41))
        Bname=os.path.join(self.Bpath,name)
        OUTname=os.path.join(self.OUTpath,name)
        pictureA=self.transform(Image.open(Aname))
        pictureB=self.transform(Image.open(Bname))
        pictureOUT=self.onetransform(Image.open(OUTname))
        pictureOUT=(pictureOUT!=0).float()
        return pictureA,pictureB,pictureOUT