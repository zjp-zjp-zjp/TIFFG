import torch
from torch import nn
from torchvision.models import resnet50,resnet152,resnet34
from simple_vit import SimpleViT
from CBAMmodel import ChannelAttention,SpatialAttention
from CrossAttention import Attention
from Decoder import Decoder
import change_detection_pytorch as cdp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import functools

device=torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

class MyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet=resnet50(pretrained=True)
        self.encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                            resnet.layer1, resnet.layer2,resnet.layer3,resnet.layer4)
        self.ca=ChannelAttention(2048)
        self.sa=SpatialAttention()
    def forward(self, img):
        feature=self.encoder(img)
        pic_feature=feature
        # print(feature.shape)
        # print(self.ca(feature).shape)
        feature=self.ca(feature)*feature
        # print(feature.shape)
        # print(self.sa(feature).shape)
        feature=self.sa(feature)*feature
        # print(feature.shape)
        return feature,pic_feature


class SeasonalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.myEncoder1=MyEncoder()
        self.myEncoder2=MyEncoder()
        self.crossAttention=Attention(2048)
        self.decoder=Decoder()

    def forward(self, img_origin,img_target):
        feature_origin,pic_feature=self.myEncoder1(img_origin)
        B, C, W, H = pic_feature.shape

        feature_target,pic_feature_target=self.myEncoder2(img_target)
        
        
        feature_origin=torch.flatten(feature_origin,start_dim=2).transpose(-1, -2)
        feature_target=torch.flatten(feature_target,start_dim=2).transpose(-1, -2)
        
        fusion_feature=self.crossAttention(feature_origin,feature_target)
        fusion_feature = fusion_feature.transpose(-1, -2).reshape((B, C, W, H))+pic_feature
        output_picture=self.decoder(fusion_feature)

        return output_picture

class CDmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seasonalModel=SeasonalModel()
        self.uplusplus=cdp.UnetPlusPlus(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your datasets)
            activation='sigmoid',
            siam_encoder=True,  # whether to use a siamese encoder
            fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
        )
    
    def forward(self, img_origin,img_target):
        output_picture=self.seasonalModel(img_origin,img_target)

        change_map=self.uplusplus(img_origin,output_picture)
        
        return change_map,output_picture

# class TestEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet=resnet50(pretrained=True)
#         self.encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
#                                             resnet.layer1, resnet.layer2,resnet.layer3,resnet.layer4)
#         resnet2=resnet152(pretrained=True)
#         self.encoder2 = nn.Sequential(resnet2.conv1, resnet2.bn1, resnet2.relu, resnet2.maxpool,
#                                             resnet2.layer1, resnet2.layer2,resnet2.layer3,resnet2.layer4)
#     def forward(self, img):
#         feature=self.encoder(img)
#         feature2=self.encoder2(img)
#         print(feature.shape,feature2.shape)


if __name__=='__main__':
    x=torch.randn((4,3,256,256))
    model=MyEncoder()
    model(x)
    # y=torch.randn((4,3,256,256))
    # x=x.to(device)
    # y=y.to(device)
    # model=CDmodel()
    # model.to(device)
    # change_map=model.forward(x,y)
    # print(change_map.shape)
    # model=TestEncoder()
    # model(x)
    # img=Image.open('00004.jpg')
    # transform = transforms.Compose([
    #     transforms.Resize(256),  # 调整图像大小为 256x256
    #     transforms.ToTensor(),  # 将图像转换为张量
    # ])

    # img=transform(img)
    # print(img.shape)
    
    
