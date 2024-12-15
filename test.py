from CDmodel import CDmodel
from Data import CDdataset, SWAUdataset,SWAUtestdataset
from torch.utils.data import DataLoader
import change_detection_pytorch as cdp
import torch
import os
from meter import AverageValueMeter
from tqdm import tqdm
import torchvision.models as models
import json
import numpy as np
import torch.nn as nn
from network.CGNet import HCGMNet,CGNet
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
resnet=models.resnet50()
num_ftrs=resnet.fc.in_features
resnet.fc=nn.Linear(num_ftrs,3)
resnet.load_state_dict(torch.load('transfer.pth',map_location=device))
resnet.to(device)
#unetpp
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/unetppweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/unetppweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/unetppweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/unetppweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/unetppweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/unetppweights_SWAU/best_model.pth'


#unet
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/unetweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/unetweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/unetweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/unetweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/unetweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/unetweights_SWAU/best_model.pth'


#fpn
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/fpnweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/fpnweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/fpnweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/fpnweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/fpnweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/fpnweights_SWAU/best_model.pth'


#manet
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/manetweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/manetweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/manetweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/manetweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/manetweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/manetweights_SWAU/best_model.pth'


#stanet
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/stanetweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/stanetweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/stanetweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/stanetweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/stanetweights_cyclegan/epoch_200.pth'

#linknet
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/linknetweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/linknetweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/linknetweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/linknetweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/linknetweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/linknetweights_SWAU/best_model.pth'

#pspnet
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/pspnetweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/pspnetweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/pspnetweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/pspnetweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/pspnetweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/pspnetweights_SWAU/best_model.pth'

#upernet
# original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/upernetweights/best_model.pth'
# refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/upernetweights_ccganunet/best_model.pth'
# cyclegan_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/new_results/upernetweights_cyclegan/best_model.pth'
# refactor_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/upernetweights_ccganunet/epoch_200.pth'
# cyclegan_cdmodel_path_single='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/upernetweights_cyclegan/epoch_200.pth'
# swau_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/singlebranchresult/upernetweights_SWAU/best_model.pth'

original_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/CGNet-CD-main/output/CDD-O/CGNet_best_iou.pth'
refactor_cdmodel_path='/archive/hot17/zjp/Cross-Seasonal-CD/CGNet-CD-main/output/CDD-MY/CGNet_best_iou.pth'
refactor_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)snow_weights/epoch_200_AB.pth'
refactor_nosnow_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)nosnow_weights/epoch_195_AB.pth'

# single_refactor_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)all_weights/epoch_200_AB.pth'
# single_cyclegan_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/cyclegan_all_weights/epoch_200_AB.pth'
# refactor_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)snow_weights/epoch_200_AB.pth'
# refactor_nosnow_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/ccGAN(unet)nosnow_weights/epoch_195_AB.pth'
# cyclegan_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/cyclegan_snow_weights/epoch_200_AB.pth'
# cyclegan_nosnow_model_path='/archive/hot17/zjp/Cross-Seasonal-CD/My_model/cyclegan_nosnow_weights/epoch_200_AB.pth'
def judge(pictureA,pictureB,judgeModel=resnet):
    pictureA_input=pictureA
    # pictureA_input=pictureA_input.unsqueeze(0)
    pictureA_input=pictureA_input.to(device)
    pictureA_output=judgeModel(pictureA_input)
    Aarray=pictureA_output.data.cpu().numpy()[0]
    resultA=np.argmax(Aarray, axis=None)

    pictureB_input=pictureB
    # pictureB_input=pictureB_input.unsqueeze(0)
    pictureB_input=pictureB_input.to(device)
    pictureB_output=judgeModel(pictureB_input)
    Barray=pictureB_output.data.cpu().numpy()[0]
    resultB=np.argmax(Barray, axis=None)
    #2雪冬
    #1无雪冬
    #0非冬
    return resultA,resultB
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
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        

    def forward(self, x):
        x_origin=x
        y=None

        if self.outermost:
            x=self.down(x)
            x=self.submodule(x)
            y=self.up(x)
        elif self.innermost:
            x=self.down(x)
            B, C, W, H =x.shape
            # print(x.shape)
            # print(fusion.shape)
            y=self.up(x)
        else:

            x=self.down(x)
            x=self.submodule(x)
            y=self.up(x)
        # print(x.shape,y.shape)
        if self.outermost:
            return y
        else:   # add skip connections
            return torch.cat([x_origin, y], 1)
def test_unetpp_original():
    model=torch.load(original_cdmodel_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    model=model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)

def test_unetpp_refactor():
    model=torch.load(refactor_cdmodel_path,map_location=device)
    transformer_model=torch.load(refactor_model_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    model=model.to(device)
    transformer_model=transformer_model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A,B)
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)

def test_unetpp_cyclegan_all():
    # transformer_model=GeneratorResNet((3,256,256), 9)
    # model=torch.load(cyclegan_cdmodel_path,map_location=device)
    # transformer_model=torch.load(cyclegan_model_path,map_location=device)
    # test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    # criterion = nn.BCELoss()
    # model=model.to(device)
    # transformer_model=transformer_model.to(device)
    new_model=torch.load(cyclegan_cdmodel_path,map_location=device)
    transformer_model=torch.load(cyclegan_model_path,map_location=device)
    transformer_nosnow_model=torch.load(cyclegan_nosnow_model_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    transformer_model=transformer_model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    A_data=valid_log
    print(valid_log)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_nosnow_model(A)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    B_data=valid_log
    calculate_all(A_data,B_data)

def calculate_all(A,B):
    # print(A,B)
    overall_precision=(A['Precision']*1667.0+B['Precision']*1241.0)/(1667.0+1241.0)
    overall_recall=(A['Recall']*1667.0+B['Recall']*1241.0)/(1667.0+1241.0)
    overall_f1_score=(A['F1-score']*1667.0+B['F1-score']*1241.0)/(1667.0+1241.0)
    valid_log={"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)

def test_unetpp_original_all():
    model=torch.load(original_cdmodel_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    model=model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"condition":"snow","Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score,"total":total_samples}
    print(valid_log)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"condition":"nosnow","Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score,"total":total_samples}
    print(valid_log)

def test_unetpp_refactor_all():
    model=torch.load(original_cdmodel_path,map_location=device)
    new_model=torch.load(refactor_cdmodel_path,map_location=device)
    transformer_model=torch.load(refactor_model_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    model=model.to(device)
    transformer_model=transformer_model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A,B)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)

def test_unetpp_refactor_all_new():
    new_model=torch.load(refactor_cdmodel_path,map_location=device)
    transformer_model=torch.load(refactor_model_path,map_location=device)
    transformer_nosnow_model=torch.load(refactor_nosnow_model_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    transformer_model=transformer_model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A,B)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    Adata=valid_log
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_nosnow_model(A,B)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    Bdata=valid_log
    calculate_all(Adata,Bdata)

def calculate_metrics(predictions, targets,swau=False):
    # 计算准确率
    predictions=(predictions>=0.5).float()
    correct = (predictions == targets).float()
    all=(targets==targets).float()
    accuracy = correct.sum() / all.sum()

    # 计算精确率、召回率、F1 分数
    true_positives = ((predictions == 1) & (targets == 1)).float().sum()
    false_positives = ((predictions == 1) & (targets == 0)).float().sum()
    false_negatives = ((predictions == 0) & (targets == 1)).float().sum()

    precision = true_positives / (true_positives + false_positives + 1e-10)  # 防止分母为零
    recall = true_positives / (true_positives + false_negatives + 1e-10)  # 防止分母为零

    f1_score = 2 * precision * recall / (precision + recall + 1e-10)  # 防止分母为零

    return accuracy.item(), precision.item(), recall.item(), f1_score.item()

def test_single_cyclegan():
    new_model=torch.load(cyclegan_cdmodel_path_single,map_location=device)
    transformer_model=torch.load(single_cyclegan_model_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    transformer_model=transformer_model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    A_data=valid_log
    print(valid_log)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    B_data=valid_log
    calculate_all(A_data,B_data)

def test_single_ccgan():
    new_model=torch.load(refactor_cdmodel_path_single,map_location=device)
    transformer_model=torch.load(single_refactor_model_path,map_location=device)
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    transformer_model=transformer_model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A,B)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    A_data=valid_log
    print(valid_log)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        A=transformer_model(A,B)
        res=new_model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    B_data=valid_log
    calculate_all(A_data,B_data)

def move_pic(array,ifsnow,transformer_model,model,new_model):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(len(array)):
        #predictions=(predictions>=0.5).float()
        num=array[i]
        if ifsnow:
            A_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test/A/%05d.jpg'%num
            B_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test/B/%05d.jpg'%num
            OUT_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test/OUT/%05d.jpg'%num
            pictureA=transform(Image.open(A_path)).unsqueeze(0)
            pictureB=transform(Image.open(B_path)).unsqueeze(0)
            pictureA=pictureA.to(device)
            pictureB=pictureB.to(device)
            O_RES=model(pictureA,pictureB)
            O_RES=(O_RES>=0.5).int().detach().cpu().numpy()[0][0]*255
            pictureA=transformer_model(pictureA,pictureB)
            MY_RES=new_model(pictureA,pictureB)
            MY_RES=(MY_RES>=0.5).int().detach().cpu().numpy()[0][0]*255
            pictureA=pictureA.squeeze(0)
            for i in range(3):
                pictureA[i] = pictureA[i] * std[i] + mean[i]
            to_pil = transforms.ToPILImage()
            restored_image = to_pil(pictureA)
            restored_image.save('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/snow/TRAN/%05d.jpg'%num)
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/snow/A/%05d.jpg'%num,cv2.imread(A_path))
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/snow/B/%05d.jpg'%num,cv2.imread(B_path))
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/snow/OUT/%05d.jpg'%num,cv2.imread(OUT_path))
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/snow/O_RES/%05d.jpg'%num,O_RES)
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/snow/MY_RES/%05d.jpg'%num,MY_RES)
        else:
            A_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test/A/%05d.jpg'%num
            B_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test/B/%05d.jpg'%num
            OUT_path='/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test/OUT/%05d.jpg'%num
            pictureA=transform(Image.open(A_path)).unsqueeze(0)
            pictureB=transform(Image.open(B_path)).unsqueeze(0)
            pictureA=pictureA.to(device)
            pictureB=pictureB.to(device)
            O_RES=model(pictureA,pictureB)
            O_RES=(O_RES>=0.5).int().detach().cpu().numpy()[0][0]*255
            pictureA=transformer_model(pictureA,pictureB)
            MY_RES=new_model(pictureA,pictureB)
            MY_RES=(MY_RES>=0.5).int().detach().cpu().numpy()[0][0]*255
            pictureA=pictureA.squeeze(0)
            for i in range(3):
                pictureA[i] = pictureA[i] * std[i] + mean[i]
            to_pil = transforms.ToPILImage()
            restored_image = to_pil(pictureA)
            restored_image.save('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/nosnow/TRAN/%05d.jpg'%num)
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/nosnow/A/%05d.jpg'%num,cv2.imread(A_path))
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/nosnow/B/%05d.jpg'%num,cv2.imread(B_path))
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/nosnow/OUT/%05d.jpg'%num,cv2.imread(OUT_path))
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/nosnow/O_RES/%05d.jpg'%num,O_RES)
            cv2.imwrite('/archive/hot17/zjp/Cross-Seasonal-CD/My_model/show/nosnow/MY_RES/%05d.jpg'%num,MY_RES)

def test_list():
    model=torch.load(original_cdmodel_path,map_location=device)
    new_model=torch.load(refactor_cdmodel_path,map_location=device)
    transformer_model=torch.load(refactor_model_path,map_location=device)
    transformer_nosnow_model=torch.load(refactor_nosnow_model_path,map_location=device)
    # test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    # test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    # test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    # criterion = nn.BCELoss()
    transformer_model=transformer_model.to(device)
    # f1s=np.array([])
    # for batch in tqdm(test_loader):
    #     A,B,OUT=batch
    #     A=A.to(device)
    #     B=B.to(device)
    #     OUT=OUT.to(device)
        
    #     res_original=model(A,B)
    #     batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res_original, OUT)
    #     original_f1=batch_f1_score
    #     A=transformer_model(A,B)
    #     res=new_model(A,B)
    #     loss=criterion(res,OUT)
    #     loss_num=loss.detach().cpu().numpy()
        
    #     batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
    #     f1s=np.append(f1s,batch_f1_score-original_f1)
    # indices = np.argsort(f1s)

    # # 使用切片操作获取前N个索引
    # top_n_indices = indices[-10:]
    # print(top_n_indices)
    # move_pic(top_n_indices,True,transformer_model,model,new_model)
    # f1s=np.array([])
    # for batch in tqdm(test_loader2):
    #     A,B,OUT=batch
    #     A=A.to(device)
    #     B=B.to(device)
    #     OUT=OUT.to(device)
        
    #     res_original=model(A,B)
    #     batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res_original, OUT)
    #     original_f1=batch_f1_score
    #     A=transformer_nosnow_model(A,B)
    #     res=new_model(A,B)
    #     loss=criterion(res,OUT)
    #     loss_num=loss.detach().cpu().numpy()
        
    #     batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
    #     f1s=np.append(f1s,batch_f1_score-original_f1)
    # indices = np.argsort(f1s)

    # # 使用切片操作获取前N个索引
    # top_n_indices = indices[-10:]
    # print(top_n_indices)
    # move_pic(top_n_indices,False,transformer_nosnow_model,model,new_model)
    # f1s=np.array([])
    # move_pic(np.array([842,1089,1212,1069,517]),True,transformer_model,model,new_model)
    move_pic(np.array([586,428,929,485,726]),False,transformer_nosnow_model,model,new_model)
    # move_pic(np.array([0]),True,transformer_model,model,new_model)
    #结果：snow【842  707 1089 1212   97 1069  123  386  344  517】，nosnow【586  428  929  485 1163   15 1092  668  401  726】
    #选择：snow【842  1089 1212   1069  517】，nosnow【586  428  929  485 726】

def test_swau():
    model=torch.load(swau_cdmodel_path,map_location=device)
    test_dataset = SWAUtestdataset('/archive/hot17/zjp/Cross-Seasonal-CD/SWAUdataset/test/snow')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = SWAUtestdataset('/archive/hot17/zjp/Cross-Seasonal-CD/SWAUdataset/test/nosnow')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    model=model.to(device)
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"condition":"snow","Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score,"total":total_samples}
    print(valid_log)
    A_data=valid_log
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.to(device)
        B=B.to(device)
        OUT=OUT.to(device)
        
        res=model(A,B)
        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT,swau=True)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"condition":"nosnow","Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score,"total":total_samples}
    print(valid_log)
    B_data=valid_log
    calculate_all(A_data,B_data)

def test_cgnet_original():
    model = CGNet().cuda()
    model.load_state_dict(torch.load(original_cdmodel_path))
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.cuda()
        B=B.cuda()
        OUT=OUT.cuda()
        
        res=model(A,B)
        output = F.sigmoid(res[1])
        pred = output.data
        loss=criterion(pred,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(pred, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"condition":"snow","Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score,"total":total_samples}
    print(valid_log)
    A_data=valid_log
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.cuda()
        B=B.cuda()
        OUT=OUT.cuda()
        
        res=model(A,B)
        output = F.sigmoid(res[1])
        pred = output.data
        loss=criterion(pred,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(pred, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"condition":"nosnow","Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score,"total":total_samples}
    print(valid_log)
    B_data=valid_log
    calculate_all(A_data,B_data)
def test_cgnet_my():
    new_model = CGNet().cuda()
    new_model.load_state_dict(torch.load(refactor_cdmodel_path))
    transformer_model=torch.load(refactor_model_path,map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    transformer_nosnow_model=torch.load(refactor_nosnow_model_path,map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.BCELoss()
    transformer_model=transformer_model.cuda()
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader):
        A,B,OUT=batch
        A=A.cuda()
        B=B.cuda()
        OUT=OUT.cuda()
        
        A=transformer_model(A,B)
        res=new_model(A,B)
        output = F.sigmoid(res[1])
        pred = output.data
        loss=criterion(pred,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(pred, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    Adata=valid_log
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(test_loader2):
        A,B,OUT=batch
        A=A.cuda()
        B=B.cuda()
        OUT=OUT.cuda()
        
        A=transformer_nosnow_model(A,B)
        res=new_model(A,B)
        output = F.sigmoid(res[1])
        pred = output.data
        loss=criterion(pred,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(pred, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    print(valid_log)
    Bdata=valid_log
    calculate_all(Adata,Bdata)
if __name__=='__main__':
    # test_unetpp_cyclegan()
    # test_unetpp_original()
    # test_unetpp_refactor()
    # print('original:\n')
    # test_unetpp_original_all()
    # print('refactor:\n')
    # test_unetpp_refactor_all_new()
    # print('cyclegan:\n')
    # test_unetpp_cyclegan_all()
    # print('cyclegan:\n')
    # test_single_cyclegan()
    # print('refactor:\n')
    # test_single_ccgan()
    # test_list()
    # test_swau()
    test_cgnet_my()
    test_cgnet_original()
    




















    # def test_unetpp_original_all():
#     model=torch.load(original_cdmodel_path,map_location=device)
#     test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
#     criterion = nn.BCELoss()
#     model=model.to(device)
#     total_accuracy = 0
#     total_precision = 0
#     total_recall = 0
#     total_f1_score = 0
#     total_loss=0
#     total_samples = 0
#     for batch in tqdm(test_loader):
#         A,B,OUT=batch
#         A=A.to(device)
#         B=B.to(device)
#         OUT=OUT.to(device)
        
#         res=model(A,B)
#         loss=criterion(res,OUT)
#         loss_num=loss.detach().cpu().numpy()
        
#         batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
#         batch_size = len(OUT)
#         total_accuracy += batch_accuracy * batch_size
#         total_precision += batch_precision * batch_size
#         total_recall += batch_recall * batch_size
#         total_f1_score += batch_f1_score * batch_size
#         total_samples += batch_size
#         total_loss+=loss_num * batch_size
#     overall_accuracy = total_accuracy / total_samples
#     overall_precision = total_precision / total_samples
#     overall_recall = total_recall / total_samples
#     overall_f1_score = total_f1_score / total_samples
#     overall_loss=total_loss / total_samples
#     valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
#     print(valid_log)

# def test_unetpp_refactor_all():
#     model=torch.load(original_cdmodel_path,map_location=device)
#     new_model=torch.load(refactor_cdmodel_path,map_location=device)
#     transformer_model=torch.load(refactor_model_path,map_location=device)
#     test_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
#     test_dataset2 = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/test')
#     test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=1)
#     criterion = nn.BCELoss()
#     model=model.to(device)
#     transformer_model=transformer_model.to(device)
#     total_accuracy = 0
#     total_precision = 0
#     total_recall = 0
#     total_f1_score = 0
#     total_loss=0
#     total_samples = 0
#     for batch in tqdm(test_loader):
#         A,B,OUT=batch
#         A=A.to(device)
#         B=B.to(device)
#         OUT=OUT.to(device)
        
#         A=transformer_model(A,B)
#         res=new_model(A,B)
#         loss=criterion(res,OUT)
#         loss_num=loss.detach().cpu().numpy()
        
#         batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
#         batch_size = len(OUT)
#         total_accuracy += batch_accuracy * batch_size
#         total_precision += batch_precision * batch_size
#         total_recall += batch_recall * batch_size
#         total_f1_score += batch_f1_score * batch_size
#         total_samples += batch_size
#         total_loss+=loss_num * batch_size
#     for batch in tqdm(test_loader2):
#         A,B,OUT=batch
#         A=A.to(device)
#         B=B.to(device)
#         OUT=OUT.to(device)
        
#         res=model(A,B)
#         loss=criterion(res,OUT)
#         loss_num=loss.detach().cpu().numpy()
        
#         batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
#         batch_size = len(OUT)
#         total_accuracy += batch_accuracy * batch_size
#         total_precision += batch_precision * batch_size
#         total_recall += batch_recall * batch_size
#         total_f1_score += batch_f1_score * batch_size
#         total_samples += batch_size
#         total_loss+=loss_num * batch_size
#     overall_accuracy = total_accuracy / total_samples
#     overall_precision = total_precision / total_samples
#     overall_recall = total_recall / total_samples
#     overall_f1_score = total_f1_score / total_samples
#     overall_loss=total_loss / total_samples
#     valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
#     print(valid_log)
