from ConditionalGanGenerator import UnetGenerator ,NLayerDiscriminator, GANLoss, ResnetGenerator
import torch
from torch import nn
import functools
import change_detection_pytorch as cdp
class CDmodelv2(nn.Module):
    def __init__(self,generator,class_num=1):
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if generator=='unet':
            self.model=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)
        elif generator=='resnet':
            self.model=ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=5)
        self.uplusplus=cdp.UnetPlusPlus(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_num,  # model output channels (number of classes in your datasets)
            siam_encoder=True,  # whether to use a siamese encoder
            fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
            activation='sigmoid'
        )

    def forward(self, img_origin,img_target):
        output_picture=self.model(img_origin,img_target)

        picture=output_picture

        change_map=self.uplusplus(img_origin,output_picture)

        return change_map,picture
    
if __name__=='__main__':
    x=torch.randn((4,3,256,256))
    y=torch.randn((4,3,256,256))
    model=CDmodelv2('resnet')
    model=model.cpu()
    z,y=model(x,y)
    print(z)
