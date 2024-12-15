from torch import nn
import torch
from CDmodel import SeasonalModel
import itertools
from Data import CDdataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import random
import change_detection_pytorch as cdp
from meter import AverageValueMeter
import os
import json
import functools
from torch.nn import init

device=torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()

epoch_n=201
epoch_delay=100

generator='unet'#'resnet' or 'unet'
season='no'#no or snow or nosnow
# if season=='no':
#     save_dir='./ccGAN('+generator+')NoSeason_weights'
# save_dir='./cyclegan_nosnow_weights'
save_dir='./cyclegan_all_weights'
# train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/train')
# val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/val')
# train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/train')
# val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/nosnow/val')
train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train')
val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val')

print('start training using generator:'+generator)

# class ReplayBuffer:
#     def __init__(self, max_size=50):
#         assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
#         self.max_size = max_size
#         self.data = []

#     def push_and_pop(self, data):
#         to_return = []
#         for element in data.data:
#             element = torch.unsqueeze(element, 0)
#             if len(self.data) < self.max_size:
#                 self.data.append(element)
#                 to_return.append(element)
#             else:
#                 if random.uniform(0, 1) > 0.5:
#                     i = random.randint(0, self.max_size - 1)
#                     to_return.append(self.data[i].clone())
#                     self.data[i] = element
#                 else:
#                     to_return.append(element)
#         return Variable(torch.cat(to_return))
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
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
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,crossAttention=None):
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
        

    def forward(self, x):
        x_origin=x
        y=None

        if self.outermost:
            x=self.down(x)
            x=self.submodule(x)
            y=self.up(x)
        elif self.innermost:
            x=self.down(x)
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
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def backward_D_basic(netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


criterionGAN = GANLoss('lsgan')  # define GAN loss.
criterionCycle = torch.nn.L1Loss()
criterionIdt = torch.nn.L1Loss()
input_shape = (3,256,256)
norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

if generator=='unet':
    G_AB=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)
    G_BA=UnetGenerator(3, 3, 5, 256, norm_layer=norm_layer, use_dropout=False)

D_AB = NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer)
D_BA = NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer)
cd_model=torch.load('best_model.pth')

init_weights(G_AB)
init_weights(D_AB)
init_weights(G_BA)
init_weights(D_BA)



G_AB=G_AB.to(device=device)
D_AB=D_AB.to(device=device)
G_BA=G_BA.to(device=device)
D_BA=D_BA.to(device=device)
criterionGAN.to(device=device)
criterionCycle.to(device=device)
criterionIdt.to(device=device)
cd_model.to(device=device)


optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(D_AB.parameters(), D_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))

# if season=='no':
#     train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train')
#     val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val')

train_loader=DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers=1)
valid_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1)

# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# # Buffers of previously generated samples
# fake_A_buffer = ReplayBuffer()
# fake_B_buffer = ReplayBuffer()
change_loss=cdp.utils.losses.CrossEntropyLoss()
metrics = [
    cdp.utils.metrics.Fscore(activation='argmax2d'),
    cdp.utils.metrics.Precision(activation='argmax2d'),
    cdp.utils.metrics.Recall(activation='argmax2d'),
]
max_score = 0

lambda_A=10.0
lambda_B=10.0
lambda_identity=0.5

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(epoch_n, 0, epoch_delay).step
)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(epoch_n,0, epoch_delay).step
)

log_list=[]
best_epoch=-1
for epoch in range(epoch_n):
    print('\nEpoch: {}'.format(epoch))
    train_logs = {}
    valid_logs={}
    for batch in tqdm(train_loader):
        _As,_Bs,OUT=batch
        A = Variable(_As.type(torch.FloatTensor))
        B = Variable(_Bs.type(torch.FloatTensor))
        A=A.to(device)
        B=B.to(device)
        fakeB=G_AB(A)
        recA=G_BA(fakeB)
        fakeA=G_BA(B)
        recB=G_AB(fakeA)
        set_requires_grad([D_AB, D_BA], False)
        optimizer_G.zero_grad()
        if lambda_identity > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = G_AB(B)
            loss_idt_A = criterionIdt(idt_A, B) * lambda_B * lambda_identity
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = G_BA(A)
            loss_idt_B = criterionIdt(idt_B, A) * lambda_A * lambda_identity
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        # GAN loss D_A(G_A(A))
        loss_G_A = criterionGAN(D_AB(fakeB), True)
        # GAN loss D_B(G_B(B))
        loss_G_B = criterionGAN(D_BA(fakeA), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterionCycle(recA, A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = criterionCycle(recB, B) * lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        optimizer_G.step()
        set_requires_grad([D_AB, D_BA], True)
        optimizer_D.zero_grad()
        backward_D_basic(D_AB,B,fakeB)
        backward_D_basic(D_BA,A,fakeA)
        optimizer_D.step()
    for batch in tqdm(valid_loader):
        _As,_Bs,_OUTs=batch
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
        As=_As.to(device)
        Bs=_Bs.to(device)
        OUTs=_OUTs.squeeze(1).to(device)

        As=G_AB(As)
        change_map=cd_model(As,Bs)
        loss=change_loss(change_map,OUTs.long())

        # update loss logs
        loss_value = loss.detach().cpu().numpy()
        loss_meter.add(loss_value)
        loss_logs = {change_loss.__name__: loss_meter.mean}
        valid_logs.update(loss_logs)

        # update metrics logs
        for metric_fn in metrics:
            metric_value = metric_fn(change_map, OUTs).detach().cpu().numpy()
            metrics_meters[metric_fn.__name__].add(metric_value)
        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        valid_logs.update(metrics_logs)
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    log_list.append({'epoch':epoch,'bestepoch':best_epoch,'train_logs':train_logs,'valid_logs':valid_logs,'generator':generator})
    print(log_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch!=0 and epoch%5==0:
        torch.save(G_AB, save_dir+f'/epoch_{epoch}_AB.pth')
        torch.save(G_BA, save_dir+f'/epoch_{epoch}_BA.pth')
    if  max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        print('max_score', max_score)
        best_epoch=epoch
        torch.save(G_AB, save_dir+f'/best_model_AB.pth')
        torch.save(G_BA, save_dir+f'/best_model_BA.pth')
        print('Model saved!')
with open(save_dir+'/alogs.json','w') as file:
    json.dump(log_list,file)