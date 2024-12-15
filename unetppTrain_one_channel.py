import torch
from torch.utils.data import DataLoader, Dataset
import os
import json
from tqdm import tqdm
import torch.nn as nn

import change_detection_pytorch as cdp
import os
from Data import CDdataset, SWAUdataset
#python unetppTrain_one_channel.py
# DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# train_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify/train')

# valid_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify/val')
# train_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train')

# valid_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val')

# train_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_classify200/train')

# valid_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_classify200/val')
# train_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_single200/train')

# valid_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/cyclegan_single200/val')
# train_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/ccgan_single200/train')

# valid_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/ccgan_single200/val')
train_dataset = SWAUdataset('/archive/hot17/zjp/Cross-Seasonal-CD/SWAUdataset/all')

valid_dataset = SWAUdataset('/archive/hot17/zjp/Cross-Seasonal-CD/SWAUdataset/all')
# train_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify200/train')

# valid_dataset = CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/ccganunet_classify200/val')


#UPerNet
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = cdp.UPerNet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your datasets)
    siam_encoder=True,  # whether to use a siamese encoder
    activation='sigmoid',
    fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
)
save_folder='./upernetweights_SWAU'
# save_folder='./upernetweights_cyclegan'
# save_folder='./upernetweights_ccganunet'
# save_folder='./upernetweights'

# pspnet
# DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# model = cdp.PSPNet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./pspnetweights_SWAU'
# save_folder='./pspnetweights_cyclegan'
# save_folder='./pspnetweights_ccganunet'
# save_folder='./pspnetweights'

# linknet
# DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# model = cdp.Linknet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./linknetweights_SWAU'
# save_folder='./linknetweights_cyclegan'
# save_folder='./linknetweights_ccganunet'
# save_folder='./linknetweights'

# unet++
# DEVICE = 'cuda:4' if torch.cuda.is_available() else 'cpu'
# model = cdp.UnetPlusPlus(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./unetppweights_SWAU'
# save_folder='./unetppweights_cyclegan'
# save_folder='./unetppweights_ccganunet'


#stanet
# model = cdp.STANet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./stanetweights_SWAU'
# save_folder='./stanetweights_cyclegan'
# save_folder='./stanetweights_ccganunet'


#fpn
# DEVICE = 'cuda:5' if torch.cuda.is_available() else 'cpu'
# model = cdp.FPN(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./fpnweights_SWAU'
# save_folder='./fpnweights_cyclegan'
# save_folder='./fpnweights_ccganunet'


#manet
# DEVICE = 'cuda:6' if torch.cuda.is_available() else 'cpu'
# model = cdp.MAnet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./manetweights_SWAU'
# save_folder='./manetweights_cyclegan'
# save_folder='./manetweights_ccganunet'


#unet
# DEVICE = 'cuda:7' if torch.cuda.is_available() else 'cpu'
# model = cdp.Unet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     activation='sigmoid',
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )
# save_folder='./unetweights_SWAU'
# save_folder='./unetweights_cyclegan'
# save_folder='./unetweights_ccganunet'

#cyclegan
# save_folder='./unetppweights_cycleGAN'
# save_folder='./stanetweights_cycleGAN'
# save_folder='./fpnweights_cycleGAN'
# save_folder='./manetweights_cycleGAN'
# save_folder='./unetweights_cycleGAN'

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

model=model.to(torch.device(DEVICE))

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=0.1)

criterion = nn.BCELoss()

EPOCH=101

max_score = 0

best_epoch=-1

def calculate_metrics(predictions, targets):
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

one_time=True

for epoch in range(EPOCH):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(train_loader):
        A,B,OUT=batch
        if(one_time):
            print(OUT)
            one_time=False
        A=A.to(torch.device(DEVICE))
        B=B.to(torch.device(DEVICE))
        OUT=OUT.to(torch.device(DEVICE))
        optimizer.zero_grad()

        res=model(A,B)

        loss=criterion(res,OUT)
        loss_num=loss.detach().cpu().numpy()
        
        loss.backward()
        optimizer.step()
        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(res, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
        # break
    overall_accuracy = total_accuracy / total_samples
    overall_precision = total_precision / total_samples
    overall_recall = total_recall / total_samples
    overall_f1_score = total_f1_score / total_samples
    overall_loss=total_loss / total_samples
    # print("Accuracy:", overall_accuracy)
    train_log={"Loss:":overall_loss,"Precision:":overall_precision,"Recall:":overall_recall,"F1-score:":overall_f1_score}
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(valid_loader):
        A,B,OUT=batch
        A=A.to(torch.device(DEVICE))
        B=B.to(torch.device(DEVICE))
        OUT=OUT.to(torch.device(DEVICE))

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
    # print("Accuracy:", overall_accuracy)
    valid_log={"Loss":overall_loss,"Precision":overall_precision,"Recall":overall_recall,"F1-score":overall_f1_score}
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if epoch!=0 and epoch%5==0:
        torch.save(model, save_folder+f'/epoch_{epoch}.pth')
    if valid_log['F1-score']>max_score:
        max_score=valid_log['F1-score']
        torch.save(model, save_folder+'/best_model.pth')
        best_epoch=epoch
    log={'epoch':epoch,'train_log':train_log,'valid_log':valid_log,'best_epoch':best_epoch}
    with open(save_folder+'/alogs.json','a') as file:
        json.dump(log,file)
        file.write("\n")
    print(log)