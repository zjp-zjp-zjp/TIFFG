from CDmodelv2 import CDmodelv2
from Data import CDdataset
from torch.utils.data import DataLoader
import change_detection_pytorch as cdp
import torch
import os
from meter import AverageValueMeter
from tqdm import tqdm
import json

device=torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
#要同步改一下zjp/Cross-Seasonal-CD/My_model/change_detection_pytorch/encoders/__init__.py的cuda

generator='unet'#resnet or unet
# weight_path='./weights_CDmodelv2('+generator+'&focalloss)'
save_folder='./CDmodelv2('+generator+'&focalloss)_snow_weights'

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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


model=CDmodelv2(generator)
# model.apply(weights_init_normal)
model.to(device)

print('train cdmodelv2 using generator '+generator)
print('focalloss')

# train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train')
# val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val')

train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/train')
val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/val')

train_loader=DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers=1)
valid_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1)

# Loss = cdp.utils.losses.FocalLoss()
Loss = cdp.losses.focal.FocalLoss(mode='binary',alpha=0.25,gamma=2,normalized=True)
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.00001),
])
scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=0.1)
EPOCH=201
max_score = 0
best_epoch=-1



for epoch in range(EPOCH):
    one_time=True
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0
    for batch in tqdm(train_loader):
        _As,_Bs,_OUTs=batch
        As=_As.to(device)
        Bs=_Bs.to(device)
        OUT=_OUTs.squeeze().to(device)
        optimizer.zero_grad()
        
        change_map,pic=model(As,Bs)
        if(one_time):
            print(change_map)
            one_time=False
        loss=Loss(change_map,OUT.long())
        loss_num=loss.detach().cpu().numpy()

        # update loss logs
        

        loss.backward()
        optimizer.step()

        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(change_map, OUT)
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
    train_log={"Loss:":overall_loss,"Precision:":overall_precision,"Recall:":overall_recall,"F1-score:":overall_f1_score}
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_loss=0
    total_samples = 0

    for batch in tqdm(valid_loader):
        _As,_Bs,_OUTs=batch
        As=_As.to(device)
        Bs=_Bs.to(device)
        OUT=_OUTs.squeeze(1).to(device) 
        change_map,pic=model(As,Bs)
        loss=Loss(change_map,OUT.long())
        loss_num=loss.detach().cpu().numpy()

        batch_accuracy, batch_precision, batch_recall, batch_f1_score = calculate_metrics(change_map, OUT)
        batch_size = len(OUT)
        total_accuracy += batch_accuracy * batch_size
        total_precision += batch_precision * batch_size
        total_recall += batch_recall * batch_size
        total_f1_score += batch_f1_score * batch_size
        total_samples += batch_size
        total_loss+=loss_num * batch_size
    scheduler_steplr.step()
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