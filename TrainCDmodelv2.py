from CDmodelv2 import CDmodelv2
from Data import CDdataset
from torch.utils.data import DataLoader
import change_detection_pytorch as cdp
import torch
import os
from meter import AverageValueMeter
from tqdm import tqdm
import json

device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
generator='unet'#resnet or unet
season='no'#no or snow or nosnow
if season=='no':
    weight_path='./CDmodelv2('+generator+')NoSeason_weights'
    #CDmodelv2(resnet)NoSeason_weights


model=CDmodelv2(generator)
model.to(device)

print('train cdmodelv2 using generator '+generator)

train_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/train')
val_dataset=CDdataset('/archive/hot17/zjp/Cross-Seasonal-CD/CDdataset/val')

# train_dataset=CDdataset('/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/train')
# val_dataset=CDdataset('/archive/cold0/zjp/Cross-Seasonal-CD/CDdataset_seasonal/snow/val')

train_loader=DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers=1)
valid_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1)

Loss = cdp.utils.losses.CrossEntropyLoss()
# Loss = cdp.losses.focal.FocalLoss(mode='binary',alpha=0.25,gamma=2,normalized=True)
metrics = [
    cdp.utils.metrics.Fscore(activation='argmax2d'),
    cdp.utils.metrics.Precision(activation='argmax2d'),
    cdp.utils.metrics.Recall(activation='argmax2d'),
]
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])
scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=0.1)
EPOCH=201
max_score = 0
best_epoch=-1

log_list=[]

for epoch in range(EPOCH):
    print('\nEpoch: {}'.format(epoch))
    train_logs = {}
    valid_logs={}
    for batch in tqdm(train_loader):
        _As,_Bs,_OUTs=batch
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
        As=_As.to(device)
        Bs=_Bs.to(device)
        OUTs=_OUTs.squeeze().to(device)
        optimizer.zero_grad()

        change_map,pic=model(As,Bs)
        loss=Loss(change_map,OUTs.long())

        # update loss logs
        loss_value = loss.detach().cpu().numpy()
        loss_meter.add(loss_value)
        loss_logs = {Loss.__name__: loss_meter.mean}
        train_logs.update(loss_logs)

        # update metrics logs
        for metric_fn in metrics:
            metric_value = metric_fn(change_map, OUTs).detach().cpu().numpy()
            metrics_meters[metric_fn.__name__].add(metric_value)
        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        train_logs.update(metrics_logs)

        loss.backward()
        optimizer.step()

    for batch in tqdm(valid_loader):
        _As,_Bs,_OUTs=batch
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
        As=_As.to(device)
        Bs=_Bs.to(device)
        OUTs=_OUTs.squeeze(1).to(device) 
        change_map,pic=model(As,Bs)
        loss=Loss(change_map,OUTs.long())

        # update loss logs
        loss_value = loss.detach().cpu().numpy()
        loss_meter.add(loss_value)
        loss_logs = {Loss.__name__: loss_meter.mean}
        valid_logs.update(loss_logs)

        # update metrics logs
        for metric_fn in metrics:
            metric_value = metric_fn(change_map, OUTs).detach().cpu().numpy()
            metrics_meters[metric_fn.__name__].add(metric_value)
        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        valid_logs.update(metrics_logs)
    scheduler_steplr.step()
    log_list.append({'epoch':epoch,'train_logs':train_logs,'valid_logs':valid_logs,'best_epoch':best_epoch,'generator':generator})
    print(log_list)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if epoch!=0 and epoch%5==0:
        filename=f'/epoch_{epoch}.pth'
        torch.save(model, weight_path+filename)
    if  max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        print('max_score', max_score)
        torch.save(model, weight_path+'/best_model.pth')
        best_epoch=epoch
        print('Model saved!')
with open(weight_path+'/alogs.json','w') as file:
    json.dump(log_list,file)