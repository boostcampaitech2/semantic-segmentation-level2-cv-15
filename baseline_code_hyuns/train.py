import os
import torch
from utils import label_accuracy_score, add_hist
import wandb

import numpy as np
import pandas as pd
from tqdm import tqdm

# 모델 저장 함수 정의
val_every = 1

saved_dir = './saved'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)

def save_model(model, model_path):
    #check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            #acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        #loggin_with_wandb("valid", epoch, len(data_loader), round(loss.item(),4), round(mIoU,4), acc)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]

        wandb.log({list(IoU_by_class[i].keys())[0] : list(IoU_by_class[i].values())[0] for i in range(len(IoU_by_class))})

        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss, mIoU

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, model_path, val_every, device):
    
    #, criterion, optimizer, saved_dir, val_every, 
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = 0

    wandb.watch(model)
    
    for epoch in range(num_epochs):
        # wandb
        run = wandb.init(
            # Set entity to specify your username or team name
            entity="hyunsoo",
            # Set the project where this run will be logged
            project="semantic-segmetation", 
            # Track hyperparameters and run metadata
            group='experiment-3_Unet++_mIoU',
            name = f"epoch_{epoch}",
            config={
            "Epoch" : epoch, 
            "loss" : "Cross_Entropy",
            "optim" : "Adam",
            }
        )
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            #print(masks.shape, outputs.shape, hist.shape)
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)


            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')


            #loggin_with_wandb("train", epoch, step, round(loss.item(),4), round(mIoU,4), acc)
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            #if avrg_loss < best_loss:
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                #best_loss = avrg_loss
                best_mIoU = val_mIoU
                wandb.log({"best_mIoU" : best_mIoU})
                #wandb.log({"best_loss" : best_loss})
                torch.save(model, model_path)

        run.finish()


