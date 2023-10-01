import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import cv2
import os
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as transforms 
from sklearn import metrics
from pydicom import dcmread
plt.style.use('fivethirtyeight')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

data_train=pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
data_train.head(5)
temp_cat_cols=['sex','anatom_site_general_challenge']
data_train=data_train.drop(['patient_id','benign_malignant'],axis=1)
data_train

plt.hist(data_train.age_approx,bins=[10,20,30,40,50,60,70,80,90,100,110],edgecolor='r')
plt.axvline(data_train.age_approx.mean(),color='r',linewidth=2)
plt.tight_layout()
data_train.image_name.unique().__len__()
def remove_cats(data_train,col_name):
    temp_df=pd.get_dummies(data_train[col_name])
    data_train=pd.concat([data_train,temp_df],axis=1)
    return data_train
data_train=remove_cats(data_train,'sex')
data_train=remove_cats(data_train,'anatom_site_general_challenge')
data_train=data_train.drop(['sex','anatom_site_general_challenge'],axis=1)
data_train=data_train.drop(['diagnosis'],axis=1)
Y=data_train.pop('target').values.reshape(-1,1).astype(np.float32)
Y.shape
data_train.fillna(value=mean,inplace=True)
data_train.age_approx.value_counts()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
sc=StandardScaler()
data_train.isna().sum()

data_train['kfold']=-1
data_train

F=StratifiedKFold(n_splits=5)
for i,(T,t) in enumerate(F.split(X=data_train.iloc[:,1:],y=Y)):
    data_train.loc[t,'kfold']=i
data_train['tar']=Y
data_train.tar.var()

class CreateDataset(Dataset):
    def __init__(self,data,transform1=None,transform2=None):
        self.data=data
        A1=self.data.image_name.to_list()
        A2=self.data.tar.to_list()
        mapper={}
        for i in range(len(A1)):
            mapper[A1[i]]=A2[i]
        self.path='../input/siim-isic-melanoma-classification/train'
        self.transform1=transform1
        self.transform2=transform2
        self.X=self.data.iloc[:,1:-2].values.astype(np.float32)
        assert self.X.shape[-1]==9
        self.X=sc.fit_transform(self.X)
        self.Ims=os.listdir(self.path)
        self.labels=[]
        for i in self.Ims:
            temp_k=i.split('.')[0]
            if mapper.get(temp_k,None)!=None:
                self.labels.append(mapper.get(temp_k))
        assert len(self.labels)==len(self.X)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,indx):
        A=self.X[indx]
        img_temp=self.Ims[indx]
        img_temp=dcmread(os.path.join(self.path,img_temp)).pixel_array/255.
        img_temp=img_temp.astype(np.float32)
        B=self.labels[indx]
        if self.transform1:
            img_temp=self.transform1(img_temp)
        A=torch.tensor(A)
        B=torch.tensor(B,dtype=torch.float32)
        return (A,img_temp,B)
def train_in_folds(fold):
        Val_imgs=data_train[data_train['kfold']==fold]
        Train_imgs=data_train[data_train['kfold']!=fold]
        my_trans2=transforms.Compose([transforms.ToTensor()])
        my_trans1=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256)),transforms.RandomRotation(degrees=45)])
        train_set=CreateDataset(data=Train_imgs,transform1=my_trans1,transform2=my_trans2)
        val_set=CreateDataset(data=Val_imgs,transform1=my_trans2,transform2=my_trans2)
        TrainLoader=DataLoader(dataset=train_set,batch_size=32,shuffle=True,drop_last=True)
        ValLoader=DataLoader(dataset=val_set,batch_size=32,shuffle=True,drop_last=True)
        from tqdm import tqdm
        optimizer=opt.Adam(merger.parameters(),lr=1e-4)
        loss_f=nn.BCEWithLogitsLoss()
        epochs=10
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
        )
        print(f'Fold {fold}')
        for i in range(epochs):
            L=0
            merger.train()
            loop=tqdm(enumerate(TrainLoader),total=len(TrainLoader),leave=False)
            for _,(a,b,c) in loop:
                merger.zero_grad()
                a,b,c=a.to(device),b.to(device),c.to(device)
                out=merger(b,a)
                c=c.view(32,1)
                loss=loss_f(out,c)
                L+=loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(merger.parameters(),max_norm=1)
                optimizer.step()
                acc=(torch.round(torch.sigmoid(out))==c).sum().float()
                loop.set_description(f'Epoch[{i+1}/{epochs}] has loss and acc')
                loop.set_postfix(loss=loss.item(),acc=acc/32)
            print('Evaluating')
            C=0
            T=0
            try:
                merger.eval()
                for a,b,c in ValLoader:
                    a,b,c=a.to(device),b.to(device),c.to(device)
                    pred=merger(b,a)
                    pred=torch.round(torch.sigmoid(pred))
                    C+=(pred==c).sum().float()
                    T+=32

                printf(f"Loss {L/len(TrainLoader)} val acc {(C/T)*100}")
            except:
                print(f"Loss {L/len(TrainLoader)}")

class LinearModel(nn.Module):
    def __init__(self,inp_shape):
        super().__init__()
        self.inp_shape=inp_shape
        self.l1=nn.Linear(32+self.inp_shape,256)
        self.l2=nn.Linear(256,512)
        self.l3=nn.Linear(512,64)
        self.l4=nn.Linear(64,1)
    def forward(self,x,u):
        x=torch.cat((x,u),dim=-1)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
class Merger(nn.Module):
    def __init__(self,lin,conv):
        super().__init__()
        self.lin=lin
        self.conv=conv
    def forward(self,x,o):
        x=self.conv(x)
        x=self.lin(o,x)
        return x

linear_model=LinearModel(9)
linear_model.to(device)
class CNNNetwork(nn.Module):
    def __init__(self,out_dim=64):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(32,64,kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,256,kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=(5,5)),
            nn.ReLU(),
            nn.Conv2d(512,64,kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.f=nn.Flatten()
        self.linear_layers=nn.Sequential(
            nn.Linear(43264,1024),
            nn.ReLU(),
            nn.Linear(1024,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
            
        )
    def forward(self,x):
        x=self.conv_layers(x)
        x=self.f(x)
        x=self.linear_layers(x)
        return x

model=CNNNetwork()
model.to(device)
merger=Merger(linear_model,model)
merger.to(device)

#82% val_acc auc 0.7345
for i in range(5):
    train_in_folds(i)

def predict_on_test_():
    pass
