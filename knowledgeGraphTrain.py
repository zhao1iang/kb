#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: zhaoliang
# Date : 2018/11/5
# TODO: add Graph attention method 

import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random
import os
import math
import time
from utils import progress_bar,print_log,Dataset,getConfig

FILEPATH='/home/lanco/zhaoliang/KB/trainData.pickle'
NODEFILE='/home/lanco/zhaoliang/KB/nodeList.pickle'
config = Config(os.path.join(os.path.dirname(FILEPATH),'config.yml'))
trainNum=0
batchSize=10
nodeNum=config.nodeNum+config.relationNum  # number of node in graph ,every node denotes a unique entity,it can  change according to specific pruning operation
hiddenLen=128  
lr=0.01
config.add('lr',lr)
trainData=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



trainloader = torch.utils.data.DataLoader(dataset=Dataset(FILEPATH,config.trainDataNum),batch_size=,shuffle=True,collate_fn=utils.padding)

def getVoc():
    voc={}
    with open(os.path.join(os.path.dirname(NODEFILE),'relationList.pickle'),'rb') as fr:
        with open(NODEFILE,'rb') as f:
            nodeList=pickle.load(f)
            relationList=pickle.load(fr)
            nodeList+=relationList
            for index,name in enumerate(nodeList):
                voc[name]=index
    return voc
voc = getVoc()

class Embedding(nn.Module): 
    def __init__(self,vocSize,hiddenLen):
        super(Embedding,self).__init__()
        self.embedding=nn.Embedding(vocSize,hiddenLen)
    def forward(self,word):
        return self.embedding(word)
embedding=Embedding(nodeNum,hiddenLen).to(device)  #hiddenLen is the lenth of the word embedding

def getIdx(path):
    idx=[]
    for i,node in enumerate(path):
        idx.append(voc[node])
    return torch.tensor(idx,device=device)



class Path2Vec(nn.Module):  #we use a LSTM network generate a vector which is corresponding to a path  [node 1,relation 1, node 2] to 
    def __init__(self,embedding,hiddenLen):
        super(Path2Vec,self).__init__()
        self.hiddenLen=hiddenLen
        self.embedding = embedding
        self.gru = nn.GRU(hiddenLen,hiddenLen)
    def forward(self,path):
        output,_ = self.gru(self.embedding(getIdx(path)).view(len(path),1,self.hiddenLen))
        return output.sum(0).view(self.hiddenLen)/len(path)

p2v=Path2Vec(embedding,hiddenLen).to(device)  #hiddenLen is the hidden state size in lstm

class ForwardNetwork(nn.Module):
    def __init__(self,embedding):
        super(ForwardNetwork,self).__init__()
        self.first=nn.Linear(hiddenLen*3,hiddenLen*5)
        self.firstRelu = nn.ReLU(inplace=True)
        self.second = nn.Linear(hiddenLen*5,hiddenLen*3)
        self.secondRelu = nn.ReLU(inplace=True)
        self.third = nn.Linear(hiddenLen*3,hiddenLen*1)

    def forward(self,embeddingA,embeddingB,embeddingPath):
        return self.third(self.secondRelu(self.second(self.firstRelu(self.first(torch.cat((embeddingA,embeddingB,embeddingPath),0))))))

f=ForwardNetwork(embedding).to(device)
optimizer_f = optim.SGD(f.parameters(), lr=lr)
optimizer_p2v = optim.SGD(p2v.parameters(),lr=lr)

def save_model(path, f,p2v, optimizer_f, optimizer_p2v,update):
    f_state_dict = f.state_dict()
    p2v_state_dict = p2v.state_dict()
    checkpoints = {
        'f': f_state_dict,
        'p2v': p2v_state_dict,
        'optimf': optimizer_f,
        'optim_p2v':optimizer_p2v,
        'update':update
    }
    torch.save(checkpoints, path)
# def print_log(file):
#     def write_log(s):
#         print(s, end='')
#         with open(file, 'a') as f:
#             f.write(s)
#     return write_log

# if not os.path.exists(os.path.join(os.path.dirname(FILEPATH),'log')):
#     os.mkdir(os.path.join(os.path.dirname(FILEPATH),'log'))
# logfile=os.path.join(os.path.dirname(FILEPATH),'log','log'+str(int(ramdom.random*10000000))+'.txt')
# if not os.path.exists(logfile):
#     os.mknod(logfile)

# print_log = print_log(logfile)

criterion=nn.MSELoss()


def train():
    try:
        for index,data in enumerate(trainData):
            A,B,path,hop=data['A'],data['B'],data['path'],int(data['hop'])
            optimizer_f.zero_grad()
            optimizer_p2v.zero_grad()
            embeddingPath=p2v(path)
            outputs=f(embedding(torch.tensor(voc[A],device=device)),embedding(torch.tensor(voc[B],device=device)),embeddingPath)
            loss=criterion(outputs, math.log(hop+1))
            loss.backward()
            optimizer_f.step()
            optimizer_p2v.step()
            if index%10000 == 0:
                progress_bar(index,trainNum)
    finally:
        chptFile=os.path.join(os.path.dirname(FILEPATH),'checkpoint'+str(int(time.time()*1000))+'.ckpt')
        if not os.path.exists(chptFile):
            os.mknod(chptFile)
        save_model(chptFile,f,p2v,optimizer_f,optimizer_p2v,index)
        print("saving in %d and if all done it should be  %d."%(index,trainNum))
train()
print('All done.')


        
            


