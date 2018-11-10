#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhaoliang
# Date: 2018/11/5

import networkx as nx
import pickle
import os
import sys
import time
from utils import progress_bar,Config
FILEPATH='/home/lanco/zhaoliang/KB/edgeList.pickle'

trainData=[]  #   {A: B: PATH: hop:}
trainNum=0
config = Config(os.path.join(os.path.dirname(FILEPATH),'config.yml'))

with open(FILEPATH,'rb') as f:
    edgeList=pickle.load(f)
G = nx.DiGraph(edgeList)
nx.write_gpickle(G, os.path.join(os.path.dirname(FILEPATH),"graph.gpickle"))


trainTxt=os.path.join(os.path.dirname(FILEPATH),"train.txt")
with open(trainTxt,'w') as sf:
    with open(os.path.join(os.path.dirname(FILEPATH),"nodeList.pickle"),'rb') as f:
        nodeList=pickle.load(f)
        nodeNum=len(nodeList)
        for index,firstNode in enumerate(nodeList):
            for k,v in G[firstNode].items():
                secondNode,firstRelation=k,v['relation']
                trainData.append({'A':firstNode,'B':secondNode,'path':[firstRelation],'hop':1})
                sf.write(firstNode+'\t'+secondNode+'\t'+firstRelation+'\t'+str(1))
                trainNum+=1
                if trainNum%1000000 == 0:
                    print('processing %d'%trainNum)
                for kk,vv in G[secondNode].items():
                    thirdNode,secondRelation=kk,vv['relation']
                    trainData.append({'A':firstNode,'B':thirdNode,'path':[firstRelation,secondNode,secondRelation],'hop':2})
                    trainNum+=1
                    sf.write(firstNode+'\t'+thirdNode+'\t'+firstRelation+'$'+secondNode+'$'+secondRelation+'\t'+str(2))
                    if trainNum%1000000 == 0:
                        print('processing %d'%trainNum)
                    # for kkk,vvv in G[thirdNode].items():
                    #     fouthNode,thirdRelation=kkk,vvv['relation']
                    #     trainData.append({'A':firstNode,'B':fouthNode,'path':[firstRelation,secondNode,secondRelation,thirdNode,thirdRelation],'hop':3})
                    #     trainNum+=1
                    #     if trainNum%10000 == 0:
                    #         print('processing %d'%trainNum)
            progress_bar(index,nodeNum)
print('there are %d elements in trainData'%trainNum)
config.add('trainDataNum',trainNum)

print("All done!")

            
            


