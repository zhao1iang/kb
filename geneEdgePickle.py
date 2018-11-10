#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zhaoliang
# Date : 2018/11/5
import networkx as nx
import re
import pickle
import os
import torch
import yaml
from utils import Voc,Config,dump_pickle

FILEPATH='/home/lanco/zhaoliang/KB/en_concept_net_extracted.csv'
ROOTPATH='/home/lanco/zhaoliang/KB/'
edgeList=[]
errorList=[]
nodeList=set()
relationList=[]

config=Config(os.path.join(ROOTPATH,'config.yml'))

voc=Voc(config)

try:
    with open(FILEPATH,'r') as file:
        for index,line in enumerate(file):
            if index%100000 == 0:
                print('processing %d'%index)
            lineSearch=re.search("/a/\[/r/(.+)/,/c/en/(.+?)/.*,/c/en/(.+)/\]",line)
            if lineSearch!=None and lineSearch.group(1)!=None and lineSearch.group(2)!=None:
                voc.addWord(lineSearch.group(3))
                voc.addWord(lineSearch.group(2))
                if lineSearch.group(1) not in relationList:
                    relationList.append(lineSearch.group(1))
            else :
                print("some error occurs!")
                print(line)
                print("index:",index)
                errorList.append({index:line})
    print(len(voc.word2count))
    voc.trim()
    print(len(voc.word2count))
    if not os.path.exists(os.path.join(ROOTPATH,'trimmed_fact.csv')):
        os.mknod(os.path.join(ROOTPATH,'trimmed_fact.csv'))
    trimFactNum=0
    with open(os.path.join(ROOTPATH,'trimmed_fact.csv'),'w') as f:
        with open(FILEPATH,'r') as file:
            for index,line in enumerate(file):
                lineSearch=re.search("/a/\[/r/(.+)/,/c/en/(.+?)/.*,/c/en/(.+)/\]",line)
                if lineSearch!=None and lineSearch.group(1)!=None and lineSearch.group(2)!=None:
                    if lineSearch.group(2)  in voc.word2index  and lineSearch.group(3) in voc.word2index:
                        edgeList.append((lineSearch.group(2),lineSearch.group(3),{"relation":lineSearch.group(1)}))
                        nodeList.add(lineSearch.group(2))
                        nodeList.add(lineSearch.group(3))
                        f.write(line)
                        trimFactNum+=1
                if index%100000 == 0:
                    print('Trimming processing %d'%index)
    print("after trimmed , there are %d facts left."%trimFactNum)

                
finally:
    nodeList=list(nodeList)
    print(nodeList[:10])
    dump_pickle('edgeList.pickle',edgeList,ROOTPATH)
    dump_pickle('errorList.pickle',errorList,ROOTPATH)
    dump_pickle('nodeList.pickle',nodeList,ROOTPATH)
    dump_pickle('relationList.pickle',relationList,ROOTPATH)
    config.add('trimFactNum',trimFactNum)
    config.add('nodeNum',len(nodeList))
    config.add('relationNum',len(relationList))
    config.add('edgeNum',len(edgeList))
    print("there are %d different nodes in graph"%len(nodeList))
    print("there are %d different relations in graph"%len(relationList))
    print('there are %d edges in graph'%len(edgeList))

print("all done")


