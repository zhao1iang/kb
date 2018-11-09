#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import re
import pickle
import os
FILEPATH='/home/zhaoliang/KB/en_concept_net_extracted.csv'
ROOTPATH='/home/zhaoliang/KB/'
edgeList=[]
errorList=[]
nodeList=[]  #store node name
relationList=[]
try:
    with open(FILEPATH,'r') as file:
        for index,line in enumerate(file):
            if index%100000 == 0:
                print('processing %d'%index)
            lineSearch=re.search("/a/\[/r/(.+)/,/c/en/(.+)/,/c/en/(.+)/\]",line)
            if lineSearch!=None and lineSearch.group(1)!=None and lineSearch.group(2)!=None:
                print(lineSearch.group(2),' ',lineSearch.group(3)," ",lineSearch.group(1))
                edgeList.append((lineSearch.group(2),lineSearch.group(3),{"relation":lineSearch.group(1)}))
                if lineSearch.group(2) not in nodeList:
                    nodeList.append(lineSearch.group(2))
                if lineSearch.group(3) not in nodeList:
                    nodeList.append(lineSearch.group(3))
                if lineSearch.group(1) not in relationList:
                    relationList.append(lineSearch.group(1))


            else :
                print("some error occurs!")
                print(line)
                print("index:",index)
                errorList.append({index:line})
                
finally:
    if not os.path.exists(os.path.join(ROOTPATH,'edgeList.pickle')):
        os.mknod(os.path.join(ROOTPATH,'edgeList.pickle'))
    if not os.path.exists(os.path.join(ROOTPATH,'errorList.pickle')):
        os.mknod(os.path.join(ROOTPATH,'errorList.pickle'))
    if not os.path.exists(os.path.join(ROOTPATH,'nodeList.pickle')):
        os.mknod(os.path.join(ROOTPATH,'nodeList.pickle'))
    if not os.path.exists(os.path.join(ROOTPATH,'relationList.pickle')):
        os.mknod(os.path.join(ROOTPATH,'relationList.pickle'))
    with open(os.path.join(ROOTPATH,'edgeList.pickle'),'wb') as f:
        pickle.dump(edgeList,f)
    with open(os.path.join(ROOTPATH,'errorList.pickle'),'wb') as f:
        pickle.dump(errorList,f)
    with open(os.path.join(ROOTPATH,'nodeList.pickle'),'wb') as f:
        pickle.dump(errorList,f)
    with open(os.path.join(ROOTPATH,'relationList.pickle'),'wb') as f:
        pickle.dump(errorList,f)
    print("there are %d different node in graph"%len(nodeList))
    print("there are %d different relation in graph"len(relation))

print("all done")


