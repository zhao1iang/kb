import time
import sys
import os
import torch
import linecache
import yaml
import pickle
import copy
# __all__=['Config','']
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time

class Config():
    def __init__(self,configFile):
        self.configFile=configFile
        with open(self.configFile,'r') as f:
            config=yaml.load(f)
            print("getting hypeparameters:\n",config)
            if config:
                for k,v in config.items():
                    setattr(self,k,v)

    def has(self,name):
        return hasattr(self,name)

    def add(self,name=None,value=None):
        if not hasattr(self,name):
            setattr(self, name, value)
            with open(self.configFile,'a') as f:
                f.write(str(name)+": "+str(value)+'\n')
        else:
            print('\'{}\' already exists in \'config\' , its values is {} , maybe you just want to change its value?'.format(name,getattr(self,name)))
        

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    current = current % total
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
def print_log(file):
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)
    return write_log

class Dataset(torch.utils.data.Dataset):

    def __init__(self, filepath=None,dataLen=None):
        self.file = filepath
        self.dataLen = dataLen
        
    def __getitem__(self, index):
        A,B,path,hop= linecache.getline(self.file, index+1).split('\t')
        return A,B,path.split(' '),int(hop)

    def __len__(self):
        return self.dataLen

class Voc():
    def __init__(self,config):
        self.word2index={}
        self.word2count={}
        self.index2word={}
        self.hasTrimmed=False
        self.index=0
        self.nodeList=[]
        self.config=config
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.index
            self.word2count[word]=1
            self.index2word[self.index]=word
            self.index+=1
        else:
            self.word2count[word]+=1
    def trim(self):
        if self.hasTrimmed:
            return 
        self.hasTrimmed = True
        trimW2I={}
        trimW2C={}
        trimI2W={}
        if hasattr(self.config,'wordNum'):
            freq = torch.tensor(
                    [self.word2count[self.index2word[i]] for i in range(len(self.word2count))])
            _, idx = torch.sort(freq, 0, True)
            idx = idx.tolist()
            for trimIndex,i in enumerate(idx[:self.config.wordNum]):
                trimW2I[self.index2word[i]]=trimIndex
                trimW2C[self.index2word[i]]=self.word2count[self.index2word[i]]
                trimI2W[trimIndex]=self.index2word[i]

        elif hasattr(self.config,'wordFreq'):
            trimIndex=0
            for i in range(self.index):
                if self.word2count[self.index2word[i]]>=self.config.wordFreq:
                    trimW2I[self.index2word[i]]=trimIndex
                    trimW2C[self.index2word[i]]=self.word2count[self.index2word[i]]
                    trimI2W[trimIndex]=self.index2word[i]
                    trimIndex+=1
        else :
            raise Exception(" trim argument is need!")
        self.word2count=trimW2C
        self.word2index=trimW2I
        self.index2word=trimI2W
    def getNodeList(self):
        return list(self.word2index)

def dump_pickle(fileName=None,pickleFile=None,ROOTPATH=None):
    if ROOTPATH:
        if not os.path.exists(os.path.join(ROOTPATH,fileName)):
            os.mknod(os.path.join(ROOTPATH,fileName))
        with open(os.path.join(ROOTPATH,fileName),'wb') as f:
            pickle.dump(pickleFile,f)
    else:
        if not os.path.exists(fileName):
            os.mknod(os.path.join(fileName))
        with open(fileName,'wb') as f:
            pickle.dump(pickleFile,f)
