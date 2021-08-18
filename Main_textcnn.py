# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:24:03 2021

@author: 10983
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from Preparing_Data_textcnn import CodeDataset
import model_textcnn
from train_textcnn import Train_Valid


torch.cuda.manual_seed_all(123)#设置随机数种子，使每一次初始化数值都相同

#parameters
embed_dim=128
BATCH_SIZE=64 #改为8试一下
N_LAYER=2  ###########
N_EPOCHS=20
USE_GPU=True #使用GPU
#gamma=0.1
max_len=229   #代码最大长度（95%分词后的代码低于此长度）
dropout=0.5
filter_num=100
filter_sizes=[3,4,5]




trainset=CodeDataset(is_train_set=True)
validset =CodeDataset(is_train_set=False)
testset=CodeDataset(is_train_set=False,is_test_set=True)

#将分词后的codes进行padding：取每个batch的列表中的最大长度作为填充目标长度
#试一试如果长度长截取后面部分
def make_tensors(batch):
    codes=[item[0] for item in batch]
    labels=[item[1] for item in batch]
    list2=[]
    for s in codes:
        feature=[]
        #如果长度大于最大长度，则截去前面部分
        if len(s)>max_len:
            latter_begin_index=len(s)-max_len
            n=latter_begin_index
            while n<len(s):
                word=s[n]
                if word in trainset.word2index:
                    feature.append(trainset.word2index[word])
                else:
                    feature.append(trainset.word2index["<unk>"])
                n+=1
        else:
            for word in s:
                if word in trainset.word2index:
                    feature.append(trainset.word2index[word])
                else:
                    feature.append(trainset.word2index["<unk>"])
            #padding:填充1使长度相等
            need_add=max_len-len(feature)
            feature=feature + [trainset.word2index["<pad>"]] * need_add
        list2.append(feature)
    #将代码向量转化为张量
    seq_tensor=torch.LongTensor(list2)
    labels=torch.LongTensor(labels)     
    return trainset.create_tensor(seq_tensor),\
          trainset.create_tensor(labels)
   
#数据的准备
#训练集
train_loader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=make_tensors)
#验证集
valid_loader=DataLoader(validset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=make_tensors)
#测试集
test_loader=DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=make_tensors)

#得到标签的种类数，决定模型最终输出维度的大小
#训练集中的类别是最全的，故使用训练集的label_num
N_LABEL=trainset.getLabelNum() 
#词典中词的个数，在使用词嵌入Embedding的时候传入的参数
N_WORDS_train=trainset.dicnum


'''
N_CHARS：整个字母表元素个数
HIDDEN_SIZE：隐层维度
N_LABEL：有多少个分类
N_LAYER：用几层
'''
classifier=model_textcnn.TextCNN(dropout,N_WORDS_train,embed_dim,N_LABEL,filter_num,filter_sizes)
#是否用GPU
if USE_GPU:
    device=torch.device('cuda:0')
    classifier.to(device)
    
#看训练的时间有多长
start=time.time()
print('Training for %d epochs(TextCNN)...' % N_EPOCHS)


train_valid=Train_Valid(classifier, train_loader, valid_loader,test_loader, start, trainset, validset,testset)
valid_acc_list,test_acc_list,loss_list=train_valid.train()

#绘图部分看准确率的变化
epoch=np.arange(1,len(valid_acc_list)+1,1)
valid_acc_list=np.array(valid_acc_list)
plt.plot(epoch,valid_acc_list)
plt.xlabel('Epoch')
plt.ylabel('Valid_Accuracy')
plt.grid()
plt.show()

#绘图部分看准确率的变化
epoch=np.arange(1,len(test_acc_list)+1,1)
test_acc_list=np.array(test_acc_list)
plt.plot(epoch,test_acc_list)
plt.xlabel('Epoch')
plt.ylabel('Test_Accuracy')
plt.grid()
plt.show()

loss_list=np.array(loss_list)
plt.plot(epoch,loss_list)
plt.xlabel('Epoch')
plt.ylabel('Train_loss')
plt.grid()
plt.show()