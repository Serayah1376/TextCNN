# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:25:37 2021

@author: 10983
"""

import torch
import torch.nn.functional as F


class TextCNN(torch.nn.Module):
    def __init__(self,dropout,vocab_size,embed_dim,class_num,filter_num,filter_sizes):
        super(TextCNN,self).__init__()
        self.class_num=class_num  #要分类的种类数
        self.filter_num=filter_num  #卷积核数目
        self.filter_sizes=filter_sizes  #每个卷积核的大小
        self.chanel_num=1   #通道数设置为1
        
        self.embedding=torch.nn.Embedding(vocab_size,embed_dim)
        
        self.convs=torch.nn.ModuleList(
            [torch.nn.Conv2d(self.chanel_num,self.filter_num,(size,embed_dim)) for size in self.filter_sizes])
        self.dropout=torch.nn.Dropout(dropout)
        self.fc=torch.nn.Linear(len(self.filter_sizes)*self.filter_num,self.class_num)
        
        
    #seq_lengths为该样本长度
    def forward(self,x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
        
