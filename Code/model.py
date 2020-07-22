#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:04:43 2020

@author: G.Mou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pickle
import time
import math, copy

from mask import mask_softmax, mask_mean, mask_max

def save_model(model, model_path, grid):
    """Save model."""
    torch.save(model.state_dict(), model_path)
    # torch.save(model.module.state_dict(), model_path) # this is for multi gpu, not used here
    with open("hyper.pkl",'wb') as f:
        pickle.dump(grid,f)
    #print("checkpoint saved") # only uncommenting for debugging
    return

def load_model(model, model_path):
    """Load model."""
    map_location = 'cpu'
    if torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

def get_model_setting(**args):
    """Load Model Settings"""
    model = JntL(**args)
    return model

class Unit(nn.Module):
    """Basic Unit Conv Operations"""
    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        kernel_size=[3,4,5],
        stride=1,
        padding=0,
        ):

        super(Unit,self).__init__()
        self.unit_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
          )

    def forward(self,myInput):
        output = self.unit_cnn(myInput)
        return output

class GAFMTF(nn.Module):
    """docstring for GAFMTF"""
    def __init__(self):
        super(GAFMTF, self).__init__()

        # cnn for GAFMTF
        # input [Batch, 3, 50, 50]
        self.network1 = nn.Sequential(
            # [3, 50, 50] -> [32, 50, 50]
            Unit(
                in_channels=3,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1),
                ),
            # [32, 50, 50] -> [64, 50, 50]
            Unit(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)),
            # [64, 50, 50] -> [64, 25, 25]
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2),
                ),
            # [64, 25, 25] -> [128, 25, 25]
            Unit(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)),
            # [128, 25, 25] -> [128, 12, 12]
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
                ),
            # [128, 12, 12] -> [256, 12, 12]
            Unit(
                in_channels=128,
                out_channels=256,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)),
            # [256, 12, 12] -> [256, 6, 6]
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
                ),
            )

        self.attn = AdditiveAttention(256, 36)

    def forward(self, x):
        x = self.network1(x)
        B,D,W,H = x.size()
        x = x.view(B,D,-1)
        # feed in [B,W*H,D] [B,36,256]
        # output [B, H] [B, 256]
        x = x.permute(0,2,1)
        return self.attn(x)

class IPT(nn.Module):
    """docstring for IPT"""
    def __init__(self):
        super(IPT, self).__init__()

        # cnn for IPT
        # input [Batch, 3, 32, 32]
        self.network2 = nn.Sequential(
            # [3, 32, 32] -> [32, 32, 32]
            Unit(
                in_channels=3,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1),
                ),

            # [32, 32, 32] -> [64, 32, 32]
            Unit(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1),
                ),
            # [64, 32, 32] -> [64, 16, 16]
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2),
                ),
            # [64, 16, 16] -> [128, 16, 16]
            Unit(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)),
            # [128, 16, 16] -> [128, 8, 8]
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
                ),
            # [128, 8, 8] -> [256, 8, 8]
            Unit(
                in_channels=128,
                out_channels=256,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)),
            # [256, 8, 8] -> [256, 4, 4]
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
                ),
            )

        self.attn = AdditiveAttention(256, 16)

    def forward(self, x):
        x = self.network2(x)
        B,D,W,H = x.size()
        x = x.view(B,D,-1)
        # feed in [B,W*H,D] [B,16,256]
        # output [B, H] [B, 256]
        x = x.permute(0,2,1)
        return self.attn(x)

class TextCNN(nn.Module):
    """docstring for TextCNN"""
    def __init__(self, dim=768, out=256):
        super(TextCNN, self).__init__()

        self.Ks = [3,4,5]
        self.topk = 5

        # cnn for word embedding
        # [1, H, dim] - >[256, H+K-1, 1]
        self.convs = nn.ModuleList([
                        Unit(
                            in_channels=1,
                            out_channels=out,
                            kernel_size=(K,dim),
                            stride=(1,1),
                            padding=(K-1,0),
                            )
                        for K in self.Ks])

        self.attn = AdditiveAttention(out, len(self.Ks)*self.topk)

    def forward(self, x):
        # [B, H, D] -> [B, 1, H, D]
        x = x.unsqueeze(1)
        # [(B, D, H+K-1), ...]*len(Ks) after convolution, w=1
        x = [conv(x).squeeze(-1) for conv in self.convs]
        # [B, D, topk*len(Ks)]  
        x = torch.cat([item.topk(self.topk, dim=-1)[0] for item in x], dim=-1) 
        # feed in [B, topk*len(Ks), D] [B, 15, 256]
        # output [B, H] [B, 256]
        x = x.permute(0,2,1)
        return self.attn(x)

class TextLSTM(nn.Module):
    """docstring for TextLSTM"""
    def __init__(self, lstm_dropout, seq_len, hidden_dim, wordDim=768):
        super(TextLSTM, self).__init__()

        # [B, H, D]
        self.LSTM = nn.LSTM(
                        input_size=wordDim,
                        hidden_size=hidden_dim,
                        num_layers=2,
                        bias=True,
                        batch_first=True,
                        dropout=lstm_dropout,
                        bidirectional=True,
                        )

        self.attn = AdditiveAttention(hidden_dim*2, seq_len)

    def forward(self, x):
        # self.LSTM.flatten_parameters()
        lstm_out , _ = self.LSTM(x)
        return self.attn(lstm_out)

class SentEmb(nn.Module):
    """docstring for SentEmb"""
    def __init__(self, hidden_dim, length):
        super(SentEmb, self).__init__()
        self.attn = AdditiveAttention(hidden_dim, length)

    def forward(self, x):
        return self.attn(x)
        
class JntL(nn.Module):
    """docstring for JntL"""
    def __init__(self, **args):
        super(JntL, self).__init__()

        # input size
        
        # fixed
        self.iptDim = args["iptDim"]
        self.gafmtfDim = args["gafmtfDim"]

        self.wordDim = args["wordDim"]
        self.wordLen = args["wordLen"]

        self.sentDim = args["sentDim"]
        self.sentLen = args["sentLen"]
        
        self.tradDim = args["tradDim"]

        # output size, hypers to be tuned
        self.tbHeads = args["tbHeads"]
        self.textHeads = args["textHeads"]

        self.TotalHeads = self.tbHeads + self.textHeads + self.tradDim

        # size & dropouts
        self.layer1_size = args["layer1_size"]
        self.layer1_dropout = args["layer1_dropout"]
        self.layer2_size = args["layer2_size"]
        self.layer2_dropout = args["layer2_dropout"]
        self.output_size = args["output_size"]

        # lstm
        self.lstm_dropout = args["lstm_dropout"]
        self.lstm_out = args["lstm_out"]
        self.cnn_out = args["cnn_out"]

        self.GAFMTF = GAFMTF()
        self.IPT = IPT()
        self.TextCNN = TextCNN(self.wordDim, self.cnn_out)
        self.TextLSTM = TextLSTM(self.lstm_dropout, self.wordLen, self.lstm_out, self.wordDim)
        self.SentEmb = SentEmb(self.sentDim, self.sentLen)

        # decision making
        self.DecisionMaking = nn.Sequential(
            nn.Dropout(self.layer1_dropout),
            nn.Linear(self.TotalHeads, self.layer1_size),
            nn.LeakyReLU(),
            nn.Dropout(self.layer2_dropout),
            nn.Linear(self.layer1_size, self.layer2_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer2_size, self.output_size),
            #nn.Sigmoid(),
            )
        self.act = nn.LeakyReLU()
        # linear projections
        self.textLinear = nn.Linear((self.cnn_out+self.lstm_out*2+self.sentDim)*3, self.textHeads)
        self.tbLinear = nn.Linear((self.iptDim+self.gafmtfDim)*3, self.tbHeads)

    def forward(self, x):
        # SentEmb, WordEmb, TradEmb, GAFMTF, IPT
        cnn_GAFMTF_out = self.GAFMTF(x[3])
        cnn_IPT_out = self.IPT(x[4])
        cnn_Word_out = self.TextCNN(x[1])
        lstm_Word_out = self.TextLSTM(x[1])
        sent_out = self.SentEmb(x[0])

        temporalBehavior = torch.cat([cnn_GAFMTF_out, cnn_IPT_out], dim=-1)
        temporalBehavior = self.act(self.tbLinear(temporalBehavior))

        text = torch.cat([cnn_Word_out, lstm_Word_out, sent_out], dim=-1)
        text = self.act(self.textLinear(text))

        result = torch.cat([temporalBehavior, text, x[2]], dim=-1)

        result = self.DecisionMaking(result)

        return result

class AdditiveAttention(nn.Module):
    
    def __init__(self, hidden_dim, length):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # attention
        self.W = nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.Tensor(hidden_dim, 1)) # context vector
        self.softmax = nn.Softmax(dim=2)

        self.linear = nn.Linear(length, 1)
        # initialization
        nn.init.normal_(self.v, 0, 0.1)
     
    def forward(self, values):
        # values: [B,T,H] (outputs to align)
        
        B,T,H = values.size()
        
        # compute energy
        query = self.linear(values.permute(0,2,1)).squeeze(-1) # [B,T,H] -> [B,H]
        query = query.unsqueeze(1).repeat(1,T,1) # [B,H] -> [B,T,H]
        feats = torch.cat((query, values), dim=2) # [B,T,H*2]
        energy = self.tanh(self.W(feats)) # [B,T,H*2] -> [B,T,H]
        
        # compute attention scores
        v = self.v.t().repeat(B,1,1) # [H,1] -> [B,1,H]
        energy = energy.permute(0,2,1) # [B,T,H] -> [B,H,T]
        scores = torch.bmm(v, energy) # [B,1,H]*[B,H,T] -> [B,1,T]
        
        # weight values
        combo = torch.bmm(scores, values).squeeze(1) # [B,1,T]*[B,T,H] -> [B,H]

        value_mean = mask_mean(values)
        value_max = mask_max(values)
        return torch.cat([combo, value_mean, value_max], dim=-1)
