#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:04:43 2020

@author: G. Mou
"""

# pytorch
from torch.utils.data import Dataset
import torch

import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler

class myDataset(Dataset):
    ''' dataset reader
    '''
    
    def __init__(self, X, y): # SentEmb, WordEmb, TradEmb, GAFMTF, IPT, userID
        self.X, self.y = X, y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        
        SentEmb = torch.tensor(self.X["SentEmb"][index], dtype=torch.float)
        WordEmb = torch.tensor(self.X["WordEmb"][index], dtype=torch.float)
        TradEmb = torch.tensor(self.X["TradEmb"][index], dtype=torch.float)
        GAFMTF = torch.tensor(self.X["GAFMTF"][index], dtype=torch.float)
        IPT = torch.tensor(self.X["IPT"][index], dtype=torch.float)
        
        y = torch.tensor(self.y[index],dtype=torch.float)
        return SentEmb, WordEmb, TradEmb, GAFMTF, IPT, y

def loadMatrix(name:str, datatype:str, path="../Data/", end=".npy"):
    return np.load("{}{}_{}{}".format(path, name, datatype, end))

def loadCSV(name:str, datatype:str, path="../Data/", end=".csv"):
    df = pd.read_csv("{}{}_{}{}".format(path, name, datatype, end))
    return df.to_numpy()

def readData(datatype:str):
    ''' method for reading different parts of data
    '''
    # text embedding
    SentEmb = loadMatrix("SentEmb", datatype)
    WordEmb = loadMatrix("WordEmb", datatype)
    # temporal
    GAFMTF = loadMatrix("GAFMTF", datatype)
    IPT = loadMatrix("IPT", datatype)

    label = loadMatrix("label", datatype)
    # hand crafted features
    TradEmb = loadMatrix("traditional_fav", datatype)

    # put them together
    X = {
        "SentEmb":  SentEmb,
        "WordEmb":  WordEmb,
        "TradEmb":  TradEmb,
        "GAFMTF":   GAFMTF,
        "IPT":      IPT,
        # "userID":   userID,
    }

    # correct/target/actual labels, just to be sure
    y = label.astype(float).reshape(-1,1)

    return X, y
