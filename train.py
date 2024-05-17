from transliteration import wandb_run_configuration,util_preprocess, pre_process,word2index, get_data, MyDataset

import argparse
import wandb 
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import requests,zipfile,io
import pandas as pd


wandb.login(key="67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e")
parser = argparse.ArgumentParser(description='Run my model function')
parser.add_argument('-wp','--wandb_project', default="Assignment3", required=False,metavar="", type=str, help=' ')
parser.add_argument('-we','--wandb_entity', default="cs23m055", required=False,metavar="", type=str, help='')
parser.add_argument('-ds','--dataset', default="aksharantar", required=False,metavar="", type=str, help=' ')
parser.add_argument('-e','--epochs', default=25, required=False,metavar="", type=int, help=' ')
parser.add_argument('-bs','--batchsize', default=256, required=False,metavar="", type=int, help=' ')
parser.add_argument('-hs','--hidden_size', default=1024, required=False,metavar="", type=int, help=' ')
parser.add_argument('-el','--encoder_layers', default=3, required=False,metavar="", type=int, help=' ')
parser.add_argument('-dl','--decoder_layers', default=3, required=False,metavar="", type=int, help=' ')
parser.add_argument('-es','--embedding_size', default=256, required=False,metavar="", type=int, help=' ')
parser.add_argument('-do','--dropout', default=0.3, required=False,metavar="", type=float, help=' ')
parser.add_argument('-ct','--cell_type', default="LSTM", required=False,metavar="", type=str,choices= ["GRU", "LSTM", "RNN"], help=' ')
parser.add_argument('-d','--bi_directional', default="Yes", required=False,metavar="", type=str,choices= ["Yes", "No"], help=' ')
parser.add_argument('-a','--attention', default="Yes", required=False,metavar="", type=str,choices= ["Yes"], help=' ')

args = parser.parse_args()

train_df,test_df,val_df,eng_to_idx,hin_to_idx,idx_to_eng,idx_to_hin,input_len,target_len=get_data()

val_x,val_y = pre_process(val_df,eng_to_idx,hin_to_idx)
test_x,test_y = pre_process(test_df,eng_to_idx,hin_to_idx)
train_x,train_y = pre_process(train_df,eng_to_idx,hin_to_idx)



train_dataset=MyDataset(train_x,train_y)
test_dataset=MyDataset(test_x,test_y)
val_dataset=MyDataset(val_x,val_y)
wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,input_len,target_len, args.epochs,args.encoder_layers,args.decoder_layers,args.batchsize,args.embedding_size,args.hidden_size,args.bi_directional,args.dropout,args.cell_type,args.attention)
# wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,args.epochs,args.encoder_layers,args.decoder_layers,args.batchsize,args.embedding_size,args.hidden_size,args.bi_directional,args.dropout,args.cell_type,args.attention)
