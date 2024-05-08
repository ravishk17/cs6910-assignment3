
# ## IMPORTING LIBRARIES

import requests,zipfile,io
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import numpy as np
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


print(device)


# ## DOWNLOADING AND UNZIPPING DATA


def download_data(url="https://drive.google.com/u/0/uc?id=1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw&export=download"):
    response=requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall()


# ## METHODS FOR GETTING CHARACTERS FOR CORPUSS AND ADDING THEIR INDICES


def get_corpus(data):
    eng_corpus=set()
    hin_corpus=set()
    for i in range(0,len(data)):
        eng_word=data[0][i]
        hin_word=data[1][i]
        for ch in eng_word:
            eng_corpus.add(ch)
        for ch in hin_word:
            hin_corpus.add(ch)
        # End Delimiter
        eng_corpus.add('#')
        hin_corpus.add('#')
        hin_corpus.add('$')
        eng_corpus.add('$')
        # Start Delimiter
#         eng_corpus.add('^')
        hin_corpus.add('^')
    return hin_corpus,eng_corpus


def word2index(data):
    hin_corpus,eng_corpus=get_corpus(data)
    engchar_idx={}
    hinchar_idx={}
    idx_engchar={}
    idx_hinchar={}
    i=0
    for char in eng_corpus:
        engchar_idx[char]=i
        idx_engchar[i]=char
        i+=1
    i=0
    for char in hin_corpus:
        hinchar_idx[char]=i
        idx_hinchar[i]=char
        i+=1
    return engchar_idx,hinchar_idx,idx_engchar,idx_hinchar,len(eng_corpus),len(hin_corpus)


# ## DATA PREPROCESSING


def maxlen(data):
    maxlen_eng=0
    maxlen_hin=0
    for i in range(0,len(data)):
        eng_word=data[0][i]
        hin_word=data[1][i]
        if(len(eng_word)>maxlen_eng):
            maxlen_eng=len(eng_word)
        if(len(hin_word)>maxlen_hin):
            maxlen_hin=len(hin_word)
    return maxlen_eng,maxlen_hin


def pre_process(data,eng_to_idx,hin_to_idx):
    eng=[]
    hin=[]
    maxlen_eng,maxlen_hin=maxlen(data)
    
    unknown= eng_to_idx['$']
    for i in range(0,len(data)):
        sz=0
        eng_word=data[0][i]
        hin_word='^'+data[1][i]
        eng_word = eng_word.ljust(maxlen_eng+1, '#')
        hin_word = hin_word.ljust(maxlen_hin+1, '#')
        idx=[]
        for char in eng_word:
            if eng_to_idx.get(char) is not None:
                idx.append(eng_to_idx[char])
            else:
                idx.append(unknown)
        eng.append(idx)
        idx=[]
        for char in hin_word:
            if hin_to_idx.get(char) is not None:
                idx.append(hin_to_idx[char])
            else:
                idx.append(unknown)
        hin.append(idx)    
    return eng,hin


# ## LOADING OUR CUSTOM DATASET TO DATALOADER


class MyDataset(Dataset):
    def __init__(self, train_x,train_y, transform=None):
        self.train_x = train_x
        self.train_y = train_y
        self.transform = transform
        
    
    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(self.train_x[idx]).to(device),torch.tensor(self.train_y[idx]).to(device)


def get_data():
    download_data()
    
    train_df=pd.read_csv("aksharantar_sampled/hin/hin_train.csv",header=None)
    test_df=pd.read_csv("aksharantar_sampled/hin/hin_test.csv",header=None)
    val_df=pd.read_csv("aksharantar_sampled/hin/hin_valid.csv",header=None)
    eng_to_idx,hin_to_idx,idx_to_eng,idx_to_hin,input_len,target_len=word2index(train_df)
    
    return train_df,test_df,val_df,eng_to_idx,hin_to_idx,idx_to_eng,idx_to_hin,input_len,target_len


train_df,test_df,val_df,eng_to_idx,hin_to_idx,idx_to_eng,idx_to_hin,input_len,target_len=get_data()

train_x,train_y = pre_process(train_df,eng_to_idx,hin_to_idx)
test_x,test_y = pre_process(test_df,eng_to_idx,hin_to_idx)
val_x,val_y = pre_process(val_df,eng_to_idx,hin_to_idx)

train_dataset=MyDataset(train_x,train_y)
test_dataset=MyDataset(test_x,test_y)
val_dataset=MyDataset(val_x,val_y)

# ## Seq2Seq MODEL


class EncoderGRU(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_of_layers,batch_size,bi_directional,dropout_p=0.1):
        super(EncoderGRU,self).__init__()
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.num_of_layers=num_of_layers
        self.bi_directional=bi_directional
        if(bi_directional=="Yes"):
            flag=True
        else:
            flag=False
        self.gru = nn.GRU(embedding_size,hidden_size,num_of_layers,bidirectional=flag)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,input,hidden):
        embedded=self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        embedded = self.dropout(embedded)
        output,hidden=self.gru(embedded,hidden)
    
        if self.bi_directional=="Yes":
            hidden=hidden.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            hidden=torch.add(hidden[0],hidden[1])/2
            
        return output,hidden

    def initHidden(self):
        if(self.bi_directional=="Yes"):
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)

class DecoderGRU(nn.Module):
    def __init__(self, output_size,hidden_size, embedding_size, decoder_layers,batch_size,dropout_p=0.1):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size,hidden_size, decoder_layers,dropout = dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.batch_size=batch_size
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
#         embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden


class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_of_layers,batch_size,bi_directional,dropout_p=0.1):
        super(EncoderRNN,self).__init__()
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.num_of_layers=num_of_layers
        self.bi_directional=bi_directional
        if(bi_directional=="Yes"):
            flag=True
        else:
            flag=False
        self.rnn = nn.RNN(embedding_size,hidden_size,num_of_layers,bidirectional=flag)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,input,hidden):
        embedded=self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        embedded = self.dropout(embedded)
        output,hidden=self.rnn(embedded,hidden)
    
        if self.bi_directional=="Yes":
            hidden=hidden.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            hidden=torch.add(hidden[0],hidden[1])/2
            
        return output,hidden

    def initHidden(self):
        if(self.bi_directional=="Yes"):
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)

class DecoderRNN(nn.Module):
    def __init__(self, output_size,hidden_size, embedding_size, decoder_layers,batch_size,dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.RNN(embedding_size,hidden_size, decoder_layers,dropout = dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.batch_size=batch_size
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
#         embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden


class EncoderLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_of_layers,batch_size,bi_directional,dropout_p=0.1):
        super(EncoderLSTM,self).__init__()
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.num_of_layers=num_of_layers
        self.bi_directional=bi_directional
        if(bi_directional=="Yes"):
            flag=True
        else:
            flag=False
        self.lstm = nn.LSTM(embedding_size,hidden_size,num_of_layers,bidirectional=flag)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,input,hidden,state):
        embedded=self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        embedded = self.dropout(embedded)
        output,(hidden,state)=self.lstm(embedded,(hidden,state))
    
        if self.bi_directional=="Yes":
            hidden=hidden.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            state=state.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            hidden=torch.add(hidden[0],hidden[1])/2
            state=torch.add(state[0],hidden[1])/2
            
        return output,hidden,state

    def initHidden(self):
        if(self.bi_directional=="Yes"):
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)
    
    def initState(self):
        if(self.bi_directional=="Yes"):
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)

class DecoderLSTM(nn.Module):
    def __init__(self, output_size,hidden_size, embedding_size, decoder_layers,batch_size,dropout_p=0.1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,decoder_layers,dropout = dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.batch_size=batch_size
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input,hidden,state):
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
#         embedded = self.dropout(embedded)
        output,(hidden,state)=self.lstm(embedded,(hidden,state))
        output = self.softmax(self.out(output))
        return output,hidden,state


# ## ATTENTION MECHANISM


class AttnDecoder(nn.Module):
    def __init__(self,output_size,hidden_size,embedding_size,decoder_layers,batch_size,cell_type,dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.batch_size=batch_size
        self.cell_type=cell_type
        self.embedding_size=embedding_size
        self.decoder_layers=decoder_layers
        
        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.U=nn.Linear(self.hidden_size,self.hidden_size,bias=False).to(device)
        self.W=nn.Linear(self.hidden_size,self.hidden_size,bias=False).to(device)
        self.V=nn.Linear(self.hidden_size,1,bias=False).to(device)
        
        self.linear=nn.Linear(self.hidden_size,output_size,bias=True)
        self.softmax=nn.Softmax(dim=1)
        self.softmax1=nn.LogSoftmax(dim=2)
        
        if(cell_type=="GRU"):
            self.gru = nn.GRU(self.embedding_size+self.hidden_size, self.hidden_size,self.decoder_layers,dropout = dropout_p)
        if(cell_type=="LSTM"):
            self.lstm = nn.LSTM(self.embedding_size+self.hidden_size, self.hidden_size,self.decoder_layers,dropout = dropout_p)
        if(cell_type=="RNN"):
            self.rnn = nn.RNN(self.embedding_size+self.hidden_size, self.hidden_size,self.decoder_layers,dropout = dropout_p)

    def forward(self, input, hidden,encoder_outputs,word_length,state=None):
        embedded = self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        T=word_length
        temp1=self.W(hidden[-1])
        temp2=self.U(encoder_outputs)
        c=torch.zeros(self.batch_size,1,self.hidden_size).to(device)
        temp1=temp1.unsqueeze(0)

        e_j=self.V(F.tanh(temp1+temp2))
        alpha_j=self.softmax(e_j)
        
        c = torch.bmm(alpha_j.permute(1,2,0),encoder_outputs.permute(1,0,2))
        
        final_input=torch.cat((embedded[0],c.squeeze(1)),1).unsqueeze(0)
    
        final_input = F.relu(final_input)
        
        if(self.cell_type=="GRU"):
            output,hidden=self.gru(final_input,hidden)
        if(self.cell_type=="RNN"):
            output,hidden=self.rnn(final_input,hidden)
        if(self.cell_type=="LSTM"):
            output, (hidden,state) =self.lstm(final_input,(hidden,state))
        
        
        output1=self.softmax1(self.linear(output))
        if(self.cell_type=="GRU" or self.cell_type=="RNN"):
            return output1, hidden, alpha_j
        if(self.cell_type=="LSTM"):
            return output1, hidden, state, alpha_j


def train(train_data,encoder,decoder,loss_fun,encoder_optimizer,decoder_optimizer,encoder_layers,decoder_layers,batch_size,hidden_size,bi_directional,cell_type,attention):
    total_loss=0
    teacher_forcing_ratio=0.5
    for i,(train_x,train_y) in enumerate(train_data):
        loss=0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        train_x=train_x.T
        train_y=train_y.T
        timesteps=len(train_x)
        
        if cell_type=='GRU' or cell_type=='RNN':
            
            encoder_hidden=encoder.initHidden()
            encoder_output,encoder_hidden=encoder(train_x,encoder_hidden)
            if(decoder_layers>encoder_layers):
                i = decoder_layers
                decoder_hidden=encoder_hidden

                while True:
                    if(i==encoder_layers):
                        break
                    # Concatenate the two tensors along the first dimension
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    i-=1

            elif(decoder_layers<encoder_layers):
                decoder_hidden=encoder_hidden[-decoder_layers:]

            else:
                decoder_hidden=encoder_hidden
        
            decoder_input = train_y[0]
            
            if(bi_directional=="Yes"):
                split_tensor= torch.split(encoder_output, hidden_size, dim=-1)
                encoder_output=torch.add(split_tensor[0],split_tensor[1])/2
            
            
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                for i in range(0,len(train_y)):
                    if(attention=="Yes"):
                        decoder_output, decoder_hidden, attn_weights=decoder(decoder_input,decoder_hidden,encoder_output,len(train_x))
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input = train_y[i] 
                    else:
                        decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input = train_y[i]  # Teacher forcing
            else:
                for i in range(0,len(train_y)):
                    if(attention=="Yes"):
                        decoder_output, decoder_hidden, attn_weights=decoder(decoder_input,decoder_hidden,encoder_output,len(train_x))
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input=index
                    else:
                        decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input=index
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss+=loss
        
        if cell_type=='LSTM':
    
            encoder_hidden=encoder.initHidden()
            encoder_state=encoder.initState()
            
            encoder_output,encoder_hidden,encoder_state=encoder(train_x,encoder_hidden,encoder_state)
        
            if(decoder_layers>encoder_layers):
                i = decoder_layers
                decoder_hidden=encoder_hidden
                decoder_state=encoder_state
                while True:
                    if(i==encoder_layers):
                        break
                    # Concatenate the two tensors along the first dimension
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    decoder_state = torch.cat([decoder_state, encoder_state[-1].unsqueeze(0)], dim=0)
                    i-=1

            elif(decoder_layers<encoder_layers):
                decoder_hidden=encoder_hidden[-decoder_layers:]
                decoder_state=encoder_state[-decoder_layers:]

            else:
                decoder_hidden=encoder_hidden
                decoder_state=encoder_state
            
            
            if(bi_directional=="Yes"):
                split_tensor= torch.split(encoder_output, hidden_size, dim=-1)
                encoder_output=torch.add(split_tensor[0],split_tensor[1])/2
            
            decoder_input = train_y[0]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                for i in range(0,len(train_y)):
                    if(attention=="Yes"):
                        decoder_output, decoder_hidden, decoder_state, attn_weights=decoder(decoder_input,decoder_hidden,encoder_output,len(train_x),decoder_state)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input= train_y[i]
                    else:
                        decoder_output, decoder_hidden,decoder_state= decoder(decoder_input, decoder_hidden,decoder_state)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input = train_y[i]  # Teacher forcing
            else:
                for i in range(0,len(train_y)):
                    if(attention=="Yes"):
                        decoder_output, decoder_hidden, decoder_state, attn_weights=decoder(decoder_input,decoder_hidden,encoder_output,len(train_x),decoder_state)
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input=index
                    else:
                        decoder_output, decoder_hidden,decoder_state= decoder(decoder_input, decoder_hidden,decoder_state)
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), train_y[i])
                        decoder_input=index
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss+=loss

        
        
    return total_loss.item()/len(train_y),encoder,decoder


def train_iter(input_data,val_data,val_y,input_len,target_len,epochs,batch_size,embedding_size,encoder_layers,decoder_layers,hidden_size,cell_type,bi_directional,dropout,attention,beam_size=0):
    lr=0.001
    if(cell_type=='GRU'):
        encoder=EncoderGRU(input_len,hidden_size,embedding_size,encoder_layers,batch_size,bi_directional,dropout).to(device)
        if(attention=="Yes"):
            decoder=AttnDecoder(target_len,hidden_size,embedding_size,decoder_layers,batch_size,cell_type,dropout).to(device)
        else:
            decoder=DecoderGRU(target_len,hidden_size,embedding_size,decoder_layers,batch_size,dropout).to(device)
        
    if(cell_type=='RNN'):
        encoder=EncoderRNN(input_len,hidden_size,embedding_size,encoder_layers,batch_size,bi_directional,dropout).to(device)
        if(attention=="Yes"):
            decoder=AttnDecoder(target_len,hidden_size,embedding_size,decoder_layers,batch_size,cell_type,dropout).to(device)
        else:
            decoder=DecoderRNN(target_len,hidden_size,embedding_size,decoder_layers,batch_size,dropout).to(device)
    
    if cell_type=='LSTM':
        encoder=EncoderLSTM(input_len,hidden_size,embedding_size,encoder_layers,batch_size,bi_directional,dropout).to(device)
        if(attention=="Yes"):
            decoder=AttnDecoder(target_len,hidden_size,embedding_size,decoder_layers,batch_size,cell_type,dropout).to(device)
        else:
            decoder=DecoderLSTM(target_len,hidden_size,embedding_size,decoder_layers,batch_size,dropout).to(device)

    encoder_optimizer=optim.Adam(encoder.parameters(),lr)
    decoder_optimizer=optim.Adam(decoder.parameters(),lr)
    loss_fun=nn.CrossEntropyLoss(reduction="sum")
    epoch_train_loss=[]
    epoch_val_loss=[]
    epoch_val_acc=[]
    for i in range(0,epochs):
        loss,encoder,decoder=train(input_data,encoder,decoder,loss_fun,encoder_optimizer,decoder_optimizer,
                                   encoder_layers,decoder_layers,batch_size,hidden_size,bi_directional,
                                   cell_type,attention)
        val_predictions,val_loss,attn_weights=eval(val_data,encoder,decoder,encoder_layers,decoder_layers,
                                  batch_size,hidden_size,bi_directional,cell_type,attention)
        
        epoch_val_loss.append(val_loss)
        epoch_train_loss.append(loss/51200)
        
        val_acc=accuracy(val_predictions,val_y)
        epoch_val_acc.append(val_acc)
        print(loss/51200,val_loss,val_acc)
    
#     train_predictions,t=eval(input_data,encoder,decoder,encoder_layers,decoder_layers,batch_size,hidden_size,bi_directional,cell_type,attention)
    return epoch_train_loss,epoch_val_loss,epoch_val_acc,encoder,decoder,encoder_layers,decoder_layers


def eval(input_data,encoder,decoder,encoder_layers,decoder_layers,batch_size,hidden_size,bi_directional,cell_type,attention,build_matrix=False):
    with torch.no_grad():
        loss_fun=nn.CrossEntropyLoss(reduction="sum")
        total_loss=0
        pred_words=list()
        attention_matrix=[]
        for x,y in input_data:
            attn=[]
            loss=0
            decoder_words=[]
            x=x.T
            y=y.T
            encoder_hidden=encoder.initHidden()
            timesteps=len(x)
            if cell_type=='GRU' or cell_type=='RNN':

                encoder_hidden=encoder.initHidden()
                encoder_output,encoder_hidden=encoder(x,encoder_hidden)
                if(decoder_layers>encoder_layers):
                    i = decoder_layers
                    decoder_hidden=encoder_hidden

                    while True:
                        if(i==encoder_layers):
                            break
                        # Concatenate the two tensors along the first dimension
                        decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                        i-=1

                elif(decoder_layers<encoder_layers):
                    decoder_hidden=encoder_hidden[-decoder_layers:]

                else:
                    decoder_hidden=encoder_hidden

                decoder_input = y[0]

                if(bi_directional=="Yes"):
                    split_tensor= torch.split(encoder_output, hidden_size, dim=-1)
                    encoder_output=torch.add(split_tensor[0],split_tensor[1])/2

                for i in range(0,len(y)):
                    if(attention=="Yes"):
                        decoder_output, decoder_hidden, attn_weights=decoder(decoder_input,decoder_hidden,encoder_output,len(x))
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), y[i])
                        index=index.squeeze()
                        decoder_input=index
                        decoder_words.append(index.tolist())
                        if(build_matrix==True):
                            attn.append(attn_weights)
                    else:
                        decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), y[i])
                        index=index.squeeze()
                        decoder_input=index
                        decoder_words.append(index.tolist())
                if(build_matrix==True):
                    attention_matrix=torch.cat(tuple(x for x in attn),dim=2).to(device)
                decoder_words=np.array(decoder_words)
                pred_words.append(decoder_words.T)
                total_loss+=loss.item()


            if cell_type=='LSTM':

                encoder_hidden=encoder.initHidden()
                encoder_state=encoder.initState()

                encoder_output,encoder_hidden,encoder_state=encoder(x,encoder_hidden,encoder_state)

                if(decoder_layers>encoder_layers):
                    i = decoder_layers
                    decoder_hidden=encoder_hidden
                    decoder_state=encoder_state
                    while True:
                        if(i==encoder_layers):
                            break
                        # Concatenate the two tensors along the first dimension
                        decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                        decoder_state = torch.cat([decoder_state, encoder_state[-1].unsqueeze(0)], dim=0)
                        i-=1

                elif(decoder_layers<encoder_layers):
                    decoder_hidden=encoder_hidden[-decoder_layers:]
                    decoder_state=encoder_state[-decoder_layers:]

                else:
                    decoder_hidden=encoder_hidden
                    decoder_state=encoder_state


                if(bi_directional=="Yes"):
                    split_tensor= torch.split(encoder_output, hidden_size, dim=-1)
                    encoder_output=torch.add(split_tensor[0],split_tensor[1])/2
                decoder_input = y[0]

                for i in range(0,len(y)):
                    if(attention=="Yes"):
                        decoder_output, decoder_hidden, decoder_state, attn_weights=decoder(decoder_input,decoder_hidden,encoder_output,len(x),decoder_state)
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), y[i])
                        index=index.squeeze()
                        decoder_input=index
                        decoder_words.append(index.tolist())
                        if(build_matrix==True):
                            attn.append(attn_weights)
                    else:
                        decoder_output, decoder_hidden,decoder_state= decoder(decoder_input, decoder_hidden,decoder_state)
                        max_prob,index=decoder_output.topk(1)
                        loss+=loss_fun(torch.squeeze(decoder_output), y[i])
                        index=index.squeeze()
                        decoder_input=index
                        decoder_words.append(index.tolist())
                if(build_matrix==True):
                    attention_matrix=torch.cat(tuple(x for x in attn),dim=2).to(device)
                decoder_words=np.array(decoder_words)
                pred_words.append(decoder_words.T)
                total_loss+=loss.item()


    predictions=[]
    for batch in pred_words:
        for word in batch:
            predictions.append(word)
    
    return predictions,total_loss/(len(predictions)*len(predictions[0])),attention_matrix


def accuracy(predictions,y):
    count=0
    for i in range(0,len(predictions)):
        p=predictions[i]
        if np.array_equal(p,y[i]):
            count+=1
    return (count/len(predictions))*100




def wandb_run_sweeps(train_dataset,val_dataset,test_dataset,train_y,val_y,test_y,input_len,target_len):
    
    config = {
        "project":"CS6910_Assignment3",
        "method": 'bayes',
        "metric": {
        'name': 'acc',
        'goal': 'maximize'
        },
        'parameters' :{
        "epochs": {"values":[15,20,25]},
        "batchsize": {"values": [64,128,256]},
        "embedding_size": {"values":[256, 512,1024]},
        "hidden_size": {"values":[256, 512,1024]},
        "encoder_layers": {"values":[2,3,4]},
        "decoder_layers": {"values":[2,3,4]},
        "cell_type": {"values":["LSTM"]},
        "bi_directional":{"values":["Yes"]},
        "dropout":{"values":[0.2,0.3,0.5]},
        "attention":{"values":["Yes"]},
        }
    }
    def train_rnn():
        wandb.init()

        name='_CT_'+str(wandb.config.cell_type)+"_BS_"+str(wandb.config.batchsize)+"_EPOCH_"+str(wandb.config.epochs)+"_ES_"+str(wandb.config.embedding_size)+"_HS_"+str(wandb.config.hidden_size)
        
        
        train_dataloader=DataLoader(train_dataset,batch_size=wandb.config.batchsize)
        test_dataloader=DataLoader(test_dataset,batch_size=wandb.config.batchsize)
        val_dataloader=DataLoader(val_dataset,batch_size=wandb.config.batchsize)
        
        epoch_train_loss,epoch_val_loss,epoch_val_acc,encoder,decoder,encoder_layers,decoder_layers=train_iter(train_dataloader,val_dataloader,val_y,input_len,target_len,wandb.config.epochs,wandb.config.batchsize,wandb.config.embedding_size,wandb.config.encoder_layers,wandb.config.decoder_layers,wandb.config.hidden_size,wandb.config.cell_type,wandb.config.bi_directional,wandb.config.dropout,wandb.config.attention)

        for i in range(wandb.config.epochs):
            wandb.log({"loss":epoch_train_loss[i]})
            wandb.log({"val_loss":epoch_val_loss[i]})
            wandb.log({"val_acc":epoch_val_acc[i]})
            wandb.log({"epoch": (i+1)})
        wandb.log({"validation_accuracy":epoch_val_acc[-1]})    
        
        train_predictions,_,_=eval(train_dataloader,encoder,decoder,wandb.config.encoder_layers,
                              wandb.config.decoder_layers,wandb.config.batchsize,wandb.config.hidden_size,
                              wandb.config.bi_directional,wandb.config.cell_type,wandb.config.attention)

        train_accuracy=accuracy(train_predictions,train_y)
        wandb.log({"train_accuracy":train_accuracy})
        
        test_predictions,_,_=eval(test_dataloader,encoder,decoder,wandb.config.encoder_layers,
                              wandb.config.decoder_layers,wandb.config.batchsize,wandb.config.hidden_size,
                              wandb.config.bi_directional,wandb.config.cell_type,wandb.config.attention)

        test_accuracy=accuracy(test_predictions,test_y)
        wandb.log({"test_accuracy":test_accuracy})
        wandb.log({"acc":epoch_val_acc[-1]})
        wandb.run.name = name
        wandb.run.save()
        wandb.run.finish()
    wandb.login(key="67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e")
    sweep_id=wandb.sweep(config,project="Assignment3")
    wandb.agent(sweep_id,function=train_rnn,count=1)

def wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,epochs,encoder_layers,decoder_layers,batchsize,embedding_size,hidden_size,bi_directional,dropout,cell_type,attention):
    
    wandb.login(key = "67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e")
    wandb.init(project="Assignment3")
    name='_CT_'+str(cell_type)+"_BS_"+str(batchsize)+"_EPOCH_"+str(epochs)+"_ES_"+str(embedding_size)+"_HS_"+str(hidden_size)


    train_dataloader=DataLoader(train_dataset,batch_size=batchsize)
    test_dataloader=DataLoader(test_dataset,batch_size=batchsize)
    val_dataloader=DataLoader(val_dataset,batch_size=batchsize)

    epoch_train_loss,epoch_val_loss,epoch_val_acc,encoder,decoder,encoder_layers,decoder_layers=train_iter(train_dataloader,val_dataloader,val_y,input_len,target_len,epochs,batchsize,embedding_size,encoder_layers,decoder_layers,hidden_size,cell_type,bi_directional,dropout,attention)

    for i in range(epochs):
        wandb.log({"loss":epoch_train_loss[i]})
        wandb.log({"val_loss":epoch_val_loss[i]})
        wandb.log({"val_acc":epoch_val_acc[i]})
        wandb.log({"epoch": (i+1)})
    wandb.log({"validation_accuracy":epoch_val_acc[-1]})    

    train_predictions,_,_=eval(train_dataloader,encoder,decoder,encoder_layers,decoder_layers,batchsize,hidden_size,bi_directional,cell_type,attention)

    train_accuracy=accuracy(train_predictions,train_y)
    wandb.log({"train_accuracy":train_accuracy})

    test_predictions,_,_=eval(test_dataloader,encoder,decoder,encoder_layers,decoder_layers,batchsize,hidden_size,bi_directional,cell_type,attention)
    test_accuracy=accuracy(test_predictions,test_y)
    wandb.log({"test_accuracy":test_accuracy})
    wandb.log({"acc":epoch_val_acc[-1]})
    
    
    test_dataset_attn=MyDataset(test_x[:batchsize],test_y[:batchsize])
    test_dataloader_attn_for_matrix=DataLoader(test_dataset_attn,batch_size=batchsize)
    test_predictions,_,attn_matrix=eval(test_dataloader_attn_for_matrix,encoder,decoder,encoder_layers,decoder_layers,batchsize,hidden_size,bi_directional,cell_type,attention,True)

    
    fig=plot_attention(test_predictions,attn_matrix)
    fig.savefig("ex.png")
    temp = plt.imread("ex.png")
    plt.show()
    image = wandb.Image(temp)
    wandb.log({"attention heatmaps":image})
    wandb.run.name = name
    wandb.run.save()
    wandb.run.finish()


def main():

    wandb_run_sweeps(train_dataset,val_dataset,test_dataset,train_y,val_y,test_y,input_len,target_len)


if __name__=="__main__":
    main()

# ## MODEL


def representation_to_hin_word(predictions,idx_to_hin):
    words=[]
    for word in predictions:
        s=''
        for char in word:
            if(idx_to_hin[char]!='#' and idx_to_hin[char]!='^'):
                s+=idx_to_hin[char]
        words.append(s)
    return words

def representation_to_eng_word(predictions,idx_to_eng):
    words=[]
    for word in predictions:
        s=''
        for char in word:
            if(idx_to_eng[char]!='#' and idx_to_eng[char]!='^'):
                s+=idx_to_eng[char]
        words.append(s)
    return words


def make_csv_file(test_x,test_y,test_predictions,idx_to_eng,idx_to_hin):
    test_eng_words=representation_to_eng_word(test_x,idx_to_eng)
    pred_test_words=representation_to_hin_word(test_predictions,idx_to_hin)
    test_hin_words=representation_to_hin_word(test_y,idx_to_hin)
    # Sample data to write to the CSV file
    data1 = [
        ['Input', 'Expected', 'Predicted']
    ]


    for i in range(len(test_eng_words)):
        data1.append([test_eng_words[i],pred_test_words[i],test_hin_words[i]])
    # Specify the file path and name for the CSV file
    csv_file_path = 'data1.csv'

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the data to the CSV file row by row
        for row in data:
            writer.writerow(row)

def plot_attention(test_predictions,attn_matrix):
    
    attn_matrix1=attn_matrix.permute(1,0,2)
    attn_matrix1=attn_matrix1[:9]
    total_words,input_length,output_length = attn_matrix1.shape


    from matplotlib.font_manager import FontProperties


    tel_font = FontProperties(fname = '/kaggle/input/hindi-font/TiroDevanagariHindi-Regular.ttf')


    fig, axes = plt.subplots(3, 3, figsize=(12,12))

    fig.tight_layout(pad=5.0)
    fig.subplots_adjust(top=0.90)
    axes = axes.ravel()

    for i in range(total_words):
        count=0
        start1=0
        end1=0
        eng_word=""
        for char in test_x[i]:
            if(idx_to_eng[char]=='^'):
                start1=count+1
            elif(idx_to_eng[char]=='#'):
                end1=count
                break
            else:
                eng_word+=idx_to_eng[char]
            count+=1

        count=0
        hin_word=""
        for char in test_predictions[i]:
            if(idx_to_hin[char]=='^'):
                start=count+1
            elif(idx_to_hin[char]=='#'):
                end=count
                break
            else:
                hin_word+=idx_to_hin[char]
            count+=1

        attn=attn_matrix1[i,start1:end1,start:end].cpu().numpy()
        sns.heatmap(attn, ax=axes[i],cmap="Greens")
        axes[i].set_yticklabels(eng_word,rotation=10)  
        axes[i].set_xticklabels(hin_word,fontproperties = tel_font,fontdict={'fontsize':16})
        axes[i].xaxis.tick_top()
    
    return fig




