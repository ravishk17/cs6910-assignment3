#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING LIBRARIES

# In[1]:


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
import wandb
import requests,zipfile,io
import pandas as pd


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# In[3]:


# device


# ## DOWNLOADING AND UNZIPPING DATA

# In[4]:


import requests  # Importing the requests library to make HTTP requests
import zipfile   # Importing the zipfile library to handle zip files
import io        # Importing the io library for input/output operations

def download_data(url="https://drive.usercontent.google.com/u/0/uc?id=1tGIO4-IPNtxJ6RQMmykvAfY_B0AaLY5A&export=download"):
    
    # Make an HTTP GET request to the specified URL and store the response
    response = requests.get(url)

    # Create a ZipFile object from the response content
    z = zipfile.ZipFile(io.BytesIO(response.content))

    # Extract all the contents of the zip file
    z.extractall()


# ## METHODS FOR GETTING CHARACTERS FOR CORPUSS AND ADDING THEIR INDICES

# In[5]:


def add_chars(word,st):
    st.add(ch for ch in word)


# In[6]:


def get_entire_collection(data):
    eng_corpus = set()  # Set to store English characters
    hin_corpus = set()  # Set to store Hindi characters
    
    for i in range(0, len(data)):
        
        # Add each character of the English word to the English corpus set
        add_chars(data[0][i],eng_corpus)
        
        # Add each character of the Hindi word to the Hindi corpus set
        add_chars(data[1][i],hin_corpus)
        
    # Add end delimiter characters to both corpora
    eng_corpus.add('#')
    hin_corpus.add('#')
    hin_corpus.add('$')
    eng_corpus.add('$')
    
    # Add start delimiter character to the Hindi corpus
    hin_corpus.add('^')
    
    return eng_corpus, len(eng_corpus), hin_corpus, len(hin_corpus)


# In[7]:


def createMapping(corpus):
    chrToIndex = {}
    idxToChr = {}
    for i,char in enumerate(corpus):
        chrToIndex[char]=i
        idxToChr[i]=char
    return chrToIndex, idxToChr


# In[8]:


# Hindi training csv
def word2index(data):
    eng_corpus, eng_vocab_size, hin_corpus, hin_vocab_size = get_entire_collection(data)  # Get Hindi and English corpora from data and their respective counts
    engchar_idx, idx_engchar = createMapping(eng_corpus) # Dictionary to map English characters to indices and indices to English characters
    hinchar_idx, idx_hinchar = createMapping(hin_corpus) # Dictionary to map Hindi characters to indices and indices to Hindi character
    return eng_vocab_size, hin_vocab_size, engchar_idx, hinchar_idx, idx_engchar, idx_hinchar


# ## DATA PREPROCESSING

# In[9]:


def getMax(data):
    mx = 0
    for word in data:
        # Update mxif the length of word is greater
        mx=max(mx,len(word))
    return mx


# In[10]:


def maxlen(data):        
    maxlen_eng = getMax(data[0]) # Variable to store the maximum length of English words
    maxlen_hin = getMax(data[1]) # Variable to store the maximum length of Hindi words
    return maxlen_eng, maxlen_hin
    


# In[11]:


# def padding(data,lst,mapping):
#     for word in data:
def util_preprocess(data, maxLen, unKnown, word_to_idx, hindi):
    sentence = []
    for word in data:
        if(hindi):
            word = '^' + word # Add start delimiter (^) to Hindi word
        # Pad the words to their respective maximum lengths
        word = word.ljust(maxLen+1, '#')
        idx = []
        for ch in word:
            if(ch in word_to_idx):
                idx.append(word_to_idx[ch])
            else:
                idx.append(unKnown)
        sentence.append(idx)
    return sentence
    
        
def pre_process(data, eng_to_idx, hin_to_idx):
    
    maxlen_eng, maxlen_hin = getMax(data[0]), getMax(data[1])  # Get the maximum lengths of English and Hindi words
       
    unKnown = eng_to_idx['$']  # Index for unknown character in English corpus

    eng = util_preprocess(data[0],maxlen_eng,unKnown,eng_to_idx,False) # List to store pre-processed English sentences
    hin = util_preprocess(data[1],maxlen_hin,unKnown,hin_to_idx,True) # List to store pre-processed Hindi sentences
    
    return eng, hin


# ## LOADING OUR CUSTOM DATASET TO DATALOADER

# In[25]:


class MyDataset(Dataset):
    def __init__(self, train_x, train_y, transform=None):
        self.transform = transform  # Optional data transformation
        self.train_y = train_y  # Target data (train_y)
        self.train_x = train_x  # Input data (train_x)
        
    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(sample)  # Apply the transformation (if any) to the sample
            
        # Return the input and target data tensors for the given index
        return torch.tensor(self.train_x[idx]).to(device), torch.tensor(self.train_y[idx]).to(device)
        
    def __len__(self):
        return len(self.train_x)  # Return the length of the dataset

def get_data(downloaded=False):
    # if(not downloaded):
    # download_data()  # Download the data (assuming it has been implemented elsewhere)
    
    # Read the train, test, and validation datasets from CSV files
    test_df = pd.read_csv("aksharantar_sampled/hin/hin_test.csv", header=None)
    train_df = pd.read_csv("aksharantar_sampled/hin/hin_train.csv", header=None)
    val_df = pd.read_csv("aksharantar_sampled/hin/hin_valid.csv", header=None)
    
    # Convert words to indices and retrieve vocabulary information
    input_len, target_len, eng_to_idx, hin_to_idx, idx_to_eng, idx_to_hin = word2index(train_df)
    
    # Return the datasets and vocabulary information
    return train_df, test_df, val_df, eng_to_idx, hin_to_idx, idx_to_eng, idx_to_hin, input_len, target_len


# ## Seq2Seq MODEL

# In[13]:


class EncoderGRU(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_of_layers,batch_size,bi_directional,dropout_p=0.1):
        super(EncoderGRU,self).__init__()
        self.batch_size=batch_size
        self.num_of_layers=num_of_layers
        self.gru = nn.GRU(embedding_size, hidden_size, num_of_layers, bidirectional = bi_directional=="Yes")
        self.dropout = nn.Dropout(dropout_p)
        self.bi_directional=bi_directional
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.input_size=input_size
        

    def forward(self,input,hidden):
        embedded=self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        embedded = self.dropout(embedded)
        output,hidden=self.gru(embedded,hidden)
    
        if self.bi_directional=="Yes":
            hidden=hidden.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            hidden=torch.add(hidden[0],hidden[1])/2
            
        return output,hidden

    def initHidden(self):
        if(self.bi_directional!="Yes"):
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)

class DecoderGRU(nn.Module):
    def __init__(self, output_size,hidden_size, embedding_size, decoder_layers,batch_size,dropout_p=0.1):
        super(DecoderGRU, self).__init__()
        self.embedding_size=embedding_size
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.LogSoftmax(dim=2)
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.batch_size=batch_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size,hidden_size, decoder_layers,dropout = dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        
        

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden



# In[14]:


class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_of_layers,batch_size,bi_directional,dropout_p=0.1):
        super(EncoderRNN,self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_size=hidden_size
        self.num_of_layers=num_of_layers
        self.batch_size=batch_size
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.bi_directional=bi_directional
        self.input_size=input_size
        self.rnn = nn.RNN(embedding_size, hidden_size, num_of_layers, bidirectional = bi_directional!="No")
        self.embedding_size=embedding_size

    def forward(self,input,hidden):
        embedded=self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        embedded = self.dropout(embedded)
        output,hidden=self.rnn(embedded,hidden)
    
        if(self.bi_directional=="Yes"):
            hidden=hidden.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            hidden=torch.add(hidden[0],hidden[1])/2
            
        return output,hidden

    def initHidden(self):
        if(self.bi_directional!="Yes"):
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)





class DecoderRNN(nn.Module):
    def __init__(self, output_size,hidden_size, embedding_size, decoder_layers,batch_size,dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.RNN(embedding_size,hidden_size, decoder_layers,dropout = dropout_p)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden


        


# In[15]:


class EncoderLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_of_layers,batch_size,bi_directional,dropout_p=0.1):
        super(EncoderLSTM,self).__init__()
        self.num_of_layers=num_of_layers
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.embedding=nn.Embedding(input_size,embedding_size) 
        self.bi_directional=bi_directional
        self.lstm = nn.LSTM(embedding_size,hidden_size,num_of_layers,bidirectional = bi_directional=="Yes")
        self.dropout = nn.Dropout(dropout_p)
        self.batch_size=batch_size

    def forward(self,input,hidden,state):
        embedded=self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        embedded = self.dropout(embedded)
        output,(hidden,state)=self.lstm(embedded,(hidden,state))
    
        if(self.bi_directional!="No"):
            hidden=hidden.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            state=state.resize(2,self.num_of_layers,self.batch_size,self.hidden_size)
            hidden=torch.add(hidden[0],hidden[1])/2
            state=torch.add(state[0],hidden[1])/2
            
        return output,hidden,state



    def initHidden(self):
        if(self.bi_directional!="Yes"):
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)      
    
    def initState(self):
        if(self.bi_directional!="Yes"):
            return torch.zeros(self.num_of_layers,self.batch_size,self.hidden_size,device=device)
        else:
            return torch.zeros(2*self.num_of_layers,self.batch_size,self.hidden_size,device=device)





class DecoderLSTM(nn.Module):
    def __init__(self, output_size,hidden_size, embedding_size, decoder_layers,batch_size,dropout_p=0.1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size        
        self.lstm = nn.LSTM(embedding_size,hidden_size,decoder_layers,dropout = dropout_p)
        self.batch_size=batch_size
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input,hidden,state):
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
        output,(hidden,state)=self.lstm(embedded,(hidden,state))
        output = self.softmax(self.out(output))
        return output,hidden,state



# ## ATTENTION MECHANISM

# In[16]:


class AttnDecoder(nn.Module):
    def __init__(self,output_size,hidden_size,embedding_size,decoder_layers,batch_size,cell_type,dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.decoder_layers=decoder_layers
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.batch_size=batch_size
        self.cell_type=cell_type
        self.embedding = nn.Embedding(output_size, embedding_size)
        

        self.U=nn.Linear(hidden_size,hidden_size,bias=False).to(device)
        self.V=nn.Linear(hidden_size,1,bias=False).to(device)
        self.W=nn.Linear(hidden_size,hidden_size,bias=False).to(device)
        
        self.linear=nn.Linear(self.hidden_size,output_size,bias=True)
        self.softmax=nn.Softmax(dim=1)
        self.softmax1=nn.LogSoftmax(dim=2)

        if(cell_type=="RNN"):
            self.rnn = nn.RNN(self.embedding_size+self.hidden_size, self.hidden_size,self.decoder_layers,dropout = dropout_p)
            
        if(cell_type=="LSTM"):
            self.lstm = nn.LSTM(self.embedding_size+self.hidden_size, self.hidden_size,self.decoder_layers,dropout = dropout_p)
            
        if(cell_type=="GRU"):
            self.gru = nn.GRU(self.embedding_size+self.hidden_size, self.hidden_size,self.decoder_layers,dropout = dropout_p)
        
        

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
        
        
        if(self.cell_type=="RNN"):
            output,hidden=self.rnn(final_input,hidden)
        elif(self.cell_type=="LSTM"):
            output, (hidden,state) =self.lstm(final_input,(hidden,state))
        else:
            output,hidden=self.gru(final_input,hidden)
        
        output1=self.softmax1(self.linear(output))
        
        if(self.cell_type=="LSTM"):
            return output1, hidden, state, alpha_j
        else:
            return output1, hidden, alpha_j
        


# In[17]:


def train_util1(decoder_layers,encoder_layers,decoder_hidden,decoder_state):
    i = decoder_layers
    while(i>encoder_layers):
        if(i==encoder_layers):
            break
        # Concatenate the two tensors along the first dimension
        decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
        decoder_state = torch.cat([decoder_state, encoder_state[-1],unsqueeze(0)], dim=0)
        i-=1
    return decoder_hidden, decoder_state



def train(train_data, encoder, decoder, loss_fun, hidden_size, bi_directional, cell_type, attention, encoder_optimizer, decoder_optimizer, encoder_layers, decoder_layers, batch_size):
    total_loss = 0
    teacher_forcing_ratio = 0.5
    
    # Iterate over the training data
    for i, (train_x, train_y) in enumerate(train_data):
        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # Transposing the dataset
        train_x = train_x.T
        train_y = train_y.T
        timesteps = len(train_x)
        
        
        # Check the cell type (LSTM)
        if cell_type == 'LSTM':
            encoder_hidden = encoder.initHidden()
            encoder_state = encoder.initState()
            
            encoder_output, encoder_hidden, encoder_state = encoder(train_x, encoder_hidden, encoder_state)
        
            if decoder_layers > encoder_layers:
                i = decoder_layers
                decoder_hidden = encoder_hidden
                decoder_state = encoder_state
                
                while True:
                    if i == encoder_layers:
                        break
                    # Concatenate the two tensors along the first dimension
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    decoder_state = torch.cat([decoder_state, encoder_state[-1].unsqueeze(0)], dim=0)
                    i -= 1
            elif decoder_layers < encoder_layers:
                decoder_hidden = encoder_hidden[-decoder_layers:]
                decoder_state = encoder_state[-decoder_layers:]
            else:
                decoder_hidden = encoder_hidden
                decoder_state = encoder_state
            
            if bi_directional != "No":
                split_tensor = torch.split(encoder_output, hidden_size, dim=-1)
                encoder_output = torch.add(split_tensor[0], split_tensor[1]) / 2
            
            decoder_input = train_y[0]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if(not use_teacher_forcing):
                for i in range(0, len(train_y)):
                    if attention == "Yes":
                        decoder_output, decoder_hidden, decoder_state, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(train_x), decoder_state)
                    else:
                        decoder_output, decoder_hidden, decoder_state = decoder(decoder_input, decoder_hidden, decoder_state)
                    max_prob, index = decoder_output.topk(1)
                    loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                    decoder_input = index
            else:
                for i in range(0, len(train_y)):
                    if attention == "Yes":
                        decoder_output, decoder_hidden, decoder_state, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(train_x), decoder_state)
                        # loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                        # decoder_input = train_y[i]
                    else:
                        decoder_output, decoder_hidden, decoder_state = decoder(decoder_input, decoder_hidden, decoder_state)
                    loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                    decoder_input = train_y[i]  # Teacher forcing
            
            
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss
            
        # Check the cell type (RNN, GRU, LSTM)
        if cell_type == 'GRU' or cell_type == 'RNN':
            # Initialize the hidden state of the encoder
            encoder_hidden = encoder.initHidden()
            
            # Pass the input through the encoder
            encoder_output, encoder_hidden = encoder(train_x, encoder_hidden)
            
            # Adjust decoder hidden state based on the number of layers
            if decoder_layers > encoder_layers:
                i = decoder_layers
                decoder_hidden = encoder_hidden
                
                while True:
                    if i == encoder_layers:
                        break
                    # Concatenate the two tensors along the first dimension
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    i -= 1
            elif decoder_layers < encoder_layers:
                decoder_hidden = encoder_hidden[-decoder_layers:]
            else:
                decoder_hidden = encoder_hidden
        
            decoder_input = train_y[0]
            
            # Apply bidirectional averaging if specified
            if bi_directional == "Yes":
                split_tensor = torch.split(encoder_output, hidden_size, dim=-1)
                encoder_output = torch.add(split_tensor[0], split_tensor[1]) / 2
            
            # Determine whether to use teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if(not use_teacher_forcing):
                # Without teacher forcing: use the predicted output as the next input
                for i in range(0, len(train_y)):
                    if attention == "Yes":
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(train_x))
                        # max_prob, index = decoder_output.topk(1)
                        # loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                        # decoder_input = index
                    else:
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    max_prob, index = decoder_output.topk(1)
                    loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                    decoder_input = index
                        
            else:
                # Teacher forcing: feed the target as the next input
                for i in range(0, len(train_y)):
                    if attention == "Yes":
                        # Pass input, hidden state, and encoder output through the decoder with attention
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(train_x))
                        # loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                        # decoder_input = train_y[i]
                    else:
                        # Pass input and hidden state through the decoder
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += loss_fun(torch.squeeze(decoder_output), train_y[i])
                    decoder_input = train_y[i]  # Teacher forcing
            
            
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss
        
        
    return total_loss.item() / len(train_y), encoder, decoder



# In[18]:


def train_iter(input_data, val_data, val_y, input_len, hidden_size, cell_type, bi_directional, dropout, attention, target_len, epochs, batch_size, embedding_size, encoder_layers, decoder_layers):
    lr = 0.001
    beam_size=0
    # Initialize the encoder and decoder based on the cell type and attention
    if(cell_type == 'RNN'):
        encoder = EncoderRNN(input_len, hidden_size, embedding_size, encoder_layers, batch_size, bi_directional, dropout).to(device)
        if(attention!='Yes'):
            decoder = DecoderRNN(target_len, hidden_size, embedding_size, decoder_layers, batch_size, dropout).to(device)
    elif(cell_type == 'GRU'):
        encoder = EncoderGRU(input_len, hidden_size, embedding_size, encoder_layers, batch_size, bi_directional, dropout).to(device)
        if(attention!='Yes'):
            decoder = DecoderGRU(target_len, hidden_size, embedding_size, decoder_layers, batch_size, dropout).to(device)
    else:
        encoder = EncoderLSTM(input_len, hidden_size, embedding_size, encoder_layers, batch_size, bi_directional, dropout).to(device)
        if(attention!='Yes'):
            decoder = DecoderLSTM(target_len, hidden_size, embedding_size, decoder_layers, batch_size, dropout).to(device)
    if(attention=='Yes'):
        decoder = AttnDecoder(target_len, hidden_size, embedding_size, decoder_layers, batch_size, cell_type, dropout).to(device)
        
    decoder_optimizer = optim.Adam(decoder.parameters(), lr)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr)
    loss_fun = nn.CrossEntropyLoss(reduction="sum")
    
    # array initialization for storing the different losses for each epochs
    
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_acc = []
    
    # Iterate over the epochs
    for i in range(0, epochs):
        loss, encoder, decoder = train(input_data, encoder, decoder, loss_fun, hidden_size, bi_directional, cell_type, attention, encoder_optimizer, decoder_optimizer, encoder_layers, decoder_layers, batch_size)
        val_predictions, val_loss, attn_weights = eval(val_data, encoder, decoder, encoder_layers, decoder_layers, batch_size, hidden_size, bi_directional, cell_type, attention)
        
        epoch_val_loss.append(val_loss)
        epoch_train_loss.append(loss / 51200)  # train_data has 51200 samples
        
        val_acc = accuracy(val_predictions, val_y)
        epoch_val_acc.append(val_acc)
        
        print(loss / 51200, val_loss, val_acc)
    
    return epoch_train_loss, epoch_val_loss, epoch_val_acc, encoder, decoder, encoder_layers, decoder_layers


# In[19]:


def eval(input_data, encoder, decoder, encoder_layers, decoder_layers, batch_size, hidden_size, bi_directional, cell_type, attention, build_matrix=False):
    with torch.no_grad():
        loss_fun = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0
        pred_words = list()
        attention_matrix = []
        
        for x, y in input_data:
            attn = []
            loss = 0
            decoder_words = []
            x = x.T
            y = y.T
            
            # Initialize the encoder hidden state
            encoder_hidden = encoder.initHidden()
            
            # Get the number of timesteps in the input sequence
            timesteps = len(x)
            
            if cell_type == 'GRU' or cell_type == 'RNN':
                # Run the input sequence through the encoder
                encoder_hidden = encoder.initHidden()
                encoder_output, encoder_hidden = encoder(x, encoder_hidden)
                
                if decoder_layers > encoder_layers:
                    i = decoder_layers
                    decoder_hidden = encoder_hidden
                    
                    while True:
                        if i == encoder_layers:
                            break
                        # Concatenate the encoder hidden state to match the decoder layers
                        decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                        i -= 1
                
                elif decoder_layers < encoder_layers:
                    decoder_hidden = encoder_hidden[-decoder_layers:]
                else:
                    decoder_hidden = encoder_hidden
                
                decoder_input = y[0]
                
                if bi_directional == "Yes":
                    # Split the encoder output tensor into two parts along the last dimension
                    split_tensor = torch.split(encoder_output, hidden_size, dim=-1)
                    # Average the two parts
                    encoder_output = torch.add(split_tensor[0], split_tensor[1]) / 2
                
                # Run the decoder for each timestep in the output sequence
                for i in range(0, len(y)):
                    if attention == "Yes":
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(x))
                        max_prob, index = decoder_output.topk(1)
                        loss += loss_fun(torch.squeeze(decoder_output), y[i])
                        index = index.squeeze()
                        decoder_input = index
                        decoder_words.append(index.tolist())
                        if build_matrix:
                            attn.append(attn_weights)
                    else:
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        max_prob, index = decoder_output.topk(1)
                        loss += loss_fun(torch.squeeze(decoder_output), y[i])
                        index = index.squeeze()
                        decoder_input = index
                        decoder_words.append(index.tolist())
                
                if build_matrix:
                    attention_matrix = torch.cat(tuple(x for x in attn), dim=2).to(device)
                
                decoder_words = np.array(decoder_words)
                pred_words.append(decoder_words.T)
                total_loss += loss.item()
            
            if cell_type == 'LSTM':
                # Run the input sequence through the encoder
                encoder_hidden = encoder.initHidden()
                encoder_state = encoder.initState()
                encoder_output, encoder_hidden, encoder_state = encoder(x, encoder_hidden, encoder_state)
                
                if decoder_layers > encoder_layers:
                    i = decoder_layers
                    decoder_hidden = encoder_hidden
                    decoder_state = encoder_state
                    
                    while True:
                        if i == encoder_layers:
                            break
                        # Concatenate the encoder hidden state and cell state to match the decoder layers
                        decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                        decoder_state = torch.cat([decoder_state, encoder_state[-1].unsqueeze(0)], dim=0)
                        i -= 1
                
                elif decoder_layers < encoder_layers:
                    decoder_hidden = encoder_hidden[-decoder_layers:]
                    decoder_state = encoder_state[-decoder_layers:]
                else:
                    decoder_hidden = encoder_hidden
                    decoder_state = encoder_state
                
                decoder_input = y[0]
                
                if bi_directional == "Yes":
                    # Split the encoder output tensor into two parts along the last dimension
                    split_tensor = torch.split(encoder_output, hidden_size, dim=-1)
                    # Average the two parts
                    encoder_output = torch.add(split_tensor[0], split_tensor[1]) / 2
                
                # Run the decoder for each timestep in the output sequence
                for i in range(0, len(y)):
                    if attention == "Yes":
                        decoder_output, decoder_hidden, decoder_state, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(x), decoder_state)
                        max_prob, index = decoder_output.topk(1)
                        loss += loss_fun(torch.squeeze(decoder_output), y[i])
                        index = index.squeeze()
                        decoder_input = index
                        decoder_words.append(index.tolist())
                        if build_matrix:
                            attn.append(attn_weights)
                    else:
                        decoder_output, decoder_hidden, decoder_state = decoder(decoder_input, decoder_hidden, decoder_state)
                        max_prob, index = decoder_output.topk(1)
                        loss += loss_fun(torch.squeeze(decoder_output), y[i])
                        index = index.squeeze()
                        decoder_input = index
                        decoder_words.append(index.tolist())
                
                if build_matrix:
                    attention_matrix = torch.cat(tuple(x for x in attn), dim=2).to(device)
                
                decoder_words = np.array(decoder_words)
                pred_words.append(decoder_words.T)
                total_loss += loss.item()
        
        predictions = []
        for batch in pred_words:
            for word in batch:
                predictions.append(word)
        
        return predictions, total_loss / (len(predictions) * len(predictions[0])), attention_matrix


# In[20]:


def accuracy(predictions,y):
    count=0
    for i in range(0,len(predictions)):
        p=predictions[i]
        if np.array_equal(p,y[i]):
            count+=1
    return (count/len(predictions))*100


# ## INTEGRATING WITH WANDB

# In[21]:


def wandb_run_sweeps(train_dataset,val_dataset,test_dataset,train_y,val_y,test_y,input_len,target_len):
    
    config = {
        "project":"Assignment3",
        "method": 'bayes',
        "metric": {
        'name': 'acc',
        'goal': 'maximize'
        },
        'parameters' :{
        "epochs": {"values":[1]},
        "batchsize": {"values": [64,128,256]},
        "embedding_size": {"values":[256,512,1024]},
        "hidden_size": {"values":[256,512,1024]},
        "encoder_layers": {"values":[2,3,4]},
        "decoder_layers": {"values":[2,3,4]},
        "cell_type": {"values":["LSTM"]},
        "bi_directional":{"values":["Yes","No"]},
        "dropout":{"values":[0.3]},
        "attention":{"values":["No","Yes"]},
        }
    }
    def train_rnn():
        wandb.init()

        name='_CT_'+str(wandb.config.cell_type)+"_BS_"+str(wandb.config.batchsize)+"_EPOCH_"+str(wandb.config.epochs)+"_ES_"+str(wandb.config.embedding_size)+"_HS_"+str(wandb.config.hidden_size)
        
        val_dataloader=DataLoader(val_dataset,batch_size=wandb.config.batchsize)
        test_dataloader=DataLoader(test_dataset,batch_size=wandb.config.batchsize)
        train_dataloader=DataLoader(train_dataset,batch_size=wandb.config.batchsize)
        
        epoch_train_loss,epoch_val_loss,epoch_val_acc,encoder,decoder,encoder_layers,decoder_layers=train_iter(train_dataloader,val_dataloader,val_y,input_len,wandb.config.hidden_size,wandb.config.cell_type,wandb.config.bi_directional,wandb.config.dropout,wandb.config.attention,target_len,wandb.config.epochs,wandb.config.batchsize,wandb.config.embedding_size,wandb.config.encoder_layers,wandb.config.decoder_layers)

        for i in range(wandb.config.epochs):
            wandb.log({"val_loss":epoch_val_loss[i]})
            wandb.log({"val_acc":epoch_val_acc[i]})
            wandb.log({"loss":epoch_train_loss[i]})
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


# In[27]:


def wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,input_len,target_len,epochs,encoder_layers,decoder_layers,batchsize,embedding_size,hidden_size,bi_directional,dropout,cell_type,attention):
    
    wandb.login(key = "67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e")
    wandb.init(project="Assignment3")
    name='_CT_'+str(cell_type)+"_BS_"+str(batchsize)+"_EPOCH_"+str(epochs)+"_ES_"+str(embedding_size)+"_HS_"+str(hidden_size)


    train_dataloader=DataLoader(train_dataset,batch_size=batchsize)
    test_dataloader=DataLoader(test_dataset,batch_size=batchsize)
    val_dataloader=DataLoader(val_dataset,batch_size=batchsize)
    
    epoch_train_loss,epoch_val_loss,epoch_val_acc,encoder,decoder,encoder_layers,decoder_layers=train_iter(train_dataloader,val_dataloader,val_y,input_len,hidden_size,cell_type,bi_directional,dropout,attention,target_len,epochs,batchsize,embedding_size,encoder_layers,decoder_layers)

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
    
    
    # test_dataset_attn=MyDataset(test_x[:batchsize],test_y[:batchsize])
    # test_dataloader_attn_for_matrix=DataLoader(test_dataset_attn,batch_size=batchsize)
    # test_predictions,_,attn_matrix=eval(test_dataloader_attn_for_matrix,encoder,decoder,encoder_layers,decoder_layers,batchsize,hidden_size,bi_directional,cell_type,attention,True)

    
    # fig=plot_attention(test_predictions,attn_matrix,test_x, idx_to_eng, idx_to_hin)
    # fig.savefig("ex.png")
    # temp = plt.imread("ex.png")
    # plt.show()
    # image = wandb.Image(temp)
    # wandb.log({"attention heatmaps":image})
    wandb.run.name = name
    wandb.run.save()
    wandb.run.finish()


# In[28]:


def main():
    train_df,test_df,val_df,eng_to_idx,hin_to_idx,idx_to_eng,idx_to_hin,input_len,target_len=get_data()

    val_x,val_y = pre_process(val_df,eng_to_idx,hin_to_idx)
    test_x,test_y = pre_process(test_df,eng_to_idx,hin_to_idx)
    train_x,train_y = pre_process(train_df,eng_to_idx,hin_to_idx)
    
    

    train_dataset=MyDataset(train_x,train_y)
    test_dataset=MyDataset(test_x,test_y)
    val_dataset=MyDataset(val_x,val_y)
    
    # wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,input_len,target_len,idx_to_eng, idx_to_hin,25,4,3,128,512,1024,"No",0.3,"LSTM","Yes")
    # wandb_run_sweeps(train_dataset,val_dataset,test_dataset,train_y,val_y,test_y,input_len,target_len)
    wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,input_len,target_len, 1,4,3,128,512,1024,"No",0.3,"LSTM","Yes")


if __name__=="__main__":
    
    main()


# In[22]:


def plot_attention(test_predictions,attn_matrix,test_x, idx_to_eng, idx_to_hin):
    
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

        start=0
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


# In[ ]:




