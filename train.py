from transliteration import wandb_run_configuration,train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y

import argparse
import wandb 
wandb.login(key="67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e1")


parser = argparse.ArgumentParser(description='Run my model function')
parser.add_argument('-wp','--wandb_project', default="CS6910_Assignment3", required=False,metavar="", type=str, help=' ')
parser.add_argument('-we','--wandb_entity', default="cs22m063", required=False,metavar="", type=str, help='')
parser.add_argument('-ds','--dataset', default="aksharantar", required=False,metavar="", type=str, help=' ')
parser.add_argument('-e','--epochs', default=20, required=False,metavar="", type=int, help=' ')
parser.add_argument('-bs','--batchsize', default=256, required=False,metavar="", type=int, help=' ')
parser.add_argument('-hs','--hidden_size', default=1024, required=False,metavar="", type=int, help=' ')
parser.add_argument('-el','--encoder_layers', default=3, required=False,metavar="", type=int, help=' ')
parser.add_argument('-dl','--decoder_layers', default=3, required=False,metavar="", type=int, help=' ')
parser.add_argument('-es','--embedding_size', default=256, required=False,metavar="", type=int, help=' ')
parser.add_argument('-do','--dropout', default=0.1, required=False,metavar="", type=int, help=' ')
parser.add_argument('-ct','--cell_type', default="LSTM", required=False,metavar="", type=str,choices= ["GRU", "LSTM", "RNN"], help=' ')
parser.add_argument('-d','--bi_directional', default="Yes", required=False,metavar="", type=str,choices= ["Yes", "No"], help=' ')
parser.add_argument('-a','--attention', default="Yes", required=False,metavar="", type=str,choices= ["Yes"], help=' ')

args = parser.parse_args()

wandb_run_configuration(train_dataset,val_dataset,test_dataset,train_y,val_y,test_x,test_y,args.epochs,args.encoder_layers,args.decoder_layers,args.batchsize,args.embedding_size,args.hidden_size,args.bi_directional,args.dropout,args.cell_type,args.attention)
