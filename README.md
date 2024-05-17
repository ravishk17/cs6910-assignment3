"# cs6910-assignment3" 
# RNN based seq2seq model

## Author: Ravish Kumar

### Link to the project report: https://wandb.ai/cs23m055/Assignment3/reports/Assignment-3--Vmlldzo3NzU0NDcx


## Required Libraries
1. `torch`
2. `random`
3. `tqdm` (for progress bar visualization)
4. `numpy`
5. `matplotlib`
6. `pandas`
7. `sklearn.utils.shuffle` (for shuffling dataset)
8. `seaborn` (for attention heatmap visualization)


### Steps to follow:

1. Install wandb by command *pip install wandb* to your system
2. Run train.py file with the appropriate arguments given below
3. Your run will be logged to wandb to my project "Assignment3" and entity="cs23m055"
4. You can view the logs that is the model training accuracy, validation accuracy, testing accuracy and respective losses




### Download the dataset in the same directory as train.py

Below are the command line arguments which can be given at runtime
| Name | Default Value | Description |
|------|---------------|-------------|
| -wp, --wandb_project | assignement2_kaggle | Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity | cs23m055 | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| -e, --epochs | 5 | number of epochs |
| -bs, --batchsize | default=256 | choices: [64,128,256] |
| -hs, --hidden_size | 1024 | choices: [256,512,1024]|
| -el, --encoder_layers | 3 | choices: [2,3,4] |
| -dl, --decoder_layers | 3 | choices: [2,3,4] |
| -es, --embedding_size | 256 | choices: [256,512,1024] |
| -do, --dropout | 0.1 | choices: [0.1, 0.2, 0.3] |
| -ct, --cell_type | LSTM | choices: ["LSTM","GRU","RNN"] |
| -d, --bi_directional | 'Yes' | choice:["Yes","No"] |
| -a, --attention | 'Yes' | choices: ["Yes","No"] |


To run the code: ``` python train.py -e 5 ``` , and so on try with different commandline arguments
