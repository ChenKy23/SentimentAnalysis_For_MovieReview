import torch
from torch.utils.data import TensorDataset, DataLoader
from model import lstm_model,rnn_model
from torch import nn
from torch.optim import Adam
import numpy as np
import pandas as pd

# Read data from files
train = pd.read_csv( "./Data/labeledTrainData.tsv.zip", header=0,  delimiter="\t", quoting=3 )
test = pd.read_csv( "./Data/testData.tsv.zip", header=0, delimiter="\t", quoting=3 )
trainLabels=train.loc[:,"sentiment"].values

# Get preprocessed data
encodedTrainFeatures = np.load("./Data/encodedTrainFeatures.npy")
encodedTestFeatures = np.load("./Data/encodedTestFeatures.npy")

batch_size=250
# Get dataLoader
trainDataSet=TensorDataset(torch.from_numpy(encodedTrainFeatures),torch.from_numpy(trainLabels))
trainDataLoader=DataLoader(dataset=trainDataSet,shuffle=True,batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vecob_size = 101247
embedding_dim = 300
hidden_dim = 512
n_layers = 2
dropout=0.3
print("vecobSize is:{}\nembedding_size is:{}\nhidden_size is:{}\nn_layers=:{}\ndropout_rate is:{}"\
      .format(vecob_size,embedding_dim,hidden_dim,n_layers,dropout))

model = lstm_model(vecob_size,n_layers,embedding_dim,hidden_dim,False,dropout).to(device)
# model = rnn_model(vecob_size,n_layers,embedding_dim,hidden_dim,False,dropout).to(device)
print(model)

criterion = nn.BCELoss()  
optimizer = Adam(model.parameters(), lr=1e-3)
grad_clip = 5

for epoch in range(20):
      model.train()
      epoch_loss = 0;
      epoch_acc = 0;
      for idx,(inputs,labals) in enumerate(trainDataLoader):
            inputs=inputs.to(device)
            labals=labals.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs =outputs.squeeze()

            #cal acc
            predict_labels = torch.round(outputs)
            acc = torch.eq(predict_labels, labals).float()
            acc=acc.mean()
            epoch_acc+=acc.item()

            #clip grad
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            #cal loss
            loss = criterion(outputs, labals.float())
            epoch_loss +=loss.item()
            loss.backward()

            # update optimizer
            optimizer.step()
            del inputs, labals, predict_labels #free some memory
      print("epoch is:",epoch,"\ttrain_loss is:",epoch_loss,"\ttrain_acc is:",epoch_acc)

path = './Model/sentimentAnalysisModel.pkl'
torch.save(model,path)


