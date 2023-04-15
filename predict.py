import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

path = './Model/sentimentAnalysisModel.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encodedTestFeatures = np.load("./Data/encodedTestFeatures.npy")
test = pd.read_csv( "./Data/testData.tsv.zip", header=0, delimiter="\t", quoting=3 )
# test
model=torch.load(path)
model.eval()
test_pre_labels=[]
bg=0
batchs=200
testDataSet=TensorDataset(torch.from_numpy(encodedTestFeatures))
testDataLoader=DataLoader(dataset=testDataSet,shuffle=False,batch_size=batchs)
with torch.no_grad():
      for i,data in enumerate(testDataLoader):
            print("i=",i,"\tinputSize=",data[0].shape)
            data=data[0].to(device)
            pre_labels = model(data)
            test_pre_labels += pre_labels.cpu()
            torch.cuda.empty_cache()

test_pre_labels=[1 if i >0.5 else 0 for i in test_pre_labels]
out = pd.DataFrame(data={"id":test["id"], "sentiment":test_pre_labels})
out.to_csv( "test_pre_res.csv", index=False, quoting=3)