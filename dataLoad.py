import nltk.data
from dataProcessUtil import getWordIndxMap,getEncodedSentences,pad_features,parseSentences
import pandas as pd
import numpy as np

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Read data from files
train = pd.read_csv( "./Data/labeledTrainData.tsv.zip", header=0,  delimiter="\t", quoting=3 )
# unlabeled_train = pd.read_csv( "./Data/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "./Data/testData.tsv.zip", header=0, delimiter="\t", quoting=3 )

trainLabels=train.loc[:,"sentiment"].values

print("Read %d labeled train reviews, %d labeled test reviews, "  % (train["review"].size, test["review"].size))

words = []
reviewLenSum=0

print("Parsing sentences from training set..")
sentencesTrain,parse_words_train,reviewLenCount=parseSentences(train["review"],tokenizer)
words+=parse_words_train
reviewLenSum+=reviewLenCount

print("Parsing sentences from test set..")
sentencesTest,parse_words_test,reviewLenCount=parseSentences(test["review"],tokenizer)
words+=parse_words_test
reviewLenSum+=reviewLenCount

avgReviewLen=round(reviewLenSum/(len(sentencesTrain)+len(sentencesTest)))
print("average sentence length: ",avgReviewLen)
print("total number of sentences: ",len(sentencesTrain)+len(sentencesTest))

idx2Word,word2Idx=getWordIndxMap(words)

print("transform sentences to encodedFeature")
encodedTrain = getEncodedSentences(word2Idx,sentencesTrain)
encodedTest = getEncodedSentences(word2Idx,sentencesTest)

encodedTrainFeatures = pad_features(encodedTrain, pad_id=word2Idx['<PAD>'], seq_length=avgReviewLen)
encodedTestFeatures = pad_features(encodedTest, pad_id=word2Idx['<PAD>'], seq_length=avgReviewLen)

np.save('./Data/encodedTrainFeatures.npy',encodedTrainFeatures)
np.save('./Data/encodedTestFeatures.npy',encodedTestFeatures)

print("wordTotalNum is:",len(set(words))+1)