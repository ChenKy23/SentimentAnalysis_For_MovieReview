from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
import numpy as np


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review,features="html.parser").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    words_of_sentences = []
    len_of_review =0
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            words = review_to_wordlist(raw_sentence, remove_stopwords)
            words_of_sentences+=words
            len_of_review+=len(words)
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return words_of_sentences,len_of_review

def getWordIndxMap(words):
    #word count
    wordCounter = Counter(words)
    #get vocab sorted by frequency
    vocab = sorted(wordCounter, key=wordCounter.get, reverse=True)
    #map words to index
    idx2Word = dict(enumerate(vocab, 1))
    idx2Word[0] = '<PAD>'
    # map index to word
    word2Idx = {word: id for id, word in idx2Word.items()}

    return idx2Word,word2Idx

def getEncodedSentences(word2Idx,sentences):
    encodedSentences = [[word2Idx[word] for word in sentence] for sentence in tqdm(sentences)]
    return encodedSentences

def pad_features(sentences, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(sentences), seq_length), pad_id, dtype=int)

    for i, sentence in enumerate(sentences):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(sentence)] = sentence[:seq_length]

    return features

def parseSentences(reviews,tokenizer):
    reviewLenCount=0;
    parse_words=[]
    parse_sentences=[]
    for review in tqdm(reviews):
        words_of_sentence, len_of_review = review_to_sentences(review, tokenizer, True)
        parse_sentences.append(words_of_sentence)
        parse_words += words_of_sentence
        reviewLenCount += len_of_review

    return parse_sentences,parse_words,reviewLenCount