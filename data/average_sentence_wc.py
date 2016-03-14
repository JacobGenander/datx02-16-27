"""
Calculate a rough estimate on the average word count in the article sentences. 
In our small test set this is 131.
"""
import nltk.data

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("texts.txt")
raw_data = fp.read()
tok_data = tokenizer.tokenize(data)
sents = len(tok_data)
alength = sum([len(s) for s in tok_data])/sent
print (alength)
print (sents)
