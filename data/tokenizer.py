from __future__ import division  # Python 2 users only
import nltk, re, pprint
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import json
import random
import string #maybe not needed

# OPEN A JSON FILE AND TOKENIZE IT
#-------------------------------------------
# make a local copy of the crawled data file    
#nltk.data.retrieve('/Users/Jacob/Documents/Data/NewsSpikesArticles/release/crawl', 
#                   'crawled_copy.json')

# number of articles to be drawn from the data
sample_size = 100

crawl_path = '/Users/Jacob/Documents/Data/NewsSpikesArticles/release/crawl'

data = []
with open(crawl_path) as data_file:    
    for line in data_file:
        data.append(json.loads(line))
        
# draw a sample from the data 
sample = random.sample(data,sample_size)

# create the regular expression tokenizer
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

articles = []
for d in sample:
    headline_tokens = tokenizer.tokenize(d.get('title').lower())
    article_tokens = tokenizer.tokenize(d.get('text','no article').lower())
    articles.append((headline_tokens, article_tokens))
    
titles = open("titles.txt",'w')
texts = open("texts.txt",'w')
for (title, text) in articles[:sample_size]:
    titles.write("%s\n" % titleTest)
    texts.write("%s\n" % textTest)
    
    
#for line in f:
#    print(line.strip()) # strip() removes the newline character at the end of the line