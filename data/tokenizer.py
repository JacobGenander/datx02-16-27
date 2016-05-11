#!/bin/
from __future__ import division  # Python 2 users only
import nltk, re, pprint
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import json
import random
import string #maybe not needed
import argparse

# OPEN A JSON FILE AND TOKENIZE IT
#-------------------------------------------
# make a local copy of the crawled data file
#nltk.data.retrieve('/Users/Jacob/Documents/Data/NewsSpikesArticles/release/crawl',
#                   'crawled_copy.json')

# number of articles to be drawn from the data
sample_size = 100

crawl_path = '/mnt/WD_Storage/ml_data/release/crawl'

def extract_from_json(filename):
    articles = []
    titles = []
    with open(filename) as data_file:
        for line in data_file:
            json_obj = json.loads(line)
            #art_id = json_obj.get('articleId')
            article = json_obj.get('text')
            title = json_obj.get('title')
            if article is not None and len(article) > 10:
                articles.append(article)
                titles.append(title)
            #else:
                #print("%d:\t%s" % (art_id, title))
                #print(json_obj)
                #raw_input()
            #data.append(json.loads(line))

    print "Done reading file"
    print "%d article-title pairs" % len(articles)

    pairs = zip(titles, articles)
    return pairs


parser = argparse.ArgumentParser(description='Extract headlines and articles from json file')
parser.add_argument('--size', metavar='amount', type=int, help='how many title-article pairs to extract')
parser.add_argument('json_file', type=str, help="path to json file")
parser.add_argument('--article_file', type=str, help="path to json file", default="articles")
parser.add_argument('--title_file', type=str, help="path to json file", default="titles")
arguments = parser.parse_args()

sample_size = arguments.size
filename = arguments.json_file
title_file = arguments.title_file
article_file = arguments.article_file

pairs = extract_from_json(filename)
# draw a sample from the data
print "Drawing sample of %d pairs" % sample_size
sample_pairs = random.sample(pairs, sample_size)

print "Opening files for writing"
titles = open("%s_%d.txt" % (title_file, sample_size), 'w')
articles = open("%s_%d.txt" % (article_file, sample_size), 'w')

print "Writing..."
for (tit, art) in sample_pairs:
    titles.write("%s\n" % tit.encode("ascii", "ignore").replace("\n", " ").lower())
    articles.write("%s\n" % art.encode("ascii", "ignore").replace("\n", " ").lower())

titles.close()
articles.close()
print "Done!"



#for (title, text) in articles[:sample_size]:


# create the regular expression tokenizer
#tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

#tokenized_samples = [
#    (tokenizer.tokenize(tit), tokenizer.tokenize(art))
#    for (tit, art) in sample
#]



#articles = []
#for d in sample:
#    headline_tokens = tokenizer.tokenize(d.get('title').lower())
#    article_tokens = tokenizer.tokenize(d.get('text','no article').lower())
#    articles.append((headline_tokens, article_tokens))
#
#
#
##for line in f:
##    print(line.strip()) # strip() removes the newline character at the end of the line
