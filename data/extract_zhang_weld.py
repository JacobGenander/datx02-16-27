"""
Small script to extract data from the Zhang and Weld dataset.
The dataset can be downloaded from https://www.cs.washington.edu/node/9473/ .
"""

import json
import random
import string

sample_size = 100000
crawl_path = 'release/crawl'

"""
Import data and draw sample.
"""
data = []
with open(crawl_path) as data_file:    
    for line in data_file:
        data.append(json.loads(line))
sample = random.sample(data,sample_size)

"""
Format data and save to disk.
"""
articles = []
for d in sample:
    articles.append((' '.join(d.get('title').lower().split()).encode('ascii', 'ignore'),' '.join(d.get('text','no article').lower().split()).encode('ascii','ignore')))

titles = open("titles.txt",'w')
texts = open("texts.txt",'w')
for (title, text) in articles[:sample_size]:
    titles.write("%s\n" % title)
    texts.write("%s\n" % text)
