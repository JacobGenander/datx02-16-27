"""
 Load data into redis and provide access.

 This unecessary loads all vectors, we only need the
 ones in the vocabulary. 

 To get a vector of string w use 
	data.lookup[w]
 To get nearest neighburs use
	data.engine.neighours(vec)

 NearPy is installed by
	pip install NearPy
 and the redis server is a tarball that can be found
 at redis.io

 We want to replace both of those with tensorflow.
"""

import numpy as np
from redis import Redis
from nearpy.storage import RedisStorage
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

# Create redis storage adapter
redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

# Get hash config from redis
config = redis_storage.load_hash_configuration('MyHash')

if config is None:
    # Config is not existing, create hash from scratch, with 10 projections
    lshash = RandomBinaryProjections('MyHash', 10)
else:
    # Config is existing, create hash with None parameters
    lshash = RandomBinaryProjections(None, None)
    # Apply configuration loaded from redis
    lshash.apply_config(config)

# Create engine for feature space of 100 dimensions and use our hash.
# This will set the dimension of the lshash only the first time, not when
# using the configuration loaded from redis. Use redis storage to store
# buckets.
engine = Engine(100, lshashes=[lshash], storage=redis_storage)

# Dimension of our vector space
dimension = 100

# Read glove vectors and store in lookup table and NearPy engine
lookup = {}
with open("glove.6B.100d.txt") as f:
    for line in f:
       data = line.split()
       key = data[0]
       value =np.array(map(float, data[1:]))
       lookup[key] = value
       engine.store_vector(value, key)

def get_engine():
    return engine
def get_lookup():
    return lookup
def close_connection():
    # Finally store hash configuration in redis for later use
    redis_storage.store_hash_configuration(lshash)
def flush_redis():
    # Empty the database and free ram.
    r_server.flushall()
