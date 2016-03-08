# Hyperparameters

# Data processing
BATCH_SIZE = 50 # Number of sentences we will train on each training step
MAX_EPOCH = 50 # This decides how many times we will run through the data
MAX_SEQ = 50 # This is the maxium length for a headline

# Network size
HIDDEN_LAYER_SIZE = 800 # Number of neurons in each layer
EMBEDDING_SIZE = 300 # Dimensions of our word vectors
NUMBER_OF_LAYERS = 2 # Layers of neurons in the network

# Learning rate and decay
LEARNING_RATE = 0.3 # Starter learning rate
LEARNING_DECAY = 1.0 # Exponential decay of the learning rate (1.0 = No learning decay)
DECAY_START = 10 # Learning decay starts after this epoch

# Misc
KEEP_PROB = 1.0 # Probability that an input/output is kept, needs to be in range (0, 1] (1 = No dropout)
INIT_RANGE = 0.3 # Initiate weights an biases within this range (-/+ init_range)

