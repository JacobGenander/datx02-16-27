# Hyperparameters

# Data processing
batch_size = 50 # Number of sentences we will train on each training step
max_epoch = 50 # This decides how many times we will run through the data

# Network size
hidden_layer_size = 800 # Number of neurons in each layer
embedding_size = 200 # Dimensions of our word vectors
number_of_layers = 2 # Layers of neurons in the network

# Learning rate and decay
learning_rate = 0.03 # Starter learning rate
learning_decay = 1.0 # Exponential decay of the learning rate (1.0 = No learning decay)
decay_start = 10 # Learning decay starts after this epoch

# Misc
dropout = 0.0 # Probability that an input/output is kept, needs to be in range [0, 1) (0 = No dropout)
init_range = 1 # Initiate weights an biases within this range (-/+ init_range)

