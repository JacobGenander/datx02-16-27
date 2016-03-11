# Dictionary holding all hyperparameters
config = {}

# Data processing
config["batch_size"] = 50 # Number of sentences we will train on each training step
config["max_epoch"] = 50 # This decides how many times we will run through the data
config["max_seq"] = 50 # This is the maxium length for a headline

# Network size
config["hidden_layer_size"] = 800 # Number of neurons in each layer
config["embedding_size"] = 300 # Dimensions of our word vectors
config["number_of_layers"] = 2 # Layers of neurons in the network

# Learning rate and decay
config["learning_rate"] = 0.3 # Starter learning rate
config["learning_decay"] = 1.0 # Exponential decay of the learning rate (1.0 = No learning decay)
config["decay_start"] = 10 # Learning decay starts after this epoch

# Misc
config["keep_prob"] = 1.0 # Probability that an input/output is kept, needs to be in range (0, 1] (1 = No dropout)
config["init_range"] = 0.3 # Initiate weights an biases within this range (-/+ init_range)

