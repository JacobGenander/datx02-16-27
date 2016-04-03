# Dictionary holding all hyperparameters
config = {}

# Data processing
config["batch_size"] = 50 # Number of sentences we will train on each training step
config["max_epoch"] = 50 # This decides how many times we will run through the data
config["num_steps"] = 50 # This is the maxium length for a headline

# Network size
config["hidden_layer_size"] = 128 # Number of neurons in each layer
config["number_of_layers"] = 2 # Layers of neurons in the network

# Learning rate and decay
config["learning_rate"] = 0.002 # Starter learning rate
config["learning_decay"] = 0.97 # Exponential decay of the learning rate (1.0 = No learning decay)
config["decay_start"] = 10 # Learning decay starts after this epoch

# Gradients
config["gradient_clip"] = 5

# Other network properties
config["keep_prob"] = 1.0 # Probability that an input/output is kept, needs to be in range (0, 1] (1 = No dropout)
config["init_range"] = 0.08 # Initiate weights an biases within this range (-/+ init_range)

