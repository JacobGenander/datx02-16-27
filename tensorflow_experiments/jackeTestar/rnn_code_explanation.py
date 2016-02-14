'''
THIS IS NOT RUNNABLE CODE! 

I've tried to explain what happens in what seems to be the most
essential steps in creating a multi-layerd LSTM cell. The code pieces come from the
ptb_word_lm.py example given by TensorFlow. Some explanations may be totally wrong.

RNNCell is the most abstract representation of an RNN cell.
When instantiated, it takes a tensor of shape [batch_size, input_size]. This means that
in our case, it takes "batch_size" many words at once. Though RNNCell is not an abstract
class, we should probably use BasicLSTMCell, since it inherits RNNCell and implements the 
needed methods.

MultiLSTMCell takes a list of RNNCells as input and stacks them onto each other.
'''



' First we import tensorflow
import tensorflow as tf

class lstmModel
'''
We need to set the input size of the network to be the number of dimensions
in the embedded word vectors - one input per dimension.
'''
    def self.input_size = 100

'''
When the data has been processed in the graph, it should result in a new vector
of the same size as the input, so we define the output to be of just that size.
'''
    def self.output_size = 100

'''
When training, we use a batches of words from different article headlines. This is good
since it speeds up the training if we use GPUs, which are good at parall tasks. A batch
size of 100 means that we train on 100 headlines for each training step. It also gives
an advantage to do so, since this is a kind of stochastic optimisation (in contrast to
training on all headlines each time).
'''
    def self.batch_size = 20

'''
When we train our network, we can do so for a number of steps back in time, meaning that
we affect states in the previous steps. Given the output, we calculate the error and
update the weights accordingly to minimize the error. Doing so we get new weights, which
can in turn be updated with respect to the error for the output in the previous time step,
and so on. In this example, we choose to do this procedure for twenty time steps back in
back in time. (NO?)

This is the number of cells which can be seen "unrolled" in some figures.
'''
    def self.num_steps = 20

'''
When we train the network, we input data to train on. Each time step takes an input and we
do the training with "batch_size" many words at a time, so the input placeholder should be
a tensor of shape [batch_size, num_steps].
'''
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    
'''
When we train the network, we compare the output to some target value to get the error.
Therefore, the placeholder for the targets should be of the same size as that of the input 
data.
'''
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
'''
There are several networks (or is it the same for all gates?), one for each gate.
They consist of one layer of neurons each, and "hidden_size" refers to the number of
neurons in each of these layers.
'''
    hidden_size = 200

'''
Since we are creating a multi-layered LSTM cell (which consists of many "basic" cells, 
we must define how many layers should be stacked on top of each other.
'''
    num_layers = 2
    
'The number of words in the vocabulary.
    vocab_size = 10000
    
'''
Now we know everything we need to know in order to create an LSTM cell.
Again, the hidden_size is the number of neurons in the cell. The forget bias I
don't really know why it is set to 0.0.
'''
    basic_lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
    
'''
We can use the LSTM cell we just created to create a MultiRNNCell which consists of many
of these basic cells. We do so by listing 
'''
    multi_cell = rnn_cell.MultiRNNCell([basic_lstm_cell] * num_layers)
    
'''
We train on the vector representation of words, called word embedding. Each word is 
represented by a vector in "hidden_size" dimensions (rather, we used the number of
dimensions as the hidden size). Now we simply put words into a matrix with each row
representing a word and the columns is its corresponding dimensions. The row number
represents the word, e.g. 0 = cat, 422 = machine.
'''
    embedding = tf.get_variable("embedding", [vocab_size, hidden_size])

'''
To perform parallell lookups of a number's corresponding word in embeddings, we use the
function embedding_lookup provided by TensorFlow. The first argument is simply the table
of word vectors. The second argument is a list (a tensor) of numbers to look up in the 
table. We specify the _input_data, since it contains the input words for the whole batch.
'''
    inputs = tf.nn.embedding_lookup(embedding, self._input_data)

'''
Here they seem to initialize the state of all cells and fill the output...
#
#
#
#
#
#
'''
    outputs = []

'''
First, all output values are concatenated along dimension 1. The resulting tensor is then
reshaped using TensorFlows function "reshape". It a given tensor into another tensor with 
the same values as the given one, but with the shape of the second argument.
'''
    output = tf.reshape(tf.concat(1, outputs),  And [-1, size])

'''
Now, we need to create the weights and biases for each internal cell's network.
The prefix "softmax" is used to indicate that there is a softmax function applied on the
output of the (internal) network. There is one weight from each input to each word.
This way, we train the network to learn a function from vector space to the vocabulary.
Given a vector (representing a word), we want the network to predict the next word.

'''
    # We should actually be in a scope to use get_variable
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])

' There is one bias per output in the layer (in this internal network)
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    
'''
We multiply the output with the weights (softmax_w) and then add the bias (softmax_b).
This results in a list of z:s (logit is the input to the activation function).
'''
    # what is output here?
    logits = tf.matmul(output, softmax_w) + softmax_b
    
'''
We now have to define the loss ("the error" ?). Here we use the loss
function defined in the seq2seq example, which takes the logits, the targets, weights and
the number of decoder symbols (e.g. the number of words in the vocabulary) as arguments.
seq2seq.sequence_loss_by_example uses nn_ops.softmax_cross_entropy_with_logits.
Returns:
    1D batch-sized float Tensor: the log-perplexity for each sequence.
'''
# "weights: list of 1D batch-sized float-Tensors of the same length as logits." WHY?
# Shouldn't we use a 2D matrix of size [inputs x vocabulary]?
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)

'''
We use the loss we defined in order to define the actual function to be minimized.
'''
    self._cost = cost = tf.reduce_sum(loss) / batch_size

'''
The learning rate indicates how big steps we should take when adjusting the weights.
As it may change, but is not trainable itself, we create a variable with the parameter
trainable = false.
'''
    self._lr = tf.Variable(0.0, trainable=False)
      
' Get a handle to the list of all trainable variables in the graph
    tvars = tf.trainable_variables()
    
'''
?
'''
    # What is clipping?
    # Can't tf.gradients in the API. Is it equivalent to compute_gradients()?
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
                                      
'''
There are several algorithms one could use for optimization. We choose the algorithm
(here, Gradient descent), set its parameters (the learning rate).
We get a handle to the chosen optimizer object, here named optimizer.
'''
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    
'''
Usually, we would the "minimize()" function of our optimizer, but since there is some
modification to the gradients going on (clipping), we need to apply the gradients 
explicitly.
'''
    # Is tf.gradients() equivalent to compute_gradients()? If not, where is it called?
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))











    



