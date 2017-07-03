'''
MNIST is a set of 60,000 training examples of handwritten digits.
All of these are 28 x 28 pixels i.e. 784 total pixels.
We have 10,000 testing unique testing examples. 
Each feature is a pixel value over here for us.

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
This is what you mean by one_hot. It is similar to imagining 10 light bulbs numbered 0 to 9,
each one glowing up (becoming 1) when corresponding button is hit to switch it on
while all others will remain switched off (remain 0).

0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
5 = [0,0,0,0,0,1,0,0,0,0]
9 = [0,0,0,0,0,0,0,0,0,1] and so on.

'''
n_nodes_hl1 = 500 # Initialising number of nodes in three hidden layers.
n_nodes_hl2 = 500 # Try comparing accuracy results by altering number of nodes in layers.
n_nodes_hl3 = 500
n_output_classes = 10 # i.e. 0,1,2....,9
batch_size = 100 # Number of MNIST examples being taken up for training our model in one pass. Training the data on batches makes the model more generalized.

# We now flatten a matrix of 28x28 values into a single dimensional array with 784 values.
# Shapes of tensors are given as [height,width] where height denotes no. of rows and width denotes no. of columns.

x = tf.placeholder('float', [None, 784]) # Placeholders are used for delaying input of a variable, passing parameters as the datatype and the size.
y = tf.placeholder('float', [None, 10]) # These placeholders get their input values usually from feed_dict (Line 101)

# random_normal returns a tensor of the specified shape filled with random normal values.
# An important point to note here is that, here we're just learning a bit about coding but for research we'll keep the bias values equal to 1 to reduce randomisation.

def NN_model(data) :

	# Just like any Tensor, Variables created with Variable() can be used as inputs for other operations in the Graph.
    # In the code snippet given below, the first parameter denotes every node in the i-1 th layer.
    # The second parameter denotes the connection every node in i-1 th layer will have with every node in i th layer with weights assigned to every connection.
    # These weights are multiplied by the data values in i-1 th layer node and then the respective biases are added to product before passing through the activation function of i th layer. 

    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_output_classes])),
                    'biases':tf.Variable(tf.random_normal([n_output_classes]))}

    
    # We could easily have used a loop and avoid writing so writing these statements for every layer separately as this would become a tedious task when number of layers are more. 
    # However, I have stuck with not using that loop right now to concentrate on making NNs clearer and easy to understand by not making coding in Python difficult to read.
    # Basically doing the same old thing below, (weights x input) + biases

    layer1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases']) # matmul, quite obviously is used for matrix multiplication, whereas add is used for addition.
    layer1 = tf.nn.relu(layer1)

    # relu is Rectified Linear Unit, one of the many activation function (like sigmoid, tanh, arctan, Softmax), removes the negative part of function i.e. f(x) = { 0 for x<0, x for x>=0 }

    layer2 = tf.add(tf.matmul(layer1,hidden_layer_2['weights']), hidden_layer_2['biases']) 
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2,hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.matmul(layer3,output_layer['weights']) + output_layer['biases']

    return output

def train_NN(x) :
    prediction = NN_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) # prediction i.e. last layer which is the output predicted by our model.

    '''
    Softmax layer results in a separate probability for each of the ten digits, and the probabilities will all add up to 1.
    Cross entropy loss is usually used to determine error rate in a softmax layer.
    Logit is a function that maps probabilities to real numbers i.e. ([0,1]) to ((-infinity, +infinity))
    Cost(also called loss) to understand how much the predicted result deviates from the expected output
    Optimizer to minimize the cost
    Lower the learning rate, more is the time taken to learn to make accurate predictions ( i.e more training time) and therefore better results.

    '''
    optimizer = tf.train.AdamOptimizer().minimize(cost) # Default parameter for AdamOptimizer (just one of the many optimising techniques) is learning rate = 0.001
    #Try gradient descent optimiser for a different(better) result -> optimizer = tf.train.GradientDescentOptimizer().minimize(cost)
    
    n_epochs = 15  # Try comparing accuracy results by altering number of epochs i.e. one forward pass + one backward pass for all samples.
    with tf.Session() as sess: # In Tensorflow, a Graph defines the computational operations we specify in our code and a Session enables execution of graphs or its parts.
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0 # Reinitialising epoch loss to 0 for calculating loss for every new epoch.
            for _ in range(int(mnist.train.num_examples/batch_size)): # Underscore denotes a random variable that will iterate through the specified range, no other use of it.
                # No. of examples divided by batch size gives the no. of times the loop needs to run (iterate) to eventually complete working on all samples in one epoch.
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) # feed_dict passing value of x and y, Refer line 32 and 33 where we talked about placeholders.
                epoch_loss = epoch_loss + c # Store loss in epoch_loss, print it and then reinitialise it to 0 at the start of next epoch.

            print('Epoch', epoch+1, 'completed out of',n_epochs,'loss:',epoch_loss)

        # argmax returns index of largest value (index of 1 over 9 other 0's), here checking for equality between predicted & expected value.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) # 1st parameter is the output our model predicts for every input given and 2nd parameter is the expected output.
        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) # cast changes a variable (here being 'correct') to another data type (here, to float).
        # correct gives us a list of true or false values that are converted to float by cast and then taken mean. Eg- [False, False, True, False] -> [0,0,1,0] -> 0.25
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) # refer to the start where we initialised 'mnist'

train_NN(x)


