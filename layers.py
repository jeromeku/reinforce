import tensorflow as tf

def MLP(states, input_dim, hidden_dim, output_dim, output_fn=tf.nn.softmax, num_layers=2, activation=tf.nn.relu, initializer=tf.random_normal_initializer):

    W1 = tf.get_variable("W1", [input_dim, hidden_dim], initializer=initializer())
    h1 = activation(tf.matmul(states, W1))

    W2 = tf.get_variable("W2", [hidden_dim, output_dim], initializer=initializer())

    logits = tf.matmul(h1,W2)
    prob = output_fn(logits)

    return logits, prob
