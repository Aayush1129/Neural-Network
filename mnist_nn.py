import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n1 = 256
n2 = 256

learning_rate = 0.1
num_epoch = 500
batch_size = 128
input_shape = 784
output_shape = 10

X = tf.placeholder(tf.float32,shape = [None,input_shape])
Y = tf.placeholder(tf.float32,shape = [None, 10])

weights = {
    'h1' : tf.Variable(tf.random_normal([input_shape, n1])),
    'h2' : tf.Variable(tf.random_normal([n1, n2])),
    'h_out' : tf.Variable(tf.random_normal([n2, output_shape]))
}

bias = {
    'b1' : tf.Variable(tf.random_normal([n1])),
    'b2' : tf.Variable(tf.random_normal([n2])),
    'b_out' : tf.Variable(tf.random_normal([output_shape]))
}

def neural_network(x):
    y1 = tf.add(tf.matmul(X, weights['h1']), bias['b1'])
    y2 = tf.add(tf.matmul(y1, weights['h2']), bias['b2'])
    y3 = tf.add(tf.matmul(y2, weights['h_out']), bias['b_out'])
    return y3

logits = neural_network(X)
pred = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        num_iteration = int(mnist.train.num_examples/batch_size)
        avg_cost = 0
        for step in range(num_iteration):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict = {X:batch_x, Y:batch_y})
            avg_cost+=c/num_iteration
            train_accur = sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y})
        print('epoch: ', '%d'%(epoch+1), 'cost: ', '%.4f'%(avg_cost), 'Train accuracy: ', '{:.4f}'.format(train_accur))

    accur= sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels})
    print('Final accuracy: ', '{:.9f}'.format(accur))








