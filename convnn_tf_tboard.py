import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def get_bias(bias_shape):
    b= tf.get_variable('bias', bias_shape, tf.float32,
            initializer= tf.constant_initializer(np.zeros(bias_shape)))
    return b

def get_weight(kernel_shape):
    W= tf.get_variable('weights', kernel_shape, tf.float32,
            initializer= tf.contrib.layers.xavier_initializer())
    return W

def conv2d(inp, W, b, relu= True, bn= False, is_training= True):
    conv= tf.nn.conv2d(inp, W, strides= [1,1,1,1], padding='SAME')

    conv= tf.nn.bias_add(conv,b)

    if bn:
        conv = tf.layers.batch_normalization(conv, training= is_training)
    if relu:
        conv = tf.nn.relu(conv)

    return conv

def pool(inp, kernel_size, strides):
    pool= tf.nn.max_pool(inp, kernel_size, strides, padding='SAME')
    return pool

def fully_connected(inp, W, b, reshape= None, relu=True, bn=True, is_training= True):
    if reshape is not None:
        inp= tf.reshape(inp, reshape)

    fc= tf.matmul(inp, W) + b

    if bn:
        fc= tf.layers.batch_normalization(fc, training= is_training)
    if relu:
        fc= tf.nn.relu(fc)

    return fc

def build_graph(input_shape, is_training= True):
    x_= tf.placeholder('float32', (None,input_shape))
    y_= tf.placeholder('int64', (None,))

    x_pre= tf.reshape(x_, (-1,28,28,1))

    with tf.variable_scope('conv1'):
        W= get_weight((5,5,1,32))
        b= get_bias((32))
        x= conv2d(x_pre, W, b, relu= True, bn= True, is_training= is_training)

    with tf.variable_scope('pool1'):
        x= pool(x, (1,2,2,1), (1,2,2,1))

    with tf.variable_scope('conv2'):
        W= get_weight((5,5,32,64))
        b= get_bias((64))
        x= conv2d(x, W, b, relu= True, bn= True, is_training= is_training)

    with tf.variable_scope('pool2'):
        x= pool(x, (1,2,2,1), (1,2,2,1))

    with tf.variable_scope('fc1'):
        W= get_weight((7*7*64,1024))
        b= get_bias((1024))
        x= fully_connected(x, W, b, reshape= (-1, 7*7*64), relu= True, bn= True, is_training= is_training)

    with tf.variable_scope('fc2'):
        W= get_weight((1024,10))
        b= get_bias((10))
        y= fully_connected(x, W, b, relu= False, bn= False)

    return y, x_, y_


def main():
    # Load MNIST dataset
    mnist= input_data.read_data_sets('MNIST_data')

    # Build Graph
    input_shape= 784
    with tf.variable_scope('model'):
        train_logits, x_, y_= build_graph(input_shape, is_training= True)
        with tf.name_scope('loss'):
            loss= tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels= y_, logits= train_logits))
            tf.summary.scalar('train_loss', loss)

    train_step= tf.train.AdamOptimizer(1e-4).minimize(loss)
    train_accuracy= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_logits, 1), y_),
            tf.float32))
    tf.summary.scalar('train_accuracy', train_accuracy)

    with tf.variable_scope('model', reuse= True):
        val_logits, x_val, y_val= build_graph(input_shape, is_training= False)
    val_accuracy= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_logits, 1), y_val),
            tf.float32))
#    tf.summary.scalar('val_accuracy', val_accuracy)

    merged= tf.summary.merge_all()
    # Begin Training Loop
    bs= 20
    max_iters= 100
    val_iters= 10
    with tf.Session() as sess:
        train_writer= tf.summary.FileWriter('summaries/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for it in range(max_iters):
            batch= mnist.train.next_batch(bs)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _, loss_, acc_= sess.run([merged, train_step, loss, train_accuracy], feed_dict={
                x_: batch[0],
                y_: batch[1]},
                options= run_options,
                run_metadata= run_metadata)
            train_writer.add_summary(summary, it)

            print('Training step ', it, ' loss is ', loss_ , ' train accuracy is ', acc_)

            if it%val_iters==0:
                v_acc= sess.run(val_accuracy, feed_dict={
                    x_val: mnist.validation.images,
                    y_val: mnist.validation.labels})
                print('validation accuracy is ', v_acc)

    # Test
    t_acc= sess.run(val_accuracy, feed_dict={
        x_val: mnist.test.images,
        y_val: mnist.test.labels})
    print('Final test accuracy is ', t_acc)

if __name__=="__main__":
    main()
