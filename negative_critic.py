import tensorflow as tf
import numpy as np
from parameter import *

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1], strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    return tf.nn.dropout(x, keepPro, name)

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME"):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
        #return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)
        return tf.nn.relu(out, name = scope.name)


class Negative_Critic(object):
    def __init__(self, lr):
        self.lr = lr

        self.graph = tf.Graph()

        with self.graph.as_default():

            #self.s = tf.placeholder(tf.float32, [None, STATE_SIZE_X, STATE_SIZE_Y, 6], "state")
            #self.s = tf.placeholder(tf.float32, [None, 7, 7, 1024], "state")
            self.s = tf.placeholder(tf.float32, [None, 7, 7, 1536], "state")
            self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
            self.r = tf.placeholder(tf.float32, [None, 1], 'r')

            #self.conv1_1 = convLayer(self.s, 3, 3, 1, 1, 64, "conv1_1")
            #self.conv1_2 = convLayer(self.conv1_1, 3, 3, 1, 1, 64, "conv1_2")
            #self.pool1 = maxPoolLayer(self.conv1_2, 2, 2, 2, 2, "pool1")

            #self.fcIn = tf.reshape(self.pool1, [-1, STATE_SIZE_X * STATE_SIZE_Y * 16])
            #self.fcIn = tf.reshape(self.s, [-1, 7 * 7 * 1024])
            self.fcIn = tf.reshape(self.s, [-1, 7 * 7 * 1536])

            self.fc1 = tf.layers.dense(
                inputs=self.fcIn,
                units=512,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., 0.01),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='fc1'
            )

            self.fc2 = tf.layers.dense(
                inputs=self.fc1,
                units=512,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., 0.01),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='fc2'
            )

            self.v = tf.layers.dense(
                inputs=self.fc2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.sess = tf.Session(graph=self.graph)

            self.saver = tf.train.Saver()

    def initialze(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

    def get_v(self, s):
        v = self.sess.run(self.v, {self.s: s})
        return v

    def get_value(self, node, s):
        with self.sess.as_default():
            with self.graph.as_default():
                value = self.sess.run(node, {self.s:s})
                return value

    def learn(self, s, r, s_):
        with self.sess.as_default():
            with self.graph.as_default():
                # s (state,)    s_ (state,)
                # => (?, state_size) s_ (?, state_size)
                #s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

                # s and s_ belong to this function(pass from outside), but self.s belongs to the class(object)

                # v_ is the return value of the next state

                v_ = self.sess.run(self.v, {self.s: s_})
                td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
                return td_error

    def save(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, "./negative_critic_models/critic_model.ckpt")

    def restore(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, "./negative_critic_models/critic_model.ckpt")

    def save_learning_rate(self):
        f = open("./negative_critic_models/critic_lr", "w")
        f.write(str(self.lr))

    def restore_learning_rate(self):
        f = open("./negative_critic_models/critic_lr", "r")
        self.lr = float(f.read())