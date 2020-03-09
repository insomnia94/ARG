import tensorflow as tf
import numpy as np
from parameter import *

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1], strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

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

class Actor(object):
    def __init__(self, n_actions, lr):
        self.lr = lr

        self.graph = tf.Graph()

        with self.graph.as_default():
            #self.s = tf.placeholder(tf.float32, [None, STATE_SIZE_X, STATE_SIZE_Y, 6], "state")
            #self.s = tf.placeholder(tf.float32, [None, 7, 7, 1024], "state")
            self.s = tf.placeholder(tf.float32, [None, 7, 7, 1536], "state")
            self.a = tf.placeholder(tf.int32, [None, 1], "act")
            self.td_error = tf.placeholder(tf.float32, [None, 1], "td_error")  # TD_error


            #self.conv1_1 = convLayer(self.s, 3, 3, 1, 1, 64, "conv1_1")
            #self.conv1_2 = convLayer(self.conv1_1, 3, 3, 1, 1, 64, "conv1_2")
            #self.pool1 = maxPoolLayer(self.conv1_2, 2, 2, 2, 2, "pool1")

            #self.fcIn = tf.reshape(self.pool1, [-1, STATE_SIZE_X * STATE_SIZE_Y * 16])
            #self.fcIn = tf.reshape(self.s, [-1, 7 * 7 * 1024])
            self.fcIn = tf.reshape(self.s, [-1, 7 * 7 * 1536])

            self.fc1 = tf.layers.dense(
                inputs=self.fcIn,
                units=512,
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

            self.acts_prob = tf.layers.dense(
                inputs=self.fc2,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

            #log_prob = tf.log(self.acts_prob[0, self.a])

            prob_ = self.acts_prob[0]
            a_ = self.a[0][0]
            prob_a = prob_[a_]
            prob_final = tf.reshape(prob_a, [1, 1])

            for i in range(1, RECORD_NUM):
                prob_ = self.acts_prob[i]
                a_ = self.a[i][0]
                prob_a = prob_[a_]
                prob_a_ = tf.reshape(prob_a, [1, 1])
                prob_final = tf.concat([prob_final, prob_a_], 0)

            log_prob = tf.log(prob_final)

            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss
            # exp_v: (1) -> log probability * reward(using TD error here)

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

            self.sess = tf.Session(graph=self.graph)

            self.saver = tf.train.Saver()

    def initialze(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, td):
        with self.sess.as_default():
            with self.graph.as_default():
                #s = s[np.newaxis, :]    # (4,) => (1,4)
                feed_dict = {self.s: s, self.a: a, self.td_error: td}
                _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
                return exp_v

    def get_value(self, node, s):
        with self.sess.as_default():
            with self.graph.as_default():
                value = self.sess.run(node, {self.s:s})
                return value

    def get_probs(self, s):
        with self.sess.as_default():
            with self.graph.as_default():
                #s = s[np.newaxis, :]  # (4,) => (1,4)
                # acts_prob: (?, 2)
                probs = self.sess.run(self.acts_prob, {self.s: s})
                return probs

    def choose_action(self, s):
        with self.sess.as_default():
            with self.graph.as_default():
                #s = s[np.newaxis, :]    # (4,) => (1,4)
                # acts_prob: (?, 2)
                probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
                # probs: (?, 2)
                # probs.shape => tuple, 0:1, 1:2 (0 represents row and 1 represents column)
                probs_shape = probs.shape[1]
                # probs_shape => 2 (number of columns)
                probs_arrange = np.arange(probs_shape)
                # probs_arrange => [0,1]
                probs_ = probs.ravel()
                # probs: (?, 2) -> (1, 2)
                # probs_: (2,)
                p = np.random.choice(probs_arrange, p=probs_)   # get probabilities for all actions
                return p   # return a int

    def save(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, "./actor_models/actor_model.ckpt")

    def restore(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, "./actor_models/actor_model.ckpt")

    def save_learning_rate(self):
        f = open("./actor_models/actor_lr", "w")
        f.write(str(self.lr))

    def restore_learning_rate(self):
        f =  open("./actor_models/actor_lr", "r")
        self.lr = float(f.read())
