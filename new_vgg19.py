import tensorflow as tf
import numpy as np
import cv2
import caffe_classes


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

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
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)

class VGG19(object):
    def __init__(self):

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.KEEPPRO = 1
            self.CLASSNUM = 1000
            self.SKIP = []
            self.MODELPATH = "vgg19.npy"

            self.X = tf.placeholder("float", [1, 224, 224, 3])
            self.conv1_1 = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1" )
            self.conv1_2 = convLayer(self.conv1_1, 3, 3, 1, 1, 64, "conv1_2")
            # (224, 224, 64) 3211264
            self.pool1 = maxPoolLayer(self.conv1_2, 2, 2, 2, 2, "pool1")
            # (112, 112, 64)  802816

            self.conv2_1 = convLayer(self.pool1, 3, 3, 1, 1, 128, "conv2_1")
            self.conv2_2 = convLayer(self.conv2_1, 3, 3, 1, 1, 128, "conv2_2")
            # (112, 112, 128)  1605632
            self.pool2 = maxPoolLayer(self.conv2_2, 2, 2, 2, 2, "pool2")
            # (56, 56, 128)  401408

            self.conv3_1 = convLayer(self.pool2, 3, 3, 1, 1, 256, "conv3_1")
            self.conv3_2 = convLayer(self.conv3_1, 3, 3, 1, 1, 256, "conv3_2")
            self.conv3_3 = convLayer(self.conv3_2, 3, 3, 1, 1, 256, "conv3_3")
            self.conv3_4 = convLayer(self.conv3_3, 3, 3, 1, 1, 256, "conv3_4")
            # (56, 56, 256) 802816
            self.pool3 = maxPoolLayer(self.conv3_4, 2, 2, 2, 2, "pool3")
            # (28, 28, 256) 200704

            self.conv4_1 = convLayer(self.pool3, 3, 3, 1, 1, 512, "conv4_1")
            self.conv4_2 = convLayer(self.conv4_1, 3, 3, 1, 1, 512, "conv4_2")
            self.conv4_3 = convLayer(self.conv4_2, 3, 3, 1, 1, 512, "conv4_3")
            self.conv4_4 = convLayer(self.conv4_3, 3, 3, 1, 1, 512, "conv4_4")
            # (28, 28, 512)  401408
            self.pool4 = maxPoolLayer(self.conv4_4, 2, 2, 2, 2, "pool4")
            # (14, 14, 512)  100352

            self.conv5_1 = convLayer(self.pool4, 3, 3, 1, 1, 512, "conv5_1")
            self.conv5_2 = convLayer(self.conv5_1, 3, 3, 1, 1, 512, "conv5_2")
            self.conv5_3 = convLayer(self.conv5_2, 3, 3, 1, 1, 512, "conv5_3")
            self.conv5_4 = convLayer(self.conv5_3, 3, 3, 1, 1, 512, "conv5_4")
            # (14, 14, 512)  100352
            self.pool5 = maxPoolLayer(self.conv5_4, 2, 2, 2, 2, "pool5")
            # (7, 7, 512)  25088

            self.fcIn = tf.reshape(self.pool5, [-1, 7*7*512])
            # 25088
            self.fc6 = fcLayer(self.fcIn, 7*7*512, 4096, True, "fc6")
            # 4096
            self.dropout1 = dropout(self.fc6, self.KEEPPRO)

            self.fc7 = fcLayer(self.dropout1, 4096, 4096, True, "fc7")
            # 4096
            self.dropout2 = dropout(self.fc7, self.KEEPPRO)

            self.fc8 = fcLayer(self.dropout2, 4096, self.CLASSNUM, True, "fc8")
            # number of classes

            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)


    def initialze(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())


    def loadModel(self):
        with self.sess.as_default():
            with self.graph.as_default():

                wDict = np.load(self.MODELPATH, encoding = "bytes").item()
                #for layers in model
                for name in wDict:
                    if name not in self.SKIP:
                        with tf.variable_scope(name, reuse = True):
                            for p in wDict[name]:
                                if len(p.shape) == 1:
                                    #bias
                                    self.sess.run(tf.get_variable('b', trainable = False).assign(p))
                                else:
                                    #weights
                                    self.sess.run(tf.get_variable('w', trainable = False).assign(p))

    def extract_conv1(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                #output = tf.reshape(self.pool1, [-1, 802816]) # 112, 112, 64
                feature = self.sess.run(self.pool1, {self.X:reshaped})
                return feature

    def extract_conv1_reshaped(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                output = tf.reshape(self.pool1, [-1, 802816]) # 112, 112, 64
                feature = self.sess.run(output, {self.X:reshaped})
                return feature

    def extract_conv2(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                #output = tf.reshape(self.pool2, [-1, 401408]) # 56, 56, 128
                feature = self.sess.run(self.pool2, {self.X:reshaped})
                return feature

    def extract_conv2_reshaped(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                output = tf.reshape(self.pool2, [-1, 401408]) # 56, 56, 128
                feature = self.sess.run(output, {self.X:reshaped})
                return feature

    def extract_conv3(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                #output = tf.reshape(self.pool3, [-1, 200704]) # 28, 28, 256
                feature = self.sess.run(self.pool3, {self.X:reshaped})
                return feature

    def extract_conv3_reshaped(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                output = tf.reshape(self.pool3, [-1, 200704]) # 28, 28, 256
                feature = self.sess.run(output, {self.X:reshaped})
                return feature

    def extract_conv4(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                #output = tf.reshape(self.pool4, [-1, 100352]) # 14, 14, 512
                feature = self.sess.run(self.pool4, {self.X:reshaped})
                return feature

    def extract_conv4_reshaped(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                output = tf.reshape(self.pool4, [-1, 100352]) # 14, 14, 512
                feature = self.sess.run(output, {self.X:reshaped})
                return feature

    def extract_conv5(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3)) # 25088
                feature = self.sess.run(self.pool5, {self.X:reshaped})
                return feature

    def extract_conv5_reshaped(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3)) # 25088
                feature = self.sess.run(self.fcIn, {self.X:reshaped})
                return feature

    def get_result(self, img):
        with self.sess.as_default():
            with self.graph.as_default():
                resized = cv2.resize(img.astype(np.float), (224, 224))
                reshaped = resized.reshape((1, 224, 224, 3))
                softmax = tf.nn.softmax(self.fc8)
                maxx = np.argmax(self.sess.run(softmax, {self.X: reshaped}))
                res = caffe_classes.class_names[maxx]
                return res

    def save(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, "./vgg_models/vgg_model.ckpt")

    def restore(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, "./vgg_models/vgg_model.ckpt")