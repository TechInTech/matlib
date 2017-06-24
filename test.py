import numpy as np
from autograd import elementwise_grad
from tensorflow.examples.tutorials.mnist import input_data

from python.neuralnet.layer import (Activation, ActivityRegularization, Dense,
                                    Dropout, Flatten, Layer)
from python.neuralnet.losses import mean_squared_error,categorical_crossentropy
from sklearn.preprocessing import OneHotEncoder
if __name__ == '__main__':
    data = input_data.read_data_sets('/home/tanxr/data/mnist', one_hot=True)
    X = data.train.images
    y = data.train.labels
    # y = OneHotEncoder(n_values=10).fit_transform(y)
    print(X.shape, y.shape)
    dense = Dense(32, input_shape=(784, ), activation='relu', use_bias=True,
                  kernel_initializer='truncated_normal', kernel_regularizer='l1', kernel_constraint='NonNeg')

    out = dense(X)
    print out.shape

    relu = Activation('relu')
    out = relu(out)

    dense10 = Dense(10)
    out = dense10(out)
    print out.shape

    softmax = Activation('softmax')
    out = softmax(out)
    print out[0], out.shape

    loss_grad = elementwise_grad(categorical_crossentropy)
    print loss_grad(out, y)
    
    # reg = ActivityRegularization(l1=0.02, l2=0.02)
    # regularization = reg(out)

    # x = X.reshape(X.shape[0], 28,28)
    # fl = Flatten()
    # print(fl(x).shape)

    # drop = Dropout(0.2)
    # x = drop(X)
    # print len(x) , len(X)
