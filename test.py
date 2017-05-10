from python.neuralnet.layer import Layer, Dense, ActivityRegularization, Flatten
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    data = input_data.read_data_sets('/home/tanxr/data/mnist')
    X = data.train.images
    y = data.train.labels
    print(X.shape, y.shape)
    dense = Dense(64, input_shape=(784, ), activation='relu', use_bias=True,
                  kernel_initializer='truncated_normal', kernel_regularizer='l1', kernel_constraint='NonNeg')

    out = dense(X)
    print out.shape

    reg = ActivityRegularization(l1=0.02, l2=0.02)
    regularization = reg(out)

    x = X.reshape(X.shape[0], 28,28)
    fl = Flatten()
    print(fl(x).shape)
