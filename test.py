#coding: utf-8
import numpy as np
from linearModel import LinearRegression, LogisticRegression, SoftMaxRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification, load_iris
from sklearn.model_selection import train_test_split
from metrics import mean_squared_error
from mlp import MLP
from decomposition import PCA, LDA, TSNE
from gmm import GMM
from kmeans import KMeans
from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(precision=4)

def regression():
    # Generate a random regression problem
    X, y = make_regression(n_samples=10000, n_features=100,
                           n_informative=75, n_targets=1, noise=0.01,bias=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = LinearRegression(lr=0.001, max_iters=3000, alpha=0.03, descent_grad=None)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # print(np.sort(np.abs(predictions.ravel() - y_test))[0:50])
    print('regression mse', mean_squared_error(y_test, predictions))



def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=100,
                               n_informative=75, random_state=1111,
                               n_classes=2, class_sep=2.5, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=1111)

    model = LogisticRegression(lr=0.01, max_iters=1000, penalty='l1', alpha=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(np.clip(predictions[:20],0.001,0.9), y_test[:20])
    # print('classification accuracy', accuracy(y_test, predictions))

def mlp():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=100,
                               n_informative=75, random_state=1111,
                               n_classes=2, class_sep=2.5, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=1111)

    # X_train=np.array([[3,3], [4,3], [1,1]])
    # y_train=np.array([1,1,-1])
    model = MLP(lr=0.05, max_iters=5000, torlerance=1e-5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('classification accuracy', np.mean(y_test == predictions))

def softmax():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=100,
                               n_informative=75, random_state=1111,
                               n_classes=10, class_sep=2.5, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=1111)

    model = SoftMaxRegression(lr=0.05, n_classes=10, max_iters=1000, alpha=1.0, penalty='l1')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    print('classification accuracy', np.mean(y_test == predictions))

def pca():
    X, y = make_classification(n_samples=1000, n_features=100,
                            n_informative=75, random_state=1111,
                            n_classes=2, class_sep=2.5)
    # ux = PCA(variance_ratio=0.99).fit_transform(X)
    # print(ux.shape)
    lda = LDA(n_components=5).fit(X,y)
    Xu = lda.transform(X)

    # plt.scatter(x=Xu[:,0][y==0], y=Xu[:,1][y==0], marker='s', color='blue', alpha=0.5, label="y=0")
    # plt.scatter(x=Xu[:,0][y==1], y=Xu[:,1][y==1], marker='o', color='red', alpha=0.5, label="y=1")
    
    # plt.xlabel('LD1')
    # plt.ylabel('LD2')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()


def tsne():
    X,y = load_iris(return_X_y=True)
    model = TSNE(lr=500.0, max_iters=5000, perplexity=50.)
    Xu = model.fit(X)
    plt.scatter(Xu[:,0], Xu[:,1], 20, y, marker='+')
    plt.show()

def gmm():
    X,y = load_iris(return_X_y=True)
    model = GMM(n_clusters=3, init='kmeans', torlerance=1e-6)
    model.fit(X)
    predictions = model.predict(X)
    print(predictions)

def kmeans():
    data = input_data.read_data_sets('/home/tanxr/data/mnist')
    X = data.train.images
    y = data.train.labels
    print(X.shape, y.shape, y[0:10])
    model=KMeans(n_clusters=10)
    model.fit(X)
    print(model.predict()[:20], y[0:20])

# regression()
# classification()
# mlp()
# softmax()
# pca()
# tsne()
# gmm()
kmeans()