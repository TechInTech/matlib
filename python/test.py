# coding: utf-8
import numpy as np
from linearModel import LinearRegression, LogisticRegression, SoftMaxRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification, load_iris, make_blobs, load_diabetes
from sklearn.model_selection import train_test_split
from metrics import mean_squared_error
from mlp import MLP
from decomposition import PCA, LDA, TSNE
from gmm import GMM
from kmeans import KMeans
# from tensorflow.examples.tutorials.mnist import input_data
from knn import KNNClassifier, KNNRegressor
from naive_bayes import NaiveBayes
from gbdt import GBDT
from sklearn.metrics import roc_auc_score
from tree import LeastSquareLoss,LogisticLoss
from ada_boost import AdaBoost
from isolation_forest import IsolationForest
from rbm import RBM
from fm import FM

np.set_printoptions(precision=4)


def regression():
    # Generate a random regression problem
    X, y = make_regression(n_samples=10000, n_features=100,
                           n_informative=75, n_targets=1, noise=0.01, bias=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = LinearRegression(lr=0.001, max_iters=3000,
                             alpha=0.03, descent_grad=None)
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

    model = LogisticRegression(
        lr=0.01, max_iters=1000, penalty='l1', alpha=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(np.clip(predictions[:20], 0.001, 0.9), y_test[:20])
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

    model = SoftMaxRegression(lr=0.05, n_classes=10,
                              max_iters=1000, alpha=1.0, penalty='l1')
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
    lda = LDA(n_components=5).fit(X, y)
    Xu = lda.transform(X)

    # plt.scatter(x=Xu[:,0][y==0], y=Xu[:,1][y==0], marker='s', color='blue', alpha=0.5, label="y=0")
    # plt.scatter(x=Xu[:,0][y==1], y=Xu[:,1][y==1], marker='o', color='red',
    # alpha=0.5, label="y=1")

    # plt.xlabel('LD1')
    # plt.ylabel('LD2')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()


def tsne():
    X, y = load_iris(return_X_y=True)
    model = TSNE(lr=500.0, max_iters=5000, perplexity=50.)
    Xu = model.fit(X)
    plt.scatter(Xu[:, 0], Xu[:, 1], 20, y, marker='+')
    plt.show()


def gmm():
    X, y = load_iris(return_X_y=True)
    model = GMM(n_clusters=3, init='kmeans', torlerance=1e-6)
    model.fit(X)
    predictions = model.predict(X)
    print(predictions)


def kmeans():
    # data = input_data.read_data_sets('/home/tanxr/data/mnist')
    # X = data.train.images
    # y = data.train.labels
    # print(X.shape, y.shape, y[0:10])
    # model=KMeans(n_clusters=10)
    # model.fit(X)
    # assign = model.predict(X)
    # for i in range(10):
    #     indices = y==i
    # print(np.mean(assign[indices] == assign[np.where(indices==True)[0][0]]))

    X, y = make_blobs(centers=4, n_samples=500, n_features=2,
                      shuffle=True)
    clusters = len(np.unique(y))
    k = KMeans(n_clusters=clusters, max_iters=150, init='++')
    k.fit(X)
    k.predict()
    k.plot()


def knn():
    X, y = make_blobs(centers=4, n_samples=500, n_features=2,
                      shuffle=True)
    model = KNNClassifier(K=4)
    model.fit(X, y)
    res = model.predict(X)
    print(np.mean(res == y))


def naivebayes():
    # model = NaiveBayes()
    # x = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [
    #              2, 2], [2, 3], [2, 3], [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]])
    # y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    # assert(len(x) == len(y))
    # model.fit(x, y)
    # print (model.predict([2, 1]))
    X, y = make_classification(n_samples=1000, n_features=100,
                               n_informative=95, random_state=1111,
                               n_classes=2, class_sep=2.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=1111)
    model = NaiveBayes(dispersed=False)
    model.fit(X_train, y_train)
    res = []
    for x in X_test:
        res.append(model.predict(x))
    print(np.mean(res == y_test))


def gbdt(kind=1):
    if kind == 1:
        # Generate a random binary classification problem.
        X, y = make_classification(n_samples=350, n_features=15, n_informative=10,
                                random_state=1111, n_classes=2,
                                class_sep=1., n_redundant=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                            random_state=1111)

        model = GBDT(n_estimators=50, max_tree_depth=4,loss=LogisticLoss(),
                    max_features=8, lr=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(predictions)
        print(predictions.min())
        print(predictions.max())
        print('classification, roc auc score: %s'
            % roc_auc_score(y_test, predictions))
    else:
        # Generate a random regression problem
        X, y = make_regression(n_samples=500, n_features=5, n_informative=5,
                            n_targets=1, noise=0.05, random_state=1111,
                            bias=0.5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            random_state=1111)

        model = GBDT(n_estimators=50, max_tree_depth=5,
                                        max_features=3, min_samples_split=10, lr=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(y_test[:10], predictions[:10])
        print('regression, mse: %s'
            % mean_squared_error(y_test.flatten(), predictions.flatten()))

def adaBoost():
    X, y = make_classification(n_samples=350, n_features=15, n_informative=10,
                            random_state=1111, n_classes=2,
                            class_sep=1., n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                        random_state=1111)

    model = AdaBoost(n_estimators=10, max_tree_depth=5,
                max_features=8)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(predictions)
    print(predictions.min())
    print(predictions.max())
    print('classification, roc auc score: %s'
        % roc_auc_score(y_test, predictions))
    
def isolation_tree():
    rng = np.random.RandomState(42)

    n_features = 2
    # Generate train data
    X = 0.3 * rng.randn(1000, n_features)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * rng.randn(200, n_features)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, n_features))

    # fit the model
    clf = IsolationForest(n_choosen=64)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    print(np.mean(y_pred_train), np.mean(y_pred_test), y_pred_outliers)
    # plot the line, the samples, and the nearest vectors to the plane
    if n_features == 2:
        xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # plt.title("IsolationForest")
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
        c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([b1, b2, c],
                ["training observations",
                    "new regular observations", "new abnormal observations"],
                loc="upper left")
        plt.show()

def print_curve(errors, n=25):
    def moving_average(a):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(errors))
    plt.show()

def rbm():
    X = np.random.uniform(0, 1, (1500, 10))
    rbm = RBM(n_hidden=10, max_epochs=200, batch_size=10, lr=0.05)
    rbm.fit(X)
    print_curve(rbm.errors)

def fm():
    X, y = load_diabetes(return_X_y=True)
    clf = FM(K=2, max_iters=200, lr=0.05)
    clf.fit(X, y)
    print_curve(clf.losses, 10)

# regression()
# classification()
# mlp()
# softmax()
# pca()
# tsne()
# gmm()
# kmeans()
# knn()
# naivebayes()
# gbdt(1)
# adaBoost()
# isolation_tree()
# rbm()
fm()
