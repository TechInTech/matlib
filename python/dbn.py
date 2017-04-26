import numpy as np
from base import BaseEstimator
from rbm import RBM
from sklearn.preprocessing import OneHotEncoder
from metrics import batch_iterator
from abc import ABCMeta, abstractmethod
from activation import SigmoidActivationFunction, ReLUActivationFunction

class DBN(BaseEstimator):
    y_required = True

    def __init__(self, lr=0.05, max_epochs=100, l2_regularization=1.0, dropout_p=0.0,
                 hidden_layers=[128, 128], rbm_lr=0.05, rbm_epochs=100, batch_size=32, active_func='sigmoid'):
        self.lr = lr
        self.max_epochs = max_epochs
        self.l2_regularization = l2_regularization
        self.hidden_layers = hidden_layers
        self.rbm_lr = rbm_lr
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.rbm_layers = []
        self.n_classes = None
        self.dropout_p = dropout_p
        self.p = 1. - self.dropout_p
        self.W = None
        self.b = None
        self.total_layers = len(self.hidden_layers) + 1
        self.active_func = active_func
        if self.active_func == 'sigmoid':
            self.active_func_class = SigmoidActivationFunction
        elif self.active_func == 'relu':
            self.active_func_class = ReLUActivationFunction
        else:
            raise ValueError('unknown active func')

    def fit(self, X, y):
        self._setup_input(X, y)
        self.n_classes = self._set_n_classes()
        self.y = self.y.reshape(-1, 1)

        x_train = self.X
        for hidden_unit in self.hidden_layers:
            layer = RBM(n_hidden=hidden_unit, lr=self.rbm_lr,
                        batch_size=self.batch_size, max_epochs=self.rbm_epochs, active_func=self.active_func)
            layer.fit(x_train)
            self.rbm_layers.append(layer)
            x_train = layer.predict(x_train)
        self._fine_tuning(self.X, self.y)

    def _fine_tuning(self, X, y):
        n_unit_input = self.rbm_layers[-1].n_hidden
        self.W = np.random.randn(
            n_unit_input, self.n_classes) / np.sqrt(n_unit_input)
        self.b = np.random.randn(self.n_classes) / np.sqrt(self.n_classes)

        y = self._transform_y(y)

        for layer in self.rbm_layers:
            layer.W /= self.p
            layer.bias_h /= self.p

        self._sgd(X, y)

        for layer in self.rbm_layers:
            layer.W = layer.W * self.p
            layer.bias_h = layer.bias_h * self.p

    def _sgd(self, X, y):
        errors = np.zeros([self.n_samples, self.n_classes])
        accum_delta_w = [np.zeros(layer.W.shape) for layer in self.rbm_layers]
        accum_delta_w.append(np.zeros(self.W.shape))
        accum_delta_b = [np.zeros(layer.bias_h.shape)
                         for layer in self.rbm_layers]
        accum_delta_b.append(np.zeros(self.b.shape))

        for iter_num in range(1, self.max_epochs + 1):
            idx = np.random.permutation(self.n_samples)
            x_train, y_train = X[idx], y[idx]
            i = 0
            for batch_x, batch_y in batch_iterator(x_train, y_train, self.batch_size):
                for arr1, arr2 in zip(accum_delta_w, accum_delta_b):
                    arr1[:], arr2[:] = .0, .0
                for xi, yi in zip(batch_x, batch_y):
                    delta_w, delta_b, pred = self._back_propagation(xi, yi)
                    for layer in range(self.total_layers):
                        accum_delta_w[layer] += delta_w[layer]
                        accum_delta_b[layer] += delta_b[layer]
                    loss = self._compute_loss(yi, pred)
                    errors[i, :] = loss
                    i += 1

                for li, layer in enumerate(self.rbm_layers):
                    layer.W = (1 - (self.lr * self.l2_regularization) / self.n_samples) * \
                        layer.W - self.lr * accum_delta_w[li] / self.batch_size
                    layer.bias_h -= self.lr * \
                        accum_delta_b[li] / self.batch_size
                self.W = (1 - (self.lr * self.l2_regularization) / self.n_samples) * \
                    self.W - self.lr * accum_delta_w[-1] / self.batch_size
                self.b -= self.lr * accum_delta_b[-1] / self.batch_size
            
            error = np.mean(np.sum(errors, 1))
            print ">> Epoch %d finished \tANN training loss %f" % (iter_num, error)

    def _compute_activations(self, sample):
        """
        Compute output values of all layers.
        :param sample: array-like, shape = (n_features, )
        :return:
        """
        input_data = sample
        if self.dropout_p > 0:
            r = np.random.binomial(1, self.p, len(input_data))
            input_data *= r
        layers_activation = list()

        for rbm in self.rbm_layers:
            input_data = rbm.predict(input_data)
            if self.dropout_p > 0:
                r = np.random.binomial(1, self.p, len(input_data))
                input_data *= r
            layers_activation.append(input_data)

        # Computing activation of output layer
        input_data = self._compute_output_units(input_data)
        layers_activation.append(input_data)

        return layers_activation

    def _back_propagation(self, xi, yi):
        deltas = list()
        layer_weights = list()
        for rbm in self.rbm_layers:
            layer_weights.append(rbm.W)
        layer_weights.append(self.W)

        layers_activation = self._compute_activations(xi)
        activation_output_layer = layers_activation[-1]
        delta_output_layer = self._compute_output_layer_delta(
            yi, activation_output_layer)
        deltas.append(delta_output_layer)

        layers = range(len(self.hidden_layers));
        layers.reverse();
        delta_previous_layer = delta_output_layer
        for layer in layers:
            neuron_activations = layers_activation[layer]
            W = layer_weights[layer + 1]
            delta = np.dot(delta_previous_layer, W.T) * \
                self.active_func_class.prime(neuron_activations)
            deltas.append(delta)
            delta_previous_layer = delta
        deltas.reverse()

        layers_activation.pop()
        layers_activation.insert(0, xi)
        layer_gradient_weights, layer_gradient_bias = [], []
        for layer in range(self.total_layers):
            neuron_activations = layers_activation[layer]
            delta = deltas[layer]
            gradient_W = np.outer(neuron_activations, delta)
            layer_gradient_weights.append(gradient_W)
            layer_gradient_bias.append(delta)
        layer_gw = np.array(layer_gradient_weights)
        layer_gb = np.array(layer_gradient_bias)
        return layer_gw, layer_gb, activation_output_layer

    @abstractmethod
    def _compute_output_units(self, vector_visible_units):
        """
        Compute activations of output units.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
        return

    @abstractmethod
    def _compute_output_layer_delta(self, actual, pred):
        return

    @abstractmethod
    def _compute_loss(self, actual, pred):
        return
    
    @abstractmethod
    def _set_n_classes(self):
        return
    
    @abstractmethod
    def _transform_y(self, y):
        return

    def _predict(self, X):
        pred = []
        for x in X:
            p = x
            for rbm in self.rbm_layers:
                p = rbm.predict(p)
            pred.append(self._compute_output_units(p))
        return np.array(pred)


class DBNClassifier(DBN):

    def _transform_y(self, y):
        return np.array(OneHotEncoder().fit_transform(y).toarray())

    def _set_n_classes(self):
        return len(np.unique(self.y))

    def _compute_output_units(self, vector_visible_units):
        scores = np.dot(vector_visible_units, self.W) + self.b
        # get unnormalized probabilities
        exp_scores = np.exp(scores)
        # normalize them for each example
        return exp_scores / np.sum(exp_scores)

    def _compute_output_layer_delta(self, actual, pred):
        dscores = np.array(pred)
        dscores[np.where(actual == 1)] -= 1
        return dscores

    def _compute_loss(self, actual, pred):
        return -np.log(pred[np.where(actual == 1)])


class DBNRegressor(DBN):

    def _transform_y(self, y):
        return y
    
    def _set_n_classes(self):
        if len(self.y.shape) == 1:
            return 1
        else:
            return self.y.shape[1]

    def _compute_output_units(self, vector_visible_units):
        return np.dot(vector_visible_units, self.W) + self.b

    def _compute_output_layer_delta(self, actual, pred):
        return -(actual - pred)

    def _compute_loss(self, actual, pred):
        return (pred - actual)**2
