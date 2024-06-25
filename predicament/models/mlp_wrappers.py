from skopt.space import Integer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import itertools
import numpy as np

#class ThreeHiddenLayerClassifier(BaseEstimator, ClassifierMixin):
#    def __init__(self, layer1=10, layer2=10, layer3=10, **kwargs):
#        self.layer1 = layer1
#        self.layer2 = layer2
#        self.layer3 = layer3
#        self.model = MLPClassifier(
#            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3],
#            **kwargs)

#    def fit(self, X, y, **kwargs):
#        if 'hidden_layer_sizes' in kwargs:
#            raise ValueError("Cannot override hidden_layer_sizes")
#        self.model.fit(X, y, **kwargs)
#        return self

#    def __getattr__(self, name):
#        if name == 'layer1':
#            return self.layer1
#        if name == 'layer2':
#            return self.layer2
#        if name == 'layer3':
#            return self.layer3
#        # `fitted_transformer`'s attributes are now accessible
#        return getattr(self.model, name)

class ThreeHiddenLayerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, layer1=10, layer2=None, layer3=None, activation='relu',
            solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, max_iter=200):
#        self.hidden_layer_sizes = hidden_layer_sizes
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.set_hidden_layer_sizes()
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.set_hidden_layer_sizes()
        self.model = None

    def get_classes(self):
        return self.model.classes_
    def set_classes(self, classes):
        self.model.classes_ = classes
    classes_ = property(get_classes, set_classes)

    def set_hidden_layer_sizes(self):
        if self.layer2 is None:
            self.hidden_layer_sizes=[self.layer1]
        elif self.layer3 is None:
            self.hidden_layer_sizes=[self.layer1, self.layer2]
        else:
            self.hidden_layer_sizes=[self.layer1, self.layer2, self.layer3]

    def fit(self, X, y):
        self.set_hidden_layer_sizes()
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                   activation=self.activation,
                                   solver=self.solver,
                                   alpha=self.alpha,
                                   batch_size=self.batch_size,
                                   learning_rate=self.learning_rate,
                                   learning_rate_init=self.learning_rate_init,
                                   max_iter=self.max_iter)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
#    def __setattr__(self, name, val):
#        reset = False
#        if name == 'layer1':
#            self.layer1 = val
#            reset = True
#        if name == 'layer2':
#            self.layer2 = val
#            reset = True
#        if name == 'layer3':
#            self.layer3 = val
#            reset = True
#        if reset:
#            setattr(
#                self.model,
#                'hidden_layer_sizes',
#                [self.layer1, self.layer2, self.layer3])
#        # `fitted_transformer`'s attributes are now accessible
#        return setattr(self.model, name, val)
        
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
        
    def __str__(self):
        return self.model.__str__()

    def __repr__(self):
        return self.model.__repr__()



# Want a classifier with some approximate functional relationship
# on the number of nodes per hidden layer 
#class BottleneckClassifier(BaseEstimator, ClassifierMixin):
#    def __init__(self, max_per_layer=20, min_per_layer=10, n_layers=3):

#        self.hidden_layer_initialiser = \
#            min_per_layer + np.arange(int(round(n_layers/2)))*(max_per_layer -

#    def fit(self, X, y, **kwargs):
#        if 'hidden_layer_sizes' in kwargs:
#            raise ValueError("Cannot override hidden_layer_sizes")
#        model = MLPClassifier(
#            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3],
#            **kwargs)
#        model.fit(X, y)
#        self.model = model
#        return self

#    def predict(self, X):
#        return self.model.predict(X)

#    def score(self, X, y):
#        return self.model.score(X, y)

