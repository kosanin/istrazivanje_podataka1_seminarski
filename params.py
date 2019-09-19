import numpy as np

# parameters for MLP classifier
# Izvor : materijali sa vezbi
mlp_params = {'solver': ['sgd'],
      'learning_rate': ['adaptive'],
      'learning_rate_init': [0.01, 0.005, 0.002, 0.001],
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'hidden_layer_sizes': [(10, ), (10, 5), (10, 10)],
      'max_iter': [500]
          }

# parameters for SVM Classsifier
# Izvor : materijali sa vezbi
svm_params = {'C': [pow(2,x) for x in range(-6,10,2)],
      'max_iter' : [5000]}


log_params = {
      'solver' : ['lbfgs', 'sag', 'saga'],
      'max_iter' : [1000]
}
