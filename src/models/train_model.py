from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_with_knn(features, labels, neighbors):

    #train the knn classifier with the passed number of neighbors
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(features, labels)

    #return the created model that will be used for the evaluation phase 
    return knn

def train_with_MLPClassifier(features, labels, hidden_layers, activation_function):
    
    #train MLP classifier model with the passed data and passed parameters
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_function)

    classifier.fit(features, labels)

    #return the created model that will be used for the evaluation phase 
    return classifier

def get_svc_best_params(features, labels):
    gird_search_parameters = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    }

    #choosing the best params for our svm model
    grid_search = GridSearchCV(SVC(), gird_search_parameters, verbose=5)
    grid_search.fit(features, labels)

    return grid_search.best_params_

def train_data_with_SVM_model_using_poly_kernel(features, labels, C, gamma, kernel_):
    model = SVC(C=C, gamma=gamma, kernel=kernel_)
    model.fit(features, labels)

    return model

