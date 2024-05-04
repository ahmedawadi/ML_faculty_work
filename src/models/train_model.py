from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 

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