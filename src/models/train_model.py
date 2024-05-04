from sklearn.neighbors import KNeighborsClassifier 

def train_with_knn(features, labels, neighbors):

    #train the knn classifier with the passed number of neighbors
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(features, labels)

    #return the created model that will be used in the predction phase 
    return knn